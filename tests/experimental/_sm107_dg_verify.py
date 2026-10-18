# SPDX-License-Identifier: Apache-2.0
# Port of DeepGEMM (MIT License, Copyright (c) 2025 DeepSeek; see licenses/LICENSE.deepgemm.txt) sm107_fp8_fp4_mega_moe.cuh / dispatch + combine semantics (test oracles).
"""Test-side oracles of the dispatch / combine roles (dispatch, NVLink protocol, cleanup, combine): pure numpy/torch, no cutlass import.

* :func:`make_inputs` / :func:`write_inputs`: random routed tokens per rank (FP8 bytes, packed UE8M0 SF words, int64 top-k ids incl. -1,
  FP32 weights) written into the symmetric-buffer input regions.
* :func:`verify_dispatch`: first-principles multiset oracle of the post-dispatch state of one rank:
  per local expert the rows of the L1 ring (FP8 bytes, SF words in the UTCCP-transformed MN-major layout, top-k weight, source
  metadata) must be exactly the tokens routed to it, the source-rank pattern must follow DeepGEMM's iterative min-peeling, the
  pool offsets must follow ``ceil(count / BLOCK_M)`` accumulation, cumulative stats must grow by the counts and every workspace
  counter except the self-cycling ones must be zero again.
* :func:`reference_combine`: DeepGEMM's combine order (ascending top-k slot, FP32 accumulation via ``add.rn.f32.bf16``, BF16 RN).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from flashinfer.experimental.sm107_deepgemm_megamoe.kernel import layout as L

TOPK = 6


def ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def utccp_sf_pos(t_in_block: int, block_m: int) -> int:
    """``transform_sf_token_idx`` for a token index inside ONE block (K:184-188 with ``t / BLOCK_M == 0``)."""
    idx = t_in_block % block_m
    return (idx & ~127) + (idx & 31) * 4 + ((idx >> 5) & 3)


def ref_peel(counts: Sequence[int], slot: int) -> Tuple[int, int]:
    """Direct transcription of K:567-615: position ``slot`` of an expert -> (source rank, index in that rank's list)."""
    remaining = list(counts)
    offset = 0
    while True:
        active = [r for r in range(len(remaining)) if remaining[r] > 0]
        n = len(active)
        assert n > 0, "slot beyond the expert's token count"
        length = min(remaining[r] for r in active)
        if slot < length * n:
            return active[slot % n], offset + slot // n
        slot -= length * n
        offset += length
        remaining = [max(x - length, 0) for x in remaining]


@dataclass
class RankInputs:
    x: np.ndarray  # (M, H) uint8 (FP8 E4M3 bytes)
    x_sf: np.ndarray  # (M, H/128) int32 packed UE8M0
    topk_idx: np.ndarray  # (M, TOPK) int64, -1 = masked
    topk_w: np.ndarray  # (M, TOPK) float32

    @property
    def num_tokens(self) -> int:
        return self.x.shape[0]


def make_inputs(
    cfg,
    num_tokens_per_rank: Sequence[int],
    seed: int = 0,
    masked_ratio: float = 0.0,
    skew: str = "none",
) -> List[RankInputs]:
    """Random inputs of every rank.  ``skew``: 'none' | 'hot' (all tokens route to the experts of rank 0 / few experts) |
    'single' (every token routes to the same top-6 experts)."""
    rng = np.random.default_rng(seed)
    out = []
    E = cfg.num_experts
    for _r, M in enumerate(num_tokens_per_rank):
        x = rng.integers(0, 256, size=(M, cfg.hidden), dtype=np.uint8)
        x_sf = rng.integers(
            -(2**31), 2**31, size=(M, cfg.hidden // 128), dtype=np.int64
        ).astype(np.int32)
        scores = rng.standard_normal((M, E)).astype(np.float32)
        if skew == "hot":
            scores[:, : max(cfg.num_experts_per_rank, 8)] += (
                50.0  # experts of rank 0 only
            )
        elif skew == "single":
            scores[:, :TOPK] += 100.0
        topk = np.argsort(-scores, axis=1)[:, :TOPK].astype(np.int64)
        if masked_ratio > 0:
            mask = rng.random(topk.shape) < masked_ratio
            topk = np.where(mask, -1, topk)
        w = rng.random((M, TOPK)).astype(np.float32)
        w = np.where(topk < 0, 0.0, w).astype(np.float32)
        out.append(RankInputs(x, x_sf, topk, w))
    return out


def write_inputs(cfg, buf_np_view, inp: RankInputs, lay) -> None:
    """Write one rank's inputs into the int8/uint8 numpy (or torch->numpy) view of its symmetric buffer (rows beyond M untouched)."""
    M = inp.num_tokens
    if M == 0:
        return

    def reg(name):
        r = lay[name]
        return r.offset, r.token_bytes

    o, tb = reg("input_token")
    buf_np_view[o : o + M * tb] = inp.x.reshape(-1)
    o, tb = reg("input_sf")
    buf_np_view[o : o + M * tb] = inp.x_sf.view(np.uint8).reshape(-1)
    o, tb = reg("input_topk_idx")
    buf_np_view[o : o + M * tb] = inp.topk_idx.view(np.uint8).reshape(-1)
    o, tb = reg("input_topk_weights")
    buf_np_view[o : o + M * tb] = inp.topk_w.view(np.uint8).reshape(-1)


def expected_expert_entries(cfg, inputs: Sequence[RankInputs], dst_rank: int):
    """For every local expert of ``dst_rank``: list of (src_rank, token, topk) routed to it."""
    epr = cfg.num_experts_per_rank
    ent = [[] for _ in range(epr)]
    for r, inp in enumerate(inputs):
        idx = inp.topk_idx
        rows, cols = np.nonzero((idx >= dst_rank * epr) & (idx < (dst_rank + 1) * epr))
        for t, k in zip(rows.tolist(), cols.tolist(), strict=False):
            ent[int(idx[t, k]) - dst_rank * epr].append((r, t, k))
    return ent


# self-cycling signals that legitimately stay non-zero after a launch
def _selfcycling_ranges(num_ranks: int):
    return [
        (
            L.SIGNALS_GRID_SYNC_COUNT,
            L.SIGNALS_L1_TASK_COUNT,
        ),  # grid_sync_count[4], nvl_barrier_counter, nvl_barrier_signals[2]
        (
            L.SIGNALS_COMBINE_READY_GRID_IDX,
            L.SIGNALS_COMBINE_READY_GRID_IDX + 8 * L.NUM_MAX_RANKS,
        ),
        (L.SIGNALS_PEER_GRID_IDX, L.SIGNALS_PEER_GRID_IDX + 8 * L.NUM_MAX_RANKS),
    ]


def check_workspace_clean(cfg, buf_np: np.ndarray) -> List[str]:
    """Every byte of ``MegaMoESignals`` outside the self-cycling words must be zero after a launch (cleanup invariant)."""
    errs = []
    head = buf_np[: L.SIGNALS_SIZEOF]
    mask = np.ones(L.SIGNALS_SIZEOF, dtype=bool)
    for a, b in _selfcycling_ranges(cfg.num_ranks):
        mask[a:b] = False
    bad = np.nonzero((head != 0) & mask)[0]
    if bad.size:
        first = int(bad[0])
        errs.append(
            f"workspace not clean: {bad.size} non-zero bytes outside the self-cycling words, first at offset {first} (value {int(head[first])})"
        )
    return errs


def verify_dispatch(
    cfg,
    rank: int,
    buf_np: np.ndarray,
    inputs: Sequence[RankInputs],
    stats_before: Optional[np.ndarray],
    stats_after: Optional[np.ndarray],
    max_errors: int = 12,
) -> List[str]:
    """Verify the post-dispatch state of rank ``rank`` (``buf_np``: uint8 view of its whole symmetric buffer)."""
    errs: List[str] = []
    R, E, EPR, H, BM = (
        cfg.num_ranks,
        cfg.num_experts,
        cfg.num_experts_per_rank,
        cfg.hidden,
        cfg.block_m,
    )
    NRT, NSRT, SFBM, NRB = (
        cfg.num_ring_tokens,
        cfg.num_sf_ring_tokens,
        cfg.sf_block_m,
        cfg.num_ring_blocks,
    )
    lay = L.buffer_layout(hidden=cfg.hidden, intermediate_hidden=cfg.intermediate_hidden, num_ranks=R, num_experts=E,
                          num_max_tokens_per_rank=cfg.num_max_tokens_per_rank, num_topk=cfg.num_topk,
                          num_ring_tokens=NRT, num_sf_ring_tokens=NSRT, num_shared_experts=cfg.num_shared_experts)  # fmt: skip
    ent = expected_expert_entries(cfg, inputs, rank)
    counts_per_src = [[0] * R for _ in range(EPR)]
    for le in range(EPR):
        for r, _t, _k in ent[le]:
            counts_per_src[le][r] += 1
    counts = [len(ent[le]) for le in range(EPR)]
    total_blocks = sum(ceil_div(c, BM) for c in counts)
    if total_blocks > cfg.num_ring_blocks:
        errs.append(
            f"test config wraps the ring ({total_blocks} pool blocks > {cfg.num_ring_blocks} ring blocks): dispatch-only cannot run this"
        )
        return errs

    ws = lay.workspace
    meta = (
        buf_np[
            ws.token_src_metadata_offset : ws.token_src_metadata_offset
            + ws.num_max_pool_tokens * 12
        ]
        .view(np.uint32)
        .reshape(-1, 3)
    )
    l1t = lay["l1_token"]
    l1_tok = buf_np[l1t.offset : l1t.offset + NRT * H].reshape(NRT, H)
    l1s = lay["l1_sf"]
    nsf = H // 128
    l1_sf = (
        buf_np[l1s.offset : l1s.offset + NSRT * 4 * nsf]
        .view(np.int32)
        .reshape(nsf, NSRT)
    )
    l1w = lay["l1_topk_weights"]
    l1_w = buf_np[l1w.offset : l1w.offset + NRT * 4].view(np.uint32)

    pool_off = 0
    n_checked = 0
    for le in range(EPR):
        cnt = counts[le]
        if cnt:
            exp_sorted = sorted(ent[le])
            got = []
            for p_in_e in range(cnt):
                p = pool_off * BM + p_in_e
                got.append((int(meta[p, 0]), int(meta[p, 1]), int(meta[p, 2])))
            # (1) multiset of (src_rank, token, topk)
            if sorted(got) != exp_sorted:
                missing = set(exp_sorted) - set(got)
                extra = set(got) - set(exp_sorted)
                errs.append(
                    f"rank {rank} expert {le}: metadata multiset mismatch (count {cnt}); missing {sorted(missing)[:4]} extra {sorted(extra)[:4]}"
                )
            # (2) the source-rank pattern follows the iterative min-peeling
            ref_ranks = [ref_peel(counts_per_src[le], q)[0] for q in range(cnt)]
            got_ranks = [g[0] for g in got]
            if ref_ranks != got_ranks:
                errs.append(
                    f"rank {rank} expert {le}: source-rank peeling pattern mismatch (first diff at {next(i for i, (a, b) in enumerate(zip(ref_ranks, got_ranks, strict=False)) if a != b)})"
                )
            # (3) row contents
            for p_in_e in range(cnt):
                if len(errs) >= max_errors:
                    return errs
                p = pool_off * BM + p_in_e
                r, t, k = got[p_in_e]
                if r >= R or t >= inputs[r].num_tokens or k >= TOPK:
                    errs.append(
                        f"rank {rank} expert {le} pos {p_in_e}: invalid metadata {(r, t, k)}"
                    )
                    continue
                ring = p % NRT
                if not np.array_equal(l1_tok[ring], inputs[r].x[t]):
                    errs.append(
                        f"rank {rank} expert {le} pos {p_in_e} (src {(r, t, k)}): token bytes differ"
                    )
                    continue
                if int(l1_w[ring]) != int(inputs[r].topk_w[t, k].view(np.uint32)):
                    errs.append(
                        f"rank {rank} expert {le} pos {p_in_e} (src {(r, t, k)}): weight bits differ"
                    )
                ring_block = (p // BM) % NRB
                sfpos = ring_block * SFBM + utccp_sf_pos(p_in_e % BM, BM)
                if not np.array_equal(l1_sf[:, sfpos], inputs[r].x_sf[t]):
                    errs.append(
                        f"rank {rank} expert {le} pos {p_in_e} (src {(r, t, k)}): SF words differ at sf pos {sfpos}"
                    )
                n_checked += 1
        pool_off += ceil_div(cnt, BM)

    # (4) cumulative stats
    if stats_before is not None and stats_after is not None:
        want = stats_before.astype(np.int64) + np.array(counts, dtype=np.int64)
        if not np.array_equal(stats_after.astype(np.int64), want):
            errs.append(
                f"rank {rank}: cumulative_local_expert_recv_stats mismatch (first diff expert {int(np.nonzero(stats_after != want)[0][0])})"
            )
    # (5) counters clean
    errs += check_workspace_clean(cfg, buf_np)
    verify_dispatch.last_rows_checked = n_checked
    verify_dispatch.last_counts = counts
    return errs


def reference_combine(rows: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """DeepGEMM combine order for one token: ``rows`` (slots, H) bf16-as-uint16, ``mask`` (slots,) bool; ascending slot, FP32
    accumulation of bf16 values (exact bf16->f32 widening + RN add), final RN cast to bf16.  Returns (H,) uint16."""
    import torch

    acc = torch.zeros(rows.shape[1], dtype=torch.float32)
    for s in range(rows.shape[0]):
        if mask[s]:
            v = (
                torch.from_numpy(rows[s].astype(np.int16))
                .view(torch.bfloat16)
                .to(torch.float32)
            )
            acc = acc + v
    return acc.to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16)
