# SPDX-License-Identifier: Apache-2.0
# Port of DeepGEMM (MIT License, Copyright (c) 2025 DeepSeek; see licenses/LICENSE.deepgemm.txt) sm107_fp8_fp4_mega_moe.cuh dispatch / NVLink protocol / combine.
"""Tests of the dispatch / combine roles of the SM107 DeepGEMM-MegaMoE port (``kernel/dispatch.py``, ``kernel/combine.py``): dispatch, NVLink protocol, cleanup, combine.

* CPU: self-tests of the first-principles oracle ``_sm107_dg_verify`` (min-peeling rank pattern, UTCCP SF position, combine accumulation order).
* GPU, staged through the REAL kernel (``jit.get_or_compile_kernel``; ``mode_mask=1`` = dispatch only, ``mode_mask=5`` = dispatch + combine, GEMM roles off,
  ``dispatch_only_sync=True`` supplies the pull->cleanup ordering that the epilogue grid sync gives in the full kernel), EP1 in this process, EP2/EP4 with one process
  per rank over torch symmetric memory (like DeepGEMM's own tests):
    - dispatch only: per local expert the multiset of (source rank, token, top-k) in the L1 ring == the routed tokens (FP8 bytes, UTCCP-transformed SF words, top-k weight bits, source-rank
      peeling pattern, pool offsets), cumulative expert-receive stats, every non-self-cycling workspace word zero after the launch, for all six BLOCK_M regimes incl. masked routes
      (``topk_idx = -1``), M = 0 on some ranks, per-rank different M (mixed BLOCK_M in one collective step);
    - dispatch + combine with host-staged expert outputs: ``y`` bit-exact against the torch reference of DeepGEMM's combine order (ascending slot, FP32 RN accumulation, BF16 RN);
    - stress: N back-to-back launches with random per-rank GPU-side launch skew (``SM107_DG_STRESS_ITERS``, default 300; 10000 in the acceptance run), per-launch device-side ``y`` check.
  A missing flag / barrier shows up as ``unspecified launch failure`` after ``timeout_sec`` (20 s here) instead of a hang.
* The full kernel (all roles) against DeepGEMM's ``y`` lives in ``test_sm107_dg_dispatch_combine_full.py``.

Run: ``pytest tests/experimental/test_sm107_dg_dispatch_combine.py``; ``python test_sm107_dg_dispatch_combine.py --world 4 [--iters N]``.
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
import time

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _sm107_dg_verify as V  # noqa: E402

E, TOPK, H, I, TMAX = 384, 6, 5120, 2304, 1920
NGPU = torch.cuda.device_count() if torch.cuda.is_available() else 0


def _is_sm107() -> bool:
    return (
        NGPU > 0
        and torch.cuda.get_device_capability(0) == (10, 7)
        and torch.cuda.get_device_properties(0).multi_processor_count == 212
    )


gpu = pytest.mark.skipif(
    not _is_sm107(), reason="needs an SM107 (cc 10.7, 212 SMs) device"
)


# ---------------------------------------------------------------------------------------------------------------------
# CPU: oracle self-tests
# ---------------------------------------------------------------------------------------------------------------------
def test_ref_peel_is_a_permutation_with_round_robin_structure():
    rng = np.random.default_rng(0)
    for _ in range(200):
        R = int(rng.choice([1, 2, 4, 8]))
        counts = [int(c) for c in rng.integers(0, 9, size=R)]
        if sum(counts) == 0:
            continue
        seen = {}
        for q in range(sum(counts)):
            r, i = V.ref_peel(counts, q)
            assert 0 <= i < counts[r]
            assert (r, i) not in seen
            seen[(r, i)] = q
        assert len(seen) == sum(counts)
        # per source rank the list order is preserved
        for r in range(R):
            order = [seen[(r, i)] for i in range(counts[r])]
            assert order == sorted(order)


def test_ref_peel_known_cases():
    # equal counts: pure round robin
    assert [V.ref_peel([2, 2, 2], q)[0] for q in range(6)] == [0, 1, 2, 0, 1, 2]
    # unequal: min-length rounds first, then the longer lists alone
    assert [V.ref_peel([1, 3], q) for q in range(4)] == [(0, 0), (1, 0), (1, 1), (1, 2)]
    assert [V.ref_peel([0, 2, 1], q) for q in range(3)] == [(1, 0), (2, 0), (1, 1)]


@pytest.mark.parametrize("block_m", (16, 32, 64, 128, 192, 240))
def test_utccp_sf_pos_is_injective_per_block(block_m):
    # K:184-188: (idx & ~127) + (idx & 31) * 4 + ((idx >> 5) & 3): distinct positions inside one block, dense for BLOCK_M = 128
    pos = [V.utccp_sf_pos(t, block_m) for t in range(block_m)]
    assert len(set(pos)) == block_m
    if block_m == 128:
        assert sorted(pos) == list(range(128))
    assert (
        V.utccp_sf_pos(0, block_m) == 0
        and V.utccp_sf_pos(1, block_m) == 4
        and (block_m < 33 or V.utccp_sf_pos(32, block_m) == 1)
    )


def test_reference_combine_order_and_rounding():
    # ascending slot, FP32 accumulation of BF16 values, one BF16 RN at the end (not a BF16 add chain)
    big, small = np.float32(2.0**24), np.float32(1.0)
    vals = torch.tensor([[big], [small], [small], [-big]], dtype=torch.float32).to(
        torch.bfloat16
    )  # bf16 can hold 2^24 and 1.0 exactly
    rows = vals.view(torch.int16).numpy().view(np.uint16).reshape(4, 1)
    out = V.reference_combine(rows, np.array([True, True, True, True]))
    # fp32: ((2^24 + 1) rounds to 2^24) ... -> 0 after the last add; a different order would give 2.0
    assert out.view(np.int16).item() == 0
    assert V.reference_combine(rows, np.array([True, False, True, False])).shape == (1,)


# ---------------------------------------------------------------------------------------------------------------------
# GPU: staged dispatch / combine through the real kernel
# ---------------------------------------------------------------------------------------------------------------------
def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _cfg(world, M, bm, S=0):
    from flashinfer.experimental.sm107_deepgemm_megamoe.heuristics import (
        get_sm107_kernel_config,
    )

    return get_sm107_kernel_config(
        world, E, TMAX, M, TOPK, H, I, S, num_sms=212, activation_clamp=10.0, block_m=bm
    )


def _layout(cfg):
    from flashinfer.experimental.sm107_deepgemm_megamoe.kernel import layout as L

    return L.buffer_layout(
        hidden=H,
        intermediate_hidden=I,
        num_ranks=cfg.num_ranks,
        num_experts=E,
        num_max_tokens_per_rank=cfg.num_max_tokens_per_rank,
        num_topk=TOPK,
        num_ring_tokens=cfg.num_ring_tokens,
        num_sf_ring_tokens=cfg.num_sf_ring_tokens,
        num_shared_experts=cfg.num_shared_experts,
    )


def _ring_wraps(world, case) -> bool:
    """True when some rank would need more pool blocks than ring blocks (dispatch-only mode has no consumer that frees ring slots: the launch would stall)."""
    ms = case["ms"]
    cfgs = [
        _cfg(world, ms[r], case.get("bm", 0), case.get("shared", 0))
        for r in range(world)
    ]
    inputs = V.make_inputs(
        cfgs[0],
        ms,
        seed=case.get("seed", 0),
        masked_ratio=case.get("masked", 0.0),
        skew=case.get("skew", "none"),
    )
    for r in range(world):
        ent = V.expected_expert_entries(cfgs[r], inputs, r)
        if (
            sum(V.ceil_div(len(e), cfgs[r].block_m) for e in ent)
            > cfgs[r].num_ring_blocks
        ):
            return True
    return False


def _write_inputs(cfg, lay, inp, buf):
    a, b = lay["input_token"].offset, lay["input_topk_weights"].end
    host = np.zeros(b, dtype=np.uint8)
    V.write_inputs(cfg, host, inp, lay)
    buf.view(torch.uint8)[a:b].copy_(torch.from_numpy(host[a:b]))


def _stage_combine(lay, M, rows_bf16, buf):
    cr = lay["combine_token"]
    for k in range(
        rows_bf16.shape[1]
    ):  # TOPK routed slots (+ the shared-expert slot kNumTopk when S=1)
        o = cr.offset + k * cr.token_bytes * cr.num_tokens
        buf.view(torch.uint8)[o : o + M * cr.token_bytes].copy_(
            torch.from_numpy(
                np.ascontiguousarray(rows_bf16[:, k, :]).view(np.uint8).reshape(-1)
            )
        )


def _expected_y(inp, rows_bf16):
    M = inp.num_tokens
    mask = torch.from_numpy(inp.topk_idx >= 0)
    if (
        rows_bf16.shape[1] > TOPK
    ):  # fused shared expert: slot kNumTopk is ALWAYS active and summed LAST
        mask = torch.cat(
            [mask, torch.ones(M, rows_bf16.shape[1] - TOPK, dtype=torch.bool)], dim=1
        )
    rows = (
        torch.from_numpy(rows_bf16.astype(np.int16))
        .view(torch.bfloat16)
        .to(torch.float32)
    )
    acc = torch.zeros(M, H, dtype=torch.float32)
    for k in range(rows_bf16.shape[1]):
        acc = torch.where(mask[:, k, None], acc + rows[:, k, :], acc)
    return acc.to(torch.bfloat16).view(torch.int16)


def _run_cases(rank, world, make_buffer, barrier, cases):
    """Run ``cases`` on this rank (all ranks run the same list). Returns a list of error strings."""
    from flashinfer.experimental.sm107_deepgemm_megamoe import jit

    errs_all = []
    for ci, case in enumerate(cases):
        ms = case["ms"]
        M = ms[rank]
        S = int(case.get("shared", 0))
        cfg = _cfg(world, M, case.get("bm", 0), S)
        lay = _layout(cfg)
        buf, sym = make_buffer(lay.total_bytes)
        inputs = V.make_inputs(
            cfg,
            ms,
            seed=case.get("seed", 0),
            masked_ratio=case.get("masked", 0.0),
            skew=case.get("skew", "none"),
        )
        _write_inputs(cfg, lay, inputs[rank], buf)
        combine = bool(case.get("combine", False))
        iters = int(case.get("iters", 1))
        sleep_us = int(case.get("sleep_us", 0))
        exp_y = None
        if combine:
            rng = np.random.default_rng(case.get("seed", 0) + 1234 + rank)
            rows = rng.standard_normal((M, TOPK + S, H)).astype(np.float32) * 3.0
            rows_bf16 = (
                torch.from_numpy(rows)
                .to(torch.bfloat16)
                .view(torch.int16)
                .numpy()
                .view(np.uint16)
            )
            if M:
                _stage_combine(lay, M, rows_bf16, buf)
                exp_y = _expected_y(inputs[rank], rows_bf16).reshape(-1).cuda()
        stats = torch.randint(
            0, 100, (cfg.num_experts_per_rank,), dtype=torch.int32, device="cuda"
        )
        stats_before = stats.cpu().numpy().copy()
        runner = jit.get_or_compile_kernel(
            cfg,
            mode_mask=5 if combine else 1,
            roles="auto",
            dispatch_only_sync=True,
            use_disk_cache=False,
            timeout_sec=20,
        )  # spin waits trap after 20 s: a missing flag is a failure, not a hang
        y = torch.full((max(M, 1) * H,), 0x5A5A, dtype=torch.int16, device="cuda")
        bad = torch.zeros((), dtype=torch.int64, device="cuda")
        torch.cuda.synchronize()
        barrier()
        rng_skew = np.random.default_rng(case.get("seed", 0) * 7 + rank)
        t0 = time.time()
        for _it in range(iters):
            if combine:
                y.fill_(0x5A5A)
            if sleep_us:
                torch.cuda._sleep(int(rng_skew.uniform(0, sleep_us) * 2000))
            runner.call_raw(y.data_ptr(), stats.data_ptr(), M, sym, 0, 0, 0, 0)
            if combine and M:
                bad += (y[: M * H] != exp_y).sum()
        torch.cuda.synchronize()
        dt = time.time() - t0
        barrier()
        buf_np = buf.view(torch.uint8).cpu().numpy()
        stats_after = stats.cpu().numpy()
        delta = stats_after.astype(np.int64) - stats_before
        errs = []
        if np.any(delta % iters):
            errs.append(f"rank {rank}: stats delta not a multiple of the launch count")
            delta = delta // iters
        else:
            delta = delta // iters
        errs += V.verify_dispatch(
            cfg, rank, buf_np, inputs, np.zeros_like(stats_before), delta
        )
        if combine and int(bad.item()):
            errs.append(
                f"rank {rank}: y != reference combine: {int(bad.item())} mismatching elements accumulated over {iters} launches"
            )
        status = "OK" if not errs else "FAIL"
        print(
            f"  [rank {rank}] case {ci} world={world} M={ms} bm={cfg.block_m} S={S} masked={case.get('masked', 0.0)} skew={case.get('skew', 'none')} combine={combine} iters={iters}: {status} "
            f"({dt * 1e3 / iters:.2f} ms/launch, rows verified {getattr(V.verify_dispatch, 'last_rows_checked', '?')})",
            flush=True,
        )
        for e in errs[:6]:
            print("     ", e, flush=True)
        errs_all += [f"case {ci}: {e}" for e in errs]
        barrier()
    return errs_all


def _worker(rank, world, port, cases):
    import torch.distributed as dist
    import torch.distributed._symmetric_memory as symm_mem

    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        world_size=world,
        rank=rank,
        device_id=torch.device(f"cuda:{rank}"),
    )
    group = dist.new_group(list(range(world)))

    cache = {}

    def make_buffer(
        nbytes,
    ):  # ONE symmetric allocation per worker, re-zeroed between cases (all ranks are past the previous case: barrier at its end)
        if nbytes in cache:
            buf, sym = cache[nbytes]
            buf.zero_()
            torch.cuda.synchronize()
            group.barrier()
            return buf, sym
        buf = symm_mem.empty(nbytes, dtype=torch.int8, device="cuda")
        handle = symm_mem.rendezvous(buf, group=group)
        buf.zero_()
        torch.cuda.synchronize()
        group.barrier()
        base = buf.data_ptr()
        cache[nbytes] = (
            buf,
            (base, tuple(int(p) - base for p in handle.buffer_ptrs), rank, world),
        )
        return cache[nbytes]

    errs = _run_cases(rank, world, make_buffer, lambda: group.barrier(), cases)
    flag = torch.tensor([len(errs)], dtype=torch.int32, device="cuda")
    dist.all_reduce(flag, group=group)
    dist.destroy_process_group()
    assert not errs, "\n".join(errs[:10])
    assert int(flag.item()) == 0


def _run(world, cases):
    cases = [c for c in cases if not _ring_wraps(world, c)]
    if world == 1:
        cache = {}

        def make_buffer(nbytes):
            if nbytes not in cache:
                cache[nbytes] = torch.zeros(nbytes, dtype=torch.int8, device="cuda")
            buf = cache[nbytes]
            buf.zero_()
            torch.cuda.synchronize()
            return buf, (buf.data_ptr(), (0,), 0, 1)

        errs = _run_cases(0, 1, make_buffer, lambda: None, cases)
        assert not errs, "\n".join(errs[:10])
    else:
        with pytest.MonkeyPatch.context() as mp:
            if (
                "CUDA_MODULE_LOADING" not in os.environ
            ):  # the spawned workers inherit it (it must be set before their first CUDA call): lazy module loading at the first
                mp.setenv(
                    "CUDA_MODULE_LOADING", "EAGER"
                )  # launch while another device's persistent kernel runs can deadlock
            torch.multiprocessing.spawn(
                _worker, args=(world, _free_port(), cases), nprocs=world, join=True
            )


BMS = (16, 32, 64, 128, 192, 240)


def _cases_ep1():
    cs = []
    for bm in BMS:
        for M in sorted({1, max(bm - 3, 1), 3 * bm + 5}):
            cs.append(dict(ms=[M], bm=bm, skew="single", seed=bm))
        cs.append(dict(ms=[bm + 7], bm=bm, masked=0.4, seed=bm + 1))
    cs += [dict(ms=[0]), dict(ms=[8], seed=3), dict(ms=[300], masked=0.3, seed=4)]
    cs += [
        dict(ms=[bm - 3 if bm > 3 else 1], bm=bm, combine=True, masked=0.2, seed=bm + 9)
        for bm in BMS
    ]
    cs += [
        dict(ms=[200], combine=True, seed=5),
        dict(ms=[1], combine=True, seed=6),
        dict(ms=[0], combine=True),
    ]
    cs += [
        dict(ms=[M], combine=True, shared=1, masked=0.2, seed=20 + M)
        for M in (1, 13, 61, 200)
    ] + [dict(ms=[0], combine=True, shared=1)]  # S=1: slot 6 summed last
    return cs


def _cases_multi(world):
    ms_equal, ms_mixed = [64] * world, ([300, 64, 1, 17] * 2)[:world]
    cs = [
        dict(ms=ms_equal, seed=1),
        dict(ms=ms_mixed, seed=2),
        dict(ms=ms_equal, masked=0.3, seed=3),
        dict(ms=[8] * world, seed=4),
    ]
    cs += [dict(ms=([0, 5, 64, 0] * 2)[:world], seed=5), dict(ms=[0] * world)]
    cs += [
        dict(ms=[1] + [64] * (world - 1), skew="single", seed=6)
    ]  # all routes concentrated: some ranks send nothing
    for bm in BMS:  # every BLOCK_M regime on every rank, then BLOCK_M mixed across ranks via per-rank M
        cs.append(dict(ms=[max(bm - 3, 1)] * world, bm=bm, masked=0.2, seed=bm))
    cs.append(
        dict(ms=[1, 17, 64, 300][:world], seed=7)
    )  # heuristic BLOCK_M differs between ranks
    cs += [
        dict(ms=ms_equal, combine=True, masked=0.1, seed=8),
        dict(ms=ms_mixed, combine=True, seed=9),
        dict(ms=([0, 5, 64, 0] * 2)[:world], combine=True, seed=10),
    ]
    cs += [
        dict(ms=ms_equal, combine=True, shared=1, masked=0.1, seed=18),
        dict(ms=ms_mixed, combine=True, shared=1, seed=19),
        dict(ms=([0, 5, 64, 0] * 2)[:world], combine=True, shared=1, seed=21),
    ]
    return cs


@gpu
def test_dispatch_combine_ep1():
    _run(1, _cases_ep1())


@gpu
@pytest.mark.skipif(NGPU < 2, reason="needs 2 GPUs")
def test_dispatch_combine_ep2():
    _run(2, _cases_multi(2))


@gpu
@pytest.mark.skipif(NGPU < 4, reason="needs 4 GPUs")
def test_dispatch_combine_ep4():
    _run(4, _cases_multi(4))


@gpu
@pytest.mark.skipif(NGPU < 4, reason="needs 4 GPUs")
def test_stress_ep4_rank_skew():
    iters = int(os.environ.get("SM107_DG_STRESS_ITERS", "300"))
    cs = [
        dict(ms=[64] * 4, iters=iters, sleep_us=400, seed=11),
        dict(ms=[64] * 4, combine=True, masked=0.1, iters=iters, sleep_us=400, seed=12),
        dict(
            ms=[300, 64, 1, 17],
            combine=True,
            iters=max(iters // 5, 10),
            sleep_us=400,
            seed=13,
        ),
    ]  # mixed BLOCK_M across ranks
    _run(4, cs)


@gpu
@pytest.mark.skipif(NGPU < 2, reason="needs 2 GPUs")
def test_stress_ep2_rank_skew():
    iters = int(os.environ.get("SM107_DG_STRESS_ITERS", "300"))
    _run(
        2,
        [
            dict(
                ms=[64, 64],
                combine=True,
                masked=0.1,
                iters=iters,
                sleep_us=400,
                seed=14,
            ),
            dict(
                ms=[1, 300],
                combine=True,
                iters=max(iters // 5, 10),
                sleep_us=400,
                seed=15,
            ),
        ],
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", type=int, default=1)
    ap.add_argument(
        "--iters",
        type=int,
        default=0,
        help="run the skew stress with this many launches instead of the functional cases",
    )
    args = ap.parse_args()
    if args.iters:
        os.environ["SM107_DG_STRESS_ITERS"] = str(args.iters)
        (test_stress_ep4_rank_skew if args.world == 4 else test_stress_ep2_rank_skew)()
    else:
        {
            1: test_dispatch_combine_ep1,
            2: test_dispatch_combine_ep2,
            4: test_dispatch_combine_ep4,
        }[args.world]()
    print("OK")
