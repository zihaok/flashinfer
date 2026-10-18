# SPDX-License-Identifier: Apache-2.0
# Adapted from the DeepGEMM Rubin SM107 MegaMoE kernel (deepseek-ai/DeepGEMM, MIT License, Copyright (c) 2025 DeepSeek; see licenses/LICENSE.deepgemm.txt; branch rubin-megamoe-a7a9277-1016 @3ec0eb4) for the FlashInfer CuteDSL port.
# Test helper of test_sm107_dg_dispatch_combine_full.py (dispatch / combine roles): FULL-kernel multi-rank STRESS test (EP>=2, one process per rank, torch.multiprocessing.spawn like DeepGEMM's own tests) of the port's kernel
# (mode_mask=7: the dispatch + combine roles, the GEMM roles and the skeleton) against DeepGEMM's `y` on the SAME
# symmetric buffers / weights / inputs.
#
# Per case: K input "sets" (set s has M_s tokens per rank, random routing, random masked slots); DeepGEMM computes the reference y of every (rank, set) with ALL
# ranks running the same set; then N back-to-back port launches per rank (no host sync) in a pseudo-random order of sets with rank-dependent GPU-side launch skew
# (`torch.cuda._sleep`, cycles drawn per (rank, launch) from a fixed table up to S_SKEW_CYCLES), every launch's y compared bit-exactly with the reference ON THE DEVICE
# (a flag per launch).  Modes: `same` (all ranks run the same set in launch i), `chaos` (every rank picks its own set per launch: mixed M / BLOCK_M / routing combinations
# across ranks in the same launch).  After the loop: cumulative stats == sum of the expected per-launch counts, workspace clean, DeepGEMM re-run on the buffer the port left.
# A hang is a failure, not a stall: the kernel's spin loops trap after timeout_sec (S_TIMEOUT_SEC, default here 60 s) and the whole run is bounded by the wrapper's timeout.
# env: R_RANKS=4 R_E=32 R_S=0|1 (fused shared expert) R_TOPK=6 R_H=5120 R_I=2304 R_TMAX=1920 R_MASKED=0.0 R_SEED=0 R_TAGMODE=epoch|gridid R_REGS=a,b,c R_BM=0 R_CLAMP=10
#      S_SETM=256,64,1000   (M per set, same on every rank)   S_MODE=same|chaos|both   S_LAUNCHES=10000   S_SKEW_CYCLES=4000000   S_CHECK_EVERY=1000   S_NY=4
#      S_SETM items: `M` plus optional flag letters: x = FULLY MASKED set (M>0 but every topk_idx = -1), h = hot routing (every token -> the same `topk` experts 0..topk-1),
#      s = hot_spread (every token -> `topk` hot experts spread over the ranks: expert j % R * El + j // R), p = perm (source rank r -> the `topk` first experts of rank (r+1) % R),
#      z = all-zero hidden states (identical rows: the vLLM warmup pattern).  Examples: S_SETM=64,300x,8192h,8192sz   R_ROUTE=rand|hot|spread|perm (default for sets without a route flag)
#      R_XZERO=1 (all sets: x = 0)   R_FULLMASK_RANKS=2,3 (those ranks run EVERY set fully masked)   R_ROUTE_SHUF=1 (hot/spread/perm: random slot order per token)
#      R_POISON_STRICT=1 : STRICT poison mode: a host barrier (synchronize + dist.barrier) before AND after the fill of every launch of either kernel, so no peer can be in flight while a rank poisons; this mode also poisons
#      `src_token_topk_idx` (written REMOTELY by the source ranks' dispatch early in a launch, so it can only be poisoned under a barrier) next to `token_src_metadata` (local only: poisoned in both modes).  Without it launches run
#      back-to-back with no per-iteration barrier (stress mode: only regions that no peer can write outside the launch window are poisoned).
#      R_SEPBUF=1 : the PORT gets its OWN symmetric buffer (DeepGEMM keeps the first one): the port can never read DeepGEMM's stale bytes.
#      R_CANARY=<launch>[:<rank>] : canary self-test of the harness: ONE staged input (the top-k weights of token 0) is corrupted in ONE launch of ONE rank; the run PASSES only if exactly that launch (and no other) is flagged as mismatching.
#      R_POISON=1|<hex byte> : before EVERY launch (DeepGEMM references, warm-ups, loop, re-run) fill every symmetric-buffer region the kernels WRITE before they READ (dispatch rings, L1/L2 activations + SF, shared L2, ALL combine slots)
#      with this byte (the harness idea: without it DeepGEMM and the port share one buffer with identical stale data, so a kernel that writes a result to a wrong address can still read back the right stale bytes). Inputs / workspace / host-staged shared SF are kept.
import os
import sys
import time
import types

os.environ.setdefault("CUTE_DSL_ARCH", "sm_107a")
os.environ.setdefault("CUDA_MODULE_LOADING", "EAGER")


def env_int(name, default):
    return int(os.environ.get(name, default))


def cast_fp8_for_mega_moe(x):
    """tests/test_mega_moe.py::_cast_fp8_for_mega_moe -> (fp8, sf, sf_tma)"""
    import torch
    from deep_gemm.utils import per_token_cast_to_fp8

    x_fp8, x_sf = per_token_cast_to_fp8(
        x, use_ue8m0=True, gran_k=32, use_packed_ue8m0=True
    )
    mn, packed_sf_k = x_sf.shape
    x_sf_tma = torch.empty_strided(
        (mn, packed_sf_k), (1, -(-mn // 4) * 4), dtype=x_sf.dtype, device=x_sf.device
    )
    x_sf_tma.copy_(x_sf)
    return x_fp8, x_sf, x_sf_tma


def to_shared_sf_layout(sf, block_m, num_max_sf_tokens):
    """tests/test_mega_moe.py::_to_shared_mega_moe_sf_layout (UTCCP-transposed inside every BLOCK_M block; depends on the BLOCK_M of the launch)."""
    import torch

    num_tokens, packed_sf_k = sf.shape
    aligned_block_m = -(-block_m // 128) * 128
    result = torch.empty_strided(
        (num_max_sf_tokens, packed_sf_k),
        (1, num_max_sf_tokens),
        dtype=sf.dtype,
        device=sf.device,
    )
    result.zero_()
    if num_tokens == 0:
        return result
    m = torch.arange(num_tokens, device=sf.device)
    blk, mi = m // block_m, m % block_m
    dst = blk * aligned_block_m + (mi // 128) * 128 + (mi % 32) * 4 + (mi % 128) // 32
    result[dst] = sf[m]
    return result


def copy_fp8_sf(dst, src, num_tokens):
    """tests/test_mega_moe.py::_copy_fp8_sf"""
    if num_tokens == 0:
        return
    if dst.shape == src.shape:
        dst.copy_(src)
        return
    dst[:num_tokens].copy_(src)
    if num_tokens < dst.shape[0]:
        dst[num_tokens:].copy_(src[-1:].expand(dst.shape[0] - num_tokens, -1))


def worker(local_rank, nranks):
    import traceback

    try:
        ok = main(local_rank, nranks)
    except Exception:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(2)
    sys.stdout.flush()
    os._exit(0 if ok else 1)


def hash32(*vals):
    h = 0x9E3779B1
    for v in vals:
        h = ((h ^ (int(v) & 0xFFFFFFFF)) * 0x85EBCA6B) & 0xFFFFFFFF
        h ^= h >> 13
        h = (h * 0xC2B2AE35) & 0xFFFFFFFF
        h ^= h >> 16
    return h


def skew_cycles(rank, it, seed, skew_max):
    """GPU-side launch skew of (rank, launch) in SM cycles: mostly none/short, sometimes long (<= skew_max)."""
    if skew_max <= 0:
        return 0
    r = hash32(rank, it, seed, 17) % 16
    if r < 8:
        return 0
    if r < 11:
        return min(20000, skew_max)
    if r < 13:
        return min(200000, skew_max)
    if r == 13:
        return min(1000000, skew_max)
    if r == 14:
        return min(4000000, skew_max)
    return skew_max


def pick_set(mode, rank, it, nsets, seed):
    if nsets == 1:
        return 0
    if mode == "same":
        return hash32(it, seed, 5) % nsets
    return hash32(rank, it, seed, 5) % nsets


def main(local_rank, nranks):
    import numpy as np
    import torch
    import torch.distributed as dist

    import deep_gemm
    from deep_gemm.utils import per_token_cast_to_fp4, per_token_cast_to_fp8
    from deep_gemm.utils.dist import init_dist

    from flashinfer.experimental.sm107_deepgemm_megamoe import jit
    from flashinfer.experimental.sm107_deepgemm_megamoe.heuristics import (
        get_sm107_kernel_config,
    )
    from flashinfer.experimental.sm107_deepgemm_megamoe.kernel import layout as L

    E = env_int("R_E", 32)
    topk = env_int("R_TOPK", 6)
    Hh = env_int("R_H", 5120)
    I = env_int("R_I", 2304)
    Tmax = env_int("R_TMAX", 1920)
    masked = float(os.environ.get("R_MASKED", "0"))
    seed = env_int("R_SEED", 0)
    clamp = float(os.environ.get("R_CLAMP", "10.0"))
    tag_mode = os.environ.get(
        "R_TAGMODE", "epoch"
    )  # the port's shipped default; R_TAGMODE=gridid = DeepGEMM-faithful parity runs
    regs_env = os.environ.get("R_REGS", "")
    regs_override = tuple(int(v) for v in regs_env.split(",")) if regs_env else None
    bm_force = env_int("R_BM", 0)
    import re

    setm, setflags = [], []
    for item in os.environ.get("S_SETM", "256,64,1000").split(","):
        mt = re.fullmatch(r"(\d+)([a-z]*)", item.strip())
        assert mt, f"bad S_SETM item {item!r}"
        setm.append(int(mt.group(1)))
        setflags.append(mt.group(2))
    route_default = os.environ.get("R_ROUTE", "rand")
    assert route_default in ("rand", "hot", "spread", "perm", "hot_spread"), (
        route_default
    )
    route_default = "spread" if route_default == "hot_spread" else route_default
    xzero_all = env_int("R_XZERO", 0)
    route_shuf = env_int("R_ROUTE_SHUF", 0)
    fullmask_ranks = [
        int(v) for v in os.environ.get("R_FULLMASK_RANKS", "").split(",") if v.strip()
    ]
    pv = os.environ.get("R_POISON", "0").strip().lower()
    poison_val = None if pv in ("", "0") else (0xFF if pv == "1" else int(pv, 16))
    poison_strict = env_int("R_POISON_STRICT", 0)
    sepbuf = env_int("R_SEPBUF", 0)
    canary_launch, canary_rank = -1, 0
    _cn = os.environ.get("R_CANARY", "")
    if _cn:
        _p = _cn.split(":")
        canary_launch = int(_p[0])
        canary_rank = int(_p[1]) if len(_p) > 1 and _p[1] else 0
    modes = (
        ["same", "chaos"]
        if os.environ.get("S_MODE", "both") == "both"
        else [os.environ.get("S_MODE", "chaos")]
    )
    nlaunch = env_int("S_LAUNCHES", 10000)
    skew_max = env_int("S_SKEW_CYCLES", 4000000)
    check_every = env_int("S_CHECK_EVERY", 1000)
    ny = env_int("S_NY", 4)
    K = len(setm)
    NUM_SHARED = env_int(
        "R_S", 0
    )  # fused shared experts (S=1: ONE merged FP8 shared expert of width S*I, replicated on every rank)

    rank, nranks, group = init_dist(local_rank, nranks)
    dev = "cuda"
    El = E // nranks
    assert E % nranks == 0
    pre = f"[r{rank}]"
    print(
        f"{pre} package: {jit.__file__}  EP{nranks} E={E} (per rank {El}) topk={topk} Tmax={Tmax} sets M={setm} flags={setflags} route={route_default} xzero={xzero_all} fullmask_ranks={fullmask_ranks} modes={modes} launches={nlaunch} skew<={skew_max} cycles "
        f"tag_mode={tag_mode} regs_override={regs_override} masked={masked} S={NUM_SHARED}",
        flush=True,
    )
    buffer = deep_gemm.get_symm_buffer_for_mega_moe(
        group,
        E,
        Tmax,
        topk,
        Hh,
        I,
        num_shared_experts=NUM_SHARED,
        mma_type="fp8xfp4",
        activation="swiglu",
    )
    bufP = (
        deep_gemm.get_symm_buffer_for_mega_moe(
            group,
            E,
            Tmax,
            topk,
            Hh,
            I,
            num_shared_experts=NUM_SHARED,
            mma_type="fp8xfp4",
            activation="swiglu",
        )
        if sepbuf
        else buffer
    )  # the PORT's buffer
    base = bufP.buffer.data_ptr()
    peer_offsets = [int(p) - base for p in bufP.handle.buffer_ptrs]
    assert len(peer_offsets) == nranks and peer_offsets[rank] == 0, peer_offsets
    symb = types.SimpleNamespace(base_ptr=base, peer_offsets=peer_offsets, group=group)

    def cast_w(w):
        g, n, k = w.shape
        q = torch.empty((g, n, k // 2), device=dev, dtype=torch.int8)
        sf = torch.empty((g, n, k // 32), device=dev, dtype=torch.float)
        for i in range(g):
            q[i], sf[i] = per_token_cast_to_fp4(w[i], use_ue8m0=True, gran_k=32)
        sf = deep_gemm.transform_sf_into_required_layout(sf, n, k, (1, 32), g)
        return q, sf

    torch.manual_seed(seed + 7919 * rank)
    l1w = torch.randn((El, 2 * I, Hh), dtype=torch.bfloat16, device=dev)
    l2w = torch.randn((El, Hh, I), dtype=torch.bfloat16, device=dev)
    l1 = cast_w(l1w)
    l2 = cast_w(l2w)
    del l1w, l2w
    tl1, tl2 = deep_gemm.transform_weights_for_mega_moe(l1, l2, activation="swiglu")
    SHARED = None
    if NUM_SHARED:  # ONE merged FP8 shared expert of width S*I (plain 2D weights + UE8M0 SF), same weights on every rank (tests/test_mega_moe.py:149-172)
        IS = I * NUM_SHARED
        g0 = torch.Generator(device=dev)
        g0.manual_seed(seed + 555)
        sh1 = cast_fp8_for_mega_moe(
            torch.randn((IS * 2, Hh), dtype=torch.bfloat16, device=dev, generator=g0)
        )[0::2]
        sh2 = cast_fp8_for_mega_moe(
            torch.randn((Hh, IS), dtype=torch.bfloat16, device=dev, generator=g0)
        )[0::2]
        SHARED = deep_gemm.transform_weights_for_mega_moe(sh1, sh2, activation="swiglu")
    pargs = (
        ()
        if SHARED is None
        else (SHARED[0][0], SHARED[0][1], SHARED[1][0], SHARED[1][1])
    )  # shared-weight pointer arguments of the runner
    dg_kw = (
        {}
        if SHARED is None
        else dict(shared_l1_weights=SHARED[0], shared_l2_weights=SHARED[1])
    )

    # ---- input sets, DeepGEMM references ----
    sets = []
    for s, M in enumerate(setm):
        g = torch.Generator(device=dev)
        g.manual_seed(seed * 1000003 + 7919 * rank + 104729 * s + 13)
        st = dict(M=M)
        flags = setflags[s]
        route = {"h": "hot", "s": "spread", "p": "perm"}.get(
            next((c for c in flags if c in "hsp"), ""), route_default
        )
        fully_masked = ("x" in flags) or (rank in fullmask_ranks)
        if M > 0:
            x = torch.randn((M, Hh), dtype=torch.bfloat16, device=dev, generator=g)
            if xzero_all or "z" in flags:
                x.zero_()  # identical rows (vLLM's MegaMoE warmup feeds all-zero hidden states)
            scores = torch.randn((M, E), dtype=torch.float, device=dev, generator=g)
            tw, ti = torch.topk(scores, topk, dim=-1, largest=True, sorted=False)
            if (
                route != "rand"
            ):  # skewed routing: every token routes to the same fixed experts (weights stay the random top-k scores)
                if route == "hot":
                    ids = list(range(topk))
                elif route == "spread":
                    ids = [(j % nranks) * El + (j // nranks) for j in range(topk)]
                else:  # perm: source rank r -> the first experts of destination rank (r + 1) % R
                    ids = [(((rank + 1) % nranks) * El + j) % E for j in range(topk)]
                assert len(set(ids)) == topk and all(0 <= i < E for i in ids), ids
                ti = (
                    torch.tensor(ids, dtype=ti.dtype, device=dev)
                    .expand(M, topk)
                    .clone()
                )
                if route_shuf:
                    order = torch.argsort(
                        torch.rand((M, topk), device=dev, generator=g), dim=-1
                    )
                    ti = torch.gather(ti, 1, order)
            if masked > 0:
                rm = torch.rand(ti.shape, dtype=torch.float, device=dev, generator=g)
                ti.masked_fill_(rm < masked, -1)
                tw.masked_fill_(ti < 0, 0)
            if fully_masked:  # M > 0 tokens, every route masked: only the shared expert (S=1) contributes
                ti.fill_(-1)
                tw.zero_()
            xq, xs = per_token_cast_to_fp8(
                x, use_ue8m0=True, gran_k=32, use_packed_ue8m0=True
            )
            st.update(x=xq, x_sf=xs, ti=ti, tw=tw)
            flat = ti.reshape(-1)
            hist = torch.bincount(flat[flat >= 0].long(), minlength=E).to(torch.int64)
        else:
            hist = torch.zeros(E, dtype=torch.int64, device=dev)
        st["hist"] = hist
        cfg_s = get_sm107_kernel_config(
            nranks,
            E,
            Tmax,
            M,
            topk,
            Hh,
            I,
            NUM_SHARED,
            num_sms=212,
            activation_clamp=clamp,
            block_m=bm_force,
        )
        st["cfg"], st["bm"] = cfg_s, cfg_s.block_m
        st["lay"] = L.buffer_layout(
            hidden=Hh,
            intermediate_hidden=I,
            num_ranks=nranks,
            num_experts=E,
            num_max_tokens_per_rank=cfg_s.num_max_tokens_per_rank,
            num_topk=topk,
            num_ring_tokens=cfg_s.num_ring_tokens,
            num_sf_ring_tokens=cfg_s.num_sf_ring_tokens,
            num_shared_experts=NUM_SHARED,
        )
        if (
            NUM_SHARED
            and bm_force == 0
            and hasattr(deep_gemm, "get_block_m_for_mega_moe")
        ):  # the shared SF staging depends on BLOCK_M: both kernels must pick the same variant
            dg_bm = deep_gemm.get_block_m_for_mega_moe(
                nranks, E, buffer.num_max_tokens_per_rank, M, topk, "fp8xfp4"
            )
            assert dg_bm == cfg_s.block_m, (
                f"BLOCK_M differs: DeepGEMM {dg_bm} vs port {cfg_s.block_m}"
            )
        sets.append(st)
    hist_local = torch.stack([st["hist"] for st in sets])  # (K, E)
    gathered = [torch.zeros_like(hist_local) for _ in range(nranks)]
    dist.all_gather(gathered, hist_local, group=group)
    HIST = (
        torch.stack(gathered).cpu().numpy()
    )  # (R, K, E) tokens routed to each global expert by (source rank, set)

    POISON_REGIONS = (
        "shared_l2_token",
        "shared_l2_sf",
        "l1_token",
        "l1_sf",
        "l1_topk_weights",
        "l2_token",
        "l2_sf",
        "combine_token",
    )

    def poison(st, buf, ws=False):
        """stream-ordered fill of every kernel-written region of the symmetric buffer `buf` (R_POISON); inputs, the host-staged shared L1 SF and the workspace signals (counters / flags) are never touched.
        `token_src_metadata` (written by this rank's own dispatch pull, read by its own epilogue) is always poisoned; `src_token_topk_idx` (written remotely by the source ranks) only with ws=True (strict mode)."""
        if poison_val is None:
            return
        raw = buf.buffer.view(torch.uint8)
        for name in POISON_REGIONS:
            r = st["lay"][name]
            if r.nbytes:
                raw[r.offset : r.end].fill_(poison_val)
        wsl = st["lay"].workspace
        raw[
            wsl.token_src_metadata_offset : wsl.token_src_metadata_offset
            + wsl.num_max_pool_tokens * L.TOKEN_SRC_METADATA_BYTES
        ].fill_(poison_val)
        if ws:
            raw[
                wsl.src_token_topk_idx_offset : wsl.src_token_topk_idx_offset
                + E * wsl.num_max_tokens_per_rank * 4
            ].fill_(poison_val)

    def prepare_launch(st, buf):
        """poison `buf` right before a launch of either kernel (strict mode: under a per-iteration host barrier, before AND after the fill)"""
        if poison_val is None:
            return
        if poison_strict:
            torch.cuda.synchronize()
            dist.barrier(group)
            poison(st, buf, ws=True)
            torch.cuda.synchronize()
            dist.barrier(group)
        else:
            poison(st, buf)

    def load_set(st):
        M = st["M"]
        if M > 0:
            for b_ in (buffer, bufP) if sepbuf else (buffer,):
                b_.x[:M].copy_(st["x"])
                b_.x_sf[:M].copy_(st["x_sf"])
                b_.topk_idx[:M].copy_(st["ti"])
                b_.topk_weights[:M].copy_(st["tw"])
                if NUM_SHARED:  # host staging of the shared-expert L1 token SF, laid out for THIS launch's BLOCK_M (both kernels read it)
                    copy_fp8_sf(
                        b_.shared_l1_acts_sf,
                        to_shared_sf_layout(
                            st["x_sf"], st["bm"], b_.shared_l1_acts_sf.shape[0]
                        ),
                        M,
                    )

    for s, st in enumerate(sets):
        M = st["M"]
        load_set(st)
        y = torch.empty((M, Hh), dtype=torch.bfloat16, device=dev)
        dist.barrier(group)
        if bm_force:
            pass  # DG_MEGA_MOE_SM107_BLOCK_M is exported by the wrapper (applies to DeepGEMM)
        prepare_launch(st, buffer)
        deep_gemm.fp8_fp4_mega_moe(
            y,
            tl1,
            tl2,
            buffer,
            activation="swiglu",
            activation_clamp=clamp,
            fast_math=True,
            **dg_kw,
        )
        torch.cuda.synchronize()
        dist.barrier(group)
        st["y_ref"] = y
        nan_ref = int(torch.isnan(y.float()).sum().item()) if M else 0
        print(
            f"{pre} DeepGEMM reference set {s} (M={M}): absmax={y.float().abs().max().item() if M else 0:.3g} nan={nan_ref}",
            flush=True,
        )

    # ---- port kernels (one per distinct config) ----
    runners = []
    for s, st in enumerate(sets):
        cfg = st["cfg"]
        lay = L.buffer_layout(
            hidden=Hh,
            intermediate_hidden=I,
            num_ranks=nranks,
            num_experts=E,
            num_max_tokens_per_rank=cfg.num_max_tokens_per_rank,
            num_topk=topk,
            num_ring_tokens=cfg.num_ring_tokens,
            num_sf_ring_tokens=cfg.num_sf_ring_tokens,
            num_shared_experts=NUM_SHARED,
        )
        assert lay.total_bytes == buffer.buffer.nbytes == bufP.buffer.nbytes, (
            lay.total_bytes,
            buffer.buffer.nbytes,
            bufP.buffer.nbytes,
        )
        runner = jit.get_or_compile_kernel(
            cfg,
            mode_mask=7,
            roles="auto",
            use_disk_cache=False,
            tag_mode=tag_mode,
            regs_override=regs_override,
            timeout_sec=float(os.environ.get("S_TIMEOUT_SEC", "60")),
        )
        runners.append(runner)
        print(
            f"{pre} set {s} M={st['M']}: cfg bm={cfg.block_m} stages={cfg.num_stages} regs={cfg.regs_dispatch}/{cfg.regs_non_epilogue}/{cfg.regs_epilogue} ring_blocks={cfg.num_ring_blocks} "
            f"-> {runner.name} (compile {runner.compile_seconds:.1f}s)",
            flush=True,
        )

    sig = bufP.buffer.view(torch.uint8)[: L.SIGNALS_SIZEOF]
    fields = (("expert_send_count", L.SIGNALS_EXPERT_SEND_COUNT, 8 * 2048), ("expert_recv_count", L.SIGNALS_EXPERT_RECV_COUNT, 8 * 2048),
              ("expert_recv_count_sum", L.SIGNALS_EXPERT_RECV_COUNT_SUM, 8 * 512), ("l1_full_count", L.SIGNALS_L1_FULL_COUNT, 4 * (1 << 20)),
              ("l1_empty_count", L.SIGNALS_L1_EMPTY_COUNT, 4 * (1 << 20)), ("l2_full_mask", L.SIGNALS_L2_FULL_MASK, 8 * (1 << 20)),
              ("l2_empty_count", L.SIGNALS_L2_EMPTY_COUNT, 4 * (1 << 20)), ("task_counts", L.SIGNALS_L1_TASK_COUNT, 32),
              ("shared_l2_full_count", L.SIGNALS_SHARED_L2_FULL_COUNT, 4 * (1 << 15)))  # fmt: skip

    all_ok = True
    stats = torch.zeros(El, dtype=torch.int32, device=dev)
    exp_stats = np.zeros(El, dtype=np.int64)
    total_launches = 0
    for mode in modes:
        ok = True
        N = nlaunch
        choice = np.array(
            [
                [pick_set(mode, r, it, K, seed) for r in range(nranks)]
                for it in range(N)
            ],
            dtype=np.int64,
        )  # (N, R)
        per_launch_stats = np.zeros(El, dtype=np.int64)
        for r in range(nranks):
            per_launch_stats += HIST[r, choice[:, r], rank * El : (rank + 1) * El].sum(
                axis=0
            )
        Ys = [
            torch.full(
                (max(st["M"] for st in sets), Hh),
                float("nan"),
                dtype=torch.bfloat16,
                device=dev,
            )
            for _ in range(ny)
        ]
        bad_flags = torch.zeros(N, dtype=torch.int32, device=dev)
        # warm-up (synchronous) launch of every set: catches gross failures before the long loop
        for s, st in enumerate(sets):
            load_set(st)
            M = st["M"]
            y = Ys[0][:M]
            y.fill_(float("nan"))
            dist.barrier(group)
            prepare_launch(st, bufP)
            runners[s](y, stats, M, symb, tl1[0], tl1[1], tl2[0], tl2[1], *pargs)
            torch.cuda.synchronize()
            eq = (
                torch.equal(y.view(torch.int16), st["y_ref"].view(torch.int16))
                if M
                else True
            )
            print(
                f"{pre}   [{mode}] warm-up launch set {s} (M={M}): y == DeepGEMM y bit-exact: {eq}",
                flush=True,
            )
            ok &= eq
            exp_stats += HIST[:, s, rank * El : (rank + 1) * El].sum(axis=0)
            total_launches += 1
        dist.barrier(group)
        cur = None
        t0 = time.time()
        for it in range(N):
            s = int(choice[it, rank])
            st = sets[s]
            M = st["M"]
            if s != cur:
                load_set(st)
                cur = s
            y = Ys[it % ny][:M]
            if M:
                y.fill_(float("nan"))
            prepare_launch(st, bufP)
            cyc = skew_cycles(rank, it, seed, skew_max)
            if cyc:
                torch.cuda._sleep(cyc)
            if (
                it == canary_launch and rank == canary_rank
            ):  # harness canary: corrupt ONE staged input of ONE launch (the top-k weights of token 0); the next launch re-stages the set
                assert M > 0, "canary launch has an empty set on the canary rank"
                bufP.topk_weights[0, :].mul_(4.0)
                cur = None
            runners[s](y, stats, M, symb, tl1[0], tl1[1], tl2[0], tl2[1], *pargs)
            if M:
                bad_flags[it] = (
                    y.view(torch.int16) != st["y_ref"].view(torch.int16)
                ).any()
            if check_every and (it + 1) % check_every == 0:
                torch.cuda.synchronize()
                nb = int(bad_flags[: it + 1].sum().item())
                print(
                    f"{pre}   [{mode}] {it + 1}/{N} launches done, {time.time() - t0:.1f}s, launches with y != DeepGEMM y so far: {nb}",
                    flush=True,
                )
        torch.cuda.synchronize()
        dt = time.time() - t0
        bad_its = torch.nonzero(bad_flags).flatten().tolist()
        exp_stats += per_launch_stats
        total_launches += N
        print(
            f"{pre}   [{mode}] {N} launches in {dt:.1f}s ({dt / N * 1e3:.3f} ms/launch incl. skew); mismatching launches: {len(bad_its)} {bad_its[:12]}",
            flush=True,
        )
        if canary_launch >= 0:
            expect = [canary_launch] if rank == canary_rank else []
            canary_ok = bad_its == expect
            print(
                f"{pre}   [{mode}] CANARY (top-k weights of token 0 corrupted at launch {canary_launch} on rank {canary_rank}): mismatching launches {bad_its[:12]}, expected {expect}: "
                f"{'DETECTED' if canary_ok else 'NOT DETECTED / unexpected mismatches'}",
                flush=True,
            )
            ok &= canary_ok
        else:
            ok &= not bad_its
        dist.barrier(group)
        stats_ok = bool(np.array_equal(stats.cpu().numpy().astype(np.int64), exp_stats))
        dirty = []
        for name, off, nbytes in fields:
            nz = int((sig[off : off + nbytes] != 0).sum().item())
            if nz:
                dirty.append((name, nz))
        nvl = sig[16:28].view(torch.int32).tolist()
        print(
            f"{pre}   [{mode}] cumulative stats == expected over {total_launches} launches: {stats_ok}; workspace clean: {not dirty} {dirty if dirty else ''}; "
            f"grid_sync_count={sig[0:16].view(torch.int32).tolist()} nvl[counter, sig0, sig1]={nvl}",
            flush=True,
        )
        ok &= stats_ok and not dirty
        flag = torch.tensor([1 if ok else 0], dtype=torch.int32, device=dev)
        dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=group)
        good = bool(flag.item())
        if rank == 0:
            print(
                f"CASE STRESS EP{nranks} E={E} sets M={setm} flags={setflags} route={route_default} mode={mode} launches={N} skew<={skew_max} tag={tag_mode} regs={regs_override} masked={masked} S={NUM_SHARED} BM(force)={bm_force} poison={'off' if poison_val is None else hex(poison_val)}{'+strict' if poison_strict else ''} sepbuf={sepbuf} canary={canary_launch}: {'PASS' if good else 'FAIL'}",
                flush=True,
            )
        all_ok &= good
    # DeepGEMM on the buffer the port left behind (clean-state compatibility)
    st = sets[0]
    load_set(st)
    y2 = torch.empty((st["M"], Hh), dtype=torch.bfloat16, device=dev)
    dist.barrier(group)
    prepare_launch(st, buffer)
    deep_gemm.fp8_fp4_mega_moe(
        y2,
        tl1,
        tl2,
        buffer,
        activation="swiglu",
        activation_clamp=clamp,
        fast_math=True,
        **dg_kw,
    )
    torch.cuda.synchronize()
    eq2 = (
        torch.equal(y2.view(torch.int16), st["y_ref"].view(torch.int16))
        if st["M"]
        else True
    )
    print(
        f"{pre}   DeepGEMM re-run on the buffer left by the port: y == first DeepGEMM y: {eq2}",
        flush=True,
    )
    flag = torch.tensor([1 if (eq2 and all_ok) else 0], dtype=torch.int32, device=dev)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=group)
    good = bool(flag.item())
    if rank == 0:
        print(f"STRESS EP{nranks} RESULT: {'PASS' if good else 'FAIL'}", flush=True)
    dist.barrier(group)
    return good


if __name__ == "__main__":
    import torch.multiprocessing as mp

    nranks = env_int(
        "R_RANKS", 4
    )  # LOCAL ranks (processes) of this node; multi-node: WORLD_SIZE = #nodes, RANK = node index, MASTER_ADDR / MASTER_PORT from the launcher (DeepGEMM init_dist convention)
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    if "WORLD_SIZE" not in os.environ:
        os.environ["MASTER_PORT"] = os.environ.get(
            "R_PORT", str(29900 + os.getpid() % 400)
        )
    mp.spawn(worker, args=(nranks,), nprocs=nranks, join=True)
