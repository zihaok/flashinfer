# SPDX-License-Identifier: Apache-2.0
# Port of DeepGEMM (MIT License, Copyright (c) 2025 DeepSeek; see licenses/LICENSE.deepgemm.txt) sm107_fp8_fp4_mega_moe.cuh / GEMM-core unit tests.
"""GPU unit tests of the GEMM core (single CTA pair, the real ctx classes + the REAL TMA-A/B, MMA and scheduler roles, harness epilogue).

* ``test_gemm_core_bit_exact``: TMA-A/B + UTCCP + K=64 ``mxf8f6f4`` 2-CTA UMMA + TMEM against an exact float64 torch reference (data chosen so that every
  product / partial sum is exactly representable in FP32: the tensor-core result must be BIT-EXACT) for all six BLOCK_M, valid_m in {1, 7, 15..17, 31, 33,
  50, 63, 100, 127, bm - 1, bm} (N = align16(valid_m) in 16..240), L1-type (K = H) and L2-type (K = I) tasks, both accumulator stages, task-ring wrap.
* ``test_scheduler_sequence_and_gemm``: the REAL scheduler warp (claims, warm-up waves, strict L2/L1 alternation, ``create_task`` warp scan, publication
  through ``st.async``) for ONE CTA pair (num_sms = 2): the published task sequence equals the Python model :func:`replay_utils.ref_task_sequence` and every
  task's GEMM result is bit-exact.

Requires a Rubin (sm_107) device and the CUTLASS DSL; skipped otherwise.  Commands: ``pytest tests/experimental/test_sm107_dg_gemm_core_unit.py -x -q``.
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

pytestmark = [
    pytest.mark.arch_rubin,
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
        reason="needs an SM107 (Rubin) device",
    ),
]

import replay_utils as R  # noqa: E402
from replay_utils import (
    decode_fp4,
    decode_fp8,
    e8m0_scale,
    make_fp4_codes,
    make_fp8_bytes,
    pack_fp4,
    pack_sf_words,
    utccp_pos,
)  # noqa: E402,F401


def _imports():
    import cutlass
    import cutlass.cute as cute
    import cuda.bindings.driver as cuda
    from cutlass.cute.runtime import make_ptr as host_make_ptr

    from flashinfer.experimental.sm107_deepgemm_megamoe.heuristics import (
        get_sm107_kernel_config,
    )
    from flashinfer.experimental.sm107_deepgemm_megamoe.kernel import layout as L

    import _sm107_dg_gemm_core_harness as H

    return cutlass, cute, cuda, host_make_ptr, get_sm107_kernel_config, L, H


def run_gemm_tasks(bm, K1, N1, E_l, tasks, seed=0, nblk=4, quiet=True):
    """tasks: list of (phase, expert, n_cluster, pool_block, valid_m)."""
    cutlass, cute, cuda, host_make_ptr, get_sm107_kernel_config, L, H = _imports()
    dev = "cuda"
    gen = torch.Generator().manual_seed(seed)
    Hh, I = K1, N1 // 2  # L1: N = 2I, K = H ; L2: N = H, K = I
    cfg = get_sm107_kernel_config(1, E_l, 1920, 1, 2, Hh, I, 0, num_sms=212, block_m=bm)
    BM, SFBM = cfg.block_m, cfg.sf_block_m
    ring, sfring = cfg.num_ring_tokens, cfg.num_sf_ring_tokens
    assert cfg.num_ring_blocks >= nblk
    lay = L.buffer_layout(
        hidden=Hh,
        intermediate_hidden=I,
        num_ranks=1,
        num_experts=E_l,
        num_max_tokens_per_rank=cfg.num_max_tokens_per_rank,
        num_topk=2,
        num_ring_tokens=ring,
        num_sf_ring_tokens=sfring,
        num_shared_experts=0,
    )
    if not quiet:
        print(
            f"cfg: bm={bm} stages={cfg.num_stages} smem={cfg.smem_bytes} tmem_alloc={cfg.tmem_cols_alloc} ring={ring} sfring={sfring} ring_blocks={cfg.num_ring_blocks} symm={lay.total_bytes / 1e6:.1f}MB"
        )
    symm = torch.zeros(lay.total_bytes, dtype=torch.uint8)

    def region(name, dtype, shape):
        r = lay[name]
        v = symm[r.offset : r.offset + r.nbytes].view(dtype)
        return v.view(*shape)

    # ---------------- token tensors (L1 input ring and L2 input ring), nblk ring blocks
    def make_acts(K, nrows):
        b = make_fp8_bytes((nrows, K), gen)
        sf_e = torch.randint(
            125, 130, (nrows, K // 32), generator=gen, dtype=torch.uint8
        )
        return b, sf_e

    acts1, acts1_sf = make_acts(Hh, nblk * BM)
    acts2, acts2_sf = make_acts(I, nblk * BM)

    def fill_token_buffers(tok_view, sf_view, acts, sf_e):
        tok_view[: nblk * BM] = acts
        words = pack_sf_words(sf_e)  # (nrows, K/128)
        for b in range(nblk):
            for t in range(BM):
                pos = b * SFBM + utccp_pos(t, SFBM)
                sf_view[:, pos] = words[b * BM + t]

    fill_token_buffers(
        region("l1_token", torch.uint8, (ring, Hh)),
        region("l1_sf", torch.int32, (Hh // 128, sfring)),
        acts1,
        acts1_sf,
    )
    fill_token_buffers(
        region("l2_token", torch.uint8, (ring, I)),
        region("l2_sf", torch.int32, (I // 128, sfring)),
        acts2,
        acts2_sf,
    )

    # ---------------- weights (packed FP4) + SF in DeepGEMM layout
    def make_weights(N, K):
        codes = make_fp4_codes((E_l * N, K), gen)
        sfe = torch.randint(
            125, 130, (E_l * N, K // 32), generator=gen, dtype=torch.uint8
        )
        wbytes = pack_fp4(codes)
        words = pack_sf_words(sfe)  # (E_l*N, K/128)
        sf = torch.zeros(E_l * (K // 128), N, dtype=torch.int32)
        for e in range(E_l):
            for r in range(N):
                p = (r // 128) * 128 + 4 * (r % 32) + ((r % 128) // 32)
                sf[e * (K // 128) : (e + 1) * (K // 128), p] = words[e * N + r]
        return codes, sfe, wbytes, sf

    codes1, sfe1, w1b, w1sf = make_weights(N1, Hh)
    codes2, sfe2, w2b, w2sf = make_weights(Hh, I)

    # ---------------- task table + readiness state in the workspace signals
    ntask = len(tasks)
    table = torch.zeros(8, 8, dtype=torch.int32)
    sig = symm[: L.SIGNALS_SIZEOF]
    l1_full = sig[
        L.SIGNALS_L1_FULL_COUNT : L.SIGNALS_L1_FULL_COUNT + 4 * cfg.num_ring_blocks
    ].view(torch.int32)
    l2_mask = sig[
        L.SIGNALS_L2_FULL_MASK : L.SIGNALS_L2_FULL_MASK + 8 * cfg.num_ring_blocks
    ].view(torch.int64)
    for i, (phase, expert, nc, pool_block, valid_m) in enumerate(tasks):
        shape_n, shape_k = (N1, Hh) if phase == 1 else (Hh, I)
        table[i] = torch.tensor(
            [phase, expert, 0, nc, pool_block, valid_m, shape_n, shape_k],
            dtype=torch.int32,
        )
        l1_full[pool_block] = BM  # generation 0: BLOCK_M tokens arrived
        l2_mask[pool_block] = (1 << cfg.num_l1_block_ns) - 1

    out = torch.zeros(8, 2, 128, BM, dtype=torch.int32, device=dev)
    taskdump = torch.zeros(
        8, 8, dtype=torch.int32, device=dev
    )  # filled by the harness epilogue with the consumed task words (unused here)
    symm_d = symm.to(dev)
    tensors = [t.to(dev) for t in (w1b, w1sf, w2b, w2sf, table)]

    def mk(t, dt):
        return host_make_ptr(dt, t.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)

    args = [
        mk(symm_d, cutlass.Uint8),
        mk(tensors[0], cutlass.Uint8),
        mk(tensors[1], cutlass.Int32),
        mk(tensors[2], cutlass.Uint8),
        mk(tensors[3], cutlass.Int32),
        mk(tensors[4], cutlass.Uint32),
        ntask,
        mk(out, cutlass.Int32),
        mk(taskdump, cutlass.Int32),
        cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    ]
    harness = H.Harness(cfg)
    harness(*args)
    torch.cuda.synchronize()
    got = out.cpu().view(torch.float32)

    # ---------------- reference
    ok = True
    for i, (phase, expert, nc, pool_block, valid_m) in enumerate(tasks):
        N = (valid_m + 15) // 16 * 16
        shape_n, K = (N1, Hh) if phase == 1 else (Hh, I)
        acts, acts_sf = (acts1, acts1_sf) if phase == 1 else (acts2, acts2_sf)
        codes, sfe = (codes1, sfe1) if phase == 1 else (codes2, sfe2)
        X = decode_fp8(acts[pool_block * BM : pool_block * BM + N]) * e8m0_scale(
            acts_sf[pool_block * BM : pool_block * BM + N]
        ).repeat_interleave(32, dim=1)
        for cta in range(2):
            r0 = expert * shape_n + (2 * nc + cta) * 128
            if os.environ.get("S1_NEG"):
                r0 += 128  # negative control: wrong reference rows must be detected
            Wm = decode_fp4(codes[r0 : r0 + 128]) * e8m0_scale(
                sfe[r0 : r0 + 128]
            ).repeat_interleave(32, dim=1)
            ref = (Wm @ X.T).to(torch.float32)  # (128, N)
            g = got[i, cta, :, :N]
            eq = torch.equal(ref, g)
            if not eq:
                ok = False
                diff = (ref - g).abs()
                bad = (ref != g).sum().item()
                print(
                    f"task {i} phase={phase} expert={expert} nc={nc} cta={cta} valid_m={valid_m} N={N}: MISMATCH {bad}/{ref.numel()} maxdiff={diff.max().item():.3g} ref[0,:4]={ref[0, :4].tolist()} got[0,:4]={g[0, :4].tolist()}"
                )
            elif not quiet:
                print(
                    f"task {i} phase={phase} expert={expert} nc={nc} cta={cta} valid_m={valid_m} N={N}: bit-exact OK"
                )
    return ok


def run_scheduler_case(bm, counts, K1=512, N1=512, seed=0, quiet=True):
    cutlass, cute, cuda, host_make_ptr, get_sm107_kernel_config, L, H = _imports()
    dev = "cuda"
    E_l = len(counts)
    gen = torch.Generator().manual_seed(seed)
    Hh, I = K1, N1 // 2
    cfg = get_sm107_kernel_config(1, E_l, 1920, 1, 2, Hh, I, 0, num_sms=2, block_m=bm)
    BM, SFBM = cfg.block_m, cfg.sf_block_m
    ring, sfring = cfg.num_ring_tokens, cfg.num_sf_ring_tokens
    tasks_ref, M = R.ref_task_sequence(
        counts,
        BM,
        cfg.num_l1_clusters,
        cfg.num_l2_clusters,
        cfg.num_sms // 2,
        N1,
        Hh,
        Hh,
        I,
    )
    assert cfg.num_ring_blocks >= M
    NT = 128
    assert len(tasks_ref) <= NT
    lay = L.buffer_layout(
        hidden=Hh,
        intermediate_hidden=I,
        num_ranks=1,
        num_experts=E_l,
        num_max_tokens_per_rank=cfg.num_max_tokens_per_rank,
        num_topk=2,
        num_ring_tokens=ring,
        num_sf_ring_tokens=sfring,
        num_shared_experts=0,
    )
    symm = torch.zeros(lay.total_bytes, dtype=torch.uint8)

    def region(name, dtype, shape):
        r = lay[name]
        return symm[r.offset : r.offset + r.nbytes].view(dtype).view(*shape)

    def make_acts(K, nrows):
        b = make_fp8_bytes((nrows, K), gen)
        sf_e = torch.randint(
            125, 130, (nrows, K // 32), generator=gen, dtype=torch.uint8
        )
        return b, sf_e

    acts1, acts1_sf = make_acts(Hh, M * BM)
    acts2, acts2_sf = make_acts(I, M * BM)

    def fill(tok_view, sf_view, acts, sf_e):
        tok_view[: M * BM] = acts
        words = pack_sf_words(sf_e)
        for b in range(M):
            for t in range(BM):
                sf_view[:, b * SFBM + utccp_pos(t, SFBM)] = words[b * BM + t]

    fill(
        region("l1_token", torch.uint8, (ring, Hh)),
        region("l1_sf", torch.int32, (Hh // 128, sfring)),
        acts1,
        acts1_sf,
    )
    fill(
        region("l2_token", torch.uint8, (ring, I)),
        region("l2_sf", torch.int32, (I // 128, sfring)),
        acts2,
        acts2_sf,
    )

    def make_weights(N, K):
        codes = make_fp4_codes((E_l * N, K), gen)
        sfe = torch.randint(
            125, 130, (E_l * N, K // 32), generator=gen, dtype=torch.uint8
        )
        wbytes = pack_fp4(codes)
        words = pack_sf_words(sfe)
        sf = torch.zeros(E_l * (K // 128), N, dtype=torch.int32)
        rows = torch.arange(N)
        pos = (rows // 128) * 128 + 4 * (rows % 32) + ((rows % 128) // 32)
        for e in range(E_l):
            sf[e * (K // 128) : (e + 1) * (K // 128), pos] = words[
                e * N : (e + 1) * N
            ].T
        return codes, sfe, wbytes, sf

    codes1, sfe1, w1b, w1sf = make_weights(N1, Hh)
    codes2, sfe2, w2b, w2sf = make_weights(Hh, I)

    sig = symm[: L.SIGNALS_SIZEOF]
    l1_full = sig[
        L.SIGNALS_L1_FULL_COUNT : L.SIGNALS_L1_FULL_COUNT + 4 * cfg.num_ring_blocks
    ].view(torch.int32)
    l2_mask = sig[
        L.SIGNALS_L2_FULL_MASK : L.SIGNALS_L2_FULL_MASK + 8 * cfg.num_ring_blocks
    ].view(torch.int64)
    recv = sig[
        L.SIGNALS_EXPERT_RECV_COUNT_SUM : L.SIGNALS_EXPERT_RECV_COUNT_SUM + 8 * E_l
    ].view(torch.int64)
    for e, c in enumerate(counts):
        recv[e] = (
            cfg.num_sms * cfg.num_ranks << 32
        ) | c  # hi32 = kNumSMs * kNumRanks (all senders arrived), lo32 = token count
    for b in range(M):
        l1_full[b] = BM
        l2_mask[b] = (1 << cfg.num_l1_block_ns) - 1

    out = torch.zeros(NT, 2, 128, BM, dtype=torch.int32, device=dev)
    taskdump = torch.full((NT, 8), -7, dtype=torch.int32, device=dev)
    symm_d = symm.to(dev)
    table = torch.zeros(8, 8, dtype=torch.int32, device=dev)
    ws = [t.to(dev) for t in (w1b, w1sf, w2b, w2sf)]

    def mk(t, dt):
        return host_make_ptr(dt, t.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)

    args = [
        mk(symm_d, cutlass.Uint8),
        mk(ws[0], cutlass.Uint8),
        mk(ws[1], cutlass.Int32),
        mk(ws[2], cutlass.Uint8),
        mk(ws[3], cutlass.Int32),
        mk(table, cutlass.Uint32),
        0,
        mk(out, cutlass.Int32),
        mk(taskdump, cutlass.Int32),
        cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    ]
    harness = H.Harness(cfg, ntasks_max=NT, real_sched=True)
    harness(*args)
    torch.cuda.synchronize()
    td = taskdump.cpu()
    got = out.cpu().view(torch.float32)

    ok = True
    n_ref = len(tasks_ref)
    # (a) task sequence
    for i, t in enumerate(tasks_ref):
        row = tuple(td[i].tolist())
        # TaskInfo words: [phase, local_expert, m_block, n_cluster, pool_block, valid_m, shape_n, shape_k]
        if row != t:
            ok = False
            print(f"task {i}: published {row} != reference {t}")
            break
    if tuple(td[n_ref].tolist()) != (-7,) * 8:
        ok = False
        print(
            f"extra task published after the reference sequence: {td[n_ref].tolist()}"
        )
    # (b) results
    bad_tasks = 0
    for i, (phase, expert, mb, nc, pool_block, valid_m, shape_n, _K) in enumerate(
        tasks_ref
    ):
        N = (valid_m + 15) // 16 * 16
        acts, acts_sf = (acts1, acts1_sf) if phase == 1 else (acts2, acts2_sf)
        codes, sfe = (codes1, sfe1) if phase == 1 else (codes2, sfe2)
        X = decode_fp8(acts[pool_block * BM : pool_block * BM + N]) * e8m0_scale(
            acts_sf[pool_block * BM : pool_block * BM + N]
        ).repeat_interleave(32, dim=1)
        for cta in range(2):
            r0 = expert * shape_n + (2 * nc + cta) * 128
            Wm = decode_fp4(codes[r0 : r0 + 128]) * e8m0_scale(
                sfe[r0 : r0 + 128]
            ).repeat_interleave(32, dim=1)
            ref = (Wm @ X.T).to(torch.float32)
            if not torch.equal(ref, got[i, cta, :, :N]):
                bad_tasks += 1
                ok = False
                if bad_tasks <= 3:
                    print(
                        f"task {i} {(phase, expert, mb, nc, pool_block, valid_m)} cta={cta}: GEMM MISMATCH"
                    )
    if not quiet:
        print(
            f"  bm={bm} counts={counts}: {n_ref} tasks ({M} m-blocks), sequence {'OK' if ok else 'BAD'}; first tasks: {tasks_ref[:3]}"
        )
    return ok


@pytest.mark.parametrize("bm", [16, 32, 64, 128, 192, 240])
def test_gemm_core_bit_exact(bm):
    vms = sorted(
        {
            v
            for v in (1, 7, 16, 17, 31, 33, 50, 63, 65, 100, 127, bm - 1, bm)
            if 1 <= v <= bm
        }
    )
    E_l = 4
    tasks = []
    for i, vm in enumerate(vms[:8]):  # the task table of the harness holds 8 tasks
        phase = 1 if i % 3 != 2 else 2
        tasks.append((phase, i % E_l, i % 2, i % 4, vm))
    assert run_gemm_tasks(bm, 512, 512, E_l, tasks, quiet=True)


@pytest.mark.parametrize(
    "bm,counts",
    [
        (16, [3, 17, 0, 40]),
        (32, [5, 33, 0, 70]),
        (64, [37, 70, 0, 100]),
        (128, [37, 133, 0, 200]),
        (128, [128, 128, 128, 128]),
        (128, [1, 0, 0, 0]),
        (128, [0, 0, 0, 0]),
        (192, [37, 200, 0, 400]),
        (240, [37, 241, 0, 500]),
    ],
)
def test_scheduler_sequence_and_gemm(bm, counts):
    assert run_scheduler_case(bm, counts, quiet=True)


# ---------------------------------------------------------------------------------------------------------------------
# S = 1: shared-expert GEMM tasks (half-K stages: 64-K SWIZZLE_64B token tiles + FP8 weight tiles, E4M3 x E4M3 idesc, SF ids 0/2, UTCCP SF redone every
# stage).  Narrow data keep every product / partial sum exactly representable in FP32 (K <= 512): the result must be BIT-EXACT.
# ---------------------------------------------------------------------------------------------------------------------
def narrow_fp8(shape, gen, e_lo, e_hi, mants):
    s = torch.randint(0, 2, shape, generator=gen, dtype=torch.int32)
    e = torch.randint(e_lo, e_hi + 1, shape, generator=gen, dtype=torch.int32)
    mi = torch.randint(0, len(mants), shape, generator=gen, dtype=torch.int32)
    m = torch.tensor(mants, dtype=torch.int32)[mi.long()]
    return ((s << 7) | (e << 3) | m).to(torch.uint8)


def run_shared_gemm_tasks(bm, K1, N1, tasks, seed=0, nblk=4, quiet=True):
    """tasks: list of (phase, n_cluster, pool_block, valid_m) with phase 3 = SharedLinear1 (K = H, N = 2*I*S), 4 = SharedLinear2 (K = I*S, N = H)."""
    cutlass, cute, cuda, host_make_ptr, get_sm107_kernel_config, L, H = _imports()
    dev = "cuda"
    gen = torch.Generator().manual_seed(seed)
    Hh, I = K1, N1 // 2
    S = 1
    IS = I * S
    cfg = get_sm107_kernel_config(1, 4, 1920, 1, 2, Hh, I, S, num_sms=212, block_m=bm)
    BM, SFBM = cfg.block_m, cfg.sf_block_m
    ring, sfring = cfg.num_ring_tokens, cfg.num_sf_ring_tokens
    T, nsf = cfg.num_max_tokens_per_rank, cfg.num_shared_sf_tokens
    assert nblk * BM <= T and cfg.num_shared_experts == S
    lay = L.buffer_layout(
        hidden=Hh,
        intermediate_hidden=I,
        num_ranks=1,
        num_experts=4,
        num_max_tokens_per_rank=T,
        num_topk=2,
        num_ring_tokens=ring,
        num_sf_ring_tokens=sfring,
        num_shared_experts=S,
    )
    if not quiet:
        print(
            f"cfg: bm={bm} stages={cfg.num_stages} smem={cfg.smem_bytes} shared_k={cfg.shared_block_k} symm={lay.total_bytes / 1e6:.1f}MB nsf={nsf}"
        )
    symm = torch.zeros(lay.total_bytes, dtype=torch.uint8)

    def region(name, dtype, shape):
        r = lay[name]
        return symm[r.offset : r.offset + r.nbytes].view(dtype).view(*shape)

    def tok(K, nrows, e_lo=6, e_hi=8, mants=(0, 1, 2, 3, 4, 5, 6, 7)):
        return narrow_fp8((nrows, K), gen, e_lo, e_hi, mants), torch.randint(
            126, 129, (nrows, K // 32), generator=gen, dtype=torch.uint8
        )

    acts1, acts1_sf = tok(Hh, nblk * BM)  # shared L1 tokens = the input token buffer
    acts2, acts2_sf = tok(
        IS, nblk * BM
    )  # shared L2 tokens = the shared L1 output buffer (here: random data)

    def fill(tok_view, sf_view, acts, sf_e):
        tok_view[: nblk * BM] = acts
        words = pack_sf_words(sf_e)
        for b in range(nblk):
            for t in range(BM):
                sf_view[:, b * SFBM + utccp_pos(t, SFBM)] = words[b * BM + t]

    fill(
        region("input_token", torch.uint8, (T, Hh)),
        region("shared_l1_sf", torch.int32, (Hh // 128, nsf)),
        acts1,
        acts1_sf,
    )
    fill(
        region("shared_l2_token", torch.uint8, (T, IS)),
        region("shared_l2_sf", torch.int32, (IS // 128, nsf)),
        acts2,
        acts2_sf,
    )

    def weights(N, K):
        w = narrow_fp8((N, K), gen, 6, 7, (0, 2, 4, 6))
        sfe = torch.randint(126, 129, (N, K // 32), generator=gen, dtype=torch.uint8)
        words = pack_sf_words(sfe)  # (N, K/128)
        rows = torch.arange(N)
        pos = (rows // 128) * 128 + 4 * (rows % 32) + ((rows % 128) // 32)
        sf = torch.zeros(K // 128, N, dtype=torch.int32)
        sf[:, pos] = words.T
        return w, sfe, sf

    sw1, sw1e, sw1_sf = weights(2 * IS, Hh)  # shared L1 weights (2*I*S, H)
    sw2, sw2e, sw2_sf = weights(Hh, IS)  # shared L2 weights (H, I*S)

    table = torch.zeros(8, 8, dtype=torch.int32)
    sig = symm[: L.SIGNALS_SIZEOF]
    sl2 = sig[
        L.SIGNALS_SHARED_L2_FULL_COUNT : L.SIGNALS_SHARED_L2_FULL_COUNT + 4 * 64
    ].view(torch.int32)
    for i, (phase, nc, pool_block, valid_m) in enumerate(tasks):
        shape_n, shape_k = (2 * IS, Hh) if phase == 3 else (Hh, IS)
        table[i] = torch.tensor(
            [phase, 0, pool_block, nc, pool_block, valid_m, shape_n, shape_k],
            dtype=torch.int32,
        )
        if phase == 4:
            sl2[pool_block] = (
                (IS // 128) * 2
            )  # all shared L1 N blocks of this token block stored (2 CTAs x IS/128 clusters... K:855-859)

    out = torch.zeros(8, 2, 128, BM, dtype=torch.int32, device=dev)
    taskdump = torch.zeros(8, 8, dtype=torch.int32, device=dev)
    symm_d = symm.to(dev)
    # routed weights are unused by shared tasks (dummies of the right shape)
    El = 4
    dummy_w1 = torch.zeros(El * 2 * I, Hh // 2, dtype=torch.uint8, device=dev)
    dummy_w1sf = torch.zeros(El * (Hh // 128), 2 * I, dtype=torch.int32, device=dev)
    dummy_w2 = torch.zeros(El * Hh, I // 2, dtype=torch.uint8, device=dev)
    dummy_w2sf = torch.zeros(El * (I // 128), Hh, dtype=torch.int32, device=dev)
    tens = [t.to(dev) for t in (sw1, sw1_sf, sw2, sw2_sf, table)]

    def mk(t, dt):
        return host_make_ptr(dt, t.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)

    args = [
        mk(symm_d, cutlass.Uint8),
        mk(dummy_w1, cutlass.Uint8),
        mk(dummy_w1sf, cutlass.Int32),
        mk(dummy_w2, cutlass.Uint8),
        mk(dummy_w2sf, cutlass.Int32),
        mk(tens[0], cutlass.Uint8),
        mk(tens[1], cutlass.Int32),
        mk(tens[2], cutlass.Uint8),
        mk(tens[3], cutlass.Int32),
        mk(tens[4], cutlass.Uint32),
        len(tasks),
        mk(out, cutlass.Int32),
        mk(taskdump, cutlass.Int32),
        0,
        cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    ]
    harness = H.HarnessShared(cfg)
    harness(*args)
    torch.cuda.synchronize()
    got = out.cpu().view(torch.float32)

    ok = True
    for i, (phase, nc, pool_block, valid_m) in enumerate(tasks):
        N = (valid_m + 15) // 16 * 16
        acts, acts_sf = (acts1, acts1_sf) if phase == 3 else (acts2, acts2_sf)
        w, sfe = (sw1, sw1e) if phase == 3 else (sw2, sw2e)
        X = decode_fp8(acts[pool_block * BM : pool_block * BM + N]) * e8m0_scale(
            acts_sf[pool_block * BM : pool_block * BM + N]
        ).repeat_interleave(32, dim=1)
        for cta in range(2):
            r0 = (2 * nc + cta) * 128
            Wm = decode_fp8(w[r0 : r0 + 128]) * e8m0_scale(
                sfe[r0 : r0 + 128]
            ).repeat_interleave(32, dim=1)
            ref = (Wm @ X.T).to(torch.float32)
            g = got[i, cta, :, :N]
            if not torch.equal(ref, g):
                ok = False
                bad = (ref != g).sum().item()
                print(
                    f"task {i} phase={phase} nc={nc} pb={pool_block} cta={cta} valid_m={valid_m} N={N}: MISMATCH {bad}/{ref.numel()} ref[0,:4]={ref[0, :4].tolist()} got[0,:4]={g[0, :4].tolist()}"
                )
            elif not quiet:
                print(
                    f"task {i} phase={phase} nc={nc} pb={pool_block} cta={cta} valid_m={valid_m} N={N}: bit-exact OK"
                )
    return ok


@pytest.mark.parametrize("bm", [16, 32, 64, 128, 192, 240])
def test_shared_gemm_core_bit_exact(bm):
    vms = sorted(
        {
            v
            for v in (
                1,
                7,
                16,
                17,
                31,
                33,
                50,
                63,
                65,
                100,
                127,
                129,
                191,
                239,
                bm - 1,
                bm,
            )
            if 1 <= v <= bm
        }
    )
    ok = True
    for c in range(0, len(vms), 8):  # the harness task table holds 8 tasks per launch
        tasks = [
            (3 if (c + i) % 3 != 2 else 4, (c + i) % 2, (c + i) % 4, vm)
            for i, vm in enumerate(vms[c : c + 8])
        ]
        ok &= run_shared_gemm_tasks(bm, 512, 512, tasks, seed=c, quiet=True)
    assert ok
