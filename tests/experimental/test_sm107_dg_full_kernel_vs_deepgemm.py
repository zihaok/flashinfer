# SPDX-License-Identifier: Apache-2.0
# Port of DeepGEMM (MIT License, Copyright (c) 2025 DeepSeek; see licenses/LICENSE.deepgemm.txt) csrc/jit_kernels/impls/sm107_fp8_fp4_mega_moe.hpp.
"""Full-kernel (``mode_mask=7``: the dispatch / combine roles + the GEMM roles + the skeleton) test of the CuteDSL SM107 MegaMoE port against DeepGEMM's OWN kernel at EP1.

For every case the DeepGEMM kernel (``fp8_fp4_mega_moe``, ``DG_MEGA_MOE_SM107=1``) and the port's kernel run on the SAME symmetric buffer / weights / inputs and must agree
BIT-EXACTLY on ``y`` (int16 view comparison), for synchronous launches and for back-to-back launches without host sync; the cumulative receive statistics must equal
``counts * launches``; every non-self-cycling workspace field must be zero afterwards; and DeepGEMM running again on the buffer the port left behind must reproduce its first ``y``.
Every case is run without (S=0) and with (S=1: one merged FP8 shared expert, host-staged ``shared_l1_acts_sf``) shared experts.

EVERY launch of EITHER kernel (DeepGEMM's reference, the port's synchronous / back-to-back / graph-replayed launches, DeepGEMM's re-run) is preceded by a POISON fill (0xFF) of every
region of the symmetric buffer that the kernels write before they read it (dispatch rings, L1/L2 activations and scale factors, shared L2, all combine slots, and the workspace
tails ``src_token_topk_idx`` + ``token_src_metadata``; inputs, host-staged shared SF and the counter area stay).  Without it a write to a wrong address is invisible because the other
kernel left the right (identical) data behind (the Int32 combine-slot overflow passed every EP1 test that way).  Canary tests prove the machinery: a corrupted input must FAIL the
comparison, the poison must be present before a launch and overwritten after it; one variant gives DeepGEMM its OWN buffer (the port then runs first on a virgin buffer).

Needs one SM107 (compute capability 10.7) GPU with 212 SMs and the ``deep_gemm`` package (skipped otherwise). Run: ``pytest tests/experimental/test_sm107_dg_full_kernel_vs_deepgemm.py -x -q``.
``test_combine_slot_beyond_2gib_with_poisoned_buffer`` runs symmetric buffers beyond 2 GiB (Int32 wrap windows of the pre-v1.7 combine-slot address) with every kernel-written
region filled with 0xFF before every launch of either kernel.
Multi-rank (EP>=2), stress and graph-replay-hazard variants: ``test_sm107_dg_dispatch_combine_full.py`` and ``test_sm107_dg_dispatch_combine_graph_replay.py``.
"""

from __future__ import annotations

import os
import socket
import types

import pytest
import torch


def _is_sm107() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (10, 7)


pytestmark = pytest.mark.skipif(
    not _is_sm107(), reason="needs an SM107 (Rubin, compute capability 10.7) device"
)

E, TOPK, H, I, T_MAX = 32, 6, 5120, 2304, 1920


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# ---------------------------------------------------------------------------------------------------------------------------------------------------------
# shared-expert (S>0) input helpers: tests/test_mega_moe.py of DeepGEMM (_cast_fp8_for_mega_moe, _to_shared_mega_moe_sf_layout, _copy_fp8_sf)
# ---------------------------------------------------------------------------------------------------------------------------------------------------------
def _cast_fp8_for_mega_moe(x):
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


def _to_shared_sf_layout(sf, block_m, num_max_sf_tokens):
    """UTCCP-transposed shared L1 token SF, laid out per BLOCK_M block (depends on the BLOCK_M of the launch)."""
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


def _stage_shared(buffer, num_shared, x_sf, block_m, num_tokens):
    if num_shared > 0 and num_tokens > 0:
        buffer.shared_l1_acts_sf.copy_(
            _to_shared_sf_layout(x_sf, block_m, buffer.shared_l1_acts_sf.shape[0])
        )


def _make_env():
    deep_gemm = pytest.importorskip("deep_gemm")
    import torch.distributed as dist
    from deep_gemm.utils import per_token_cast_to_fp4

    if (
        torch.cuda.get_device_properties(
            torch.cuda.current_device()
        ).multi_processor_count
        != 212
    ):
        pytest.skip("the SM107 MegaMoE kernel is built for a 212-SM device")
    owns_group = not dist.is_initialized()
    if owns_group:
        dist.init_process_group(
            "nccl",
            init_method=f"tcp://127.0.0.1:{_free_port()}",
            world_size=1,
            rank=0,
            device_id=torch.device("cuda", torch.cuda.current_device()),
        )
    group = dist.new_group([0])
    dev = "cuda"
    buffers = {
        s: deep_gemm.get_symm_buffer_for_mega_moe(
            group,
            E,
            T_MAX,
            TOPK,
            H,
            I,
            num_shared_experts=s,
            mma_type="fp8xfp4",
            activation="swiglu",
        )
        for s in (0, 1)
    }

    def cast_w(w):
        g, n, k = w.shape
        q = torch.empty((g, n, k // 2), device=dev, dtype=torch.int8)
        sf = torch.empty((g, n, k // 32), device=dev, dtype=torch.float)
        for i in range(g):
            q[i], sf[i] = per_token_cast_to_fp4(w[i], use_ue8m0=True, gran_k=32)
        return q, deep_gemm.transform_sf_into_required_layout(sf, n, k, (1, 32), g)

    gen = torch.Generator(device=dev)
    gen.manual_seed(1234)
    l1 = cast_w(
        torch.randn((E, 2 * I, H), dtype=torch.bfloat16, device=dev, generator=gen)
    )
    l2 = cast_w(torch.randn((E, H, I), dtype=torch.bfloat16, device=dev, generator=gen))
    tl1, tl2 = deep_gemm.transform_weights_for_mega_moe(l1, l2, activation="swiglu")
    sh1 = _cast_fp8_for_mega_moe(
        torch.randn((2 * I, H), dtype=torch.bfloat16, device=dev, generator=gen)
    )[0::2]  # one merged FP8 shared expert (S=1)
    sh2 = _cast_fp8_for_mega_moe(
        torch.randn((H, I), dtype=torch.bfloat16, device=dev, generator=gen)
    )[0::2]
    shared = deep_gemm.transform_weights_for_mega_moe(sh1, sh2, activation="swiglu")
    yield types.SimpleNamespace(
        deep_gemm=deep_gemm,
        buffers=buffers,
        tl1=tl1,
        tl2=tl2,
        shared=shared,
        group=group,
    )
    if owns_group:
        dist.destroy_process_group()


@pytest.fixture(scope="module")
def env():
    with pytest.MonkeyPatch.context() as mp:
        # DeepGEMM reads DG_MEGA_MOE_SM107 when its SM107 MegaMoE kernel is JIT-compiled / launched: set for the tests of this module only (restored afterwards), never at import time
        mp.setenv("DG_MEGA_MOE_SM107", os.environ.get("DG_MEGA_MOE_SM107", "1"))
        yield from _make_env()


POISON_REGIONS = (
    "shared_l2_token",
    "shared_l2_sf",
    "l1_token",
    "l1_sf",
    "l1_topk_weights",
    "l2_token",
    "l2_sf",
    "combine_token",
)  # written by the kernels before they read them


def _layout(cfg, num_experts, num_shared):
    """``layout::MegaMoEBuffer`` of the launch's configuration (identical for every BLOCK_M: the symmetric buffer is allocated once)."""
    from flashinfer.experimental.sm107_deepgemm_megamoe.kernel import layout as L

    return L.buffer_layout(hidden=H, intermediate_hidden=I, num_ranks=1, num_experts=num_experts, num_max_tokens_per_rank=cfg.num_max_tokens_per_rank, num_topk=TOPK,
                           num_ring_tokens=cfg.num_ring_tokens, num_sf_ring_tokens=cfg.num_sf_ring_tokens, num_shared_experts=num_shared)  # fmt: skip


def _poison(buffer, lay, value=0xFF):
    """Fill every kernel-written region of the symmetric buffer: the regions of ``POISON_REGIONS`` and the workspace tails (``src_token_topk_idx`` + ``token_src_metadata``, written
    by the dispatch before they are read).  Inputs (x, x_sf, topk, the host-staged shared L1 SF) and the counter area (``MegaMoESignals``) stay.  Stream-ordered: call it right
    before EVERY launch of either kernel, so that a write to a wrong address can no longer hide behind the stale, identical data a previous launch left in the right place."""
    raw = buffer.buffer.view(torch.uint8)
    for name in POISON_REGIONS:
        r = lay[name]
        if r.nbytes:
            raw[r.offset : r.end].fill_(value)
    ws = lay.workspace
    raw[ws.src_token_topk_idx_offset : ws.nbytes].fill_(value)


def _signals_dirty(buffer):
    """Names of the non-self-cycling MegaMoESignals fields that are not zero (must be empty after every launch)."""
    from flashinfer.experimental.sm107_deepgemm_megamoe.kernel import layout as L

    sig = buffer.buffer.view(torch.uint8)[: L.SIGNALS_SIZEOF]
    fields = (("expert_send_count", L.SIGNALS_EXPERT_SEND_COUNT, 8 * 2048), ("expert_recv_count", L.SIGNALS_EXPERT_RECV_COUNT, 8 * 2048),
              ("expert_recv_count_sum", L.SIGNALS_EXPERT_RECV_COUNT_SUM, 8 * 512), ("l1_full_count", L.SIGNALS_L1_FULL_COUNT, 4 * (1 << 20)),
              ("l1_empty_count", L.SIGNALS_L1_EMPTY_COUNT, 4 * (1 << 20)), ("l2_full_mask", L.SIGNALS_L2_FULL_MASK, 8 * (1 << 20)),
              ("l2_empty_count", L.SIGNALS_L2_EMPTY_COUNT, 4 * (1 << 20)), ("task_counts", L.SIGNALS_L1_TASK_COUNT, 32),
              ("shared_l2_full_count", L.SIGNALS_SHARED_L2_FULL_COUNT, 4 * (1 << 15)))  # fmt: skip
    return [
        name
        for name, off, nbytes in fields
        if int((sig[off : off + nbytes] != 0).sum().item())
    ]


# (num_shared_experts, num_tokens, forced BLOCK_M or 0 = DeepGEMM's heuristic, fraction of masked top-k slots, tag_mode)
CASES = [
    (0, 1, 0, 0.0, "gridid"),
    (0, 64, 0, 0.0, "gridid"),
    (0, 257, 64, 0.0, "gridid"),
    (0, 1000, 240, 0.0, "gridid"),
    (0, 1920, 192, 0.0, "gridid"),  # pool blocks > ring blocks: ring wrap
    (0, 700, 16, 0.0, "epoch"),
    (
        0,
        500,
        128,
        0.4,
        "gridid",
    ),  # masked routes (tokens with fewer than top-k valid slots, all-masked rows)
    (0, 0, 0, 0.0, "gridid"),
    (1, 1, 0, 0.0, "epoch"),
    (1, 64, 0, 0.0, "epoch"),
    (1, 257, 64, 0.0, "epoch"),
    (1, 1000, 240, 0.0, "epoch"),
    (1, 1920, 192, 0.0, "epoch"),
    (1, 700, 16, 0.0, "gridid"),
    (1, 500, 128, 0.4, "epoch"),
    (1, 0, 0, 0.0, "epoch"),
]


@pytest.mark.parametrize(
    "num_shared,num_tokens,block_m,masked,tag_mode",
    CASES,
    ids=[f"S{c[0]}_M{c[1]}_bm{c[2]}_mask{c[3]}_{c[4]}" for c in CASES],
)
def test_full_kernel_matches_deepgemm_bit_exact(
    env, monkeypatch, num_shared, num_tokens, block_m, masked, tag_mode
):
    from deep_gemm.utils import per_token_cast_to_fp8

    from flashinfer.experimental.sm107_deepgemm_megamoe import jit
    from flashinfer.experimental.sm107_deepgemm_megamoe.heuristics import (
        get_sm107_kernel_config,
    )

    deep_gemm, buffer, tl1, tl2 = (
        env.deep_gemm,
        env.buffers[num_shared],
        env.tl1,
        env.tl2,
    )
    shared = env.shared if num_shared else None
    M, dev = num_tokens, "cuda"
    if block_m:
        monkeypatch.setenv(
            "DG_MEGA_MOE_SM107_BLOCK_M", str(block_m)
        )  # DeepGEMM and the port (via block_m=) use the same variant
    cfg = get_sm107_kernel_config(
        1,
        E,
        T_MAX,
        M,
        TOPK,
        H,
        I,
        num_shared,
        num_sms=212,
        activation_clamp=10.0,
        block_m=block_m,
    )
    lay = _layout(cfg, E, num_shared)
    assert lay.total_bytes == buffer.buffer.nbytes
    if (
        num_shared and not block_m and hasattr(deep_gemm, "get_block_m_for_mega_moe")
    ):  # the shared SF staging depends on BLOCK_M: both kernels must pick the same variant
        assert (
            deep_gemm.get_block_m_for_mega_moe(
                1, E, buffer.num_max_tokens_per_rank, M, TOPK, "fp8xfp4"
            )
            == cfg.block_m
        )
    gen = torch.Generator(device=dev)
    gen.manual_seed(7000 + M)
    x = torch.randn((M, H), dtype=torch.bfloat16, device=dev, generator=gen)
    scores = torch.randn((M, E), dtype=torch.float, device=dev, generator=gen)
    topk_weights, topk_idx = torch.topk(
        scores, TOPK, dim=-1, largest=True, sorted=False
    )
    if masked > 0:
        topk_idx.masked_fill_(
            torch.rand(topk_idx.shape, device=dev, generator=gen) < masked, -1
        )
        topk_weights.masked_fill_(topk_idx < 0, 0)
    if M > 0:
        x_fp8, x_sf = per_token_cast_to_fp8(
            x, use_ue8m0=True, gran_k=32, use_packed_ue8m0=True
        )
        buffer.x[:M].copy_(x_fp8)
        buffer.x_sf[:M].copy_(x_sf)
        buffer.topk_idx[:M].copy_(topk_idx)
        buffer.topk_weights[:M].copy_(topk_weights)
        _stage_shared(buffer, num_shared, x_sf, cfg.block_m, M)
    flat = topk_idx.reshape(-1)
    counts = torch.bincount(flat[flat >= 0].long(), minlength=E)

    def dg(y):
        _poison(buffer, lay)
        kw = (
            dict(shared_l1_weights=shared[0], shared_l2_weights=shared[1])
            if shared is not None
            else {}
        )
        deep_gemm.fp8_fp4_mega_moe(
            y,
            tl1,
            tl2,
            buffer,
            activation="swiglu",
            activation_clamp=10.0,
            fast_math=True,
            **kw,
        )

    y_ref = torch.empty((M, H), dtype=torch.bfloat16, device=dev)
    dg(y_ref)
    torch.cuda.synchronize()
    assert not torch.isnan(y_ref.float()).any()

    runner = jit.get_or_compile_kernel(
        cfg, mode_mask=7, roles="auto", tag_mode=tag_mode
    )
    sh_args = (
        ()
        if shared is None
        else (shared[0][0], shared[0][1], shared[1][0], shared[1][1])
    )
    sym = types.SimpleNamespace(
        base_ptr=buffer.buffer.data_ptr(), peer_offsets=[0], group=env.group
    )
    stats = torch.zeros(E, dtype=torch.int32, device=dev)
    sf_before = buffer.shared_l1_acts_sf.clone() if num_shared else None
    launches = 0
    for _ in range(2):  # synchronous launches
        y = torch.full((M, H), float("nan"), dtype=torch.bfloat16, device=dev)
        _poison(buffer, lay)
        runner(y, stats, M, sym, tl1[0], tl1[1], tl2[0], tl2[1], *sh_args)
        torch.cuda.synchronize()
        launches += 1
        assert torch.equal(y.view(torch.int16), y_ref.view(torch.int16)), (
            f"y differs from DeepGEMM (launch {launches})"
        )
    ys = torch.full((4, M, H), float("nan"), dtype=torch.bfloat16, device=dev)
    for i in range(
        4
    ):  # back-to-back launches, no host synchronization in between (the stream-ordered poison fill runs between them)
        _poison(buffer, lay)
        runner(ys[i], stats, M, sym, tl1[0], tl1[1], tl2[0], tl2[1], *sh_args)
        launches += 1
    torch.cuda.synchronize()
    for i in range(4):
        assert torch.equal(ys[i].view(torch.int16), y_ref.view(torch.int16)), (
            f"back-to-back launch {i} differs from DeepGEMM"
        )
    assert torch.equal(stats.cpu(), (counts * launches).to(torch.int32).cpu()), (
        "cumulative_local_expert_recv_stats != counts x launches"
    )
    assert _signals_dirty(buffer) == [], "workspace not clean after the port's launches"
    if num_shared:
        assert torch.equal(buffer.shared_l1_acts_sf, sf_before), (
            "the host-staged shared SF must not be modified by the kernel"
        )
    y2 = torch.empty((M, H), dtype=torch.bfloat16, device=dev)
    dg(y2)  # DeepGEMM on the buffer the port left behind
    torch.cuda.synchronize()
    assert torch.equal(y2.view(torch.int16), y_ref.view(torch.int16)), (
        "DeepGEMM result changed after the port ran on the same buffer"
    )


def test_graph_replay_epoch_tag_and_variant_switch(env):
    """CUDA-graph replays (``tag_mode='epoch'``: the launch-unique combine tag) and switching between BLOCK_M variants with changing inputs, back to back on one buffer (S=1)."""
    from deep_gemm.utils import per_token_cast_to_fp8

    from flashinfer.experimental.sm107_deepgemm_megamoe import jit
    from flashinfer.experimental.sm107_deepgemm_megamoe.heuristics import (
        get_sm107_kernel_config,
    )

    deep_gemm, buffer, tl1, tl2, shared = (
        env.deep_gemm,
        env.buffers[1],
        env.tl1,
        env.tl2,
        env.shared,
    )
    dev = "cuda"
    sym = types.SimpleNamespace(
        base_ptr=buffer.buffer.data_ptr(), peer_offsets=[0], group=env.group
    )
    sh_args = (shared[0][0], shared[0][1], shared[1][0], shared[1][1])
    sets = {}
    for M in (7, 300, 1500):
        gen = torch.Generator(device=dev)
        gen.manual_seed(900 + M)
        x = torch.randn((M, H), dtype=torch.bfloat16, device=dev, generator=gen)
        tw, ti = torch.topk(
            torch.randn((M, E), dtype=torch.float, device=dev, generator=gen),
            TOPK,
            dim=-1,
            largest=True,
            sorted=False,
        )
        xq, xs = per_token_cast_to_fp8(
            x, use_ue8m0=True, gran_k=32, use_packed_ue8m0=True
        )
        cfg = get_sm107_kernel_config(
            1, E, T_MAX, M, TOPK, H, I, 1, num_sms=212, activation_clamp=10.0
        )
        lay = _layout(cfg, E, 1)
        buffer.x[:M].copy_(xq)
        buffer.x_sf[:M].copy_(xs)
        buffer.topk_idx[:M].copy_(ti)
        buffer.topk_weights[:M].copy_(tw)
        _stage_shared(buffer, 1, xs, cfg.block_m, M)
        y = torch.empty((M, H), dtype=torch.bfloat16, device=dev)
        _poison(buffer, lay)
        deep_gemm.fp8_fp4_mega_moe(
            y,
            tl1,
            tl2,
            buffer,
            shared_l1_weights=shared[0],
            shared_l2_weights=shared[1],
            activation="swiglu",
            activation_clamp=10.0,
            fast_math=True,
        )
        torch.cuda.synchronize()
        sets[M] = dict(
            xq=xq,
            xs=xs,
            ti=ti,
            tw=tw,
            y=y,
            bm=cfg.block_m,
            runner=jit.get_or_compile_kernel(
                cfg, mode_mask=7, roles="auto", tag_mode="epoch"
            ),
        )

    def load(M):
        st = sets[M]
        buffer.x[:M].copy_(st["xq"])
        buffer.x_sf[:M].copy_(st["xs"])
        buffer.topk_idx[:M].copy_(st["ti"])
        buffer.topk_weights[:M].copy_(st["tw"])
        _stage_shared(buffer, 1, st["xs"], st["bm"], M)

    # variant switching: every launch picks another (M, BLOCK_M) and rewrites the inputs (incl. the BLOCK_M-dependent shared SF), no host sync
    order = [7, 1500, 300, 7, 300, 1500, 1500, 7, 300] * 4
    stats = torch.zeros(E, dtype=torch.int32, device=dev)
    outs = []
    for M in order:
        load(M)
        y = torch.full((M, H), float("nan"), dtype=torch.bfloat16, device=dev)
        _poison(buffer, lay)
        sets[M]["runner"](y, stats, M, sym, tl1[0], tl1[1], tl2[0], tl2[1], *sh_args)
        outs.append((M, y))
    torch.cuda.synchronize()
    for i, (M, y) in enumerate(outs):
        assert torch.equal(y.view(torch.int16), sets[M]["y"].view(torch.int16)), (
            f"variant-switching launch {i} (M={M}) differs from DeepGEMM"
        )
    assert _signals_dirty(buffer) == []
    # CUDA-graph replays of one variant (inputs of the last set stay in the buffer)
    M = order[-1]
    load(M)
    st = sets[M]
    yg = torch.full((M, H), float("nan"), dtype=torch.bfloat16, device=dev)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        st["runner"](yg, stats, M, sym, tl1[0], tl1[1], tl2[0], tl2[1], *sh_args)
    for i in range(16):
        yg.fill_(float("nan"))
        _poison(buffer, lay)  # outside the graph, stream-ordered before the replay
        g.replay()
        torch.cuda.synchronize()
        assert torch.equal(yg.view(torch.int16), st["y"].view(torch.int16)), (
            f"graph replay {i} differs from DeepGEMM"
        )
    assert _signals_dirty(buffer) == []


# ---------------------------------------------------------------------------------------------------------------------------------------------------------
# Int32 address-overflow regression (DESIGN.md section 4.1 rule 13): symmetric buffers beyond 2 GiB, POISONED before every launch of either kernel
# ---------------------------------------------------------------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def wrap_env(env):
    """Two symmetric buffers inside the Int32 wrap windows of the old combine-slot arithmetic (EP1): E=96 S=0 T_max=32768 (-> 34560: top-k slot 5 beyond 2 GiB) and E=32 S=1 T_max=26880
    (the shared slot 6 beyond 2 GiB); the E=32 weights are the module's."""
    from deep_gemm.utils import per_token_cast_to_fp4

    deep_gemm, dev = env.deep_gemm, "cuda"
    gen = torch.Generator(device=dev)
    gen.manual_seed(4321)

    def cast_w(w):
        g, n, k = w.shape
        q = torch.empty((g, n, k // 2), device=dev, dtype=torch.int8)
        sf = torch.empty((g, n, k // 32), device=dev, dtype=torch.float)
        for i in range(g):
            q[i], sf[i] = per_token_cast_to_fp4(w[i], use_ue8m0=True, gran_k=32)
        return q, deep_gemm.transform_sf_into_required_layout(sf, n, k, (1, 32), g)

    E96 = 96
    l1 = cast_w(
        torch.randn((E96, 2 * I, H), dtype=torch.bfloat16, device=dev, generator=gen)
    )
    l2 = cast_w(
        torch.randn((E96, H, I), dtype=torch.bfloat16, device=dev, generator=gen)
    )
    tl1_96, tl2_96 = deep_gemm.transform_weights_for_mega_moe(
        l1, l2, activation="swiglu"
    )
    del l1, l2
    bufs = {
        (0, 96): deep_gemm.get_symm_buffer_for_mega_moe(
            env.group,
            96,
            32768,
            TOPK,
            H,
            I,
            num_shared_experts=0,
            mma_type="fp8xfp4",
            activation="swiglu",
        ),
        (1, 32): deep_gemm.get_symm_buffer_for_mega_moe(
            env.group,
            32,
            26880,
            TOPK,
            H,
            I,
            num_shared_experts=1,
            mma_type="fp8xfp4",
            activation="swiglu",
        ),
    }
    weights = {(0, 96): (tl1_96, tl2_96, None), (1, 32): (env.tl1, env.tl2, env.shared)}
    yield types.SimpleNamespace(buffers=bufs, weights=weights)


WRAP_CASES = [  # (num_shared, num_experts, requested T_max, tokens, forced BLOCK_M or 0 = heuristic)
    (0, 96, 32768, 64, 0),  # slot 5 beyond 2 GiB (combine offset 543 MB + 5 x 354 MB)
    (0, 96, 32768, 700, 240),
    (1, 32, 26880, 64, 0),  # the shared slot 6 beyond 2 GiB
    (1, 32, 26880, 300, 128),
]


@pytest.mark.parametrize(
    "num_shared,num_experts,t_max,num_tokens,block_m",
    WRAP_CASES,
    ids=[f"S{c[0]}_E{c[1]}_T{c[2]}_M{c[3]}_bm{c[4]}" for c in WRAP_CASES],
)
def test_combine_slot_beyond_2gib_with_poisoned_buffer(
    env, wrap_env, monkeypatch, num_shared, num_experts, t_max, num_tokens, block_m
):
    from deep_gemm.utils import per_token_cast_to_fp8

    from flashinfer.experimental.sm107_deepgemm_megamoe import jit
    from flashinfer.experimental.sm107_deepgemm_megamoe.heuristics import (
        get_sm107_kernel_config,
    )

    deep_gemm, dev, M, Ex = env.deep_gemm, "cuda", num_tokens, num_experts
    buffer = wrap_env.buffers[(num_shared, Ex)]
    tl1, tl2, shared = wrap_env.weights[(num_shared, Ex)]
    if block_m:
        monkeypatch.setenv("DG_MEGA_MOE_SM107_BLOCK_M", str(block_m))
    cfg = get_sm107_kernel_config(
        1,
        Ex,
        t_max,
        M,
        TOPK,
        H,
        I,
        num_shared,
        num_sms=212,
        activation_clamp=10.0,
        block_m=block_m,
    )
    lay = _layout(cfg, Ex, num_shared)
    assert lay.total_bytes == buffer.buffer.nbytes
    ct = lay["combine_token"]
    assert (
        ct.offset + (ct.num_slots - 1) * ct.token_bytes * ct.num_tokens > (1 << 31) - 1
    ), "the case must lie inside the Int32 wrap window (otherwise it proves nothing)"
    if num_shared and not block_m and hasattr(deep_gemm, "get_block_m_for_mega_moe"):
        assert (
            deep_gemm.get_block_m_for_mega_moe(
                1, Ex, buffer.num_max_tokens_per_rank, M, TOPK, "fp8xfp4"
            )
            == cfg.block_m
        )
    gen = torch.Generator(device=dev)
    gen.manual_seed(8000 + M)
    x = torch.randn((M, H), dtype=torch.bfloat16, device=dev, generator=gen)
    topk_weights, topk_idx = torch.topk(
        torch.randn((M, Ex), dtype=torch.float, device=dev, generator=gen),
        TOPK,
        dim=-1,
        largest=True,
        sorted=False,
    )
    x_fp8, x_sf = per_token_cast_to_fp8(
        x, use_ue8m0=True, gran_k=32, use_packed_ue8m0=True
    )
    buffer.x[:M].copy_(x_fp8)
    buffer.x_sf[:M].copy_(x_sf)
    buffer.topk_idx[:M].copy_(topk_idx)
    buffer.topk_weights[:M].copy_(topk_weights)
    _stage_shared(buffer, num_shared, x_sf, cfg.block_m, M)
    flat = topk_idx.reshape(-1)
    counts = torch.bincount(flat[flat >= 0].long(), minlength=Ex)

    def dg(y):
        _poison(buffer, lay)
        kw = (
            dict(shared_l1_weights=shared[0], shared_l2_weights=shared[1])
            if shared is not None
            else {}
        )
        deep_gemm.fp8_fp4_mega_moe(
            y,
            tl1,
            tl2,
            buffer,
            activation="swiglu",
            activation_clamp=10.0,
            fast_math=True,
            **kw,
        )

    y_ref = torch.empty((M, H), dtype=torch.bfloat16, device=dev)
    dg(y_ref)
    torch.cuda.synchronize()
    assert not torch.isnan(y_ref.float()).any()
    runner = jit.get_or_compile_kernel(cfg, mode_mask=7, roles="auto", tag_mode="epoch")
    sh_args = (
        ()
        if shared is None
        else (shared[0][0], shared[0][1], shared[1][0], shared[1][1])
    )
    sym = types.SimpleNamespace(
        base_ptr=buffer.buffer.data_ptr(), peer_offsets=[0], group=env.group
    )
    stats = torch.zeros(Ex, dtype=torch.int32, device=dev)
    launches = 0
    for _ in range(2):
        y = torch.full((M, H), float("nan"), dtype=torch.bfloat16, device=dev)
        _poison(buffer, lay)
        runner(y, stats, M, sym, tl1[0], tl1[1], tl2[0], tl2[1], *sh_args)
        torch.cuda.synchronize()
        launches += 1
        assert torch.equal(y.view(torch.int16), y_ref.view(torch.int16)), (
            f"y differs from DeepGEMM (launch {launches}): a combine slot beyond 2 GiB was addressed wrongly?"
        )
    ys = torch.full((4, M, H), float("nan"), dtype=torch.bfloat16, device=dev)
    for i in range(4):  # back-to-back, poison between the launches (stream-ordered)
        _poison(buffer, lay)
        runner(ys[i], stats, M, sym, tl1[0], tl1[1], tl2[0], tl2[1], *sh_args)
        launches += 1
    torch.cuda.synchronize()
    for i in range(4):
        assert torch.equal(ys[i].view(torch.int16), y_ref.view(torch.int16)), (
            f"back-to-back launch {i} differs from DeepGEMM"
        )
    assert torch.equal(stats.cpu(), (counts * launches).to(torch.int32).cpu())
    assert _signals_dirty(buffer) == []
    y2 = torch.empty((M, H), dtype=torch.bfloat16, device=dev)
    dg(y2)
    torch.cuda.synchronize()
    assert torch.equal(y2.view(torch.int16), y_ref.view(torch.int16)), (
        "DeepGEMM result changed after the port ran on the same buffer"
    )


# ---------------------------------------------------------------------------------------------------------------------------------------------------------
# canaries: the comparison and the poison machinery must be able to FAIL; DeepGEMM on its own buffer (the port then runs first, on a virgin buffer)
# ---------------------------------------------------------------------------------------------------------------------------------------------------------
def _stage_inputs(buffer, num_tokens, seed):
    """Random routed inputs (S=0) for `num_tokens` tokens into the INPUT regions of `buffer`; the same seed gives identical inputs in different buffers.  Returns topk_idx."""
    from deep_gemm.utils import per_token_cast_to_fp8

    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)
    x = torch.randn((num_tokens, H), dtype=torch.bfloat16, device="cuda", generator=gen)
    topk_weights, topk_idx = torch.topk(
        torch.randn((num_tokens, E), dtype=torch.float, device="cuda", generator=gen),
        TOPK,
        dim=-1,
        largest=True,
        sorted=False,
    )
    x_fp8, x_sf = per_token_cast_to_fp8(
        x, use_ue8m0=True, gran_k=32, use_packed_ue8m0=True
    )
    buffer.x[:num_tokens].copy_(x_fp8)
    buffer.x_sf[:num_tokens].copy_(x_sf)
    buffer.topk_idx[:num_tokens].copy_(topk_idx)
    buffer.topk_weights[:num_tokens].copy_(topk_weights)
    return topk_idx


def _dg_launch(env, buffer, lay, y):
    _poison(buffer, lay)
    env.deep_gemm.fp8_fp4_mega_moe(
        y,
        env.tl1,
        env.tl2,
        buffer,
        activation="swiglu",
        activation_clamp=10.0,
        fast_math=True,
    )


def _port_launch(env, buffer, cfg, num_tokens, y, stats):
    from flashinfer.experimental.sm107_deepgemm_megamoe import jit

    runner = jit.get_or_compile_kernel(cfg, mode_mask=7, roles="auto", tag_mode="epoch")
    sym = types.SimpleNamespace(
        base_ptr=buffer.buffer.data_ptr(), peer_offsets=[0], group=env.group
    )
    runner(y, stats, num_tokens, sym, env.tl1[0], env.tl1[1], env.tl2[0], env.tl2[1])


def _s0_cfg(num_tokens):
    from flashinfer.experimental.sm107_deepgemm_megamoe.heuristics import (
        get_sm107_kernel_config,
    )

    return get_sm107_kernel_config(
        1, E, T_MAX, num_tokens, TOPK, H, I, 0, num_sms=212, activation_clamp=10.0
    )


def test_canary_corrupted_input_fails_the_comparison(env):
    """The port reads the INPUT regions of the symmetric buffer: corrupting one token after DeepGEMM computed the reference must make exactly that row differ (the bit-exact
    comparison of the other tests is therefore not vacuous), and restoring the input must restore bit-exactness."""
    M, bad_row = 64, 5
    buffer, cfg = env.buffers[0], _s0_cfg(64)
    lay = _layout(cfg, E, 0)
    _stage_inputs(buffer, M, seed=31)
    y_ref = torch.empty((M, H), dtype=torch.bfloat16, device="cuda")
    _dg_launch(env, buffer, lay, y_ref)
    torch.cuda.synchronize()
    stats = torch.zeros(E, dtype=torch.int32, device="cuda")

    saved = buffer.x[bad_row].clone()
    buffer.x[bad_row].zero_()  # corrupt ONE token after the reference exists
    y = torch.full((M, H), float("nan"), dtype=torch.bfloat16, device="cuda")
    _poison(buffer, lay)
    _port_launch(env, buffer, cfg, M, y, stats)
    torch.cuda.synchronize()
    differing = (
        (y.view(torch.int16) != y_ref.view(torch.int16))
        .any(dim=1)
        .nonzero()
        .flatten()
        .tolist()
    )
    assert not torch.equal(y.view(torch.int16), y_ref.view(torch.int16)), (
        "the corrupted input was not noticed: the bit-exact comparison would be vacuous"
    )
    assert differing == [bad_row], (
        f"exactly the corrupted token's row must differ, got {differing}"
    )

    buffer.x[bad_row].copy_(saved)  # restore: bit-exact again
    y2 = torch.full((M, H), float("nan"), dtype=torch.bfloat16, device="cuda")
    _poison(buffer, lay)
    _port_launch(env, buffer, cfg, M, y2, stats)
    torch.cuda.synchronize()
    assert torch.equal(y2.view(torch.int16), y_ref.view(torch.int16))
    assert _signals_dirty(buffer) == []


def test_canary_poison_is_present_before_and_overwritten_after_a_launch(env):
    """The poison fill must reach every kernel-written region (incl. the workspace tails) and must not touch inputs or the counter area; after a launch the regions the kernel
    writes for this input (all top-k slots of token 0, the first dispatch / metadata records) must have been overwritten, y must be free of NaN."""
    from flashinfer.experimental.sm107_deepgemm_megamoe.kernel import layout as L

    M = 64
    buffer, cfg = env.buffers[0], _s0_cfg(64)
    lay = _layout(cfg, E, 0)
    topk_idx = _stage_inputs(buffer, M, seed=47)
    x_before, sf_before, ti_before = (
        buffer.x[:M].clone(),
        buffer.x_sf[:M].clone(),
        buffer.topk_idx[:M].clone(),
    )
    raw = buffer.buffer.view(torch.uint8)
    ws = lay.workspace
    sig_before = raw[: L.SIGNALS_SIZEOF].clone()
    spans = {
        name: (lay[name].offset, lay[name].end)
        for name in POISON_REGIONS
        if lay[name].nbytes
    }
    spans["workspace tails (src_token_topk_idx + token_src_metadata)"] = (
        ws.src_token_topk_idx_offset,
        ws.nbytes,
    )

    _poison(buffer, lay)
    for name, (a, b) in spans.items():
        assert int((raw[a:b] != 0xFF).sum().item()) == 0, (
            f"{name} is not fully poisoned"
        )
    assert torch.equal(raw[: L.SIGNALS_SIZEOF], sig_before), (
        "the counter area (MegaMoESignals) must not be poisoned"
    )
    assert (
        torch.equal(buffer.x[:M], x_before)
        and torch.equal(buffer.x_sf[:M], sf_before)
        and torch.equal(buffer.topk_idx[:M], ti_before)
    ), "inputs must not be poisoned"

    y = torch.full((M, H), float("nan"), dtype=torch.bfloat16, device="cuda")
    stats = torch.zeros(E, dtype=torch.int32, device="cuda")
    _port_launch(env, buffer, cfg, M, y, stats)
    torch.cuda.synchronize()
    assert not torch.isnan(y.float()).any()
    ct = lay["combine_token"]
    for slot in range(
        TOPK
    ):  # every top-k slot of token 0 is written by the L2 epilogue of its expert
        a = ct.offset + slot * ct.token_bytes * ct.num_tokens
        assert int((raw[a : a + ct.token_bytes] != 0xFF).sum().item()) > 0, (
            f"combine slot {slot} of token 0 was not written by the kernel"
        )
    e0 = int(
        topk_idx[0, 0].item()
    )  # token 0's first expert: its `count` entries of src_token_topk_idx ((local_expert * R + src_rank) * T + slot, u32) were written by the dispatch
    count = int((topk_idx == e0).sum().item())
    base = ws.src_token_topk_idx_offset + e0 * cfg.num_max_tokens_per_rank * 4
    assert int((raw[base : base + count * 4] != 0xFF).sum().item()) > 0, (
        "src_token_topk_idx of the first routed expert was not written"
    )
    assert (
        int(
            (
                raw[ws.token_src_metadata_offset : ws.token_src_metadata_offset + 12]
                != 0xFF
            )
            .sum()
            .item()
        )
        > 0
    ), "token_src_metadata of pool token 0 was not written"
    assert (
        int(
            (raw[lay["l1_token"].offset : lay["l1_token"].offset + H] != 0xFF)
            .sum()
            .item()
        )
        > 0
    ), "the first pulled token of the L1 ring was not written"
    assert _signals_dirty(buffer) == []


@pytest.fixture(scope="module")
def own_buffers(env):
    """Two FRESH symmetric buffers (zero workspace, never launched on): one for DeepGEMM, one for the port."""
    make = lambda: env.deep_gemm.get_symm_buffer_for_mega_moe(
        env.group,
        E,
        T_MAX,
        TOPK,
        H,
        I,
        num_shared_experts=0,
        mma_type="fp8xfp4",
        activation="swiglu",
    )  # noqa: E731
    return types.SimpleNamespace(dg=make(), port=make())


def test_deepgemm_and_port_on_separate_buffers(env, own_buffers):
    """DeepGEMM runs on ITS OWN buffer and the port on another one: the port's first launch ever is on a virgin buffer (it cannot inherit DeepGEMM's self-cycling workspace
    state or its stale rings), every launch of both kernels is poisoned, and the results must still be bit-identical."""
    M = 300
    cfg = _s0_cfg(M)
    lay = _layout(cfg, E, 0)
    topk_idx = _stage_inputs(own_buffers.dg, M, seed=77)
    _stage_inputs(own_buffers.port, M, seed=77)  # identical inputs, different buffer
    flat = topk_idx.reshape(-1)
    counts = torch.bincount(flat[flat >= 0].long(), minlength=E)
    y_ref = torch.empty((M, H), dtype=torch.bfloat16, device="cuda")
    _dg_launch(env, own_buffers.dg, lay, y_ref)
    torch.cuda.synchronize()
    stats = torch.zeros(E, dtype=torch.int32, device="cuda")
    launches = 0
    for _ in range(
        2
    ):  # the first of these is the first launch ever on `own_buffers.port`
        y = torch.full((M, H), float("nan"), dtype=torch.bfloat16, device="cuda")
        _poison(own_buffers.port, lay)
        _port_launch(env, own_buffers.port, cfg, M, y, stats)
        torch.cuda.synchronize()
        launches += 1
        assert torch.equal(y.view(torch.int16), y_ref.view(torch.int16)), (
            f"port on a virgin buffer differs from DeepGEMM on its own buffer (launch {launches})"
        )
    ys = torch.full((4, M, H), float("nan"), dtype=torch.bfloat16, device="cuda")
    for i in range(4):
        _poison(own_buffers.port, lay)
        _port_launch(env, own_buffers.port, cfg, M, ys[i], stats)
        launches += 1
    torch.cuda.synchronize()
    for i in range(4):
        assert torch.equal(ys[i].view(torch.int16), y_ref.view(torch.int16)), (
            f"back-to-back launch {i} differs from DeepGEMM"
        )
    assert torch.equal(stats.cpu(), (counts * launches).to(torch.int32).cpu())
    assert (
        _signals_dirty(own_buffers.port) == [] and _signals_dirty(own_buffers.dg) == []
    )
    y2 = torch.empty((M, H), dtype=torch.bfloat16, device="cuda")
    _dg_launch(env, own_buffers.dg, lay, y2)
    torch.cuda.synchronize()
    assert torch.equal(y2.view(torch.int16), y_ref.view(torch.int16))


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-x", "-q", "-s"] + sys.argv[1:]))
