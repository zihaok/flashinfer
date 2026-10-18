# SPDX-License-Identifier: Apache-2.0
# Port of DeepGEMM (MIT License, Copyright (c) 2025 DeepSeek; see licenses/LICENSE.deepgemm.txt) sm107_fp8_fp4_mega_moe.cuh dispatch / NVLink protocol / combine.
"""FULL-kernel multi-rank tests of the dispatch / combine roles (together with the GEMM roles) against DeepGEMM's own ``y``.

One process per rank over torch symmetric memory (``_sm107_dg_full_ep.py``, like DeepGEMM's own tests): K input sets with per-rank random routing, DeepGEMM's ``fp8_fp4_mega_moe`` provides the reference ``y`` of every
set (all ranks running the same set), then the port's kernel (``jit.get_or_compile_kernel(cfg, mode_mask=7)``) runs N back-to-back launches in a pseudo-random order of sets with GPU-side rank-dependent launch
skew; every launch's ``y`` is compared bit-exactly ON THE DEVICE.  Modes: ``same`` (all ranks run the same set) and ``chaos`` (every rank picks its own set per launch: mixed M / BLOCK_M / routing across ranks in one
collective step).  After the loop: cumulative expert-receive stats, workspace clean, DeepGEMM re-run on the buffer the port left behind.

Poison (default ON here, ``R_POISON=0`` to disable): before EVERY launch of BOTH kernels every kernel-written region of the symmetric buffer (dispatch rings, L1/L2 activations + SF, shared L2, ALL combine slots, ``token_src_metadata``)
is filled with 0xFF, so a wrong-address / dropped remote write cannot be masked by the identical stale bytes DeepGEMM left in the shared buffer.  ``R_POISON_STRICT=1`` runs the same fill under a per-iteration host barrier
(synchronize + barrier before AND after the fill) and also poisons the remotely written ``src_token_topk_idx``; ``R_SEPBUF=1`` gives the port its own buffer; ``R_CANARY`` corrupts one staged input of one launch (the run passes
only if exactly that launch is flagged).
Acceptance run (10^4 launches per mode, EP4): ``SM107_DG_STRESS_ITERS=10000``; the default is short.
Run: ``pytest tests/experimental/test_sm107_dg_dispatch_combine_full.py`` (2-4 SM107 GPUs, one per rank).
"""

from __future__ import annotations

import os
import socket
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
NGPU = torch.cuda.device_count() if torch.cuda.is_available() else 0


def _is_sm107() -> bool:
    return (
        NGPU > 0
        and torch.cuda.get_device_capability(0) == (10, 7)
        and torch.cuda.get_device_properties(0).multi_processor_count == 212
    )


pytestmark = [
    pytest.mark.skipif(
        not _is_sm107(), reason="needs an SM107 (cc 10.7, 212 SMs) device"
    )
]


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _run_full(nranks: int, **env):
    pytest.importorskip("deep_gemm")
    import _sm107_dg_full_ep

    env = {
        "R_POISON": os.environ.get("R_POISON", "1"),
        **env,
    }  # poison ON by default: fill every kernel-written region of the symmetric buffer with 0xFF before EVERY launch of BOTH kernels (R_POISON=0 disables)
    saved = {
        k: os.environ.get(k)
        for k in list(env)
        + ["R_RANKS", "MASTER_ADDR", "MASTER_PORT", "DG_MEGA_MOE_SM107"]
    }
    try:
        os.environ.update({k: str(v) for k, v in env.items()})
        os.environ.update(
            R_RANKS=str(nranks),
            MASTER_ADDR="127.0.0.1",
            MASTER_PORT=str(_free_port()),
            DG_MEGA_MOE_SM107="1",
        )
        torch.multiprocessing.spawn(
            _sm107_dg_full_ep.worker, args=(nranks,), nprocs=nranks, join=True
        )  # a rank exits non-zero on any mismatch / error
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _iters(default: int) -> int:
    return int(os.environ.get("SM107_DG_STRESS_ITERS", default))


@pytest.mark.skipif(NGPU < 2, reason="needs 2 GPUs")
def test_full_kernel_ep2_vs_deepgemm_mixed_m():
    # per-rank M differs between sets and (in `chaos`) between ranks of one launch: BLOCK_M 240/192/128/64/16 variants mixed
    _run_full(
        2,
        R_E=32,
        S_SETM="1000,64,7,300,0,1900",
        S_MODE="both",
        S_LAUNCHES=_iters(300),
        S_SKEW_CYCLES=2000000,
        S_CHECK_EVERY=0,
        R_MASKED=0.2,
    )


@pytest.mark.skipif(NGPU < 4, reason="needs 4 GPUs")
def test_full_kernel_ep4_vs_deepgemm_mixed_m():
    _run_full(
        4,
        R_E=32,
        S_SETM="256,64,1000,7,1900,300",
        S_MODE="both",
        S_LAUNCHES=_iters(300),
        S_SKEW_CYCLES=4000000,
        S_CHECK_EVERY=0,
        R_MASKED=0.2,
    )


@pytest.mark.skipif(NGPU < 4, reason="needs 4 GPUs")
@pytest.mark.parametrize("tag_mode", ("gridid", "epoch"))
def test_full_kernel_ep4_skew_stress(tag_mode):
    # eager back-to-back launches under large rank skew (up to 10 ms): no hang, no mismatch, both tag modes
    _run_full(
        4,
        R_E=32,
        S_SETM="256,64,1000",
        S_MODE="chaos",
        S_LAUNCHES=_iters(300),
        S_SKEW_CYCLES=20000000,
        S_CHECK_EVERY=0,
        R_TAGMODE=tag_mode,
    )


@pytest.mark.skipif(NGPU < 4, reason="needs 4 GPUs")
def test_full_kernel_ep4_production_expert_count():
    # E = 384 (the V4.1 expert count, 96 experts per rank: DeepGEMM's own 96/88/160 register split)
    _run_full(
        4,
        R_E=384,
        S_SETM="64,512,1900",
        S_MODE="both",
        S_LAUNCHES=_iters(100),
        S_SKEW_CYCLES=2000000,
        S_CHECK_EVERY=0,
    )


@pytest.mark.skipif(NGPU < 2, reason="needs 2 GPUs")
def test_full_kernel_ep2_shared_expert_vs_deepgemm():
    # S=1: ONE fused FP8 shared expert (host-staged shared L1 SF per launch's BLOCK_M), shared slot 6 of the combine; mixed M / BLOCK_M, M=0 sets, masked routes
    _run_full(
        2,
        R_E=32,
        R_S=1,
        S_SETM="1000,64,7,300,0,1900",
        S_MODE="both",
        S_LAUNCHES=_iters(300),
        S_SKEW_CYCLES=2000000,
        S_CHECK_EVERY=0,
        R_MASKED=0.2,
    )


@pytest.mark.skipif(NGPU < 4, reason="needs 4 GPUs")
def test_full_kernel_ep4_shared_expert_stress():
    _run_full(
        4,
        R_E=32,
        R_S=1,
        S_SETM="256,64,1000,7,1900,300",
        S_MODE="both",
        S_LAUNCHES=_iters(300),
        S_SKEW_CYCLES=4000000,
        S_CHECK_EVERY=0,
        R_MASKED=0.2,
    )


# ---- harness strength: strict poison under a per-iteration barrier, canary, separate buffers (reviewer finding TS-01) ------------------------------------------------------------------------------------
@pytest.mark.skipif(NGPU < 4, reason="needs 4 GPUs")
@pytest.mark.parametrize("shared", (0, 1))
def test_full_kernel_ep4_poison_strict_barrier(shared):
    # every launch of BOTH kernels runs under a per-iteration host barrier with the buffer poisoned (also src_token_topk_idx); no launch overlap, no peer in flight while a rank poisons
    _run_full(
        4,
        R_E=32,
        R_S=shared,
        R_POISON_STRICT=1,
        S_SETM="256,64,1000,7,1900,300",
        S_MODE="both",
        S_LAUNCHES=_iters(40),
        S_SKEW_CYCLES=2000000,
        S_CHECK_EVERY=0,
        R_MASKED=0.2,
    )


@pytest.mark.skipif(NGPU < 4, reason="needs 4 GPUs")
@pytest.mark.parametrize("shared", (0, 1))
def test_full_kernel_ep4_separate_deepgemm_buffer_poisoned(shared):
    # the port runs on its OWN symmetric buffer (DeepGEMM keeps the other one): it can never read DeepGEMM's stale bytes
    _run_full(
        4,
        R_E=32,
        R_S=shared,
        R_SEPBUF=1,
        S_SETM="256,64,1000,7,1900,300",
        S_MODE="both",
        S_LAUNCHES=_iters(100),
        S_SKEW_CYCLES=2000000,
        S_CHECK_EVERY=0,
        R_MASKED=0.2,
    )


@pytest.mark.skipif(NGPU < 2, reason="needs 2 GPUs")
def test_full_kernel_ep2_canary_corrupted_staged_input_is_detected():
    # harness self-test: the top-k weights of token 0 are corrupted in launch 12 of rank 0; the harness must flag exactly that launch (it exits non-zero, and the test fails, if it does not)
    _run_full(
        2,
        R_E=32,
        R_CANARY="12:0",
        S_SETM="300",
        S_MODE="same",
        S_LAUNCHES=30,
        S_SKEW_CYCLES=0,
        S_CHECK_EVERY=0,
        R_MASKED=0.0,
    )


# ---- skewed routing and mixed per-rank token counts (all poisoned, bitwise vs DeepGEMM on every launch) --------------------------------------------------------------------------------
@pytest.mark.skipif(NGPU < 4, reason="needs 4 GPUs")
@pytest.mark.parametrize("shared,bm", ((0, 240), (1, 192)))
def test_full_kernel_ep4_hot_expert_routing_ring_wrap(shared, bm):
    # every token of every rank -> the same 6 experts (all on rank 0: ~5 ring generations at E=32, M=2048), hot experts spread over the ranks, rank r -> experts of rank r + 1,
    # all-zero hidden states (identical rows: vLLM's MegaMoE warmup), plain random routing; forced BLOCK_M for both kernels
    _run_full(
        4,
        R_E=32,
        R_S=shared,
        R_BM=bm,
        DG_MEGA_MOE_SM107_BLOCK_M=bm,
        R_TMAX=2048,
        S_SETM="2048h,2048s,2048p,2048hz,2048",
        S_MODE="both",
        S_LAUNCHES=_iters(60),
        S_SKEW_CYCLES=2000000,
        S_CHECK_EVERY=0,
    )


@pytest.mark.skipif(NGPU < 4, reason="needs 4 GPUs")
def test_full_kernel_ep4_mixed_rank_counts_fully_masked_shared_expert():
    # S=1: 0 / 1 / 7-token ranks, fully masked sets (M > 0 tokens, every topk_idx = -1: only the shared expert contributes), mixed BLOCK_M per rank in one step (chaos), shared SF layouts differ per rank
    _run_full(
        4,
        R_E=32,
        R_S=1,
        S_SETM="0,1,7,64,300,1900,300x,1900x",
        S_MODE="chaos",
        S_LAUNCHES=_iters(300),
        S_SKEW_CYCLES=4000000,
        S_CHECK_EVERY=0,
    )
