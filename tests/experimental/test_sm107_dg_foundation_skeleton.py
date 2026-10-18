# SPDX-License-Identifier: Apache-2.0
# Port of DeepGEMM (MIT License, Copyright (c) 2025 DeepSeek; see licenses/LICENSE.deepgemm.txt) sm107_fp8_fp4_mega_moe.cuh / kernel skeleton tests.
"""GPU tests of the kernel skeleton with STUB roles (kernel/kernel.py, DESIGN.md): launch geometry, prologue, no hang, graph capture.

Run: ``pytest tests/experimental/test_sm107_dg_foundation_skeleton.py``  or  ``python tests/experimental/test_sm107_dg_foundation_skeleton.py [--bm 128] [--dep 4] [--modes 7,0,1,2,4]``.
Every test uses ``KernelBuild(roles="stubs", debug_marks=True)``: all 16 warps of all CTAs bump visit counters in the "trace" buffer
(entry counters [0..15], exit counters [16..31]), so a missing rendezvous / hang / early exit is visible.
"""

from __future__ import annotations

import argparse
import time
from typing import Dict, List, Optional, Tuple

import pytest
import torch


def _is_sm107() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (10, 7)


pytestmark = pytest.mark.skipif(
    not _is_sm107(), reason="needs an SM107 (Rubin, compute capability 10.7) device"
)

BLOCK_MS = (16, 32, 64, 128, 192, 240)


def make_cfg(
    block_m: int,
    num_ranks: int = 4,
    *,
    num_shared_experts: int = 0,
    trace: bool = False,
    num_sms: Optional[int] = None,
):
    from flashinfer.experimental.sm107_deepgemm_megamoe.heuristics import (
        get_sm107_kernel_config,
    )

    return get_sm107_kernel_config(
        num_ranks, 384, 8192, 1024, 6, 5120, 2304, num_shared_experts,
        num_sms=num_sms, activation_clamp=10.0, block_m=block_m, trace=trace,
    )  # fmt: skip


class Launcher:
    """Compiles one ``KernelBuild`` with dummy buffers and launches it."""

    def __init__(self, build, num_tokens: int = 0):
        import cuda.bindings.driver as cuda  # noqa: F401
        import cutlass
        import cutlass.cute as cute
        from cutlass import Int32
        from cutlass.cute.runtime import make_ptr
        from cutlass.cute.typing import AddressSpace

        from flashinfer.experimental.sm107_deepgemm_megamoe.kernel import comm
        from flashinfer.experimental.sm107_deepgemm_megamoe.kernel.kernel import (
            DeepGemmSm107MegaMoeKernel,
            role_sources,
        )

        self.build = build
        self.cfg = build.cfg
        self.role_sources = role_sources()
        dev = torch.device("cuda")
        self.dummy = torch.zeros(
            1 << 20, dtype=torch.uint8, device=dev
        )  # stands for the symmetric buffer / weights
        self.counters = torch.zeros(64, dtype=torch.int64, device=dev)
        self.y = torch.zeros(16, dtype=torch.bfloat16, device=dev)
        self.stats = torch.zeros(
            self.cfg.num_experts_per_rank, dtype=torch.int32, device=dev
        )
        gp = lambda dt, t, al=16: make_ptr(
            dt, t.data_ptr(), AddressSpace.gmem, assumed_align=al
        )  # noqa: E731
        R = self.cfg.num_ranks
        self.sym = comm.SymBufferHost(
            base_address=cutlass.Int64(self.dummy.data_ptr()), offsets=tuple(cutlass.Int64(0) for _ in range(R)),
            rank=Int32(0), max_ranks=R,
        )  # fmt: skip
        self.args = (
            gp(cutlass.BFloat16, self.y), gp(cutlass.Int32, self.stats), Int32(num_tokens), self.sym,
            gp(cutlass.Uint8, self.dummy), gp(cutlass.Int32, self.dummy), gp(cutlass.Uint8, self.dummy), gp(cutlass.Int32, self.dummy),
            gp(cutlass.Uint8, self.dummy), gp(cutlass.Int32, self.dummy), gp(cutlass.Uint8, self.dummy), gp(cutlass.Int32, self.dummy),
            gp(cutlass.Int64, self.counters), gp(cutlass.Uint8, self.dummy),
        )  # fmt: skip
        self.kernel = DeepGemmSm107MegaMoeKernel(build)
        self._cute = cute
        self._cuda = cuda
        self.compiled = None
        self.compile_seconds = None

    def stream(self):
        return self._cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    def compile(self):
        t0 = time.time()
        self.compiled = self._cute.compile(self.kernel, *self.args, self.stream())
        self.compile_seconds = time.time() - t0
        return self.compiled

    def launch(self):
        self.compiled(*self.args, self.stream())

    def marks(self) -> Tuple[List[int], List[int]]:
        torch.cuda.synchronize()
        c = self.counters.tolist()
        return c[0:16], c[16:32]


def run_case(block_m: int, num_ranks: int, mode_mask: int, *, keep: bool = False, graph: bool = False, iters: int = 1,
             gridsync_iters: int = 0) -> Dict:  # fmt: skip
    from flashinfer.experimental.sm107_deepgemm_megamoe.kernel.config import KernelBuild

    cfg = make_cfg(block_m, num_ranks)
    build = KernelBuild(
        cfg,
        mode_mask=mode_mask,
        roles="stubs",
        debug_marks=True,
        debug_gridsync_iters=gridsync_iters,
    )
    L = Launcher(build)
    L.compile()
    L.counters.zero_()
    torch.cuda.synchronize()
    t_launch = time.time()
    for _ in range(iters):
        L.launch()
    entry, exit_ = L.marks()
    t_launch = time.time() - t_launch
    n = cfg.num_sms * iters
    ok = entry == [n] * 16 and exit_ == [n] * 16
    out = dict(
        bm=block_m,
        dep=num_ranks,
        mode=mode_mask,
        regs=build.regs,
        smem=cfg.smem_bytes,
        ok=ok,
        compile_s=L.compile_seconds,
        entry_min=min(entry),
        exit_min=min(exit_),
        expect=n,
        name=build.name(),
        launch_s=t_launch,
    )
    if gridsync_iters:
        out["us_per_gridsync"] = t_launch / (gridsync_iters * iters) * 1e6
    if graph:
        L.counters.zero_()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            L.launch()
        for _ in range(20):
            g.replay()
        entry, exit_ = L.marks()
        out["graph_ok"] = (
            entry == [20 * cfg.num_sms] * 16 and exit_ == [20 * cfg.num_sms] * 16
        )
    return out


@pytest.mark.parametrize("block_m", BLOCK_MS)
@pytest.mark.parametrize("num_ranks", [4, 8])
def test_skeleton_all_roles_visit(block_m, num_ranks):
    r = run_case(block_m, num_ranks, 7, iters=3)
    assert r["ok"], r


@pytest.mark.parametrize("mode_mask", range(8))
def test_skeleton_mode_masks(mode_mask):
    r = run_case(128, 8, mode_mask, iters=2)
    assert r["ok"], r


@pytest.mark.parametrize("block_m", (16, 128, 240))
def test_skeleton_all_ctas_coresident_gridsync(block_m):
    """212 CTAs with the REAL launch geometry (512 threads, 2-CTA clusters, max smem) must all be resident: 20000 software grid syncs."""
    r = run_case(block_m, 4, 7, gridsync_iters=20000)
    assert r["ok"], r


def test_skeleton_trace_build():
    """``cfg.trace=True``: DeepGEMM-compatible stamps (slot 0 globaltimer + slot 60 clock64 right after the PDL wait) for every CTA."""
    import cutlass
    from cutlass.cute.runtime import make_ptr
    from cutlass.cute.typing import AddressSpace

    from flashinfer.experimental.sm107_deepgemm_megamoe.kernel.config import KernelBuild
    from flashinfer.experimental.sm107_deepgemm_megamoe.kernel.trace import (
        trace_buffer_u64s,
        NUM_SLOTS,
    )

    cfg = make_cfg(128, 4, trace=True)
    build = KernelBuild(
        cfg, roles="stubs"
    )  # no debug_marks: the trace buffer is the real one
    L = Launcher(build)
    tb = torch.zeros(trace_buffer_u64s(cfg.num_sms), dtype=torch.int64, device="cuda")
    args = list(L.args)
    args[12] = make_ptr(
        cutlass.Int64, tb.data_ptr(), AddressSpace.gmem, assumed_align=16
    )
    L.args = tuple(args)
    L.compile()
    L.launch()
    torch.cuda.synchronize()
    t = tb[: cfg.num_sms * NUM_SLOTS].view(cfg.num_sms, NUM_SLOTS)
    assert (t[:, 0] != 0).all() and (t[:, 60] != 0).all(), (
        "slots 0/60 must be stamped by every CTA"
    )
    span_us = (t[:, 0].max() - t[:, 0].min()).item() / 1e3
    assert span_us < 5e5, (
        span_us
    )  # all CTAs started within 0.5 s of each other (they are co-resident)
    assert (t[:, 1:60] == 0).all(), "stub roles stamp nothing else"


def test_launch_tag_modes():
    """``ctx.launch_tag``: gridid mode == %gridid+1 (unique per direct launch, repeated per CUDA-graph replay); epoch mode == (nvl_barrier_counter>>1)+1."""
    from flashinfer.experimental.sm107_deepgemm_megamoe.kernel.config import KernelBuild
    from flashinfer.experimental.sm107_deepgemm_megamoe.kernel import layout as Ly

    cfg = make_cfg(128, 4)
    # gridid mode
    L = Launcher(KernelBuild(cfg, roles="stubs", debug_marks=True, tag_mode="gridid"))
    L.compile()
    tags = []
    for _ in range(3):
        L.launch()
        torch.cuda.synchronize()
        tags.append(int(L.counters[4].item()) if False else int(L.counters[32].item()))
    assert len(set(tags)) == 3 and tags == sorted(tags) and all(t > 0 for t in tags), (
        tags
    )
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        L.launch()
    replay_tags = []
    for _ in range(3):
        g.replay()
        torch.cuda.synchronize()
        replay_tags.append(int(L.counters[32].item()))
    assert len(set(replay_tags)) == 1, (
        replay_tags
    )  # MEASURED hazard: the same tag on every replay of the node
    # epoch mode: tag follows the workspace counter (self-cycling, +2 per launch in the real kernel)
    L2 = Launcher(KernelBuild(cfg, roles="stubs", debug_marks=True, tag_mode="epoch"))
    L2.compile()
    ctr = L2.dummy[
        Ly.SIGNALS_NVL_BARRIER_COUNTER : Ly.SIGNALS_NVL_BARRIER_COUNTER + 4
    ].view(torch.int32)
    got = []
    for n in range(4):
        ctr.fill_(2 * n)
        L2.launch()
        torch.cuda.synchronize()
        got.append(int(L2.counters[32].item()))
    assert got == [1, 2, 3, 4], got


def test_skeleton_pdl_launch():
    """``pdl=True``: programmatic dependent launch attribute + the kernel's ``griddepcontrol.wait`` after the prologue: back-to-back launches, no hang."""
    from flashinfer.experimental.sm107_deepgemm_megamoe.kernel.config import KernelBuild

    cfg = make_cfg(128, 4)
    L = Launcher(KernelBuild(cfg, roles="stubs", debug_marks=True, pdl=True))
    L.compile()
    L.counters.zero_()
    x = torch.randn(1 << 20, device="cuda")
    for _ in range(20):
        (x * 2.0).sum()  # a primary kernel in the stream before every launch
        L.launch()
    entry, exit_ = L.marks()
    assert entry == [20 * cfg.num_sms] * 16 and exit_ == [20 * cfg.num_sms] * 16


def test_skeleton_graph_capture():
    r = run_case(128, 4, 7, graph=True)
    assert r["ok"] and r["graph_ok"], r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bm", default=",".join(map(str, BLOCK_MS)))
    ap.add_argument("--dep", default="4,8")
    ap.add_argument("--modes", default="7")
    ap.add_argument("--graph", action="store_true")
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--gridsync", type=int, default=0)
    a = ap.parse_args()
    rows = []
    for dep in map(int, a.dep.split(",")):
        for bm in map(int, a.bm.split(",")):
            for mode in map(int, a.modes.split(",")):
                r = run_case(
                    bm,
                    dep,
                    mode,
                    graph=a.graph,
                    iters=a.iters,
                    gridsync_iters=a.gridsync,
                )
                rows.append(r)
                print(
                    f"BM={bm:3d} DEP{dep} mode={mode} regs={r['regs']} smem={r['smem']} compile={r['compile_s']:.2f}s "
                    f"ok={r['ok']} graph_ok={r.get('graph_ok')} entry_min={r['entry_min']} exit_min={r['exit_min']} expect={r['expect']}"
                    + (
                        f" gridsync={a.gridsync} us/sync={r['us_per_gridsync']:.2f}"
                        if a.gridsync
                        else ""
                    ),
                    flush=True,
                )
    print(
        "ALL OK"
        if all(r["ok"] and r.get("graph_ok", True) for r in rows)
        else "FAILURES"
    )


if __name__ == "__main__":
    main()
