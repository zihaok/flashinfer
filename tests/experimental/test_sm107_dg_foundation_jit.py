# SPDX-License-Identifier: Apache-2.0
# Port of DeepGEMM (MIT License, Copyright (c) 2025 DeepSeek; see licenses/LICENSE.deepgemm.txt) csrc/jit_kernels/impls/sm107_fp8_fp4_mega_moe.hpp.
"""GPU tests of ``jit.py`` with STUB roles: cold compile vs disk-cache vs memo, TVM-FFI ABI (pointers as ints, peer table tuple, explicit stream),
per-BLOCK_M names, CUDA-graph capture of the runner, compile-inside-capture error, DEP32 peer table.

Run: ``pytest tests/experimental/test_sm107_dg_foundation_jit.py`` (use a fresh FLASHINFER_WORKSPACE_BASE so that the first compile is cold).
"""

from __future__ import annotations

import os
import sys
import time
from types import SimpleNamespace

import pytest
import torch


def _is_sm107() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (10, 7)


pytestmark = pytest.mark.skipif(
    not _is_sm107(), reason="needs an SM107 (Rubin, compute capability 10.7) device"
)


def _cfg(block_m: int, num_ranks: int):
    from flashinfer.experimental.sm107_deepgemm_megamoe.heuristics import (
        get_sm107_kernel_config,
    )

    return get_sm107_kernel_config(
        num_ranks,
        384,
        8192,
        1024,
        6,
        5120,
        2304,
        0,
        activation_clamp=10.0,
        block_m=block_m,
    )


class _Group:
    def __init__(self, rank):
        self._r = rank

    def rank(self):
        return self._r


def _fake_sym(buf: torch.Tensor, num_ranks: int, rank: int = 0):
    return SimpleNamespace(
        base_ptr=buf.data_ptr(),
        peer_offsets=[0] * num_ranks,
        group=_Group(rank),
        num_ranks=num_ranks,
    )


def _run(runner, cfg, counters, dummy, y, stats, iters: int = 1):
    sym = _fake_sym(dummy, cfg.num_ranks)
    counters.zero_()
    for _ in range(iters):
        runner(y, stats, 0, sym, dummy, dummy, dummy, dummy, trace=counters)
    torch.cuda.synchronize()
    c = counters.tolist()
    return (
        c[0:16] == [cfg.num_sms * iters] * 16 and c[16:32] == [cfg.num_sms * iters] * 16
    )


def _buffers(cfg):
    dev = torch.device("cuda")
    return (torch.zeros(1 << 20, dtype=torch.uint8, device=dev), torch.zeros(64, dtype=torch.int64, device=dev),
            torch.zeros(16, dtype=torch.bfloat16, device=dev), torch.zeros(cfg.num_experts_per_rank, dtype=torch.int32, device=dev))  # fmt: skip


def test_cold_warm_memo_and_launch():
    from flashinfer.experimental.sm107_deepgemm_megamoe import jit

    cfg = _cfg(128, 4)
    dummy, counters, y, stats = _buffers(cfg)
    t0 = time.time()
    r1 = jit.get_or_compile_kernel(cfg, roles="stubs", debug_marks=True)
    cold = time.time() - t0
    assert _run(r1, cfg, counters, dummy, y, stats, iters=3)
    t0 = time.time()
    r2 = jit.get_or_compile_kernel(cfg, roles="stubs", debug_marks=True)
    memo = time.time() - t0
    assert r2 is r1
    jit.clear_memo()
    t0 = time.time()
    r3 = jit.get_or_compile_kernel(cfg, roles="stubs", debug_marks=True)
    warm = time.time() - t0
    assert (
        r3.source == "disk-cache"
        or os.environ.get("FLASHINFER_CUTE_DSL_DISABLE_CACHE") == "1"
    ), r3.source
    assert _run(r3, cfg, counters, dummy, y, stats, iters=2)
    print(
        f"JIT cold={cold:.3f}s (source={r1.source}) memo={memo * 1e6:.1f}us warm(disk cache)={warm * 1e3:.1f}ms name={r1.name}"
    )


def test_is_compiled_query_for_capture_readiness():
    """The package's `validate_capture_ready` asks whether a variant is already in the in-process memo (no host compile can happen inside a capture)."""
    from flashinfer.experimental.sm107_deepgemm_megamoe import jit

    cfg = _cfg(32, 4)
    jit.clear_memo()
    assert not jit.is_compiled(cfg, roles="stubs", debug_marks=True)
    r = jit.get_or_compile_kernel(cfg, roles="stubs", debug_marks=True)
    assert jit.is_compiled(cfg, roles="stubs", debug_marks=True)
    assert not jit.is_compiled(
        cfg, roles="stubs", debug_marks=True, tag_mode="gridid"
    )  # a different variant
    assert r.name in jit.compiled_variant_names()
    jit.clear_memo()
    assert not jit.is_compiled(
        cfg, roles="stubs", debug_marks=True
    )  # disk-cache-only variants are not "compiled" for capture purposes


@pytest.mark.parametrize("block_m", (16, 32, 64, 128, 192, 240))
def test_variants_have_distinct_names_and_run(block_m):
    from flashinfer.experimental.sm107_deepgemm_megamoe import jit

    cfg = _cfg(block_m, 4)
    dummy, counters, y, stats = _buffers(cfg)
    r = jit.get_or_compile_kernel(cfg, roles="stubs", debug_marks=True)
    assert f"bm{block_m}_" in r.name
    assert _run(r, cfg, counters, dummy, y, stats)


def test_name_covers_every_knob():
    from flashinfer.experimental.sm107_deepgemm_megamoe import jit

    cfg = _cfg(128, 4)
    base = jit.make_build(cfg, roles="stubs").name()
    variants = {
        "mode_mask": jit.make_build(cfg, roles="stubs", mode_mask=3).name(),
        "pdl": jit.make_build(cfg, roles="stubs", pdl=True).name(),
        "ptxas": jit.make_build(cfg, roles="stubs", ptxas_options="").name(),
        "opt": jit.make_build(cfg, roles="stubs", opt_level=2).name(),
        "debug_marks": jit.make_build(cfg, roles="stubs", debug_marks=True).name(),
        "roles": jit.make_build(cfg, roles="auto").name(),
        "block_m": jit.make_build(_cfg(192, 4), roles="stubs").name(),
        "ranks": jit.make_build(_cfg(128, 8), roles="stubs").name(),
    }
    assert len({base, *variants.values()}) == 1 + len(variants), variants
    assert all(len(n) < 100 for n in variants.values())


def test_regs_override_tag_mode_validation_and_naming():
    """`regs_override` (setmaxnreg split) and `tag_mode` / `dispatch_only_sync` are validated, part of the name/digest and exposed by jit."""
    from flashinfer.experimental.sm107_deepgemm_megamoe import jit

    cfg = _cfg(128, 4)  # DEP4 at E=384 -> DeepGEMM's own split is 96/88/160
    base = jit.make_build(cfg, roles="stubs")
    assert base.regs == (cfg.regs_dispatch, cfg.regs_non_epilogue, cfg.regs_epilogue)
    tuned = jit.make_build(cfg, roles="stubs", regs_override=(96, 88, 160))
    faithful_dep8 = jit.make_build(cfg, roles="stubs", regs_override=(48, 40, 208))
    assert tuned.regs == (96, 88, 160) and "_r96-88-160_" in tuned.name()
    assert (
        faithful_dep8.regs == (48, 40, 208) and "_r48-40-208_" in faithful_dep8.name()
    )
    assert len({base.name(), tuned.name(), faithful_dep8.name()}) == 3
    assert (
        jit.make_build(cfg, roles="stubs", regs_override=[96, 88, 160]).name()
        == tuned.name()
    )  # list normalised to a tuple
    assert (
        jit.make_build(cfg, roles="stubs", tag_mode="gridid").name() != base.name()
    )  # default is "epoch"
    assert jit.make_build(cfg, roles="stubs", tag_mode="epoch").name() == base.name()
    assert (
        jit.make_build(cfg, roles="stubs", dispatch_only_sync=True).name()
        != base.name()
    )
    # (80, 88, 176) sums to 128*80 + 128*88 + 256*176 = 66560 > 64512; the others violate the setmaxnreg operand rules
    for bad in (
        (96, 88),
        (95, 88, 160),
        (16, 88, 160),
        (264, 40, 40),
        (96, 96, 192),
        (80, 88, 176),
    ):
        with pytest.raises(ValueError):
            jit.make_build(cfg, roles="stubs", regs_override=bad)
    with pytest.raises(ValueError):
        jit.make_build(cfg, roles="stubs", tag_mode="bogus")
    assert (
        jit.make_build(cfg, roles="stubs", timeout_sec=10).name() != base.name()
    )  # the spin-wait timeout is baked into the PTX -> part of the cache key
    assert jit.make_build(cfg, roles="stubs", timeout_sec=300.0).name() == base.name()
    with pytest.raises(ValueError):
        jit.make_build(cfg, roles="stubs", timeout_sec=0)


def test_default_register_policy_is_spill_free_and_faithful_split_is_available():
    """Product default: 96/88/160 at EVERY EP size (DeepGEMM's 48/40/208 at experts-per-rank <= 64 spills in the dispatch pull loop); `faithful_regs=True` = DeepGEMM's split."""
    from flashinfer.experimental.sm107_deepgemm_megamoe import jit

    cfg8 = _cfg(
        128, 8
    )  # E=384, DEP8 -> 48 experts per rank: DeepGEMM's own split is 48/40/208
    assert (cfg8.regs_dispatch, cfg8.regs_non_epilogue, cfg8.regs_epilogue) == (
        48,
        40,
        208,
    )
    default = jit.make_build(cfg8, roles="stubs")
    faithful = jit.make_build(cfg8, roles="stubs", faithful_regs=True)
    assert default.regs == (96, 88, 160) and faithful.regs == (48, 40, 208)
    assert default.name() != faithful.name()
    cfg4 = _cfg(
        128, 4
    )  # E/R = 96 > 64: DeepGEMM already uses 96/88/160, so both policies coincide
    assert (
        jit.make_build(cfg4, roles="stubs").regs
        == jit.make_build(cfg4, roles="stubs", faithful_regs=True).regs
        == (96, 88, 160)
    )
    with pytest.raises(ValueError):
        jit.make_build(
            cfg8, roles="stubs", faithful_regs=True, regs_override=(96, 88, 160)
        )
    assert jit.make_build(cfg8, roles="stubs", regs_override=(56, 40, 200)).regs == (
        56,
        40,
        200,
    )  # the 56/40/200 variant stays expressible


@pytest.mark.parametrize("regs", ((96, 88, 160), (48, 40, 208), (64, 56, 192)))
def test_regs_override_runs_with_stub_roles(regs):
    """The skeleton honours any legal split (setmaxnreg per role); results of real roles are checked by the full-kernel EP1 test."""
    from flashinfer.experimental.sm107_deepgemm_megamoe import jit

    cfg = _cfg(128, 4)
    dummy, counters, y, stats = _buffers(cfg)
    r = jit.get_or_compile_kernel(
        cfg, roles="stubs", debug_marks=True, regs_override=regs
    )
    assert r.build.regs == regs
    assert _run(r, cfg, counters, dummy, y, stats, iters=2)


def test_shared_experts_refused_until_roles_support_them():
    """S>0 with real roles that do not declare SUPPORTS_SHARED_EXPERTS must raise at launch (not silently drop the shared expert); stubs and S=0 are unaffected."""
    import torch  # noqa: F401

    from flashinfer.experimental.sm107_deepgemm_megamoe import jit
    from flashinfer.experimental.sm107_deepgemm_megamoe.heuristics import (
        get_sm107_kernel_config,
    )
    from flashinfer.experimental.sm107_deepgemm_megamoe.kernel.kernel import (
        shared_experts_supported,
    )

    cfg = get_sm107_kernel_config(
        4, 384, 8192, 1024, 6, 5120, 2304, 1, activation_clamp=10.0, block_m=128
    )
    dummy, counters, y, stats = _buffers(cfg)
    stub = jit.get_or_compile_kernel(cfg, roles="stubs", debug_marks=True)
    assert stub.unsupported is None
    assert _run(stub, cfg, counters, dummy, y, stats)
    real = jit.get_or_compile_kernel(cfg, roles="auto")  # compiling is allowed
    dispatch_only = jit.get_or_compile_kernel(
        cfg, roles="auto", mode_mask=1
    )  # staged dispatch-only builds at S>0 stay launchable (staged tests)
    assert dispatch_only.unsupported is None
    if shared_experts_supported("auto"):
        assert real.unsupported is None
    else:
        assert real.unsupported is not None
        sym = _fake_sym(dummy, cfg.num_ranks)
        with pytest.raises(NotImplementedError, match="shared experts"):
            real(y, stats, 0, sym, dummy, dummy, dummy, dummy)
        with pytest.raises(NotImplementedError, match="shared experts"):
            real.bind(sym, dummy, dummy, dummy, dummy, stats=stats)


def test_bind_fast_path_and_overhead():
    from flashinfer.experimental.sm107_deepgemm_megamoe import jit

    cfg = _cfg(128, 4)
    dummy, counters, y, stats = _buffers(cfg)
    runner = jit.get_or_compile_kernel(cfg, roles="stubs", debug_marks=True)
    sym = _fake_sym(dummy, cfg.num_ranks)
    launch = runner.bind(sym, dummy, dummy, dummy, dummy, stats=stats, trace=counters)
    counters.zero_()
    for _ in range(5):
        launch(y, 0)
    torch.cuda.synchronize()
    c = counters.tolist()
    assert c[0:16] == [cfg.num_sms * 5] * 16
    # host-side overhead of one call (includes the TVM-FFI dispatch and the async launch; stub kernel runs ~10 us on the GPU)
    n = 2000
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        launch(y, 0)
    t_bind = (time.perf_counter() - t0) / n * 1e6
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        runner(y, stats, 0, sym, dummy, dummy, dummy, dummy, trace=counters)
    t_call = (time.perf_counter() - t0) / n * 1e6
    torch.cuda.synchronize()
    print(
        f"host launch overhead: bind={t_bind:.1f} us/call, runner()={t_call:.1f} us/call"
    )
    assert t_bind < 200 and t_call < 400  # generous sanity bound (real numbers printed)


def test_graph_capture_of_runner_and_compile_error_in_capture():
    from flashinfer.experimental.sm107_deepgemm_megamoe import jit

    cfg = _cfg(64, 4)
    dummy, counters, y, stats = _buffers(cfg)
    runner = jit.get_or_compile_kernel(cfg, roles="stubs", debug_marks=True)
    sym = _fake_sym(dummy, cfg.num_ranks)
    counters.zero_()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        runner(y, stats, 0, sym, dummy, dummy, dummy, dummy, trace=counters)
    for _ in range(10):
        g.replay()
    torch.cuda.synchronize()
    c = counters.tolist()
    assert c[0:16] == [cfg.num_sms * 10] * 16 and c[16:32] == [cfg.num_sms * 10] * 16
    # compiling a NEW variant inside a capture must raise (not silently corrupt the graph)
    other = _cfg(32, 4)
    g2 = torch.cuda.CUDAGraph()
    with pytest.raises(RuntimeError, match="CUDA graph capture"), torch.cuda.graph(g2):
        jit.get_or_compile_kernel(other, roles="stubs", debug_marks=True, mode_mask=5)


@pytest.mark.parametrize("num_ranks", (16, 32))
def test_large_peer_table(num_ranks):
    """DEP16/32: by-value peer table of 16/32 offsets through TVM-FFI (DSL >= 4.7 lifts the old 16-entry limit)."""
    from flashinfer.experimental.sm107_deepgemm_megamoe import jit

    cfg = _cfg(128, num_ranks)
    dummy, counters, y, stats = _buffers(cfg)
    r = jit.get_or_compile_kernel(cfg, roles="stubs", debug_marks=True)
    assert _run(r, cfg, counters, dummy, y, stats)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-x", "-q", "-s"] + sys.argv[1:]))
