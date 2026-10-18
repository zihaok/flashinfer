# SPDX-License-Identifier: Apache-2.0
# Port of DeepGEMM (MIT License, Copyright (c) 2025 DeepSeek; see licenses/LICENSE.deepgemm.txt) sm107_fp8_fp4_mega_moe.cuh / API tests.
"""Eager launch path of ``fp8_fp4_mega_moe``: the validated-call records and the torch-function guard.

The first call with a set of operands validates them (``check_mega_moe_arguments``) and stores a record (validated geometry, weak references to the symmetric buffer, one bound launcher
per BLOCK_M); later calls compare the operands' (data_ptr, shape, stride, dtype) with the record and launch through the bound launcher.  These tests drive the REAL ``jit.MegaMoeRunner``
(fake compiled function: the arguments of the compiled ABI are captured) and count the validations, so that

* a warm call validates nothing and launches with exactly the arguments the validating path produced,
* everything that can invalidate a record does: a changed weight / SF / stats pointer, shape, layout or dtype (weights swapped in place), a symmetric buffer re-created over the same
  memory (different object) or destroyed, a different T_max / shared-expert count / BLOCK_M, the per-die mode, the kernel options, the kernel factory,
* the per-call operand ``y`` is still validated on the warm path,
* the call does not enter Python-level ``TorchFunctionMode``s (``torch.set_default_device("cuda")`` installs one: every tensor method call would cost ~1 us),
* records pin no tensor (weights can be freed).
"""

import gc
import weakref

import pytest

torch = pytest.importorskip("torch")

from flashinfer.experimental.sm107_deepgemm_megamoe import compat  # noqa: E402

_RUBIN = torch.cuda.is_available() and torch.cuda.get_device_capability() == (10, 7)
pytestmark = [
    pytest.mark.arch_rubin,
    pytest.mark.skipif(not _RUBIN, reason="needs an SM107 (Rubin, cc 10.7) device"),
]

E, H, I, TOPK = 8, 1024, 512, 2


def _ep1_group():
    import atexit
    import socket

    import torch.distributed as dist

    if not dist.is_initialized():
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        dist.init_process_group(
            "gloo", init_method=f"tcp://127.0.0.1:{port}", rank=0, world_size=1
        )
        atexit.register(lambda: dist.is_initialized() and dist.destroy_process_group())
    return dist.new_group([0])


def _sf(g, n, k):
    k128 = -(-k // 128)
    n4 = -(-n // 4) * 4
    shape, strides = (
        ((g, n, k128), (k128 * n4, 1, n4)) if g is not None else ((n, k128), (1, n4))
    )
    return torch.empty_strided(shape, strides, dtype=torch.int32, device="cuda").zero_()


def _weights(S=0):
    l1 = (
        torch.zeros((E, 2 * I, H // 2), dtype=torch.int8, device="cuda"),
        _sf(E, 2 * I, H),
    )
    l2 = (torch.zeros((E, H, I // 2), dtype=torch.int8, device="cuda"), _sf(E, H, I))
    if S:
        s1 = (
            torch.zeros((2 * I, H), dtype=torch.float8_e4m3fn, device="cuda"),
            _sf(None, 2 * I, H),
        )
        s2 = (
            torch.zeros((H, I), dtype=torch.float8_e4m3fn, device="cuda"),
            _sf(None, H, I),
        )
    else:
        s1 = s2 = None
    return l1, l2, s1, s2


class _Env:
    """Real ``jit.MegaMoeRunner`` objects around a fake compiled function; counts validations and factory requests."""

    def __init__(self, monkeypatch):
        from flashinfer.experimental.sm107_deepgemm_megamoe import frontend, jit

        self.frontend, self.jit = frontend, jit
        self.captured, self.built, self.validations = [], [], 0
        real_check = frontend.check_mega_moe_arguments

        def counting_check(*a, **k):
            self.validations += 1
            return real_check(*a, **k)

        monkeypatch.setattr(frontend, "check_mega_moe_arguments", counting_check)

        def factory(cfg):
            self.built.append(
                (cfg.block_m, cfg.num_shared_experts, cfg.num_max_tokens_per_rank)
            )
            return jit.MegaMoeRunner(
                jit.make_build(cfg),
                lambda *args: self.captured.append(args),
                0.0,
                "fake",
            )

        frontend.set_kernel_factory(factory)
        self.factory = factory

    def close(self):
        self.frontend.set_kernel_options()
        self.frontend.set_per_die_mode(None)
        self.frontend.set_kernel_factory(None)


@pytest.fixture
def env(monkeypatch):
    e = _Env(monkeypatch)
    yield e
    e.close()


def _y(m):
    return torch.empty((m, H), dtype=torch.bfloat16, device="cuda")


def _call(y, l1, l2, buf, s1=None, s2=None, stats=None, **kw):
    return compat.fp8_fp4_mega_moe(
        y, l1, l2, buf, s1, s2, stats, activation_clamp=10.0, **kw
    )


def _two_block_ms(T=1920):
    seen = {}
    for m in (1, 7, 100, 500, 1500, 1920):
        seen.setdefault(compat.get_block_m_for_mega_moe(1, E, T, m, TOPK, "fp8xfp4"), m)
    (bm_a, m_a), (bm_b, m_b) = sorted(seen.items())[:2]
    return (bm_a, m_a), (bm_b, m_b)


def test_warm_calls_skip_validation_and_launch_with_the_validated_arguments(env):
    buf = compat.SymmBuffer(_ep1_group(), E, 1920, TOPK, H, I, num_shared_experts=1)
    l1, l2, s1, s2 = _weights(S=1)
    stats = torch.zeros(E, dtype=torch.int32, device="cuda")
    y = _y(100)
    try:
        _call(y, l1, l2, buf, s1, s2, stats)
        assert (
            env.validations == 1
            and len(env.captured) == 1
            and len(env.frontend._call_records) == 1
        )
        for _ in range(3):
            _call(y, l1, l2, buf, s1, s2, stats)
        assert env.validations == 1 and len(env.captured) == 4
        assert all(
            c == env.captured[0] for c in env.captured
        )  # the bound launcher passes exactly what the validating path passed (same y, same stream)
        (
            y_ptr,
            stats_ptr,
            num_tokens,
            sym,
            l1w,
            l1s,
            l2w,
            l2s,
            sl1w,
            sl1s,
            sl2w,
            sl2s,
            trace,
            domains,
            stream,
        ) = env.captured[-1]
        assert (y_ptr, stats_ptr, num_tokens) == (
            y.data_ptr(),
            stats.data_ptr(),
            100,
        ) and sym == (buf.base_ptr, (0,), 0, 1)
        assert (l1w, l1s, l2w, l2s) == (
            l1[0].data_ptr(),
            l1[1].data_ptr(),
            l2[0].data_ptr(),
            l2[1].data_ptr(),
        )
        assert (sl1w, sl1s, sl2w, sl2s) == (
            s1[0].data_ptr(),
            s1[1].data_ptr(),
            s2[0].data_ptr(),
            s2[1].data_ptr(),
        )
        assert (trace, domains) == (
            0,
            0,
        ) and stream == torch.cuda.current_stream().cuda_stream
        # a new output tensor every call (the normal eager pattern) is not a miss
        for m in (100, 64, 33):
            y2 = _y(m)
            _call(y2, l1, l2, buf, s1, s2, stats)
            assert env.captured[-1][0] == y2.data_ptr() and env.captured[-1][2] == m
        assert env.validations >= 1 and len(env.built) == len(
            {b[0] for b in env.built}
        )  # one runner per BLOCK_M variant
    finally:
        buf.destroy()


def test_each_block_m_validates_once_then_runs_warm(env):
    buf = compat.SymmBuffer(_ep1_group(), E, 1920, TOPK, H, I)
    l1, l2, _, _ = _weights()
    (bm_a, m_a), (bm_b, m_b) = _two_block_ms()
    assert bm_a != bm_b
    try:
        for m in (m_a, m_b, m_a, m_b, m_a, m_b, m_a):
            _call(_y(m), l1, l2, buf)
        assert (
            env.validations == 2
            and sorted(b[0] for b in env.built) == sorted((bm_a, bm_b))
            and len(env.frontend._call_records) == 1
        )
        assert [c[2] for c in env.captured] == [m_a, m_b, m_a, m_b, m_a, m_b, m_a]
    finally:
        buf.destroy()


def test_weights_swapped_or_reshaped_are_revalidated_but_in_place_updates_are_not(env):
    buf = compat.SymmBuffer(_ep1_group(), E, 1920, TOPK, H, I)
    l1, l2, _, _ = _weights()
    y = _y(100)
    try:
        _call(y, l1, l2, buf)
        _call(y, l1, l2, buf)
        assert env.validations == 1
        l1[0].copy_(
            torch.ones_like(l1[0])
        )  # content updated in place (weight reload): same pointers -> no validation
        l2[1].zero_()
        _call(y, l1, l2, buf)
        assert env.validations == 1
        fresh = torch.zeros_like(l1[0])
        l1[
            0
        ].data = fresh  # the parameter now owns other memory: a miss, and the launch uses the new pointer
        _call(y, l1, l2, buf)
        assert env.validations == 2 and env.captured[-1][4] == fresh.data_ptr()
        _call(y, l1, l2, buf)
        assert env.validations == 2
        new_sf = _sf(E, 2 * I, H)
        l1 = (l1[0], new_sf)  # a different scale-factor tensor
        _call(y, l1, l2, buf)
        assert env.validations == 3 and env.captured[-1][5] == new_sf.data_ptr()
        # a layout that fails validation is rejected on the miss, never launched
        n = len(env.captured)
        bad = (l1[0].to(torch.float8_e4m3fn), l1[1])  # wrong dtype, new memory
        with pytest.raises(AssertionError, match="packed FP4"):
            _call(y, bad, l2, buf)
        with pytest.raises(AssertionError, match="int32"):
            _call(y, (l1[0], l1[1].to(torch.float32)), l2, buf)
        assert len(env.captured) == n
        # a view of the same memory with another dtype / strides is not the recorded operand
        transposed = (l2[0].transpose(1, 2).contiguous().transpose(1, 2), l2[1])
        with pytest.raises(AssertionError, match="contiguous"):
            _call(y, l1, transposed, buf)
        assert len(env.captured) == n
    finally:
        buf.destroy()


def test_stats_and_shared_operands_are_part_of_the_record(env):
    buf = compat.SymmBuffer(_ep1_group(), E, 1920, TOPK, H, I, num_shared_experts=1)
    l1, l2, s1, s2 = _weights(S=1)
    stats = torch.zeros(E, dtype=torch.int32, device="cuda")
    y = _y(100)
    try:
        _call(y, l1, l2, buf, s1, s2, stats)
        _call(y, l1, l2, buf, s1, s2, stats)
        assert env.validations == 1
        _call(y, l1, l2, buf, s1, s2, None)  # no stats: another operand set
        assert env.validations == 2 and env.captured[-1][1] == 0
        _call(
            y, l1, l2, buf, s1, s2, torch.zeros(E, dtype=torch.int32, device="cuda")
        )  # new stats tensor
        assert env.validations == 3
        n = len(env.captured)
        with pytest.raises(AssertionError, match="int32"):
            _call(
                y, l1, l2, buf, s1, s2, torch.zeros(E, dtype=torch.int64, device="cuda")
            )
        with pytest.raises(AssertionError, match="elements"):
            _call(
                y,
                l1,
                l2,
                buf,
                s1,
                s2,
                torch.zeros(E + 1, dtype=torch.int32, device="cuda"),
            )
        assert (
            env.validations == 5 and len(env.captured) == n
        )  # both rejected on the validating path, nothing launched
        s1_new = (torch.zeros_like(s1[0]), s1[1])
        _call(y, l1, l2, buf, s1_new, s2, stats)  # shared weights swapped
        assert env.validations == 6 and env.captured[-1][8] == s1_new[0].data_ptr()
    finally:
        buf.destroy()


def test_symmetric_buffer_recreated_over_the_same_memory_or_destroyed(env):
    group = _ep1_group()
    a = compat.SymmBuffer(group, E, 1920, TOPK, H, I)
    l1, l2, _, _ = _weights()
    y = _y(100)
    try:
        _call(y, l1, l2, a)
        _call(y, l1, l2, a)
        assert env.validations == 1
        b = compat.SymmBuffer(
            group, E, 1920, TOPK, H, I, base=a
        )  # another buffer OBJECT over the very same memory (same address, same peer table)
        assert b.base_ptr == a.base_ptr and b is not a
        _call(y, l1, l2, b)
        assert env.validations == 2 and env.captured[-1][3] == (b.base_ptr, (0,), 0, 1)
        _call(y, l1, l2, b)
        _call(y, l1, l2, a)  # a's record is still its own
        assert env.validations == 2
        # an object that died and whose id() may be recycled by a new buffer never matches the old record
        del b
        gc.collect()
        c = compat.SymmBuffer(group, E, 1920, TOPK, H, I, base=a)
        _call(y, l1, l2, c)
        assert env.validations == 3
        # a destroyed buffer is an error on the warm path exactly like on the validating path
        c.destroy()
        with pytest.raises((AttributeError, AssertionError)):
            _call(y, l1, l2, c)
        assert (
            env.validations == 4 or env.validations == 3
        )  # the miss went to the validating path, which raised
    finally:
        a.destroy()


def test_geometry_shared_experts_and_t_max_key_separate_records(env):
    group = _ep1_group()
    l1, l2, s1, s2 = _weights(S=1)
    y = _y(100)
    bufs = {
        "S0_T1920": compat.SymmBuffer(group, E, 1920, TOPK, H, I),
        "S1_T1920": compat.SymmBuffer(group, E, 1920, TOPK, H, I, num_shared_experts=1),
        "S1_T3840": compat.SymmBuffer(group, E, 3840, TOPK, H, I, num_shared_experts=1),
    }
    try:
        for _ in range(3):
            _call(y, l1, l2, bufs["S0_T1920"])
            _call(y, l1, l2, bufs["S1_T1920"], s1, s2)
            _call(y, l1, l2, bufs["S1_T3840"], s1, s2)
        assert env.validations == 3 and len(env.captured) == 9
        assert {(b[1], b[2]) for b in env.built} == {(0, 1920), (1, 1920), (1, 3840)}
        # a call whose token count exceeds the buffer is rejected on the warm path
        with pytest.raises(AssertionError, match="exceeds"):
            _call(_y(1921), l1, l2, bufs["S0_T1920"])
    finally:
        for b in bufs.values():
            b.destroy()


def test_per_die_mode_kernel_options_and_factory_invalidate(env):
    frontend = env.frontend
    buf = compat.SymmBuffer(_ep1_group(), E, 1920, TOPK, H, I)
    l1, l2, _, _ = _weights()
    y = _y(100)
    try:
        _call(y, l1, l2, buf)
        _call(y, l1, l2, buf)
        assert env.validations == 1
        frontend.set_per_die_mode(
            False
        )  # forced plain scheduling: a different mode, validated again, then warm again
        _call(y, l1, l2, buf)
        _call(y, l1, l2, buf)
        assert env.validations == 2
        frontend.set_per_die_mode(None)
        n_built = len(env.built)
        frontend.set_kernel_options(
            tag_mode="gridid"
        )  # other build options -> another runner
        _call(y, l1, l2, buf)
        _call(y, l1, l2, buf)
        assert env.validations >= 3 and len(env.built) == n_built + 1
        v = env.validations
        with frontend.kernel_options_scope(frontend.KernelOptions(block_m=64)):
            _call(y, l1, l2, buf)
            assert env.built[-1][0] == 64
        _call(
            y, l1, l2, buf
        )  # the scope restored the options: the earlier record is valid again
        assert env.validations == v + 1
        frontend.set_kernel_factory(
            env.factory
        )  # a new factory forgets runners AND the bound launchers made from them
        assert not frontend._call_records
        n_built = len(env.built)
        _call(y, l1, l2, buf)
        assert len(env.built) > n_built
    finally:
        buf.destroy()


def test_per_call_output_is_still_validated_on_the_warm_path(env):
    buf = compat.SymmBuffer(_ep1_group(), E, 1920, TOPK, H, I)
    l1, l2, _, _ = _weights()
    y = _y(100)
    try:
        _call(y, l1, l2, buf)
        _call(y, l1, l2, buf)
        n = len(env.captured)
        with pytest.raises(AssertionError, match="bfloat16"):
            _call(
                torch.empty((100, H), dtype=torch.float16, device="cuda"), l1, l2, buf
            )
        with pytest.raises(AssertionError, match="bfloat16"):
            _call(
                torch.empty((100, H + 128), dtype=torch.bfloat16, device="cuda"),
                l1,
                l2,
                buf,
            )
        with pytest.raises(AssertionError, match="contiguous"):
            _call(
                torch.empty((H, 100), dtype=torch.bfloat16, device="cuda").t(),
                l1,
                l2,
                buf,
            )
        with pytest.raises(AssertionError, match="16 B aligned"):
            _call(
                torch.empty(100 * H + 8, dtype=torch.bfloat16, device="cuda")[
                    1 : 1 + 100 * H
                ].view(100, H),
                l1,
                l2,
                buf,
            )
        with pytest.raises(AssertionError, match="exceeds"):
            _call(_y(1921), l1, l2, buf)
        assert len(env.captured) == n
        _call(_y(0), l1, l2, buf)  # an empty batch is a valid launch
        assert env.captured[-1][2] == 0
    finally:
        buf.destroy()


def test_the_call_does_not_enter_torch_function_modes(env):
    """``torch.set_default_device("cuda")`` (DeepGEMM's init_dist, vLLM) installs a Python ``TorchFunctionMode`` that every tensor method call goes through."""
    from torch.overrides import TorchFunctionMode

    class Count(TorchFunctionMode):
        n = 0

        def __torch_function__(self, func, types, args=(), kwargs=None):
            Count.n += 1
            return func(*args, **(kwargs or {}))

    buf = compat.SymmBuffer(_ep1_group(), E, 1920, TOPK, H, I, num_shared_experts=1)
    l1, l2, s1, s2 = _weights(S=1)
    stats = torch.zeros(E, dtype=torch.int32, device="cuda")
    y = _y(100)
    y_other = _y(7)
    try:
        _call(y, l1, l2, buf, s1, s2, stats)  # validating call (and the first record)
        with Count():
            _call(y, l1, l2, buf, s1, s2, stats)  # warm
            assert Count.n == 0, (
                f"{Count.n} tensor calls went through the Python torch-function mode on the warm path"
            )
            Count.n = 0
            _call(
                y_other, l1, l2, buf, s1, s2, stats
            )  # another BLOCK_M: validates (cold path of the same record)
            assert Count.n == 0, (
                f"{Count.n} tensor calls went through the Python torch-function mode on the validating path"
            )
    finally:
        buf.destroy()


def test_records_pin_no_tensors(env):
    buf = compat.SymmBuffer(_ep1_group(), E, 1920, TOPK, H, I, num_shared_experts=1)
    l1, l2, s1, s2 = _weights(S=1)
    y = _y(100)
    try:
        _call(y, l1, l2, buf, s1, s2)
        _call(y, l1, l2, buf, s1, s2)
        refs = [weakref.ref(t) for t in (l1[0], l1[1], l2[0], l2[1], s1[0], s2[0])]
        del l1, l2, s1, s2
        gc.collect()
        assert all(r() is None for r in refs), (
            "the call records keep weight tensors alive"
        )
    finally:
        buf.destroy()


def test_runners_without_a_bound_launcher_are_called_directly(env):
    """Custom kernel factories / test doubles (plain callables) keep the pre-record behavior: validate and call on every call."""
    calls = []
    env.frontend.set_kernel_factory(lambda cfg: lambda *a: calls.append(a))
    buf = compat.SymmBuffer(_ep1_group(), E, 1920, TOPK, H, I)
    l1, l2, _, _ = _weights()
    y = _y(100)
    try:
        for _ in range(3):
            _call(y, l1, l2, buf)
        assert (
            env.validations == 3 and len(calls) == 3 and not env.frontend._call_records
        )
        assert calls[0][0] is y and calls[0][3] is buf and calls[0][4] is l1[0]
    finally:
        buf.destroy()


def test_backend_options_object_is_stable_and_follows_the_config(env):
    """The moe_ep backend enters ``kernel_options_scope(self._options())`` on every launch: one options object per option set (the call records are keyed on it)."""
    from flashinfer.experimental.sm107_deepgemm_megamoe import backend
    from flashinfer.experimental.sm107_deepgemm_megamoe import config as cfgmod

    cfg = cfgmod.Sm107_Mxfp8_Mxfp4_Bf16_DeepgemmCutedsl_MegaMoeConfig(
        intermediate_size=I,
        top_k=TOPK,
        gate_up_clamp=10.0,
        allow_unvalidated_geometry=True,
    )
    kernel = backend.Sm107DeepgemmCutedslMegaKernelBackend(cfg)
    first = kernel._options()
    assert kernel._options() is first and first.tag_mode == "epoch"
    cfg.tag_mode = (
        "gridid"  # the user-facing config is a mutable dataclass: the options follow it
    )
    second = kernel._options()
    assert (
        second is not first
        and second.tag_mode == "gridid"
        and kernel._options() is second
    )

    buf = compat.SymmBuffer(_ep1_group(), E, 1920, TOPK, H, I)
    l1, l2, _, _ = _weights()
    y = _y(100)
    try:
        for _ in range(
            4
        ):  # the backend's launch pattern: a fresh scope around every call
            with env.frontend.kernel_options_scope(kernel._options()):
                _call(y, l1, l2, buf)
        assert env.validations == 1 and len(env.captured) == 4
    finally:
        buf.destroy()


def test_equal_option_sets_built_by_different_callers_share_records(env):
    """A fresh ``KernelOptions(...)`` around every launch (the pre-memo backend pattern, user code) is interned: it must not create a record per call."""
    frontend = env.frontend
    buf = compat.SymmBuffer(_ep1_group(), E, 1920, TOPK, H, I)
    l1, l2, _, _ = _weights()
    y = _y(100)
    try:
        for _ in range(4):
            with frontend.kernel_options_scope(
                frontend.KernelOptions(block_m=64, max_stages=3)
            ):
                _call(y, l1, l2, buf)
        assert (
            env.validations == 1
            and len(env.captured) == 4
            and len(frontend._call_records) == 1
        )
        assert frontend.set_kernel_options(
            frontend.KernelOptions(tag_mode="gridid")
        ) is frontend.set_kernel_options(frontend.KernelOptions(tag_mode="gridid"))
    finally:
        buf.destroy()
