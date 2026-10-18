# SPDX-License-Identifier: Apache-2.0
# Port of DeepGEMM (MIT License, Copyright (c) 2025 DeepSeek; see licenses/LICENSE.deepgemm.txt) sm107_fp8_fp4_mega_moe.cuh / numerics tests.
"""Non-default L1-epilogue numerics vs DeepGEMM at EP1, bit-exact: ``activation_clamp`` (None / inf: compiled out; 6.0 BF16-exact; 7.3 and 0.1 NOT BF16-exact; 0.0),
``activation_alpha`` != 1 (``sigmoid(alpha * gate)``), ``activation_beta`` != 0 (``up + beta`` after the clamp), ``fast_math=False`` (IEEE ``expf`` / true division), at
several BLOCK_M, with and without the shared expert (S = 1) and the multi-expert shared slot (S = 2).

Every case runs ``deep_gemm.fp8_fp4_mega_moe(activation_clamp=, activation_alpha=, activation_beta=, fast_math=)`` and the port with the SAME numerics on the same staged
inputs: the FULL kernel (``MODE_ALL``) must reproduce DeepGEMM's ``y`` bit-for-bit (3 back-to-back launches) and, where the routed pool fits the ring, the GEMM-only replay
must reproduce ``l2_token`` / ``l2_sf`` / ``combine_token`` / ``y`` (see ``replay_utils``).  The pre-activations are ~N(0, 5120 * x_scale^2): ``x_scale`` 1 saturates every
clamp (std ~ 72; also exercises the ``expf`` overflow / underflow tails), 1/16 puts the std at ~ 4.5 (6.0 / 7.3 clamp a fraction of the values, 0.1 nearly all), 1/512
at ~ 0.14 (0.1 clamps a fraction).  ``test_knobs_change_deepgemm_output`` proves that every knob changes DeepGEMM's own result in those regimes (the comparisons are
not vacuous).  Each distinct (BLOCK_M, S, clamp, alpha, beta, fast_math) is one DeepGEMM JIT compile + one DSL compile on a cold cache.

    pytest tests/experimental/test_sm107_dg_gemm_core_numerics.py -x -q
    SM107_NUMERICS_FULL=1 pytest tests/experimental/test_sm107_dg_gemm_core_numerics.py -q -k full_product     # clamp x alpha x beta x BLOCK_M x S (216 compiles)
"""

import math
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

deep_gemm = pytest.importorskip("deep_gemm")


@pytest.fixture(scope="module", autouse=True)
def _deepgemm_sm107_kernel():
    """DeepGEMM dispatches to its SM107 MegaMoE kernel only with ``DG_MEGA_MOE_SM107=1`` (otherwise the reference below would be its SM100 kernel): select it for the tests of this module only."""
    with pytest.MonkeyPatch.context() as mp:
        if "DG_MEGA_MOE_SM107" not in os.environ:
            mp.setenv("DG_MEGA_MOE_SM107", "1")
        yield


def _port_has_ieee_fast_math() -> bool:
    """``fast_math=False`` (IEEE ``expf`` / true division) is implemented by ``ptx_g.expf_ieee`` + ``ptx_g.div_rn_f32`` + the epilogue branch; builds without them
    still refuse it at trace time, the tests that need it are skipped there (they run, and must pass, as soon as the build has the branch)."""
    try:
        from flashinfer.experimental.sm107_deepgemm_megamoe.kernel import ptx_g
    except Exception:  # noqa: BLE001 - no DSL on this machine: the whole module is skipped anyway
        return False
    return hasattr(ptx_g, "expf_ieee") and hasattr(ptx_g, "div_rn_f32")


HAS_IEEE = _port_has_ieee_fast_math()
needs_ieee = pytest.mark.skipif(
    not HAS_IEEE,
    reason="this build has no fast_math=False branch (ptx_g.expf_ieee missing)",
)

INF = math.inf
MS = (
    64,
    300,
    1500,
)  # 1500 wraps the ring of the E = 32 environment (full-kernel check only)
SCALES = (1.0, 1 / 16, 1 / 512)


@pytest.fixture(scope="module")
def env32():
    return R.make_env(32, 1920)


@pytest.fixture(scope="module")
def env32_s1():
    return R.make_env(32, 1920, num_shared_experts=1)


@pytest.fixture(scope="module")
def env32_s2():
    return R.make_env(32, 1920, num_shared_experts=2)


def _unit_failures(env, bm, numerics, ms=MS, scales=SCALES, replay=True):
    """One compile unit: the FULL kernel vs DeepGEMM's y for every (M, x_scale) and the GEMM-only replay of the cases whose pool fits the ring.  Returns failure strings."""
    fails = []
    tag = f"S={env.num_shared_experts} BM={bm} {numerics}"
    for m in ms:
        for sc in scales:
            res = R.full_kernel_case(env, m, bm, numerics=numerics, x_scale=sc)
            if not res.passed:
                fails.append(f"{tag} full M={m} x_scale={sc:g}: {res} {res.message}")
            if replay:
                res = R.replay_case(env, m, bm, numerics=numerics, x_scale=sc)
                if not res.skipped and not res.passed:
                    fails.append(f"{tag} replay M={m} x_scale={sc:g}: {res}")
    return fails


def _num(clamp, alpha, beta, fast=True):
    return dict(clamp=clamp, alpha=alpha, beta=beta, fast_math=fast)


# (S, BLOCK_M, clamp, alpha, beta, fast_math)
UNITS = [
    # --- clamp: compile-out (None == inf), BF16-exact (6.0), BF16-inexact (7.3, 0.1), degenerate (0.0)
    (0, 128, None, 1.0, 0.0, True),
    (0, 128, INF, 1.0, 0.0, True),
    (0, 128, 6.0, 1.0, 0.0, True),
    (0, 128, 7.3, 1.0, 0.0, True),
    (0, 128, 0.1, 1.0, 0.0, True),
    (0, 128, 0.0, 1.0, 0.0, True),
    # --- alpha / beta alone and combined (clamp 10: DeepSeek-V4.1; None: no clamp; 7.0 / 1.702 / +1: the GPT-OSS style SwiGLU)
    (0, 128, 10.0, 0.7, 0.0, True),
    (0, 128, 10.0, 1.702, 0.0, True),
    (0, 128, 10.0, 1.0, 1.0, True),
    (0, 128, 10.0, 1.0, -0.5, True),
    (0, 128, None, 1.702, -0.5, True),
    (0, 128, 7.3, 0.7, 1.0, True),
    (0, 128, 7.0, 1.702, 1.0, True),
    (0, 128, INF, 0.0, 0.0, True),
    (0, 128, INF, -1.0, 0.0, True),
    # --- the other BLOCK_M extremes and the shared-expert builds (S = 1: shared tasks run the same epilogue code on E4M3 x E4M3 stages)
    (0, 16, 7.3, 1.702, -0.5, True),
    (0, 240, 0.1, 0.7, 1.0, True),
    (0, 16, None, 0.7, 0.0, True),
    (0, 240, 6.0, 1.0, -0.5, True),
    (1, 128, 7.3, 0.7, 1.0, True),
    (1, 16, 6.0, 1.702, -0.5, True),
    (1, 240, None, 0.7, 1.0, True),
    (1, 128, 0.1, 1.0, 0.0, True),
    # --- fast_math = False (IEEE expf / true division): defaults, clamp 10, alpha / beta / clamp combinations, other BLOCK_M, shared expert
    (0, 128, None, 1.0, 0.0, False),
    (0, 128, 10.0, 1.0, 0.0, False),
    (0, 128, 6.0, 0.7, 1.0, False),
    (0, 128, 7.3, 1.702, -0.5, False),
    (0, 128, 0.1, 1.0, 0.0, False),
    (0, 16, 10.0, 1.702, 0.0, False),
    (0, 240, None, 0.7, 1.0, False),
    (1, 128, 10.0, 1.0, 0.0, False),
    (1, 240, 7.3, 1.702, -0.5, False),
    (1, 16, None, 1.0, 0.0, False),
]


def _unit_id(u):
    s, bm, c, a, b, f = u
    return f"S{s}-BM{bm}-clamp{c}-a{a}-b{b}-{'fast' if f else 'ieee'}"


def _unit_params(units):
    return [
        pytest.param(u, id=_unit_id(u), marks=() if u[5] else needs_ieee) for u in units
    ]


@pytest.mark.parametrize("unit", _unit_params(UNITS))
def test_numerics_vs_deepgemm(env32, env32_s1, unit):
    s, bm, clamp, alpha, beta, fast = unit
    env = env32_s1 if s else env32
    fails = _unit_failures(env, bm, _num(clamp, alpha, beta, fast))
    assert not fails, "\n".join(fails)


FULL_PRODUCT = [
    (s, bm, c, a, b, True)
    for s in (0, 1)
    for bm in (16, 128, 240)
    for c in (INF, 7.3, 0.1, 6.0)
    for a in (1.0, 0.7, 1.702)
    for b in (0.0, 1.0, -0.5)
]


@pytest.mark.skipif(
    not os.environ.get("SM107_NUMERICS_FULL"),
    reason="set SM107_NUMERICS_FULL=1: the full clamp x alpha x beta x BLOCK_M x S product (216 compile units)",
)
@pytest.mark.parametrize("unit", FULL_PRODUCT, ids=[_unit_id(u) for u in FULL_PRODUCT])
def test_numerics_full_product(env32, env32_s1, unit):
    s, bm, clamp, alpha, beta, fast = unit
    fails = _unit_failures(env32_s1 if s else env32, bm, _num(clamp, alpha, beta, fast))
    assert not fails, "\n".join(fails)


def test_knobs_change_deepgemm_output(env32):
    """The comparisons above are not vacuous: in the regime where each knob matters, DeepGEMM's OWN y differs from the default-numerics y (clamp None, alpha 1, beta 0)."""
    bm, m = 128, 300
    base = {
        sc: R.full_kernel_case(
            env32, m, bm, numerics=_num(None, 1.0, 0.0), x_scale=sc, keep_ref=True
        ).y_ref
        for sc in SCALES
    }

    def changed(numerics, sc):
        y = R.full_kernel_case(
            env32, m, bm, numerics=numerics, x_scale=sc, keep_ref=True
        ).y_ref
        return float(
            (y.view(torch.int16) != base[sc].view(torch.int16)).float().mean().item()
        )

    assert (
        changed(_num(6.0, 1.0, 0.0), 1 / 16) > 0.01
    )  # clamp 6.0 clamps ~18 % of the pre-activations
    assert changed(_num(7.3, 1.0, 0.0), 1 / 16) > 0.01
    assert changed(_num(0.1, 1.0, 0.0), 1 / 512) > 0.01  # the std-0.14 regime
    assert changed(_num(0.1, 1.0, 0.0), 1 / 16) > 0.5  # saturated
    for sc in SCALES:
        assert changed(_num(None, 0.7, 0.0), sc) > 0.01
        assert changed(_num(None, 1.702, 0.0), sc) > 0.01
        assert changed(_num(None, 1.0, 1.0), sc) > 0.01
        assert changed(_num(None, 1.0, -0.5), sc) > 0.01


def test_inf_and_none_clamp_are_the_same_variant(env32):
    """``activation_clamp=None`` and ``float('inf')`` compile the clamp out (one DeepGEMM / DSL compile key): identical results on both sides."""
    for m in (64, 300):
        a = R.full_kernel_case(
            env32, m, 128, numerics=_num(None, 1.0, 0.0), keep_ref=True
        )
        b = R.full_kernel_case(
            env32, m, 128, numerics=_num(INF, 1.0, 0.0), keep_ref=True
        )
        assert a.passed and b.passed
        assert torch.equal(a.y_ref.view(torch.int16), b.y_ref.view(torch.int16))


@needs_ieee
@pytest.mark.parametrize("dg_fast", [False, True])
def test_fast_math_variants_are_distinguishable(env32, dg_fast):
    """Negative control of the fast_math comparisons: the port built with the OTHER fast_math value must NOT reproduce DeepGEMM (otherwise the two branches would be
    indistinguishable and the fast_math=False cases above would test nothing).  Pre-activations of std ~ 0.14 (x_scale 1/512), where the approximate and the IEEE forms
    differ in the last bits of ``gate * sigmoid(gate)``."""
    res = R.replay_case(
        env32,
        300,
        128,
        numerics=_num(None, 1.0, 0.0, dg_fast),
        x_scale=1 / 512,
        cfg_replace={"fast_math": not dg_fast},
    )
    assert not res.passed and (
        res.bad_l2_rows > 0 or res.bad_l2_sf_tokens > 0 or res.bad_combine_rows > 0
    ), str(res)
    # ... and with matching fast_math it passes (same data, same compile keys as the units above)
    res = R.replay_case(
        env32, 300, 128, numerics=_num(None, 1.0, 0.0, dg_fast), x_scale=1 / 512
    )
    assert res.passed, str(res)


# ---------------------------------------------------------------------------------------------------------------------
# num_shared_experts = 2: one merged shared expert of intermediate width 2 * I (DeepGEMM derives S from the shared weights' shape); the shared slot of the combine
# buffer is still ONE slot (index kNumTopk) that sums over K = S * I
# ---------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "bm,m", [(16, 24), (16, 100), (128, 300), (128, 700), (240, 400), (240, 900)]
)
def test_shared2_replay(env32_s2, bm, m):
    res = R.replay_case(env32_s2, m, bm)
    if res.skipped:
        pytest.skip(res.message)
    assert res.passed, str(res)


@pytest.mark.parametrize(
    "bm,m",
    [
        (16, 24),
        (16, 300),
        (32, 100),
        (64, 600),
        (128, 300),
        (128, 1500),
        (192, 700),
        (240, 900),
        (240, 1900),
        (128, 1),
        (128, 0),
    ],
)
def test_shared2_full_kernel(env32_s2, bm, m):
    res = R.full_kernel_case(env32_s2, m, bm)
    assert res.passed, str(res) + " " + res.message


@pytest.mark.parametrize("fast", [True, pytest.param(False, marks=needs_ieee)])
def test_shared2_with_numerics(env32_s2, fast):
    fails = _unit_failures(
        env32_s2, 128, _num(7.3, 0.7, 1.0, fast), ms=(64, 300), scales=(1.0, 1 / 16)
    )
    assert not fails, "\n".join(fails)


# ---------------------------------------------------------------------------------------------------------------------
# The kFastMath=false building blocks against nvcc 13.4 (DeepGEMM's JIT flags): ``ptx_g.expf_ieee`` vs ``expf`` over ALL 2^32 float bit patterns, ``ptx_g.div_rn_f32`` vs
# ``a / b``; control: the fast-path forms (``ex2.approx(x * log2 e)``) vs ``__expf``.  The reference kernels are compiled with nvcc and launched through the driver API.
# ---------------------------------------------------------------------------------------------------------------------
_REF_CU = r"""
extern "C" __global__ void ref_expf(float* y, unsigned base, unsigned n) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = expf(__uint_as_float(base + i));
}
extern "C" __global__ void ref_fastexpf(float* y, unsigned base, unsigned n) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = __expf(__uint_as_float(base + i));
}
extern "C" __global__ void ref_div(float* y, const float* a, const float* b, unsigned n) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a[i] / b[i];
}
"""
_NVCC_FLAGS = [
    "--gpu-architecture=sm_100f",
    "-O3",
    "--compiler-options=-O3",
    "-std=c++20",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
]  # DeepGEMM's flags (meta.json)


def _nvcc_reference_module(tmp_path):
    import shutil
    import subprocess

    import cuda.bindings.driver as cuda

    nvcc = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
    if not os.path.exists(nvcc):
        pytest.skip("nvcc not found")
    src, out = tmp_path / "ref.cu", tmp_path / "ref.cubin"
    src.write_text(_REF_CU)
    r = subprocess.run(
        [nvcc, str(src), "--cubin", "--output-file", str(out)] + _NVCC_FLAGS,
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        pytest.skip(f"nvcc failed: {r.stderr[:300]}")
    torch.zeros(
        1, device="cuda"
    )  # the primary context must exist before the driver-API module load
    err, mod = cuda.cuModuleLoadData(out.read_bytes())
    assert int(err) == 0, err
    return mod


def _launch_nvcc_kernel(mod, name, grid, *args):
    import ctypes

    import cuda.bindings.driver as cuda

    err, fn = cuda.cuModuleGetFunction(mod, name.encode())
    assert int(err) == 0, (name, err)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    ptrs = (ctypes.c_void_p * len(args))(*[ctypes.addressof(a) for a in args])
    (err,) = cuda.cuLaunchKernel(
        fn, grid, 1, 1, 256, 1, 1, 0, stream, ctypes.addressof(ptrs), 0
    )
    assert int(err) == 0, err


def _i32(u):
    return u - (1 << 32) if u >= (1 << 31) else u


def _bit_mismatches(a, b):
    """Elements whose bit patterns differ (NaN == NaN whatever the payload)."""
    fa, fb = a.view(torch.float32), b.view(torch.float32)
    return int(((a != b) & ~(torch.isnan(fa) & torch.isnan(fb))).sum().item())


@pytest.mark.parametrize("kind", [pytest.param("ieee", marks=needs_ieee), "fast"])
def test_expf_building_blocks_vs_nvcc_all_float32(tmp_path, kind):
    """``ptx_g.expf_ieee`` == nvcc ``expf`` (and the fast form == ``__expf``) for every one of the 2^32 float bit patterns."""
    import ctypes

    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    from cutlass import Int32, Int64

    from flashinfer.experimental.sm107_deepgemm_megamoe.kernel import ptx, ptx_g

    class Probe:
        @cute.jit
        def __call__(self, y_addr: Int64, base: Int32, n: Int32, stream: cuda.CUstream):
            self.kernel(y_addr, base, n).launch(
                grid=[(n + 255) // 256, 1, 1], block=[256, 1, 1], stream=stream
            )

        @cute.kernel
        def kernel(self, y_addr: Int64, base: Int32, n: Int32):
            tidx, _, _ = cute.arch.thread_idx()
            bidx, _, _ = cute.arch.block_idx()
            i = bidx * 256 + tidx
            if i < n:
                a = ptx_g.bits_f32(base + i)
                if cutlass.const_expr(kind == "ieee"):
                    r = ptx_g.expf_ieee(a)
                else:
                    r = ptx.ex2_approx(
                        a * cutlass.Float32(1.4426950408889634)
                    )  # __expf(a): mul.f32 log2e + ex2.approx.f32
                ptx.st_f32(y_addr + Int64(i) * 4, r)

    mod = _nvcc_reference_module(tmp_path)
    chunk_log2 = 26
    n = 1 << chunk_log2
    y_ref = torch.empty(n, dtype=torch.int32, device="cuda")
    y_dsl = torch.empty(n, dtype=torch.int32, device="cuda")
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    compiled = cute.compile(
        Probe(), Int64(y_dsl.data_ptr()), Int32(0), Int32(n), stream
    )
    sent_ref, sent_dsl = (
        _i32(0xBF800000),
        _i32(0xC0000000),
    )  # negative: expf never returns one, so an unwritten element cannot pass
    bad = unwritten = 0
    for c in range(1 << (32 - chunk_log2)):
        base = c << chunk_log2
        y_ref.fill_(sent_ref)
        y_dsl.fill_(sent_dsl)
        _launch_nvcc_kernel(
            mod,
            "ref_expf" if kind == "ieee" else "ref_fastexpf",
            n // 256,
            ctypes.c_void_p(y_ref.data_ptr()),
            ctypes.c_uint32(base),
            ctypes.c_uint32(n),
        )
        compiled(Int64(y_dsl.data_ptr()), Int32(_i32(base)), Int32(n), stream)
        torch.cuda.synchronize()
        unwritten += int((y_ref == sent_ref).sum().item()) + int(
            (y_dsl == sent_dsl).sum().item()
        )
        bad += _bit_mismatches(y_dsl, y_ref)
        if (
            c == 15
        ):  # control: the buffers hold real expf values (inputs x in [2^-7, 2))
            xin = (
                (torch.arange(n, device="cuda", dtype=torch.int64) + base)
                .to(torch.int32)
                .view(torch.float32)
            )
            ref_exp = torch.exp(xin)
            assert (
                float(
                    ((y_dsl.view(torch.float32) - ref_exp).abs() / ref_exp).max().item()
                )
                < 1e-5
            )
    assert unwritten == 0 and bad == 0, (
        f"{kind}: {bad} mismatching results, {unwritten} unwritten elements"
    )


@needs_ieee
def test_div_rn_f32_vs_nvcc(tmp_path):
    """``ptx_g.div_rn_f32`` == nvcc ``a / b`` (and torch's IEEE division) on 2^24 random bit-pattern pairs and all pairs of special values (+-0, +-inf, NaN, subnormals ...)."""
    import ctypes

    import cuda.bindings.driver as cuda
    import cutlass.cute as cute
    from cutlass import Int32, Int64

    from flashinfer.experimental.sm107_deepgemm_megamoe.kernel import ptx, ptx_g

    class Probe:
        @cute.jit
        def __call__(
            self,
            y_addr: Int64,
            a_addr: Int64,
            b_addr: Int64,
            n: Int32,
            stream: cuda.CUstream,
        ):
            self.kernel(y_addr, a_addr, b_addr, n).launch(
                grid=[(n + 255) // 256, 1, 1], block=[256, 1, 1], stream=stream
            )

        @cute.kernel
        def kernel(self, y_addr: Int64, a_addr: Int64, b_addr: Int64, n: Int32):
            tidx, _, _ = cute.arch.thread_idx()
            bidx, _, _ = cute.arch.block_idx()
            i = bidx * 256 + tidx
            if i < n:
                ptx.st_f32(
                    y_addr + Int64(i) * 4,
                    ptx_g.div_rn_f32(
                        ptx.ld_f32(a_addr + Int64(i) * 4),
                        ptx.ld_f32(b_addr + Int64(i) * 4),
                    ),
                )

    mod = _nvcc_reference_module(tmp_path)
    gen = torch.Generator(device="cuda")
    gen.manual_seed(1234)
    a = torch.randint(
        -(2**31), 2**31 - 1, (1 << 24,), dtype=torch.int32, device="cuda", generator=gen
    )
    b = torch.randint(
        -(2**31), 2**31 - 1, (1 << 24,), dtype=torch.int32, device="cuda", generator=gen
    )
    special = [
        0x00000000,
        0x80000000,
        0x7F800000,
        0xFF800000,
        0x7FC00000,
        0x00000001,
        0x80000001,
        0x007FFFFF,
        0x807FFFFF,
        0x00800000,
        0x80800000,
        0x7F7FFFFF,
        0xFF7FFFFF,
        0x3F800000,
        0xBF800000,
        0x40000000,
        0x40400000,
        0x3DCCCCCD,
        0x3F000000,
        0x3F800001,
        0x3F7FFFFF,
        0x7E800000,
        0x01000000,
        0x00400000,
        0x00000100,
    ]
    sp = torch.tensor([_i32(v) for v in special], dtype=torch.int32, device="cuda")
    a = torch.cat([a, sp.repeat_interleave(len(special))])
    b = torch.cat([b, sp.repeat(len(special))])
    n = a.numel()
    y_ref = torch.full(
        (n,), _i32(0x7FC12345), dtype=torch.int32, device="cuda"
    )  # NaN payloads a division never produces
    y_dsl = torch.full((n,), _i32(0x7FC23456), dtype=torch.int32, device="cuda")
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    compiled = cute.compile(
        Probe(),
        Int64(y_dsl.data_ptr()),
        Int64(a.data_ptr()),
        Int64(b.data_ptr()),
        Int32(n),
        stream,
    )
    _launch_nvcc_kernel(
        mod,
        "ref_div",
        (n + 255) // 256,
        ctypes.c_void_p(y_ref.data_ptr()),
        ctypes.c_void_p(a.data_ptr()),
        ctypes.c_void_p(b.data_ptr()),
        ctypes.c_uint32(n),
    )
    compiled(
        Int64(y_dsl.data_ptr()),
        Int64(a.data_ptr()),
        Int64(b.data_ptr()),
        Int32(n),
        stream,
    )
    torch.cuda.synchronize()
    assert (
        int((y_ref == _i32(0x7FC12345)).sum().item()) == 0
        and int((y_dsl == _i32(0x7FC23456)).sum().item()) == 0
    )  # every element written
    assert _bit_mismatches(y_dsl, y_ref) == 0
    assert (
        _bit_mismatches(
            y_dsl, (a.view(torch.float32) / b.view(torch.float32)).view(torch.int32)
        )
        == 0
    )


# ---------------------------------------------------------------------------------------------------------------------
# cfg fields that are hashed into the compile key / parity manifest but NOT implemented by the CuteDSL kernel are refused at trace time instead
# of compiling an unchanged kernel under a different name.  The host factory only produces the defaults (tma_store_stages 2, epi_prefetch_stages 0, weight_prefetch 0,
# debug_flags 0), so no product path can trip these.
# ---------------------------------------------------------------------------------------------------------------------
_UNIMPLEMENTED_OPTIONS = [
    ("epi_prefetch_stages", 1),
    ("tma_store_stages", 3),
    ("debug_flags", 1),
    ("weight_prefetch", 1),
]


@pytest.mark.parametrize("field,value", _UNIMPLEMENTED_OPTIONS)
def test_unimplemented_cfg_options_are_refused_cpu(field, value):
    """CPU level: the role-entry checks accept the defaults, and refuse any other value with a message that names the cfg field and says 'not implemented by the CuteDSL port'."""
    import dataclasses
    import types

    from flashinfer.experimental.sm107_deepgemm_megamoe.heuristics import (
        get_sm107_kernel_config,
    )
    from flashinfer.experimental.sm107_deepgemm_megamoe.kernel import (
        epilogue,
        tma_warps,
    )

    cfg = get_sm107_kernel_config(
        1,
        32,
        1920,
        300,
        6,
        5120,
        2304,
        0,
        num_sms=212,
        activation_clamp=10.0,
        block_m=128,
    )
    check = (
        tma_warps.require_default_tma_options
        if field == "weight_prefetch"
        else epilogue.require_default_epilogue_options
    )
    check(cfg)  # the factory's defaults pass
    bad = types.SimpleNamespace(
        **{f.name: getattr(cfg, f.name) for f in dataclasses.fields(cfg)}
    )
    setattr(bad, field, value)
    with pytest.raises(
        NotImplementedError,
        match=rf"cfg\.{field}={value}.*not implemented by the CuteDSL port",
    ):
        check(bad)


def test_unimplemented_cfg_option_is_refused_at_compile():
    """Compile level: a config with weight_prefetch=1 (hashed into the compile key, ignored by TMA-B) must raise while the kernel is traced, naming the field."""
    import dataclasses

    from flashinfer.experimental.sm107_deepgemm_megamoe import jit as sjit
    from flashinfer.experimental.sm107_deepgemm_megamoe.heuristics import (
        get_sm107_kernel_config,
    )

    cfg = get_sm107_kernel_config(
        1,
        32,
        1920,
        300,
        6,
        5120,
        2304,
        0,
        num_sms=212,
        activation_clamp=10.0,
        block_m=128,
    )
    bad = dataclasses.replace(cfg, weight_prefetch=1)
    assert (
        bad.cache_key() != cfg.cache_key()
    )  # it IS part of the compile key: the unchanged kernel would have been cached under a different name
    with pytest.raises(
        Exception, match=r"weight_prefetch.*not implemented by the CuteDSL port"
    ):
        sjit.compile_kernel_object(sjit.make_build(bad, mode_mask=7))
