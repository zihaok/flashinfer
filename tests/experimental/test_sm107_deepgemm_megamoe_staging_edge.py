# SPDX-License-Identifier: Apache-2.0
# Port of DeepGEMM (MIT License, Copyright (c) 2025 DeepSeek; see licenses/LICENSE.deepgemm.txt) sm107_fp8_fp4_mega_moe.cuh / staging tests.
"""Edge-case inputs of the MegaMoE input staging: the single-launch Triton kernel (``staging_triton``, the default path) against the torch
fallback (``staging.per_token_cast_to_fp8`` / ``prepare_megamoe_inputs_torch``) and against DeepGEMM's own ``per_token_cast_to_fp8``
(``use_ue8m0=True, gran_k=32, use_packed_ue8m0=True``).

The existing staging tests use ``randn`` data only.  Here the inputs sit on the places where the arithmetic of the three implementations could differ:
``-0.0``, all-zero groups (amax clamp 1e-4), bf16 sub-normals, fp16 inputs, a group amax of exactly ``448 * 2^k`` and one bf16 ulp above / below it
(the Triton kernel divides with ``div.full.f32``, the reference multiplies by ``1 / scale``), values that saturate the E4M3 range, and non-finite values.

* Finite inputs and +-Inf lanes: all three implementations are bit-identical (``x_fp8`` bytes and packed ``x_sf``), for the untiled (``num_tokens < 64``)
  and the 8-row-tiled (``num_tokens >= 64``, with a partial last tile) specializations of the Triton kernel.
* NaN: DeepGEMM's cast and the torch fallback propagate it (``amax`` is NaN, the scale exponent byte of the group is 254, the NaN lane stays an E4M3 NaN); the
  Triton kernel ignores NaN lanes when it computes the group scale (``tl.max`` / ``tl.maximum`` have max-number semantics), so the scale of a group that
  contains a NaN differs (known deviation, ``test_triton_staging_nan_group_scale_matches_deepgemm`` is a strict xfail).  What is pinned for NaN inputs: the NaN
  lane stays NaN, every other row and every other group of the row is bit-identical in all implementations.
"""

import pytest

torch = pytest.importorskip("torch")

from flashinfer.experimental.sm107_deepgemm_megamoe import staging as ST  # noqa: E402

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 9),
    reason="needs a CUDA GPU with FP8 conversions (compute capability >= 8.9)",
)

H, TOPK, T_MAX = 5120, 6, 1920
GROUPS = H // 32
SMALL_M, TILED_M = (
    3,
    70,
)  # Triton kernel: BLOCK_M = 1 below 64 tokens, 8 from 64 on (70 = 8 tiles + a partial one)
E4M3_NAN_BYTES = (0x7F, 0xFF)
UE8M0_NAN_BYTE = 254  # exponent clamp of a NaN scale (DeepGEMM ceil_to_ue8m0)


def _bits(t):
    return (
        t.contiguous().view(torch.uint8)
        if t.dtype == torch.float8_e4m3fn
        else t.contiguous()
    )


def _dg_cast_or_none():
    """DeepGEMM's cast when ``deep_gemm`` is importable, else ``None`` (the torch fallback is the reference then; it is bit-exact with DeepGEMM's cast)."""
    try:
        from deep_gemm.utils.math import per_token_cast_to_fp8
    except ImportError:
        return None
    return per_token_cast_to_fp8


def _stage(fn, x):
    """Stage ``x`` through ``fn`` (a ``prepare_megamoe_inputs*`` function) with trivial routing; returns ``(x_fp8 bytes, x_sf)`` of the live rows."""
    m = x.shape[0]
    rows = max(T_MAX, m)
    ids = torch.zeros((m, TOPK), dtype=torch.int64, device="cuda")
    w = torch.zeros((m, TOPK), dtype=torch.float32, device="cuda")
    x_out = torch.zeros((rows, H), dtype=torch.float8_e4m3fn, device="cuda")
    sf_out = torch.zeros((rows, H // 128), dtype=torch.int32, device="cuda")
    idx_out = torch.zeros((rows, TOPK), dtype=torch.int64, device="cuda")
    w_out = torch.zeros((rows, TOPK), dtype=torch.float32, device="cuda")
    fn(x, w, ids, x_out, sf_out, idx_out, w_out)
    torch.cuda.synchronize()
    return _bits(x_out[:m]).clone(), sf_out[:m].clone()


def _triton(x):
    pytest.importorskip("triton")
    from flashinfer.experimental.sm107_deepgemm_megamoe.staging_triton import (
        prepare_megamoe_inputs_triton,
    )

    return _stage(prepare_megamoe_inputs_triton, x)


def _torch(x):
    return _stage(ST.prepare_megamoe_inputs_torch, x)


def _deepgemm(x):
    cast = _dg_cast_or_none()
    if cast is None:
        return None
    fp8, sf = cast(x, use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)
    return _bits(fp8).clone(), sf.clone()


def _randn(m, dtype, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn((m, H), generator=g, device="cuda").to(dtype)


def _embed(edge, m_total, seed):
    """``edge`` rows first, the rest random: the same rows are staged by the untiled and by the tiled specialization of the Triton kernel."""
    x = _randn(m_total, edge.dtype, seed)
    x[: edge.shape[0]] = edge
    return x


def _next_bf16(v, step):
    """The bf16 value ``step`` ulps away from the (positive) bf16 scalar tensor ``v``."""
    return (v.view(torch.int16) + step).view(torch.bfloat16)


# ----------------------------------------------------------------------------------------------------------------------------------------------
# edge rows (three rows per case)
# ----------------------------------------------------------------------------------------------------------------------------------------------
def _case_negative_zero(dtype):
    x = _randn(3, dtype, 1)
    x[0, :] = -0.0
    x[1, :64] = -0.0
    x[2, ::2] = -0.0
    x[2, 1::64] = 0.0
    return x


def _case_zero_groups(dtype):
    x = _randn(3, dtype, 2)
    x[0, :] = 0.0
    x[1].view(GROUPS, 32)[::2] = 0.0
    x[2, : H - 32] = 0.0
    return x


def _case_bf16_subnormals(dtype):
    x = (
        _randn(3, dtype, 3) * 1e-39
    )  # bf16 shares fp32's exponent range: 1e-39 is sub-normal
    x[1, :] = 1e-39
    x[2, :] = -9.18e-41  # the smallest bf16 sub-normal magnitude
    return x


def _case_amax_448_pow2(dtype):
    """Group ``g`` holds one value of exactly ``448 * 2^k`` (k = -20..20), the second row one bf16 ulp above it, the third one ulp below it."""
    x = torch.zeros((3, H), dtype=torch.bfloat16, device="cuda")
    for g in range(GROUPS):
        k = g % 41 - 20
        v = torch.tensor(448.0 * 2.0**k, dtype=torch.float32, device="cuda").to(
            torch.bfloat16
        )
        x[0, g * 32 + 3] = v
        x[1, g * 32 + 3] = _next_bf16(v, 1)
        x[2, g * 32 + 3] = _next_bf16(v, -1)
        x[0, g * 32 + 4] = -v / 2  # a second, smaller lane of the opposite sign
    return x


def _case_e4m3_saturation(dtype):
    x = _randn(3, dtype, 5) * 448.0
    x[1, ::32] = (
        448.0 * 1.0078125
    )  # the bf16 value just above the E4M3 maximum of a group whose amax is 448 * 1.0078
    x[2] = (
        _randn(1, dtype, 6)[0].clamp(-1.0, 1.0) * 3.0e38
    )  # near the bf16 maximum: the scale exponent is large but not clamped
    return x


def _case_tiny_amax(dtype):
    x = _randn(3, dtype, 7) * 1e-30  # every group clamps to the 1e-4 amax floor
    x[1, 100] = 1.5e-4  # just above the clamp
    x[2, 200] = 9.9e-5  # just below it
    return x


FINITE_CASES = {
    "negative_zero": _case_negative_zero,
    "zero_groups": _case_zero_groups,
    "bf16_subnormals": _case_bf16_subnormals,
    "amax_448_pow2": _case_amax_448_pow2,
    "e4m3_saturation": _case_e4m3_saturation,
    "tiny_amax": _case_tiny_amax,
}


@pytest.mark.parametrize("m_total", [SMALL_M, TILED_M], ids=["untiled", "tiled"])
@pytest.mark.parametrize("case", sorted(FINITE_CASES))
def test_triton_staging_is_bit_identical_on_finite_edge_cases(case, m_total):
    pytest.importorskip("triton")
    x = _embed(FINITE_CASES[case](torch.bfloat16), m_total, seed=100 + m_total)
    ref = _torch(x)
    tri = _triton(x)
    assert torch.equal(tri[0], ref[0]), (
        f"{case}: Triton x_fp8 differs from the torch fallback"
    )
    assert torch.equal(tri[1], ref[1]), (
        f"{case}: Triton x_sf differs from the torch fallback"
    )
    default = _stage(
        ST.prepare_megamoe_inputs, x
    )  # the dispatcher (Triton when available)
    assert torch.equal(default[0], ref[0]) and torch.equal(default[1], ref[1])
    dg = _deepgemm(x)  # None when deep_gemm is not importable
    if dg is not None:
        assert torch.equal(ref[0], dg[0]) and torch.equal(ref[1], dg[1]), (
            f"{case}: the torch fallback differs from DeepGEMM's cast"
        )


@pytest.mark.parametrize("m_total", [SMALL_M, TILED_M], ids=["untiled", "tiled"])
def test_triton_staging_is_bit_identical_for_fp16_inputs(m_total):
    pytest.importorskip("triton")
    x = _embed(_randn(3, torch.float16, 8) * 3, m_total, seed=200 + m_total)
    x[0, :32] = 6.0e-8  # fp16 sub-normal range
    x[1, :32] = 65504.0  # the fp16 maximum
    x[1, 40] = -0.0
    x[2, ::2] = 0.0
    ref = _torch(x)
    tri = _triton(x)
    assert torch.equal(tri[0], ref[0]) and torch.equal(tri[1], ref[1])
    dg = _deepgemm(x)
    if dg is not None:
        assert torch.equal(ref[0], dg[0]) and torch.equal(ref[1], dg[1])


def _infinite_rows():
    x = _randn(4, torch.bfloat16, 9)
    x[0, 5] = float("inf")
    x[1, 5] = float("-inf")
    x[2, 64:96] = float("inf")  # a whole group
    x[3, 7], x[3, 8] = float("inf"), float("-inf")  # both signs in one group
    return x


@pytest.mark.parametrize("m_total", [SMALL_M + 1, TILED_M], ids=["untiled", "tiled"])
def test_infinite_lanes_are_staged_identically_by_all_implementations(m_total):
    """+-Inf: the group scale is the clamped exponent 254 and the data saturates the same way in DeepGEMM's cast, the torch fallback and the Triton kernel."""
    pytest.importorskip("triton")
    x = _embed(_infinite_rows(), m_total, seed=300 + m_total)
    ref = _torch(x)
    tri = _triton(x)
    assert torch.equal(tri[0], ref[0]) and torch.equal(tri[1], ref[1])
    dg = _deepgemm(x)
    if dg is not None:
        assert torch.equal(ref[0], dg[0]) and torch.equal(ref[1], dg[1])
    sf_bytes = ref[1].view(torch.uint8).view(m_total, GROUPS)
    assert (
        int(sf_bytes[0, 0]) == UE8M0_NAN_BYTE
        and int(sf_bytes[1, 0]) == UE8M0_NAN_BYTE
        and int(sf_bytes[2, 2]) == UE8M0_NAN_BYTE
    )
    assert (
        int(sf_bytes[0, 1]) != UE8M0_NAN_BYTE
    )  # the other groups of the row keep their own scale


def _nan_rows():
    x = _randn(4, torch.bfloat16, 10)
    x[0, :32] = float("nan")  # a whole group
    x[1, 70] = float("nan")  # one lane of group 2
    x[2, 40] = float("nan")
    x[2, 41] = float("inf")  # NaN and Inf in one group
    return x  # row 3 stays finite


NAN_GROUP = {
    0: 0,
    1: 70 // 32,
    2: 40 // 32,
}  # row -> index of the group that contains the NaN
NAN_LANE = {0: 0, 1: 70, 2: 40}


@pytest.mark.parametrize("m_total", [SMALL_M + 1, TILED_M], ids=["untiled", "tiled"])
def test_torch_fallback_propagates_nan_like_deepgemm(m_total):
    """DeepGEMM's cast and the torch fallback: a NaN makes the group's scale exponent 254 and stays an E4M3 NaN; all other groups are unaffected."""
    x = _embed(_nan_rows(), m_total, seed=400 + m_total)
    ref = _torch(x)
    dg = _deepgemm(x)
    if dg is not None:
        assert torch.equal(ref[0], dg[0]) and torch.equal(ref[1], dg[1])
    sf_bytes = ref[1].view(torch.uint8).view(m_total, GROUPS)
    for row, grp in NAN_GROUP.items():
        assert int(sf_bytes[row, grp]) == UE8M0_NAN_BYTE
        assert int(ref[0][row, NAN_LANE[row]]) in E4M3_NAN_BYTES
        other = torch.arange(GROUPS, device="cuda") != grp
        assert (sf_bytes[row][other] != UE8M0_NAN_BYTE).all()


@pytest.mark.parametrize("m_total", [SMALL_M + 1, TILED_M], ids=["untiled", "tiled"])
def test_triton_staging_keeps_nan_lanes_nan_and_leaves_everything_else_identical(
    m_total,
):
    """What the Triton kernel guarantees for NaN inputs: the NaN lane stays an E4M3 NaN, rows without NaN and the groups of a NaN row that do not contain the NaN are
    bit-identical to DeepGEMM's cast (the scale of the NaN group itself is the known deviation, see the xfail below)."""
    pytest.importorskip("triton")
    x = _embed(_nan_rows(), m_total, seed=500 + m_total)
    ref = _torch(x)
    tri = _triton(x)
    nan_rows = sorted(NAN_GROUP)
    clean_rows = [r for r in range(m_total) if r not in nan_rows]
    assert torch.equal(tri[0][clean_rows], ref[0][clean_rows]) and torch.equal(
        tri[1][clean_rows], ref[1][clean_rows]
    )
    for row in nan_rows:
        assert int(tri[0][row, NAN_LANE[row]]) in E4M3_NAN_BYTES
        grp = NAN_GROUP[row]
        keep = torch.ones(H, dtype=torch.bool, device="cuda")
        keep[grp * 32 : (grp + 1) * 32] = False
        assert torch.equal(tri[0][row][keep], ref[0][row][keep])
        sf_t, sf_r = tri[1][row].view(torch.uint8), ref[1][row].view(torch.uint8)
        other = torch.arange(GROUPS, device="cuda") != grp
        assert torch.equal(sf_t[:GROUPS][other], sf_r[:GROUPS][other])


@pytest.mark.xfail(
    strict=True,
    reason="known deviation: the Triton kernel ignores NaN lanes when it computes the group scale (tl.max / tl.maximum have max-number semantics); "
    "DeepGEMM's cast and the torch fallback propagate the NaN (scale exponent byte 254). Finite inputs and +-Inf are bit-identical. Remove the marker when the kernel propagates NaN.",
)
@pytest.mark.parametrize("m_total", [SMALL_M + 1, TILED_M], ids=["untiled", "tiled"])
def test_triton_staging_nan_group_scale_matches_deepgemm(m_total):
    pytest.importorskip("triton")
    x = _embed(_nan_rows(), m_total, seed=600 + m_total)
    tri = _triton(x)
    ref = (
        _deepgemm(x) or _torch(x)
    )  # DeepGEMM's cast, else the torch fallback (bit-exact with it, see test_torch_fallback_propagates_nan_like_deepgemm)
    assert torch.equal(tri[1], ref[1]) and torch.equal(tri[0], ref[0])
