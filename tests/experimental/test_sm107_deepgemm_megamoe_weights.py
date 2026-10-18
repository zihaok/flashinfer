# SPDX-License-Identifier: Apache-2.0
# Port of DeepGEMM (MIT License, Copyright (c) 2025 DeepSeek; see licenses/LICENSE.deepgemm.txt) sm107_fp8_fp4_mega_moe.cuh / weight-transform tests.
"""``transform_weights_for_mega_moe`` of the SM107 MegaMoE package: pure torch, bit-exact against DeepGEMM's.

* independent index-math checks (explicit Python loops, no shared code with the implementation): gate/up 8-row interleave of
  the L1 weights and SF, UTCCP 4x32 SF transpose, strides preserved (MN-major SF);
* bit-exact equality with ``deep_gemm.transform_weights_for_mega_moe`` (skipped when ``deep_gemm`` is absent) on random packed
  FP4 weights / int32 UE8M0 scale factors of the V4.1 and V4 shapes, routed (3D) and shared (2D) tensors, BF16 path;
* the module must not import ``deep_gemm``.
"""

import importlib
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")

from flashinfer.experimental.sm107_deepgemm_megamoe import weights as W  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def mn_major_sf(num_groups, n, k, gen):
    """Random int32 UE8M0-packed SF ``(G, N, ceil(K/128))`` in the TMA layout of ``transform_sf_into_required_layout``:
    strides ``(ceil(K/128) * align(N, 4), 1, align(N, 4))``."""
    k128 = -(-k // 128)
    n4 = -(-n // 4) * 4
    sf = torch.empty_strided(
        (num_groups, n, k128), (k128 * n4, 1, n4), dtype=torch.int32, device=DEVICE
    )
    # random UE8M0 bytes in [100, 140] per K group
    b = torch.randint(
        100,
        141,
        (num_groups, n, k128, 4),
        generator=gen,
        dtype=torch.int32,
        device="cpu",
    ).to(DEVICE)
    sf.copy_(b[..., 0] | (b[..., 1] << 8) | (b[..., 2] << 16) | (b[..., 3] << 24))
    return sf


def packed_fp4(shape, gen):
    return torch.randint(
        -128, 128, shape, generator=gen, dtype=torch.int8, device="cpu"
    ).to(DEVICE)


def test_module_does_not_import_deep_gemm():
    code = "import sys; import flashinfer.experimental.sm107_deepgemm_megamoe.weights; assert 'deep_gemm' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.mark.parametrize("gran", [8])
def test_interleave_index_math(gran):
    gen = torch.Generator().manual_seed(1)
    for shape in ((2, 64, 24), (1, 128, 5), (32, 5)):
        t = torch.randint(-128, 128, shape, generator=gen, dtype=torch.int8).to(DEVICE)
        out = W.interleave_weights(t)
        assert out.shape == t.shape and out.dtype == t.dtype
        t3 = t if t.dim() == 3 else t.unsqueeze(0)
        o3 = out if out.dim() == 3 else out.unsqueeze(0)
        n = t3.shape[1]
        half = n // 2
        for g in range(half // gran):
            # [g0..g7, u0..u7, g8..g15, u8..u15, ...]
            assert torch.equal(
                o3[:, 2 * gran * g : 2 * gran * g + gran],
                t3[:, gran * g : gran * g + gran],
            )
            assert torch.equal(
                o3[:, 2 * gran * g + gran : 2 * gran * (g + 1)],
                t3[:, half + gran * g : half + gran * (g + 1)],
            )


def test_utccp_transpose_index_math_and_strides():
    gen = torch.Generator().manual_seed(2)
    sf = mn_major_sf(2, 256, 5120, gen)
    out = W.transpose_sf_for_utccp(sf)
    assert (
        out.shape == sf.shape
        and out.dtype == torch.int32
        and out.stride() == sf.stride()
    )  # MN-major preserved
    for g in range(2):
        for blk in range(
            2
        ):  # inside every 128-row block: stored position 4*c + a holds row 32*a + c
            for a in range(4):
                for c in (0, 1, 17, 31):
                    assert torch.equal(
                        out[g, blk * 128 + 4 * c + a], sf[g, blk * 128 + 32 * a + c]
                    )
    with pytest.raises(AssertionError):
        W.transpose_sf_for_utccp(mn_major_sf(1, 192, 256, gen))  # MN % 128 != 0


def test_transform_composition_matches_documented_layout():
    # L1: interleave(gate/up, 8 rows) then UTCCP; L2: only UTCCP.  Check one transformed SF row against first principles.
    gen = torch.Generator().manual_seed(3)
    two_i, hidden, e_l = 256, 512, 2
    l1_w, l2_w = (
        packed_fp4((e_l, two_i, hidden // 2), gen),
        packed_fp4((e_l, hidden, two_i // 2 // 2), gen),
    )
    l1_sf, l2_sf = (
        mn_major_sf(e_l, two_i, hidden, gen),
        mn_major_sf(e_l, hidden, two_i // 2, gen),
    )
    (tw1, tsf1), (tw2, tsf2) = W.transform_weights_for_mega_moe(
        (l1_w, l1_sf), (l2_w, l2_sf)
    )
    assert (
        torch.equal(tw2, l2_w)
        and tw1.shape == l1_w.shape
        and tsf1.stride() == l1_sf.stride()
        and tsf2.stride() == l2_sf.stride()
    )
    half = two_i // 2
    inter_row = (
        lambda r: (r // 16) * 8 + (r % 8)
        if (r % 16) < 8
        else half + (r // 16) * 8 + (r % 8)
    )  # noqa: E731  (row r after interleave -> source row)
    for r in (0, 3, 8, 15, 16, 31, 64, 127, 128, 200, 255):
        # row r of the interleaved SF (r = 128*B + 32*a + c) is stored at position 128*B + 4*c + a (UTCCP 4x32 transpose)
        stored = 128 * (r // 128) + 4 * (r % 32) + (r % 128) // 32
        assert torch.equal(tsf1[0, stored], l1_sf[0, inter_row(r)]), r
        assert torch.equal(tw1[0, r], l1_w[0, inter_row(r)]), r


def _dg():
    pytest.importorskip("deep_gemm")
    return importlib.import_module("deep_gemm")


@pytest.mark.parametrize(
    "hidden,inter,e_l", [(5120, 2304, 3), (4096, 2048, 2), (512, 256, 4)]
)
def test_bit_exact_vs_deepgemm_routed(hidden, inter, e_l):
    dg = _dg()
    gen = torch.Generator().manual_seed(10 + hidden)
    l1 = (
        packed_fp4((e_l, 2 * inter, hidden // 2), gen),
        mn_major_sf(e_l, 2 * inter, hidden, gen),
    )
    l2 = (
        packed_fp4((e_l, hidden, inter // 2), gen),
        mn_major_sf(e_l, hidden, inter, gen),
    )
    mine = W.transform_weights_for_mega_moe(l1, l2)
    ref = dg.transform_weights_for_mega_moe(l1, l2)
    for a, b in zip(mine, ref, strict=False):
        for x, y in zip(a, b, strict=False):
            assert (
                x.dtype == y.dtype and x.shape == y.shape and x.stride() == y.stride()
            )
            assert torch.equal(x, y)


def test_bit_exact_vs_deepgemm_shared_2d_and_bf16():
    dg = _dg()
    gen = torch.Generator().manual_seed(5)
    hidden, inter = 5120, 2304
    # shared experts: FP8 weights (2I, H) / (H, I) + MN-major SF (G = 1 squeezed to 2D)
    sw1 = torch.randint(
        -100, 100, (2 * inter, hidden), generator=gen, dtype=torch.int8
    ).to(DEVICE)
    sf1 = mn_major_sf(1, 2 * inter, hidden, gen)[0]
    sw2 = torch.randint(-100, 100, (hidden, inter), generator=gen, dtype=torch.int8).to(
        DEVICE
    )
    sf2 = mn_major_sf(1, hidden, inter, gen)[0]
    mine = W.transform_weights_for_mega_moe((sw1, sf1), (sw2, sf2))
    ref = dg.transform_weights_for_mega_moe((sw1, sf1), (sw2, sf2))
    for a, b in zip(mine, ref, strict=False):
        for x, y in zip(a, b, strict=False):
            assert x.stride() == y.stride() and torch.equal(x, y)
    # BF16 path: L1 interleave only
    w1 = torch.randn((2, 2 * 256, 128), generator=gen).to(torch.bfloat16).to(DEVICE)
    w2 = torch.randn((2, 128, 256), generator=gen).to(torch.bfloat16).to(DEVICE)
    m1, m2 = W.transform_weights_for_mega_moe(w1, w2)
    r1, r2 = dg.transform_weights_for_mega_moe(w1, w2)
    assert torch.equal(m1, r1) and m2 is w2 and torch.equal(m2, r2)
    with pytest.raises(AssertionError):
        W.transform_weights_for_mega_moe(w1, w2, activation="relu")


@pytest.mark.parametrize(
    "g,n,k", [(3, 4608, 5120), (2, 5120, 2304), (1, 256, 512), (None, 640, 4096)]
)
def test_sf_to_required_layout_bit_exact_vs_deepgemm(g, n, k):
    dg = _dg()
    if DEVICE != "cuda":
        pytest.skip(
            "deep_gemm.transform_sf_into_required_layout has no CPU kernel (needs a CUDA device)"
        )
    gen = torch.Generator().manual_seed(n + k)
    shape = (g, n, k // 32) if g is not None else (n, k // 32)
    exp = torch.randint(90, 160, shape, generator=gen, dtype=torch.int32).to(DEVICE)
    sf = (exp << 23).view(torch.float32)  # exact powers of two
    mine = W.sf_to_required_layout(sf, n, k, g)
    ref = dg.transform_sf_into_required_layout(sf, n, k, (1, 32), g)
    assert (
        mine.dtype == ref.dtype == torch.int32
        and mine.shape == ref.shape
        and mine.stride() == ref.stride()
    )
    assert torch.equal(mine, ref)
    # E8M0 bytes give the same words as the float32 representation
    assert torch.equal(W.sf_to_required_layout(exp.to(torch.uint8), n, k, g), mine)
    # first-principles: byte b of word w of row r = exponent of K group 4w + b
    r = n // 2
    lead = (0,) * (mine.dim() - 2)
    word = int(mine[lead + (r, 1)])
    assert [(word >> (8 * b)) & 0xFF for b in range(4)] == [
        int(exp[lead + (r, 4 + b)]) for b in range(4)
    ]
