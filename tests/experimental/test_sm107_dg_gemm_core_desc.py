# SPDX-License-Identifier: Apache-2.0
# Port of DeepGEMM (MIT License, Copyright (c) 2025 DeepSeek; see licenses/LICENSE.deepgemm.txt) sm107_fp8_fp4_mega_moe.cuh / mma/sm100.cuh descriptors.
"""CPU tests of the GEMM-role constants and models (no GPU needed; the CUTLASS DSL must be importable because ``ptx_g`` wraps ``llvm.inline_asm``).

* instruction-descriptor test vectors (routed M=256: N = 16..240 with SF id 0 / 2; shared FP8 weights N = 32),
* bit-field layout of the instruction descriptor and of the shared-memory descriptor high words,
* the UTCCP-transposed SF position function (a permutation inside every 128-token group),
* the Python model of the scheduler (``replay_utils.ref_task_sequence``): task multiset, per-phase order, L1-before-L2 dependency, strict L2/L1 alternation,
  and its warm-up wave formula against the package's ``get_num_l1_warmup_waves``.
"""

import os
import random
import sys

import pytest

pytest.importorskip("cutlass.cute")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import replay_utils as R  # noqa: E402
from flashinfer.experimental.sm107_deepgemm_megamoe.kernel import ptx_g  # noqa: E402

# (routed, M = 256): N -> (sf id 0, sf id 2)
EXPECTED_ROUTED = {
    16: (0x90840280, 0xD08402A0),
    32: (0x90880280, 0xD08802A0),
    64: (0x90900280, 0xD09002A0),
    128: (0x90A00280, 0xD0A002A0),
    192: (0x90B00280, 0xD0B002A0),
    240: (0x90BC0280, 0xD0BC02A0),
}
EXPECTED_SHARED_N32 = (0x90880000, 0xD0880020)


@pytest.mark.parametrize("n", sorted(EXPECTED_ROUTED))
def test_routed_idesc_vectors(n):
    v0, v2 = EXPECTED_ROUTED[n]
    assert ptx_g.idesc_with_n_and_sf_id(ptx_g.IDESC_BASE_ROUTED, n, 0) == v0
    assert ptx_g.idesc_with_n_and_sf_id(ptx_g.IDESC_BASE_ROUTED, n, 2) == v2


def test_shared_idesc_vectors():
    assert (
        ptx_g.idesc_with_n_and_sf_id(ptx_g.IDESC_BASE_SHARED, 32, 0)
        == EXPECTED_SHARED_N32[0]
    )
    assert (
        ptx_g.idesc_with_n_and_sf_id(ptx_g.IDESC_BASE_SHARED, 32, 2)
        == EXPECTED_SHARED_N32[1]
    )


def test_idesc_fields():
    base = ptx_g.IDESC_BASE_ROUTED
    assert base == 0x90800280
    assert (
        base >> 7
    ) & 7 == 5  # a_format = E2M1 (unpacked-smem code under kind::mxf8f6f4)
    assert (base >> 10) & 7 == 0  # b_format = E4M3
    assert (base >> 23) & 1 == 1  # scale format UE8M0
    assert (base >> 24) & 0x1F == 16  # m_dim = 256 >> 4
    assert (base >> 31) & 1 == 1  # k_size: K = 64 per UMMA
    for n in range(16, 241, 16):
        v = ptx_g.idesc_with_n_and_sf_id(base, n, 2)
        assert (v >> 17) & 0x3F == n >> 3  # n_dim
        assert (v >> 29) & 3 == 2 and (v >> 4) & 3 == 2  # a_sf_id / b_sf_id
    assert (
        base & ~(7 << 7) == ptx_g.IDESC_BASE_SHARED
    )  # shared FP8 weights: a_format = E4M3 (0)


def test_smem_desc_hi_words():
    def decode(hi):
        return dict(
            sbo=(hi & 0x3FFF) << 4, version=(hi >> 14) & 3, layout=(hi >> 29) & 7
        )

    assert decode(ptx_g.SMEM_DESC_HI_SW128_SBO1024) == dict(
        sbo=1024, version=1, layout=2
    )  # tokens, SW128
    assert decode(ptx_g.SMEM_DESC_HI_SW64_SBO512) == dict(
        sbo=512, version=1, layout=4
    )  # packed FP4 weights (and shared FP8 half-K tiles), SW64
    assert decode(ptx_g.SMEM_DESC_HI_SF_SBO128) == dict(
        sbo=128, version=1, layout=0
    )  # SF (UTCCP source), no swizzle


def test_smem_desc_lo_and_stage_advance():
    assert ptx_g.smem_desc_lo(0x12340) == (0x12340 >> 4) & 0x3FFF
    # K=64 advance inside a 128-K stage: tokens 64 B (>> 4 = 4), packed FP4 weights 32 B (>> 4 = 2)
    assert 64 >> 4 == 4 and 64 // 2 >> 4 == 2


@pytest.mark.parametrize("sf_block", [128, 256])
def test_utccp_pos_is_a_permutation(sf_block):
    pos = [R.utccp_pos(i) for i in range(sf_block)]
    assert sorted(pos) == list(range(sf_block))
    assert all(
        p // 128 == i // 128 for i, p in enumerate(pos)
    )  # stays inside its 128-token group
    assert (
        R.utccp_pos(1) == 4 and R.utccp_pos(32) == 1 and R.utccp_pos(33) == 5
    )  # (idx & 31) * 4 + ((idx >> 5) & 3)


# ---------------------------------------------------------------------------------------------------------------------
# scheduler model
# ---------------------------------------------------------------------------------------------------------------------
CASES = [
    ([3, 17, 0, 40], 16),
    ([5, 33, 0, 70], 32),
    ([37, 70, 0, 100], 64),
    ([37, 133, 0, 200], 128),
    ([128, 128, 128, 128], 128),
    ([1, 0, 0, 0], 128),
    ([37, 200, 0, 400], 192),
    ([37, 241, 0, 500], 240),
    ([600, 1, 255, 3, 0, 128], 128),
]


@pytest.mark.parametrize("counts,bm", CASES)
@pytest.mark.parametrize(
    "shape", [(2, 2), (18, 20)]
)  # (num_l1_n_clusters, num_l2_n_clusters): test shapes and V4.1 (4608/256, 5120/256)
def test_ref_scheduler_invariants(counts, bm, shape):
    l1n, l2n = shape
    seq, nblocks = R.ref_task_sequence(counts, bm, l1n, l2n, 1, 4608, 5120, 5120, 2304)
    l1 = [t for t in seq if t[0] == 1]
    l2 = [t for t in seq if t[0] == 2]
    assert len(l1) == nblocks * l1n and len(l2) == nblocks * l2n
    # every (block, n) exactly once per phase, claimed block-major / n-minor
    assert [(t[4], t[3]) for t in l1] == [
        (m, n) for m in range(nblocks) for n in range(l1n)
    ]
    assert [(t[4], t[3]) for t in l2] == [
        (m, n) for m in range(nblocks) for n in range(l2n)
    ]
    # valid_m: full blocks, then the partial last block of every expert
    flat = [
        (e, mb, min(c - mb * bm, bm))
        for e, c in enumerate(counts)
        for mb in range(-(-c // bm))
    ]
    assert [(t[1], t[2], t[5]) for t in l1[::l1n]] == flat
    # L2 of a block comes after all its L1 tasks
    last_l1, first_l2 = {}, {}
    for i, t in enumerate(seq):
        if t[0] == 1:
            last_l1[t[4]] = i
        else:
            first_l2.setdefault(t[4], i)
    for m in range(nblocks):
        assert last_l1[m] < first_l2[m]
    # after the first L2 task the phases strictly alternate (L2, L1, L2, L1, ...) while L1 tasks remain, then only L2 tasks follow
    phases = [t[0] for t in seq]
    start = phases.index(2) if 2 in phases else len(phases)
    tail = phases[start:]
    n_l1_left = tail.count(1)
    expect = [2, 1] * n_l1_left
    assert tail[: 2 * n_l1_left] == expect and all(
        p == 2 for p in tail[2 * n_l1_left :]
    )


@pytest.mark.parametrize("num_blocks", [1, 2, 5, 32, 97, 400])
@pytest.mark.parametrize("nc", [1, 53, 106])
@pytest.mark.parametrize("shape", [(2, 2), (18, 20), (16, 20), (20, 18)])
def test_warmup_formula_matches_host(num_blocks, nc, shape):
    heuristics = pytest.importorskip(
        "flashinfer.experimental.sm107_deepgemm_megamoe.heuristics"
    )
    l1n, l2n = shape
    assert R.warmup_waves(
        num_blocks, nc, l1n, l2n
    ) == heuristics.get_num_l1_warmup_waves(num_blocks, nc, l1n, l2n)


def test_ref_scheduler_random_cases():
    rng = random.Random(0)
    for _ in range(50):
        bm = rng.choice([16, 32, 64, 128, 192, 240])
        counts = [
            rng.choice([0, 1, rng.randint(1, 3 * bm), bm, 2 * bm])
            for _ in range(rng.randint(1, 8))
        ]
        seq, nblocks = R.ref_task_sequence(
            counts, bm, 18, 20, 1, 4608, 5120, 5120, 2304
        )
        assert len(seq) == nblocks * (18 + 20)
        assert nblocks == sum(-(-c // bm) for c in counts)


# ---------------------------------------------------------------------------------------------------------------------
# shared-expert (S = 1) phases of the scheduler model
# ---------------------------------------------------------------------------------------------------------------------
V41 = dict(
    l1n=18, l2n=20, shape_n1=4608, shape_k1=5120, shape_n2=5120, shape_k2=2304
)  # DeepSeek-V4.1-Flash, one shared expert


@pytest.mark.parametrize("bm", [16, 32, 64, 128, 192, 240])
@pytest.mark.parametrize(
    "num_tokens", [1, 7, 80, 81, 300, 640, 641, 960, 961, 1500, 1920]
)
def test_shared_phase_order_and_early_late(bm, num_tokens):
    counts = [37, 133, 0, 200]
    seq, _ = R.ref_full_task_sequence(
        counts,
        bm,
        V41["l1n"],
        V41["l2n"],
        1,
        V41["shape_n1"],
        V41["shape_k1"],
        V41["shape_n2"],
        V41["shape_k2"],
        num_tokens,
        212,
        shared=(V41["shape_n1"], V41["shape_k1"], V41["shape_n2"], V41["shape_k2"]),
    )
    nm = -(-num_tokens // bm)
    s1 = [t for t in seq if t[0] == 3]
    s2 = [t for t in seq if t[0] == 4]
    assert len(s1) == nm * 18 and len(s2) == nm * 20
    assert [(t[2], t[3]) for t in s1] == [(m, n) for m in range(nm) for n in range(18)]
    assert [(t[2], t[3]) for t in s2] == [(m, n) for m in range(nm) for n in range(20)]
    # pool block == m block, valid_m = min(num_tokens - m*BM, BM), no expert
    assert all(
        t[1] == 0 and t[4] == t[2] and t[5] == min(num_tokens - t[2] * bm, bm)
        for t in s1 + s2
    )
    first_routed = next(i for i, t in enumerate(seq) if t[0] in (1, 2))
    last_routed = max(i for i, t in enumerate(seq) if t[0] in (1, 2))
    early = nm * 20 <= 106
    assert R.shared_l2_is_early(num_tokens, bm, V41["shape_n2"], 212) == early
    # shared L1 always first; shared L2 before the routed tasks when early, after them otherwise
    assert [t[0] for t in seq[: len(s1)]] == [3] * len(s1)
    if early:
        assert [t[0] for t in seq[len(s1) : len(s1) + len(s2)]] == [4] * len(
            s2
        ) and first_routed == len(s1) + len(s2)
    else:
        assert first_routed == len(s1) and [t[0] for t in seq[last_routed + 1 :]] == [
            4
        ] * len(s2)


# ---------------------------------------------------------------------------------------------------------------------
# L2 promotion policy of the weight tensor maps (tma_warps.l2_promotion_bytes)
# ---------------------------------------------------------------------------------------------------------------------
def test_l2_promotion_policy():
    import types

    tw = pytest.importorskip(
        "flashinfer.experimental.sm107_deepgemm_megamoe.kernel.tma_warps"
    )
    auto_big = 256 if tw.HAVE_TENSOR_MAP_API else 0
    for bm in (16, 32, 64):
        assert tw.l2_promotion_bytes(types.SimpleNamespace(block_m=bm)) == 0
    for bm in (128, 192, 240):
        assert tw.l2_promotion_bytes(types.SimpleNamespace(block_m=bm)) == auto_big
        assert (
            tw.l2_promotion_bytes(
                types.SimpleNamespace(block_m=bm, tma_l2_promotion=None)
            )
            == auto_big
        )
    assert (
        tw.l2_promotion_bytes(types.SimpleNamespace(block_m=240, tma_l2_promotion=0))
        == 0
    )  # forced: keep the DSL's own descriptors
    if tw.HAVE_TENSOR_MAP_API:
        assert (
            tw.l2_promotion_bytes(
                types.SimpleNamespace(block_m=16, tma_l2_promotion=256)
            )
            == 256
        )
        assert (
            tw.l2_promotion_bytes(
                types.SimpleNamespace(block_m=16, tma_l2_promotion=64)
            )
            == 64
        )
    with pytest.raises(ValueError):
        tw.l2_promotion_bytes(types.SimpleNamespace(block_m=16, tma_l2_promotion=100))
    # die-local weights (per-die): 256 B at every BLOCK_M (DEP4 per-die S=1 A/B: -3.1..-3.5 % at BM16..64); plain weights below BLOCK_M 128 keep the DSL's descriptors
    for bm in (16, 32, 64, 128, 192, 240):
        assert (
            tw.l2_promotion_bytes(
                types.SimpleNamespace(block_m=bm, localized_weights=True)
            )
            == auto_big
        )
        assert tw.l2_promotion_bytes(
            types.SimpleNamespace(block_m=bm, localized_weights=False)
        ) == (auto_big if bm >= 128 else 0)
