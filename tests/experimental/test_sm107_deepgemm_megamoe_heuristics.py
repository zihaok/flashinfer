# SPDX-License-Identifier: Apache-2.0
# Port of DeepGEMM (MIT License, Copyright (c) 2025 DeepSeek; see licenses/LICENSE.deepgemm.txt) sm107_fp8_fp4_mega_moe.cuh / host heuristics tests.
"""Tests of the pure-Python host heuristics of the experimental SM107 DeepGEMM-style MegaMoE.

CPU tests (no torch, no GPU): every table of DeepGEMM's host-side math (shared-memory layout, symmetric-buffer regions, signal offsets, ring sizes) for the
DeepSeek-V4.1-Flash shape (H=5120, I=2304, E=384, top-6, 212 SMs), compile-key coverage and dataclass invariants.

GPU tests (a CUDA device + ``deep_gemm`` importable, i.e. DeepGEMM 3ec0eb4 on a Rubin node): zero-mismatch comparison against
DeepGEMM's C++ host functions through ``deep_gemm._C`` over the grid {both shapes} x {EP sizes} x {T_max} x {M in 0..16384};
the stage / dynamic-smem numbers (not exposed by the C++ API) are compared with the ``DG_PRINT_CONFIGS=1`` output of a real
launch in ``test_stages_smem_vs_dg_print_configs`` (slow: compiles DeepGEMM kernels; opt in with
``FLASHINFER_SM107_DG_PRINT_CONFIGS=1``).  DeepGEMM's own ``get_num_sms_for_mega_moe`` asserts that the SM count equals the
device's, so the C++ oracle can only be evaluated at the device SM count (212 on Rubin VR200).
"""

import dataclasses
import math
import os
import re
import subprocess
import sys

import pytest

from flashinfer.experimental.sm107_deepgemm_megamoe import config as cfgmod
from flashinfer.experimental.sm107_deepgemm_megamoe import heuristics as h

V41 = dict(
    num_experts=384, num_topk=6, hidden=5120, intermediate_hidden=2304
)  # DeepSeek-V4.1-Flash
V4 = dict(
    num_experts=256, num_topk=6, hidden=4096, intermediate_hidden=2048
)  # DeepSeek-V4-Flash (secondary shape)
SMS = 212
BLOCK_MS = (16, 32, 64, 128, 192, 240)


def kcfg(R=4, T=8192, M=1024, S=1, shape=V41, sms=SMS, **kw):
    return h.get_sm107_kernel_config(
        R, shape["num_experts"], T, M, shape["num_topk"], shape["hidden"], shape["intermediate_hidden"], S,
        num_sms=sms, **kw,
    )  # fmt: skip


# ---------------------------------------------------------------------------------------------------------------------
# BLOCK_M regimes (V4.1), transcribed from DeepGEMM's heuristic
# ---------------------------------------------------------------------------------------------------------------------
_REGIMES_SPEC = {
    4: "16: 1-160 / 32: 161-384 / 64: 385-903 / 128: 904-1874 / 192: 1875-2858 / 240: 2859-3600 / 128: 3601-3847 / 192: 3848-5838 / 240: 5839-7337 / 192: 7338-8839 / 240: 8840-11098 / 192: 11099-11852 / 240: 11853-16384",
    8: "16: 1-80 / 32: 81-192 / 64: 193-451 / 128: 452-937 / 192: 938-1429 / 240: 1430-1800 / 128: 1801-1923 / 192: 1924-2919 / 240: 2920-3668 / 192: 3669-4419 / 240: 4420-5549 / 192: 5550-5926 / 240: 5927-16384",
    16: "16: 1-40 / 32: 41-96 / 64: 97-225 / 128: 226-468 / 192: 469-714 / 240: 715-900 / 128: 901-961 / 192: 962-1459 / 240: 1460-1834 / 192: 1835-2209 / 240: 2210-2774 / 192: 2775-2963 / 240: 2964-16384",
    32: "16: 1-20 / 32: 21-48 / 64: 49-112 / 128: 113-234 / 192: 235-357 / 128: 358-480 / 192: 481-729 / 240: 730-917 / 192: 918-1104 / 240: 1105-1387 / 192: 1388-1481 / 240: 1482-16384",
}  # fmt: skip


def _parse_regimes(text):
    out = []
    for seg in text.split("/"):
        bm, rng = seg.strip().split(":")
        a, b = rng.strip().split("-")
        out.append((int(a), int(b), int(bm)))
    return out


@pytest.mark.parametrize("R", [4, 8, 16, 32])
def test_block_m_regimes_match_reference(R):
    got = h.get_block_m_regimes(R, 384, 6, 16384)
    assert got == _parse_regimes(_REGIMES_SPEC[R])
    # the selection is non-monotonic at large M (nb >= 2 re-selects 128/192)
    bms = [bm for _, _, bm in got]
    assert bms != sorted(bms)
    assert set(bms) == set(BLOCK_MS)


# sweep table: M -> BLOCK_M for DEP4/8/16/32
_SWEEP = {
    1: (16, 16, 16, 16), 8: (16, 16, 16, 16), 16: (16, 16, 16, 16), 32: (16, 16, 16, 32), 64: (16, 16, 32, 64),
    128: (16, 32, 64, 128), 256: (32, 64, 128, 192), 512: (64, 128, 192, 192), 1024: (128, 192, 192, 192),
    2048: (192, 192, 192, 240), 3072: (240, 240, 240, 240), 4096: (192, 192, 240, 240), 8192: (192, 240, 240, 240),
}  # fmt: skip


@pytest.mark.parametrize("M", sorted(_SWEEP))
def test_block_m_sweep_table(M):
    for R, want in zip((4, 8, 16, 32), _SWEEP[M], strict=False):
        assert h.get_block_m_for_mega_moe(R, 384, 8192, M, 6, "fp8xfp4") == want, (R, M)


def test_block_m_basic_properties():
    assert (
        h.get_block_m_for_mega_moe(4, 384, 0, 0, 6) == 16
    )  # M = 0 (a rank without tokens)
    # num_max_tokens_per_rank is ignored, mma_type 'fp8xfp8' selects the same BLOCK_M, unknown mma types are rejected
    for M in (1, 100, 1000, 5000):
        a = h.get_block_m_for_mega_moe(8, 384, 1920, M, 6, "fp8xfp4")
        assert (
            a
            == h.get_block_m_for_mega_moe(8, 384, 99999, M, 6, "fp8xfp8")
            == h.get_block_m_for_mega_moe(8, 384, 1, M, 6, "bf16xbf16")
        )
    with pytest.raises(AssertionError):
        h.get_block_m_for_mega_moe(8, 384, 1920, 10, 6, "fp4xfp4")
    with pytest.raises(AssertionError):
        h.get_block_m_for_mega_moe(8, 384, 1920, -1, 6, "fp8xfp4")
    # the `E/R < 14` bump: DEP32 of E=384 has 12 experts per rank
    assert (
        h.get_block_m_for_mega_moe(32, 384, 0, 358, 6) == 128
        and h.get_block_m_for_mega_moe(4, 384, 0, 3601, 6) == 128
    )
    # variants to precompile for T_max = 9600 (vLLM warmup enumerates exactly these)
    for R in (4, 8, 16, 32):
        assert h.get_block_m_variants(R, 384, 6, 9600) == BLOCK_MS
    assert h.get_block_m_variants(4, 384, 6, 100) == (16,)


def test_block_m_float32_semantics():
    # e = float32(M) * R * topk / E evaluated in float32 (left to right, each operation rounded), cov = e + sqrtf(e),
    # ceil(cov / 240) in float32.  Compare against an independent, explicitly float32 transliteration of the C++ for
    # expert counts where the quotient is inexact in binary (E = 100, 257, ...) and thresholds are hit by rounding.
    import ctypes

    def f32(x):
        return ctypes.c_float(x).value

    def reference(R, E, M, topk=6):
        e = f32(f32(f32(float(M)) * R) * topk)
        e = f32(e / E)
        cov = f32(e + f32(math.sqrt(e)))
        bm = 16 if e <= 10 else 32
        if e > 24:
            nb = int(math.ceil(f32(cov / 240)))
            if nb == 1 and cov > 192 and E // R < 14:
                nb = 2
            for c in (64, 128, 192, 240):
                bm = c
                if nb * c >= cov:
                    break
        return bm

    for E, R in (
        (100, 3),
        (384, 1),
        (384, 3),
        (384, 6),
        (384, 12),
        (384, 24),
        (257, 1),
        (96, 12),
        (7, 1),
    ):
        for M in range(0, 4000, 3):
            assert h.get_block_m_for_mega_moe(R, E, 0, M, 6) == reference(R, E, M), (
                E,
                R,
                M,
            )
    # exact thresholds in exact arithmetic: DEP4/E=384 -> e = M/16: e == 10 at M = 160 (16), M = 161 -> e > 10 (32);
    # e == 24 at M = 384 (still 32), M = 385 -> 64
    assert [
        h.get_block_m_for_mega_moe(4, 384, 0, m, 6) for m in (160, 161, 384, 385)
    ] == [16, 32, 32, 64]


# ---------------------------------------------------------------------------------------------------------------------
# fixed geometry + TMEM; stages / smem; SharedStorage offsets
# ---------------------------------------------------------------------------------------------------------------------
# BLOCK_M: (LOAD_BLOCK_M, STORE_BLOCK_M, SF_BLOCK_M, WG_BLOCK_M, store blocks / WG, UMMA N)
_GEOMETRY = {16: (8, 8, 128, 8, 1, 16), 32: (16, 16, 128, 16, 1, 32), 64: (32, 16, 128, 32, 2, 64),
             128: (64, 32, 128, 64, 2, 128), 192: (96, 32, 256, 96, 3, 192), 240: (120, 40, 256, 120, 3, 240)}  # fmt: skip
# BLOCK_M: (accum + SFA + SFB, alloc, SFA col, SFB col)
_TMEM = {16: (40, 64, 32, 36), 32: (72, 128, 64, 68), 64: (136, 256, 128, 132), 128: (264, 512, 256, 260),
         192: (396, 512, 384, 392), 240: (492, 512, 480, 488)}  # fmt: skip
# BLOCK_M: (stages, per-stage bytes, fixed107, requested smem, exact end, sizeof, reusable, stages if fixed were clean)
_STAGES = {
    16: (13, 10256, 97828, 231156, 141044, 141312, 140608, 21),
    32: (12, 11280, 93988, 229348, 147428, 147456, 147008, 19),
    64: (11, 13328, 85796, 232404, 158676, 158720, 158272, 16),
    128: (8, 17424, 78116, 217508, 160164, 160768, 159808, 12),
    192: (7, 22032, 69924, 224148, 174996, 175104, 174656, 9),
    240: (6, 25104, 66084, 216708, 175748, 176128, 175424, 8),
}  # fmt: skip
# member -> offsets per BLOCK_M
_SMEM_OFFSETS = {
    "expert_token_count": (0, 0, 0, 0, 0, 0),
    "dispatch_send_buffer": (2048, 2048, 2048, 2048, 2048, 2048),
    "smem_d": (3072, 3072, 3072, 3072, 3072, 3072),
    "smem_a": (7168, 11264, 11264, 19456, 19456, 23552),
    "smem_b": (20480, 35840, 56320, 84992, 105472, 115712),
    "smem_sfa": (126976, 134144, 146432, 150528, 162816, 164864),
    "smem_sfb": (133632, 140288, 152064, 154624, 169984, 171008),
    "amax_reduction": (140288, 146432, 157696, 158720, 173568, 174080),
    "task_infos": (140544, 146944, 158208, 159744, 174592, 175360),
    "dispatch_barriers": (140608, 147008, 158272, 159808, 174656, 175424),
    "full_barriers": (140640, 147040, 158304, 159840, 174688, 175456),
    "empty_barriers": (140744, 147136, 158392, 159904, 174744, 175504),
    "tmem_full_barriers": (140848, 147232, 158480, 159968, 174800, 175552),
    "tmem_empty_barriers": (140864, 147248, 158496, 159984, 174816, 175568),
    "combine_barriers": (140880, 147264, 158512, 160000, 174832, 175584),
    "task_info_full_barriers": (141008, 147392, 158640, 160128, 174960, 175712),
    "task_info_empty_barriers": (141024, 147408, 158656, 160144, 174976, 175728),
    "tmem_ptr_in_smem": (141040, 147424, 158672, 160160, 174992, 175744),
}  # fmt: skip
_SMEM_D_SIZE = (4096, 8192, 8192, 16384, 16384, 20480)
_SMEM_A_SIZE = (13312, 24576, 45056, 65536, 86016, 92160)
_SMEM_B_SIZE = (106496, 98304, 90112, 65536, 57344, 49152)


@pytest.mark.parametrize("bm", BLOCK_MS)
def test_geometry_tmem_stages_smem_tables(bm):
    i = BLOCK_MS.index(bm)
    for R in (
        4,
        8,
        16,
        32,
    ):  # stage count / smem are DEP- and M-independent once BLOCK_M is fixed
        c = kcfg(R=R, M=1, block_m=bm)
        assert (
            c.load_block_m,
            c.store_block_m,
            c.sf_block_m,
            c.wg_block_m,
            c.num_store_blocks_per_wg,
            c.umma_n,
        ) == _GEOMETRY[bm]
        used, alloc, sfa, sfb = _TMEM[bm]
        assert (
            c.tmem_cols_used,
            c.tmem_cols_alloc,
            c.tmem_sfa_col,
            c.tmem_sfb_col,
        ) == (used, alloc, sfa, sfb)
        stages, per_stage, fixed107, smem, end, sizeof, reusable, clean = _STAGES[bm]
        assert (
            c.num_stages,
            c.smem_bytes,
            c.smem_end,
            c.smem_sizeof,
            c.smem_reusable_bytes,
        ) == (stages, smem, end, sizeof, reusable)
        pipe = h.get_sm107_pipeline_config(384, 5120, bm, sf_block_m=c.sf_block_m)
        assert (pipe.smem_per_stage, pipe.smem_fixed, pipe.stages_if_fixed_clean) == (
            per_stage,
            fixed107,
            clean,
        )
        assert c.smem_bytes == fixed107 + stages * per_stage
        for name, vals in _SMEM_OFFSETS.items():
            assert c.smem_offsets[name] == vals[i], (name, bm)
        assert c.smem_offsets["reusable_bytes"] == c.smem_offsets["dispatch_barriers"]
        assert (
            c.smem_sizes["smem_d"],
            c.smem_sizes["smem_a"],
            c.smem_sizes["smem_b"],
        ) == (_SMEM_D_SIZE[i], _SMEM_A_SIZE[i], _SMEM_B_SIZE[i])
        # combine overlay needs 3 slots x 8 warps x chunk bytes <= reusable (K:1674)
        assert (
            3 * 8 * c.combine_chunk_bytes <= c.smem_reusable_bytes
            and c.num_combine_chunks == 2
            and c.combine_chunk_bytes == 5120
        )


def test_stage_numbers_vs_sm100_sizing():
    # the SM100-sized MegaMoEConfig line (first DG_PRINT_CONFIGS line; NOT the shipped SM107 values)
    want = {
        16: (11, 230100, 8, 8),
        32: (10, 226244, 16, 16),
        64: (9, 225204, 16, 16),
        128: (7, 219540, 32, 32),
        192: (6, 221572, 32, 32),
        240: (6, 229508, 24, 8),
    }
    for bm, (stages, smem, l1, l2) in want.items():
        c = h.get_mega_moe_config(
            4,
            384,
            96,
            9600,
            1,
            6,
            5120,
            2304,
            30720,
            491520,
            h.MMA_MXFP8FP4,
            force_block_m=bm,
        )
        assert (c.num_stages, c.smem_size, c.store_block_m_l1, c.store_block_m_l2) == (
            stages,
            smem,
            l1,
            l2,
        ), bm
        assert (
            c.block_n == 128
            and c.block_k == 128
            and c.num_bytes_per_pull == 5120
            and c.sf_block_m == max(128, -(-bm // 128) * 128)
        )
    assert str(c).startswith(
        "MegaMoEConfig(block_m=240, block_n=128, block_k=128, load_block_m=120, load_block_n=128"
    )


def test_shipped_stage_knobs():
    # env knobs of the SM107 launcher become explicit arguments: capacity (DG_MEGA_MOE_SM107_SMEM), cap (..._STAGES)
    c = kcfg(M=1024)
    assert (c.num_stages, c.smem_bytes) == (8, 217508)
    capped = kcfg(M=1024, max_stages=7)
    assert capped.num_stages == 7 and capped.smem_bytes == 78116 + 7 * 17424
    # too few stages shrink the reusable region below what the combine overlay needs (K:1675 static assert): rejected
    with pytest.raises(AssertionError, match="combine overlay"):
        kcfg(M=1024, max_stages=5)
    big = kcfg(M=1024, smem_capacity=334848)  # "oversized smem" experiment
    assert (
        big.num_stages > 8
        and big.smem_bytes <= 334848
        and big.smem_bytes > h.SM100_SMEM_CAPACITY
    )


@pytest.mark.parametrize(
    "R,regs,lanes",
    [
        (1, (96, 88, 160), 12),
        (2, (96, 88, 160), 6),
        (4, (96, 88, 160), 3),
        (8, (48, 40, 208), 2),
        (16, (48, 40, 208), 1),
        (32, (48, 40, 208), 1),
    ],
)
def test_per_ep_constants(R, regs, lanes):
    c = kcfg(R=R, M=64)
    assert (
        c.num_registers == regs
        and c.num_experts_per_lane == lanes
        and c.num_ranks_per_lane == 1
    )
    assert regs[0] * 128 + regs[1] * 128 + regs[2] * 256 <= 64512
    assert c.num_experts_per_rank == 384 // R
    # boundary of the register split: E/R <= 64 -> 48/40/208 (V4: 256 experts, DEP4 has 64 per rank)
    assert kcfg(R=4, shape=V4, M=64).num_registers == (48, 40, 208)
    assert kcfg(R=2, shape=V4, M=64).num_registers == (96, 88, 160)


def test_constants_and_derived():
    c = kcfg(R=4, M=1024)
    assert c.num_threads == 512 and (
        c.num_dispatch_threads,
        c.num_non_epilogue_threads,
        c.num_epilogue_threads,
    ) == (128, 128, 256)
    assert (
        c.l1_shape_n,
        c.l1_shape_k,
        c.l2_shape_n,
        c.l2_shape_k,
        c.shared_l2_shape_k,
    ) == (4608, 5120, 5120, 2304, 2304)
    assert (
        c.num_l1_block_ns,
        c.num_l2_block_ns,
        c.num_l1_clusters,
        c.num_l2_clusters,
    ) == (36, 40, 18, 20)
    assert (
        c.num_tokens_per_warp == 5
        and c.num_activate_lanes == 30
        and c.pull_uint4_per_lane == 10
        and c.num_bytes_per_pull == 5120
    )
    assert (
        c.combine_chunk_uint4_per_lane == 10
        and c.num_shared_sf_tokens == 153600
        and c.num_max_tokens_per_rank == 9600
    )
    assert (
        c.umma_k == 64
        and c.block_k == 128
        and c.shared_block_k == 64
        and (c.accum_stages, c.tma_store_stages, c.schedule_stages) == (2, 2, 2)
    )
    assert c.clamp == math.inf and kcfg(M=1024, activation_clamp=10.0).clamp == 10.0
    assert (
        kcfg(M=1024, S=0).shared_l2_shape_k == 0
        and not kcfg(M=1024, S=0).has_shared
        and c.has_shared
    )


# ---------------------------------------------------------------------------------------------------------------------
# ring sizing; symmetric buffer
# ---------------------------------------------------------------------------------------------------------------------
_RING_T = (1920, 3840, 5760, 9600, 15360)
_RING_SPEC = {  # DEP -> [(ring tokens, SF ring tokens) per T_max]
    4: [(11520, 184320), (17280, 276480), (21120, 337920), (30720, 491520), (44160, 706560)],
    8: [(15360, 245760), (24960, 399360), (34560, 552960), (51840, 829440), (80640, 1290240)],
    16: [(24960, 399360), (42240, 675840), (61440, 983040), (97920, 1566720), (153600, 2457600)],
    32: [(42240, 675840), (78720, 1259520), (115200, 1843200), (190080, 3041280), (299520, 4792320)],
}  # fmt: skip
_RING_BLOCKS_9600 = {4: (1920, 960, 480, 240, 160, 128), 8: (3240, 1620, 810, 405, 270, 216), 16: (6120, 3060, 1530, 765, 510, 408), 32: (11880, 5940, 2970, 1485, 990, 792)}  # fmt: skip


@pytest.mark.parametrize("R", [4, 8, 16, 32])
def test_ring_sizes_match_reference(R):
    for T, want in zip(_RING_T, _RING_SPEC[R], strict=False):
        for sms in (208, 212):  # identical for 208 and 212 SMs
            assert h.get_ring_sizes(R, 384, T, 6, 5120, 2304, sms) == want, (R, T, sms)
    for bm, nb in zip(BLOCK_MS, _RING_BLOCKS_9600[R], strict=False):
        assert kcfg(R=R, T=9600, M=1, block_m=bm).num_ring_blocks == nb
    # hardware-confirmed point: DEP4, T_max 8192 -> 9600: ring 30720, SF ring 491520
    c = kcfg(R=4, T=8192, M=1024)
    assert (
        c.num_max_tokens_per_rank,
        c.num_ring_tokens,
        c.num_sf_ring_tokens,
        c.block_m,
        c.num_stages,
        c.smem_bytes,
    ) == (9600, 30720, 491520, 128, 8, 217508)


def test_ring_v4_constant_and_sm_independence():
    # V4 (l1n == l2n == 16): no frontier growth -> the ring is the constant 5760 / 92160 for all EP and T_max
    for R in (1, 2, 4, 8, 16, 32):
        for T in (1920, 3840, 9600):
            assert (
                h.get_ring_sizes(R, 256, T, 6, 4096, 2048, 208) == (5760, 92160)
                or R == 1
            )
    # more SMs -> more warm-up clusters -> never a smaller ring
    for R in (4, 32):
        assert (
            h.get_ring_sizes(R, 384, 9600, 6, 5120, 2304, 148)[0]
            <= h.get_ring_sizes(R, 384, 9600, 6, 5120, 2304, 212)[0]
        )


_SYMM_MIB = {
    4: (298.2, 514.8, 1123.8),
    8: (339.4, 597.3, 1350.7),
    16: (442.4, 783.0, 1845.5),
    32: (628.1, 1174.7, 2835.1),
}


@pytest.mark.parametrize("R", [4, 8, 16, 32])
def test_symm_buffer_mib_matches_reference(R):
    for T, mib in zip((1920, 3840, 9600), _SYMM_MIB[R], strict=False):
        lay = h.build_symm_buffer_layout(
            R, 384, T, 6, 5120, 2304, "fp8xfp4", "swiglu", 1, SMS
        )
        assert round(lay.num_bytes / 2**20, 1) == mib, (R, T, lay.num_bytes / 2**20)


def test_symm_buffer_worked_layout_dep8():
    lay = h.build_symm_buffer_layout(
        8, 384, 9600, 6, 5120, 2304, "fp8xfp4", "swiglu", 1, SMS
    )
    want = [  # (region, offset, bytes) of the V4.1 DEP8 / T_max 9600 layout
        ("workspace", 0, 41554304), ("input_token", 41554304, 49152000), ("input_sf", 90706304, 1536000),
        ("input_topk_idx", 92242304, 460800), ("input_topk_weights", 92703104, 230400), ("shared_l1_sf", 92933504, 24576000),
        ("shared_l2_token", 117509504, 22118400), ("shared_l2_sf", 139627904, 11059200), ("l1_token", 150687104, 265420800),
        ("l1_sf", 416107904, 132710400), ("l1_topk_weights", 548818304, 207360), ("l2_token", 549025664, 119439360),
        ("l2_sf", 668465024, 59719680), ("combine_token", 728184704, 688128000),
    ]  # fmt: skip
    for name, off, nbytes in want:
        assert lay.regions[name] == (off, nbytes), name
    assert lay.num_bytes == 1416312704
    assert (
        lay.regions["src_token_topk_idx"] == (21140864, 14745600)
        and lay.regions["token_src_metadata"][1] == 5667840
    )
    assert lay.num_max_pool_tokens == 472320 and lay.workspace_bytes == 41554304
    # views: aliasing, strides and dtypes of slice_symm_buffer_from_layout
    v = lay.views
    assert (
        v["shared_l1_acts"] == v["x"]
        and v["x"].shape == (9600, 5120)
        and v["x_sf"].shape == (9600, 40)
    )
    assert v["l1_acts_sf"].strides == (1, lay.num_sf_ring_tokens) and v[
        "shared_l2_acts_sf"
    ].shape == (153600, 18)
    assert (
        v["topk_idx"].dtype == "int64"
        and v["topk_weights"].dtype == "float32"
        and v["x"].dtype == "float8_e4m3fn"
    )
    # S = 0: shared regions are empty, the ring follows the top-k weights directly
    l0 = h.build_symm_buffer_layout(
        8, 384, 9600, 6, 5120, 2304, "fp8xfp4", "swiglu", 0, SMS
    )
    assert (
        l0.regions["shared_l1_sf"][1]
        == l0.regions["shared_l2_token"][1]
        == l0.regions["shared_l2_sf"][1]
        == 0
    )
    assert (
        l0.l1_token_base == l0.input_topk_weights_base + 9600 * 6 * 4
        and "shared_l2_acts" not in l0.views
    )
    assert (
        l0.regions["combine_token"][1] == 6 * 9600 * 10240
        and lay.regions["combine_token"][1] == 7 * 9600 * 10240
    )


_SIGNALS = {
    "grid_sync_count": 0, "nvl_barrier_counter": 16, "nvl_barrier_signals": 20, "l1_task_count": 28, "l2_task_count": 36,
    "shared_l1_task_count": 44, "shared_l2_task_count": 52, "combine_ready_grid_idx": 128, "peer_grid_idx": 768,
    "expert_send_count": 1408, "expert_recv_count": 17792, "expert_recv_count_sum": 34176, "l1_full_count": 38272,
    "l1_empty_count": 4232576, "l2_full_mask": 8426880, "l2_empty_count": 16815488, "shared_l2_full_count": 21009792,
}  # fmt: skip


def test_signals_layout_matches_reference():
    assert h.SIGNALS_OFFSETS == _SIGNALS and h.SIGNALS_BYTES == 21140864
    lay = h.build_symm_buffer_layout(
        4, 384, 1920, 6, 5120, 2304, "fp8xfp4", "swiglu", 1, SMS
    )
    assert (
        lay.src_token_topk_idx_offset == 21140864
        and lay.token_src_metadata_offset == 21140864 + 384 * 1920 * 4
    )
    assert lay.workspace_bytes % 16 == 0 and lay.num_bytes == sum(n for _, n in (lay.regions[k] for k in (
        "workspace", "input_token", "input_sf", "input_topk_idx", "input_topk_weights", "shared_l1_sf", "shared_l2_token",
        "shared_l2_sf", "l1_token", "l1_sf", "l1_topk_weights", "l2_token", "l2_sf", "combine_token")))  # fmt: skip


def test_token_alignment_and_alignment_idempotence():
    assert h.get_token_alignment_for_mega_moe() == 1920
    a = kcfg(T=8192, M=64)
    b = kcfg(T=9600, M=64)
    c = kcfg(T=1, M=0)
    assert (
        a == b
        and a.num_max_tokens_per_rank == 9600
        and c.num_max_tokens_per_rank == 1920
    )
    with pytest.raises(AssertionError):
        kcfg(T=1920, M=1921)  # num_tokens > num_max_tokens_per_rank


# ---------------------------------------------------------------------------------------------------------------------
# config dataclass: compile key coverage, invariants, host asserts
# ---------------------------------------------------------------------------------------------------------------------
def test_compile_key_covers_every_codegen_input():
    base = kcfg(R=4, T=9600, M=1024, activation_clamp=10.0)
    keys = {base.cache_key()}
    # every primary input that changes generated code must change the digest
    variants = {
        "hidden": kcfg(shape=dict(V41, hidden=4608), activation_clamp=10.0),
        "intermediate_hidden": kcfg(shape=dict(V41, intermediate_hidden=2048), activation_clamp=10.0),
        "num_experts": kcfg(shape=dict(V41, num_experts=256), activation_clamp=10.0),
        "num_topk": kcfg(shape=dict(V41, num_topk=8), activation_clamp=10.0),
        "num_shared_experts": kcfg(S=0, activation_clamp=10.0),
        "num_ranks": kcfg(R=8, activation_clamp=10.0),
        "num_max_tokens_per_rank": kcfg(T=15360, activation_clamp=10.0),
        "num_sms": kcfg(sms=208, activation_clamp=10.0),
        "block_m": kcfg(M=64, activation_clamp=10.0),
        "clamp": kcfg(activation_clamp=9.0),
        "no clamp": kcfg(),
        "alpha": kcfg(activation_clamp=10.0, activation_alpha=1.5),
        "beta": kcfg(activation_clamp=10.0, activation_beta=0.25),
        "fast_math": kcfg(activation_clamp=10.0, fast_math=False),
        "max_stages": kcfg(activation_clamp=10.0, max_stages=7),
        "smem capacity": kcfg(activation_clamp=10.0, smem_capacity=334848),
        "trace": kcfg(activation_clamp=10.0, trace=True),
        "per_die": kcfg(activation_clamp=10.0, per_die=True),
        "localized_weights": kcfg(activation_clamp=10.0, localized_weights=True),
    }  # fmt: skip
    for what, c in variants.items():
        assert c.cache_key() not in keys, f"compile key does not cover {what}"
        keys.add(c.cache_key())
    # ... and what is a runtime argument must NOT change it: num_tokens that maps to the same BLOCK_M
    assert (
        kcfg(R=4, T=9600, M=1000, activation_clamp=10.0).cache_key()
        == kcfg(R=4, T=9600, M=1800, activation_clamp=10.0).cache_key()
    )
    # the digest inputs are exactly the declared compile-key fields, in order
    assert (
        tuple(n for n, _ in base.kernel_name_digest_inputs)
        == cfgmod.Sm107KernelConfig._COMPILE_KEY_FIELDS
    )
    assert re.fullmatch(r"sm107dg_bm128_dep4_s1_[0-9a-f]{16}", base.short_name())


def test_config_is_frozen_hashable_and_consistent():
    c = kcfg()
    assert hash(c) == hash(kcfg()) and c == kcfg() and c != kcfg(M=1)
    with pytest.raises(dataclasses.FrozenInstanceError):
        c.block_m = 64
    with pytest.raises(
        ValueError
    ):  # stale derived fields after replace() of a primary field are rejected
        dataclasses.replace(c, block_m=64)
    assert isinstance(c.to_dict()["smem_offsets"], dict)
    # field names are a contract with the kernel modules: never renamed
    contract = (  # noqa: SIM905
        "block_m block_n block_k umma_k load_block_m load_block_n store_block_m sf_block_m sf_block_n wg_block_m num_stages smem_bytes "
        "num_sms num_ranks num_experts num_experts_per_rank hidden intermediate_hidden num_shared_experts num_topk "
        "num_max_tokens_per_rank num_ring_tokens num_sf_ring_tokens num_ring_blocks num_bytes_per_pull num_dispatch_threads "
        "num_non_epilogue_threads num_epilogue_threads num_threads regs_dispatch regs_non_epilogue regs_epilogue tmem_cols_alloc "
        "tmem_sfa_col tmem_sfb_col accum_stages tma_store_stages schedule_stages smem_offsets clamp alpha beta fast_math per_die "
        "num_locality_domains num_experts_per_lane kernel_name_digest_inputs"
    ).split()  # fmt: skip
    for name in contract:
        assert hasattr(c, name), name
    assert (
        c.block_n,
        c.block_k,
        c.umma_k,
        c.accum_stages,
        c.tma_store_stages,
        c.schedule_stages,
        c.per_die,
        c.num_locality_domains,
    ) == (128, 128, 64, 2, 2, 2, False, 1)


def test_host_asserts_and_per_die_rules():
    with pytest.raises(AssertionError):
        kcfg(M=1, block_m=96)  # host candidate but not instantiable (K:1238)
    with pytest.raises(AssertionError):
        kcfg(M=1, block_m=8)
    with pytest.raises(AssertionError):
        h.get_sm107_kernel_config(
            5, 384, 1920, 1, 6, 5120, 2304, 1, num_sms=212
        )  # E % R != 0
    with pytest.raises(AssertionError):
        kcfg(sms=211)  # odd SM count
    with pytest.raises(AssertionError):
        kcfg(activation_clamp=-1.0)
    # per-die: both N dims must split into 2 domains x whole 256-clusters (2I % 512, H % 512), and the device must have 2 domains
    assert kcfg(per_die=True).num_locality_domains == 2
    assert kcfg(per_die=True, device_num_locality_domains=1).num_locality_domains == 1
    narrow = dict(
        V41, intermediate_hidden=2432
    )  # 2 * 2432 % 512 != 0 (the L1 N dimension does not split into 2 domains of whole clusters)
    assert (
        kcfg(per_die=True, shape=narrow).num_locality_domains == 1
        and kcfg(per_die=False).num_locality_domains == 1
    )
    with pytest.raises(AssertionError):
        kcfg(localized_weights=True, shape=narrow)
    one_chunk = dict(V41, hidden=3072)  # H <= 4096: one combine chunk of H * 2 bytes
    assert (
        kcfg(shape=one_chunk).num_combine_chunks == 1
        and kcfg(shape=one_chunk).combine_chunk_bytes == 6144
    )
    with pytest.raises(
        AssertionError, match="512"
    ):  # the register-pull dispatch needs hidden % 512 == 0 (2816 passes every other check)
        kcfg(shape=dict(V41, hidden=2816))
    with pytest.raises(AssertionError, match="combine chunking"):
        kcfg(
            shape=dict(V41, hidden=4864)
        )  # H*2/2 B per chunk is not a multiple of 32 x 16 B (K:1671)
    # the user-level config derives the same kernel config
    uc = cfgmod.Sm107MegaMoeConfig(num_experts=384, num_max_tokens_per_rank=8192, num_topk=6, hidden=5120, intermediate_hidden=2304,
                                   num_shared_experts=1, num_ranks=4, num_sms=212, activation_clamp=10.0)  # fmt: skip
    assert (
        uc.kernel_config(1024) == kcfg(R=4, T=8192, M=1024, activation_clamp=10.0)
        and uc.aligned_num_max_tokens_per_rank == 9600
    )
    with pytest.raises(ValueError):
        cfgmod.Sm107MegaMoeConfig(
            num_experts=384,
            num_max_tokens_per_rank=1920,
            num_topk=6,
            hidden=5120,
            intermediate_hidden=2304,
            activation="swiglustep",
        )


def test_tma_l2_promotion_knob_is_validated_keyed_and_forwarded():
    """The weight-TMA L2-promotion knob (None = the kernel's own policy): validated, part of the compile key, forwarded by the call-independent config."""
    auto = kcfg(activation_clamp=10.0)
    assert (
        auto.tma_l2_promotion is None
        and "tma_l2_promotion" in cfgmod.Sm107KernelConfig._COMPILE_KEY_FIELDS
    )
    keys = {auto.cache_key()}
    for v in (0, 64, 128, 256):
        c = kcfg(activation_clamp=10.0, tma_l2_promotion=v)
        assert c.tma_l2_promotion == v and c.cache_key() not in keys, v
        keys.add(c.cache_key())
    for bad in (32, 512, -1):
        with pytest.raises(ValueError):
            kcfg(activation_clamp=10.0, tma_l2_promotion=bad)
    kw = dict(
        num_experts=384,
        num_max_tokens_per_rank=8192,
        num_topk=6,
        hidden=5120,
        intermediate_hidden=2304,
        num_shared_experts=1,
        num_ranks=4,
        num_sms=212,
        activation_clamp=10.0,
    )
    assert cfgmod.Sm107MegaMoeConfig(**kw, tma_l2_promotion=0).kernel_config(
        1024
    ) == kcfg(R=4, T=8192, M=1024, activation_clamp=10.0, tma_l2_promotion=0)
    assert cfgmod.Sm107MegaMoeConfig(**kw).kernel_config(1024).tma_l2_promotion is None
    with pytest.raises(ValueError):
        cfgmod.Sm107MegaMoeConfig(**kw, tma_l2_promotion=96)


# ---------------------------------------------------------------------------------------------------------------------
# per-die (locality domain) geometry handed to the kernel roles: cluster split, tile coordinates, 3D weight TMA maps
# ---------------------------------------------------------------------------------------------------------------------
def test_perdie_cluster_split_and_scheduler_constants():
    c = kcfg(R=4, per_die=True, localized_weights=True)
    assert (c.num_locality_domains, c.num_l1_clusters, c.num_l2_clusters) == (
        2,
        18,
        20,
    )  # 4608 / 128 / 2, 5120 / 128 / 2
    assert (c.num_l1_clusters_per_domain, c.num_l2_clusters_per_domain) == (
        9,
        10,
    )  # half of the N-block clusters in each domain
    assert (c.num_min_steal_waves, c.domain_counter_stride) == (
        2,
        1,
    )  # kNumMinStealWaves; one counter slot per domain
    plain = kcfg(R=4)
    assert (
        plain.num_locality_domains,
        plain.num_l1_clusters_per_domain,
        plain.num_l2_clusters_per_domain,
    ) == (1, 18, 20)
    # every compile-relevant per-die input keys the cache
    assert len({kcfg(per_die=False).cache_key(), kcfg(per_die=True).cache_key(), c.cache_key(),
                kcfg(per_die=True, localized_weights=True, localized_shared_weights=True).cache_key()}) == 4  # fmt: skip


def test_perdie_tile_coords():
    coords = lambda e, nb, kb, **kw: h.get_perdie_weight_tile_coords(
        e, nb, kb, 4608, **kw
    )  # noqa: E731  (V4.1 L1: N = 2 * 2304 = 36 N blocks)
    assert coords(5, 0, 3) == (384, 5 * 2304, 0)
    assert coords(5, 17, 0) == (
        0,
        5 * 2304 + 17 * 128,
        0,
    )  # last block of the first half
    assert coords(5, 18, 0) == (
        0,
        5 * 2304,
        1,
    )  # first block of the second half -> slice 1, row 0 of the expert
    assert coords(5, 35, 39) == (39 * 128, 5 * 2304 + 17 * 128, 1)
    # shared experts: no expert offset; half-K stages advance K by SHARED_BLOCK_K (64)
    assert h.get_perdie_weight_tile_coords(0, 20, 7, 4608, is_shared=True) == (
        7 * 128,
        2 * 128,
        1,
    )
    assert h.get_perdie_weight_tile_coords(
        0, 20, 7, 4608, is_shared=True, shared_half_k=True
    ) == (7 * 64, 2 * 128, 1)
    # (slice, row) addresses every row block of every expert exactly once, rows stay inside the slice
    seen = {
        (half, row)
        for e in range(3)
        for nb in range(36)
        for _, row, half in [coords(e, nb, 0)]
    }
    assert len(seen) == 3 * 36 and all(
        0 <= row < 3 * 2304 and row % 128 == 0 for _, row in seen
    )
    # L2 (N = H = 5120): 40 blocks, halves of 2560
    assert h.get_perdie_weight_tile_coords(2, 20, 1, 5120) == (
        128,
        2 * 2560,
        1,
    ) and h.get_perdie_weight_tile_coords(2, 19, 1, 5120) == (
        128,
        2 * 2560 + 19 * 128,
        0,
    )


def test_perdie_tma_specs_v41():
    c = kcfg(R=4, per_die=True, localized_weights=True, localized_shared_weights=True)
    s = h.get_perdie_weight_tma_spec(
        c, "routed_l1", row_stride_bytes=2560, domain_stride_bytes=1 << 20
    )
    assert (
        s.dtype,
        s.global_dims,
        s.global_strides_bytes,
        s.box_dims,
        s.swizzle_bytes,
    ) == ("16U4_ALIGN8B", (5120, 96 * 2304, 2), (2560, 1 << 20), (128, 128, 1), 64)
    s = h.get_perdie_weight_tma_spec(
        c, "routed_l2", row_stride_bytes=1152, domain_stride_bytes=1 << 20
    )
    assert (s.dtype, s.global_dims, s.box_dims) == (
        "16U4_ALIGN8B",
        (2304, 96 * 2560, 2),
        (128, 128, 1),
    )
    s = h.get_perdie_weight_tma_spec(
        c, "shared_l1", row_stride_bytes=5120, domain_stride_bytes=1 << 20
    )
    assert (s.dtype, s.global_dims, s.box_dims, s.swizzle_bytes) == (
        "UINT8",
        (5120, 2304, 2),
        (c.shared_block_k, 128, 1),
        64,
    )
    s = h.get_perdie_weight_tma_spec(
        c, "shared_l2", row_stride_bytes=2304, domain_stride_bytes=1 << 20
    )
    assert (s.dtype, s.global_dims, s.box_dims) == (
        "UINT8",
        (2304, 2560, 2),
        (c.shared_block_k, 128, 1),
    )
    with pytest.raises(ValueError):
        h.get_perdie_weight_tma_spec(
            c, "bogus", row_stride_bytes=1, domain_stride_bytes=1
        )


def test_unsupported_reason_predicate():
    ok = (
        lambda R=4,
        E=384,
        T=8192,
        k=6,
        H=5120,
        I=2304,
        S=1,
        **kw: h.get_mega_moe_unsupported_reason(R, E, T, k, H, I, S, num_sms=SMS, **kw)
    )  # noqa: E731
    for R in (
        1,
        2,
        4,
        8,
        16,
        32,
    ):  # the production shape at every supported EP size, with and without the fused shared expert
        assert ok(R=R) is None and ok(R=R, S=0) is None
    assert "mma_type" in ok(mma_type="bf16xbf16") and "mma_type" in ok(
        mma_type="fp8xfp8"
    )
    assert "activation" in ok(activation="swiglustep") and "activation" in ok(
        activation="situ"
    )
    assert "num_ranks" in ok(R=64, E=384) and "num_ranks" in ok(R=0)
    assert ok(E=385) is not None  # E % R != 0 (also not in the validated table)
    # structural reasons (derivation of every BLOCK_M variant), with the validated-geometry table switched off
    free = lambda **kw: ok(allow_unvalidated=True, **kw)  # noqa: E731
    assert free(E=256, H=4096, I=2048) is None  # DeepSeek-V4-Flash: derivable
    assert free(E=385) is not None
    assert free(H=4864) is not None and "combine" in free(
        H=4864
    )  # combine chunking (K:1671)
    assert free(I=2300) is not None and free(H=5100) is not None  # not multiples of 128
    assert free(k=32, S=1) is not None  # top-k + shared must fit one warp


def test_support_predicate_admits_only_validated_geometries():
    ok = (
        lambda R=4,
        E=384,
        T=8192,
        k=6,
        H=5120,
        I=2304,
        S=1,
        **kw: h.get_mega_moe_unsupported_reason(R, E, T, k, H, I, S, num_sms=SMS, **kw)
    )  # noqa: E731
    table = h.get_validated_geometries()
    assert table == h.VALIDATED_GEOMETRIES and (5120, 2304, 6) in table
    for (hidden, inter, topk), (experts, evidence) in table.items():
        assert evidence and all(isinstance(e, int) for e in experts)
        for E in experts:
            for R in (
                1,
                2,
                4,
            ):  # every admitted shape is structurally derivable at the small EP sizes (and E=384 at every supported one, above)
                if E % R == 0 and E // R >= h.MIN_VALIDATED_EXPERTS_PER_RANK:
                    assert ok(R=R, E=E, k=topk, H=hidden, I=inter) is None, (
                        R,
                        E,
                        topk,
                        hidden,
                        inter,
                    )
    # derivable but never run on a GPU: refused with a reason naming the table; the escape hatch admits it
    for kw in (
        dict(E=256, H=4096, I=2048),
        dict(E=128),
        dict(E=64, k=3),
        dict(E=128, H=2048, I=768, k=8),
        dict(E=288),
        dict(E=384, k=8),
    ):
        reason = ok(**kw)
        assert (
            reason is not None and "not been validated" in reason and "5120" in reason
        ), kw
    assert ok(E=256, H=4096, I=2048, allow_unvalidated=True) is None
    assert ok(E=288, allow_unvalidated=True) is None


def test_support_predicate_experts_per_rank_dimension_and_the_draft_geometry():
    """The EP size is a validation dimension: at least 12 experts per rank (E=384 at DEP32 is the smallest value that ran); the MTP / DSpark draft geometry
    (hidden 5120, intermediate 2304, top-3, 128 experts) is admitted at R in {1, 2, 4, 8} only."""
    ok = (
        lambda R=4,
        E=384,
        T=8192,
        k=6,
        H=5120,
        I=2304,
        S=1,
        **kw: h.get_mega_moe_unsupported_reason(R, E, T, k, H, I, S, num_sms=SMS, **kw)
    )  # noqa: E731
    assert (
        h.MIN_VALIDATED_EXPERTS_PER_RANK == 12
        and h.VALIDATED_GEOMETRIES[(5120, 2304, 3)][0] == (128,)
        and h.VALIDATED_GEOMETRIES[(5120, 2304, 3)][1]
    )
    # V4.1: 12 experts per rank (E=384, DEP32) is admitted, fewer are not
    assert ok(R=32, E=384) is None and ok(R=24, E=384) is None  # 16 experts per rank
    assert ok(R=8, E=96) is None and ok(R=2, E=32) is None  # 12 and 16 experts per rank
    for R, E in ((16, 96), (4, 32), (32, 192)):  # 6, 8, 6 experts per rank
        reason = ok(R=R, E=E)
        assert (
            reason is not None
            and "not been validated" in reason
            and f"{E // R} experts per rank" in reason
            and "12" in reason
        ), (R, E, reason)
    # the draft geometry: R in {1, 2, 4, 8} (128, 64, 32, 16 experts per rank) admitted, DEP16 / DEP32 (8 / 4 experts per rank) refused
    for R in (1, 2, 4, 8):
        assert ok(R=R, E=128, k=3) is None, R
    for R in (16, 32):
        reason = ok(R=R, E=128, k=3)
        assert (
            reason is not None
            and "not been validated" in reason
            and f"{128 // R} experts per rank" in reason
        ), (R, reason)
    assert (
        ok(E=128, k=6) is not None
        and ok(E=64, k=3) is not None
        and ok(E=128, k=3, H=4096, I=2048) is not None
    )  # only the evidence's exact shape
    assert (
        ok(R=32, E=128, k=3, allow_unvalidated=True) is None
    )  # the development escape hatch skips the table, EP dimension included


def test_support_predicate_is_total_and_enforces_hidden_multiple_of_512():
    """``get_mega_moe_unsupported_reason`` returns a reason string for every input (a DeepGEMM-API caller falls through on it); the register-pull dispatch needs
    hidden % 512 == 0, which is checked by the predicate and by ``get_sm107_kernel_config`` instead of at JIT trace time."""
    free = (
        lambda R=4,
        E=384,
        T=8192,
        k=6,
        H=5120,
        I=2304,
        S=1,
        **kw: h.get_mega_moe_unsupported_reason(
            R, E, T, k, H, I, S, num_sms=SMS, allow_unvalidated=True, **kw
        )
    )  # noqa: E731
    reason = free(H=2304, I=2304)
    assert reason is not None and "hidden" in reason and "512" in reason
    with pytest.raises(AssertionError, match="512"):
        h.get_sm107_kernel_config(4, 384, 8192, 1, 6, 2304, 2304, 0, num_sms=SMS)
    for kw in (
        dict(T=0),
        dict(T=-5),
        dict(I=-128),
        dict(I=0),
        dict(H=0),
        dict(k=-1),
        dict(k=0),
        dict(E=0),
        dict(R=0),
        dict(R=33),
    ):
        assert isinstance(free(**kw), str), kw
    for args in (
        (2, 512, 0, 1, 2304, 0, 1),
        (2, 512, 8192, -1, 2304, 2304, 1),
        (2, None, 8192, 1, 2304, 2304, 1),
        (2, "512", 8192, 1, 2304, 2304, 1),
    ):
        assert isinstance(
            h.get_mega_moe_unsupported_reason(
                *args, num_sms=400, allow_unvalidated=True
            ),
            str,
        ), args  # ZeroDivisionError / ValueError / TypeError before
    assert isinstance(
        h.get_mega_moe_unsupported_reason(
            2, 512, 8192, 1, 2304, 2304, 1, num_sms=-2, allow_unvalidated=True
        ),
        str,
    )
    assert free() is None  # the V4.1 shape is unaffected


def test_support_predicate_refuses_options_the_kernel_does_not_implement():
    ok = (
        lambda R=4,
        E=384,
        T=8192,
        k=6,
        H=5120,
        I=2304,
        S=1,
        **kw: h.get_mega_moe_unsupported_reason(R, E, T, k, H, I, S, num_sms=SMS, **kw)
    )  # noqa: E731
    assert (
        ok(fast_math=True) is None
        and ok(S=0) is None
        and ok(S=1) is None
        and ok(localized_shared_weights=False) is None
    )
    assert (
        ok(fast_math=False) is None
        and ok(fast_math=False, allow_unvalidated=True) is None
    )  # both fast_math values are implemented
    assert "die-local shared" in ok(
        localized_shared_weights=True
    ) and "die-local routed weights" in ok(
        localized_shared_weights=True, allow_unvalidated=True
    )
    for S in (2, 3):
        reason = ok(S=S)
        assert (
            reason is not None
            and f"num_shared_experts={S}" in reason
            and "one merged shared expert" in reason
        )
    assert "num_shared_experts=-1" in ok(S=-1)
    # the options are refused before the geometry is looked at (a call that DeepGEMM serves must never reach the port with them)
    assert "not been validated" in ok(E=256, H=4096, I=2048, fast_math=False)


def test_secondary_shape_v4_consistent_with_analysts():
    # V4-Flash (H=4096, I=2048, E=256, 208 SMs): requested smem 230132/228324/231380/216484/223124/215684
    want = {16: 230132, 32: 228324, 64: 231380, 128: 216484, 192: 223124, 240: 215684}
    for bm, smem in want.items():
        c = kcfg(R=4, shape=V4, M=1, sms=208, block_m=bm)
        assert c.smem_bytes == smem and c.num_stages == _STAGES[bm][0]
        assert (
            c.num_combine_chunks == 2
        )  # 3*8*8192 = 196,608 > reusable (smem clause decisive for V4)
        assert h.get_ring_sizes(4, 256, 9600, 6, 4096, 2048, 208) == (5760, 92160)


# ---------------------------------------------------------------------------------------------------------------------
# GPU tests: zero mismatches against DeepGEMM's C++ host functions (deep_gemm._C)
# ---------------------------------------------------------------------------------------------------------------------
def _require_deep_gemm():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip(
            "CUDA device required (DeepGEMM host functions query the device SM count)"
        )
    dg = pytest.importorskip("deep_gemm")
    return torch, dg


def _divisors(E, upto=32):
    return [r for r in range(1, upto + 1) if E % r == 0]


@pytest.mark.parametrize("shape", [V41, V4], ids=["v41", "v4"])
def test_block_m_vs_deepgemm_cpp(shape):
    _, dg = _require_deep_gemm()
    E, topk = shape["num_experts"], shape["num_topk"]
    mismatches, n = [], 0
    for R in _divisors(E):
        for M in range(
            0, 16385
        ):  # every M; T_max is ignored by the C++ function (checked on a subsample below)
            want = dg._C.get_block_m_for_mega_moe(R, E, 9600, M, topk, "fp8xfp4")
            got = h.get_block_m_for_mega_moe(R, E, 9600, M, topk, "fp8xfp4")
            n += 1
            if want != got:
                mismatches.append((R, M, want, got))
        for T in (1920, 3840, 15360):
            for M in range(0, 16385, 97):
                assert dg._C.get_block_m_for_mega_moe(
                    R, E, T, M, topk, "fp8xfp8"
                ) == h.get_block_m_for_mega_moe(R, E, T, M, topk, "fp8xfp8")
    assert n > 90000 and not mismatches, mismatches[:10]
    assert (
        dg._C.get_token_alignment_for_mega_moe()
        == h.get_token_alignment_for_mega_moe()
        == 1920
    )


@pytest.mark.parametrize("shape", [V41, V4], ids=["v41", "v4"])
def test_symm_buffer_layout_vs_deepgemm_cpp(shape):
    torch, dg = _require_deep_gemm()
    sms = torch.cuda.get_device_properties(
        0
    ).multi_processor_count  # the C++ asserts num_sms == device SM count
    E, topk, H, I = (
        shape["num_experts"],
        shape["num_topk"],
        shape["hidden"],
        shape["intermediate_hidden"],
    )
    ops = dg._C._torch_ops
    mismatches, n = [], 0
    for R in _divisors(E):
        for T in (1920, 3840, 9600, 15360):
            for S in (0, 1):
                for mma in ("fp8xfp4", "fp8xfp8", "bf16xbf16"):
                    nbytes, info = ops.get_symm_buffer_size_for_mega_moe(
                        R, E, T, topk, H, I, mma, "swiglu", S
                    )
                    lay = h.get_symm_buffer_size_for_mega_moe(
                        R, E, T, topk, H, I, mma, "swiglu", S, sms
                    )
                    n += 1
                    if (nbytes, list(info)) != (lay[0], lay[1]):
                        mismatches.append((R, T, S, mma, nbytes, list(info), lay))
    assert n > 100 and not mismatches, mismatches[:3]


_PROBE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "_sm107_dg_print_probe.py"
)


@pytest.mark.skipif(
    os.environ.get("FLASHINFER_SM107_DG_PRINT_CONFIGS") != "1",
    reason="slow: compiles six DeepGEMM kernels",
)
def test_stages_smem_vs_dg_print_configs():
    """Run DeepGEMM at EP1 (one launch per BLOCK_M, V4.1 shape) with DG_PRINT_CONFIGS=1 and compare BOTH printed lines:
    ``SM107 MegaMoE: stages=, smem=`` (shipped values) and ``MegaMoEConfig(...)`` (SM100-sized config)."""
    torch, _ = _require_deep_gemm()
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    env = dict(os.environ, DG_MEGA_MOE_SM107="1", DG_PRINT_CONFIGS="1")
    proc = subprocess.run([sys.executable, _PROBE, "--hidden", "5120", "--intermediate-hidden", "2304", "--num-experts", "384",
                           "--num-topk", "6", "--num-max-tokens-per-rank", "1920"],
                          capture_output=True, text=True, env=env, timeout=7200)  # fmt: skip
    out = proc.stdout + proc.stderr
    rows = re.findall(r"PROBE bm=(\d+) M=(\d+)\n(?:.*\n)*?(MegaMoEConfig\(num_ranks=1[^\n]*\)): (MegaMoEConfig\([^\n]*\))\n"
                      r"(?:.*\n)*?SM107 MegaMoE: stages=(\d+), smem=(\d+)", out)  # fmt: skip
    assert len(rows) == len(BLOCK_MS), out[-3000:]
    for bm, M, key, line_a, stages, smem in rows:
        bm, M = int(bm), int(M)
        mine = h.get_sm107_kernel_config(
            1, 384, 1920, M, 6, 5120, 2304, 0, block_m=bm, num_sms=sms
        )
        assert (int(stages), int(smem)) == (mine.num_stages, mine.smem_bytes), (
            bm,
            stages,
            smem,
        )
        ring, sf_ring = h.get_ring_sizes(1, 384, 1920, 6, 5120, 2304, sms)
        base = h.get_mega_moe_config(
            1, 384, 384, 1920, M, 6, 5120, 2304, ring, sf_ring, force_block_m=bm
        )
        assert str(base) == line_a and key == h.get_mega_moe_print_key(
            1, 384, 5120, 2304, 1920, M, 6
        ), (bm, key, line_a)


_RING_ORACLE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "_sm107_dg_ring_oracle.cu"
)


def _deep_gemm_include_dir():
    try:
        import deep_gemm
    except Exception:
        return None
    d = os.path.join(os.path.dirname(deep_gemm.__file__), "include")
    return d if os.path.isdir(os.path.join(d, "deep_gemm", "scheduler")) else None


def test_ring_sizes_vs_deepgemm_headers_at_any_sm_count(tmp_path):
    """The only SM-count dependent host math (``sched::get_num_max_live_pool_blocks`` -> ring tokens) compiled from DeepGEMM's OWN
    headers with nvcc (the Python C++ API can only be evaluated at the device SM count) and compared with the port at SM counts
    {120, 132, 148, 152, 160, 196, 208, 212}, both shapes, T_max, EP sizes and a direct grid of the live-block function."""
    import shutil

    nvcc = shutil.which("nvcc") or (
        "/usr/local/cuda/bin/nvcc"
        if os.path.exists("/usr/local/cuda/bin/nvcc")
        else None
    )
    inc = _deep_gemm_include_dir()
    if nvcc is None or inc is None:
        pytest.skip("needs nvcc and DeepGEMM's include directory")
    exe = str(tmp_path / "ring_oracle")
    cmd = [nvcc, "-std=c++20", "-O1", "--expt-relaxed-constexpr", "--expt-extended-lambda", "-arch=sm_100f", "-I", inc,
           "--diag-suppress=39,161,174,177,186,940,20014", "-Xcompiler", "-Wno-deprecated-declarations", "-o", exe, _RING_ORACLE]  # fmt: skip
    build = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    assert build.returncode == 0, build.stderr[-3000:]
    out = subprocess.run(
        [exe], capture_output=True, text=True, timeout=300, check=True
    ).stdout.splitlines()
    n_ring = n_live = 0
    for line in out:
        f = line.split()
        if f[0] == "LIVE":
            nb, sms, want = int(f[1]), int(f[2]), int(f[3])
            assert h.get_num_max_live_pool_blocks(nb, sms, 5120, 2304) == want, (
                nb,
                sms,
            )
            n_live += 1
        else:
            R, E, T, topk, H, I, sms, ring, sf_ring = (int(x) for x in f)
            assert h.get_ring_sizes(R, E, T, topk, H, I, sms) == (ring, sf_ring), (
                R,
                E,
                T,
                topk,
                H,
                I,
                sms,
            )
            n_ring += 1
    assert n_ring >= 600 and n_live >= 4000, (n_ring, n_live)
