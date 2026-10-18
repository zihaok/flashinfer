# SPDX-License-Identifier: Apache-2.0
# Port of DeepGEMM (MIT License, Copyright (c) 2025 DeepSeek; see licenses/LICENSE.deepgemm.txt) sm107_fp8_fp4_mega_moe.cuh / GEMM-core replay tests.
"""EP1 replay of DeepGEMM's post-dispatch state through the port's GEMM part (``mode_mask = MODE_GEMM``): bit-exact ``l2_token`` / ``l2_sf`` /
``combine_token`` and torch-reduced ``y`` vs DeepGEMM's own results, for all six BLOCK_M (see ``replay_utils`` for the method).

Needs one Rubin GPU (sm_107, 212 SMs) and DeepGEMM (``deep_gemm``).  Cases whose pool blocks exceed the ring (DeepGEMM wrapped the ring itself) are skipped.
Each distinct BLOCK_M costs one DeepGEMM JIT compile + one DSL compile on a cold cache.

    pytest tests/experimental/test_sm107_dg_gemm_core_replay.py -x -q
    pytest tests/experimental/test_sm107_dg_gemm_core_replay.py -q -k stress      # 1000-launch hang / determinism check
"""

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


@pytest.fixture(scope="module")
def env32():
    return R.make_env(32, 1920)


@pytest.fixture(scope="module")
def env8():
    return R.make_env(8, 1920, group=None)


def _check(res):
    if res.skipped:
        pytest.skip(res.message)
    assert res.passed, str(res)


# (BLOCK_M, M, masked fraction): partial last blocks everywhere (E = 32, top-6), masked routes (-1 slots), tiny M
CASES_E32 = [
    (16, 1, 0.0),
    (16, 5, 0.0),
    (16, 24, 0.0),
    (16, 300, 0.3),
    (32, 8, 0.0),
    (32, 50, 0.0),
    (32, 100, 0.0),
    (32, 400, 0.3),
    (64, 100, 0.0),
    (64, 256, 0.0),
    (64, 600, 0.0),
    (64, 900, 0.3),
    (128, 300, 0.0),
    (128, 500, 0.0),
    (128, 700, 0.3),
    (192, 300, 0.0),
    (192, 700, 0.0),
    (192, 900, 0.3),
    (240, 400, 0.0),
    (240, 700, 0.0),
    (240, 900, 0.3),
]


@pytest.mark.parametrize("bm,m,masked", CASES_E32)
def test_replay_e32(env32, bm, m, masked):
    _check(R.replay_case(env32, m, bm, masked))


@pytest.mark.parametrize("bm", [16, 32, 64, 128, 192, 240])
@pytest.mark.parametrize("m", [0, 1, 2])
def test_replay_tiny_m(env32, bm, m):
    _check(R.replay_case(env32, m, bm))


# E = 8: several m-blocks per expert at the large BLOCK_M (the ring of E = 8 is small: cases beyond it are skipped)
@pytest.mark.parametrize(
    "bm,m,masked",
    [
        (128, 400, 0.0),
        (128, 600, 0.0),
        (128, 800, 0.3),
        (192, 300, 0.0),
        (192, 500, 0.0),
        (192, 700, 0.3),
        (240, 300, 0.0),
        (240, 600, 0.0),
        (240, 700, 0.3),
    ],
)
def test_replay_e8_multi_block(env8, bm, m, masked):
    _check(R.replay_case(env8, m, bm, masked))


def test_replay_natural_block_m(env32):
    """DeepGEMM's own BLOCK_M heuristic (the package's port must pick the same variant, otherwise the layouts would not match)."""
    for m in (24, 100, 600):
        _check(R.replay_case(env32, m, 0))


@pytest.mark.parametrize("bm,m", [(128, 700), (16, 24), (240, 900)])
def test_replay_stress_1000_launches(env32, bm, m):
    res = R.replay_case(env32, m, bm, 0.0, stress_launches=1000)
    _check(res)


# ---------------------------------------------------------------------------------------------------------------------
# S = 1 (one shared expert): the shared GEMM tasks (half-K stages, E4M3 x E4M3) and the shared slot of the combine buffer
# ---------------------------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def env32_s1():
    return R.make_env(32, 1920, num_shared_experts=1)


@pytest.fixture(scope="module")
def env4_top2_s1():
    return R.make_env(4, 1920, topk=2, num_shared_experts=1)


# shared L2 tasks go BEFORE the routed ones when `ceil(M / BLOCK_M) * 20 <= 106` (early), else after them (late)
CASES_S1_E32 = [
    (16, 24, 0.0),  # early (2 m-blocks)
    (16, 100, 0.0),  # late (7 m-blocks)
    (32, 100, 0.0),  # early
    (32, 400, 0.3),  # late
    (64, 256, 0.0),  # early
    (64, 600, 0.0),  # late
    (128, 300, 0.0),  # early
    (128, 700, 0.3),  # late (6 m-blocks)
    (192, 300, 0.0),  # early
    (192, 900, 0.3),  # early (5 m-blocks, the boundary)
    (240, 400, 0.0),  # early
    (240, 900, 0.3),  # early
]


@pytest.mark.parametrize("bm,m,masked", CASES_S1_E32)
def test_replay_shared_e32(env32_s1, bm, m, masked):
    _check(R.replay_case(env32_s1, m, bm, masked))


@pytest.mark.parametrize("bm", [16, 128, 240])
@pytest.mark.parametrize("m", [0, 1, 2])
def test_replay_shared_tiny_m(env32_s1, bm, m):
    _check(R.replay_case(env32_s1, m, bm))


# late shared L2 at the large BLOCK_M needs >= 6 token blocks (M > 5 * BLOCK_M): fewer experts / top-2 keep the routed pool inside the ring
@pytest.mark.parametrize("bm,m", [(192, 1200), (240, 1250), (240, 1700)])
def test_replay_shared_late_large_block_m(env4_top2_s1, bm, m):
    _check(R.replay_case(env4_top2_s1, m, bm))


def test_replay_shared_stress_1000_launches(env32_s1):
    _check(R.replay_case(env32_s1, 300, 128, 0.0, stress_launches=1000))


# ---------------------------------------------------------------------------------------------------------------------
# FULL kernel (dispatch + GEMM + combine, mode_mask = MODE_ALL) at EP1 vs DeepGEMM's y, S = 0 and S = 1 (ring wrap included): the end-to-end version of the replay
# ---------------------------------------------------------------------------------------------------------------------
FULL_CASES = [
    (16, 24, 0.0),
    (16, 300, 0.3),
    (32, 100, 0.0),
    (64, 600, 0.0),
    (128, 300, 0.0),
    (128, 1500, 0.3),
    (192, 700, 0.0),
    (192, 1900, 0.0),
    (240, 900, 0.3),
    (240, 1900, 0.0),
    (16, 1, 0.0),
    (128, 0, 0.0),
]


@pytest.mark.parametrize("bm,m,masked", FULL_CASES)
def test_full_kernel_ep1_s0(env32, bm, m, masked):
    res = R.full_kernel_case(env32, m, bm, masked)
    assert res.passed, str(res) + " " + res.message


@pytest.mark.parametrize("bm,m,masked", FULL_CASES)
def test_full_kernel_ep1_s1(env32_s1, bm, m, masked):
    res = R.full_kernel_case(env32_s1, m, bm, masked)
    assert res.passed, str(res) + " " + res.message


# ---------------------------------------------------------------------------------------------------------------------
# L2 promotion of the weight tensor maps: the host tensor maps (256 B) and the atoms' own descriptors (128 B) must give identical results at every BLOCK_M
# (the auto policy runs 256 B for BLOCK_M >= 128 and the DSL default below; the forced values go through `cfg.tma_l2_promotion`)
# ---------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "bm,m,promo",
    [
        (16, 300, 256),
        (64, 600, 256),
        (128, 700, 0),
        (192, 700, 0),
        (240, 900, 0),
        (128, 700, 128),
        (128, 700, 64),
    ],
)
def test_replay_forced_l2_promotion(env32, bm, m, promo):
    _check(R.replay_case(env32, m, bm, 0.0, cfg_replace={"tma_l2_promotion": promo}))


@pytest.mark.parametrize(
    "bm,m,promo", [(128, 1500, 256), (240, 1900, 0), (16, 300, 256)]
)
def test_full_kernel_forced_l2_promotion(env32_s1, bm, m, promo):
    res = R.full_kernel_case(
        env32_s1, m, bm, 0.0, cfg_replace={"tma_l2_promotion": promo}
    )
    assert res.passed, str(res) + " " + res.message
