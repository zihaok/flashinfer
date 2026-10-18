# SPDX-License-Identifier: Apache-2.0
# Port of DeepGEMM (MIT License, Copyright (c) 2025 DeepSeek; see licenses/LICENSE.deepgemm.txt) sm107_fp8_fp4_mega_moe.cuh / scheduler/sm107_mega_moe.cuh.
"""CPU tests of the per-die (locality-domain) scheduling rules and weight addressing of the SM107 port (no GPU needed, the CUTLASS DSL must be importable).

* a Python model of ``MegaMoESchedulerLD::get_next_task_idx`` (SCH:196-222) with ``kNumLocalityDomains = 2`` driven by many randomly interleaving scheduler pairs: every
  ``(m_block, n_cluster)`` task of a GEMM is claimed EXACTLY once, own-queue claims stay inside the domain's N-cluster range ``[d*C/2, (d+1)*C/2)``, stealing happens only when
  the GEMM has >= ``kNumMinStealWaves`` waves of tasks (``num_m_blocks * C >= 2 * (kNumSMs / 2)``), with a domain that has no pair the queue is drained by thieves only above that threshold;
* the L2 wait numbers ``(pool_block + 1) * (kNumL1Clusters / D)`` of BOTH domains;
* the 3D weight tile coordinates of ``kernel/perdie.py`` against the package's reference model (``heuristics.get_perdie_weight_tile_coords``), its compile-time slice stride against
  ``heuristics.get_perdie_domain_stride_bytes`` and the geometry of the 3D tensor map (``get_perdie_weight_tma_spec``).
"""

import os
import random
import sys

import pytest

pytest.importorskip("cutlass.cute")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flashinfer.experimental.sm107_deepgemm_megamoe import heuristics as H  # noqa: E402
from flashinfer.experimental.sm107_deepgemm_megamoe.kernel import perdie  # noqa: E402

NUM_SMS = 212
NUM_MIN_STEAL_WAVES = 2


class QueueModel:
    """Counters of ONE GEMM (L1, L2 or a shared one) for ``D`` locality domains: the claim helper of SCH:196-222 as plain Python."""

    def __init__(self, num_domains: int, nclusters: int, num_m_blocks: int):
        self.D, self.C, self.M = num_domains, nclusters, num_m_blocks
        self.counter = [0] * num_domains
        assert nclusters % num_domains == 0

    def _claim(self, d):
        idx = self.counter[d]
        self.counter[d] += 1  # atom.add happens even when the claim then fails
        nl = self.C // self.D
        if idx >= self.M * nl:
            return None
        return idx // nl, d * nl + idx % nl

    def claim(self, dom):
        """``get_next_task_idx<C>(counter, num_m_blocks, dom)`` with stealing; returns ``(m_block, n_cluster, stolen)`` or ``None``."""
        got = self._claim(dom if self.D > 1 else 0)
        if got is not None:
            return got + (False,)
        if self.D > 1 and self.M * self.C >= NUM_MIN_STEAL_WAVES * (NUM_SMS // 2):
            got = self._claim(dom ^ 1)
            if got is not None:
                return got + (True,)
        return None


def drain(queue: QueueModel, pair_domains, rng: random.Random):
    """Random interleaving of the claim attempts of ``len(pair_domains)`` scheduler pairs until every pair sees an empty queue."""
    claimed, stolen = [], []
    active = list(range(len(pair_domains)))
    while active:
        p = rng.choice(active)
        got = queue.claim(pair_domains[p])
        if got is None:
            active.remove(p)
            continue
        m, n, was_stolen = got
        claimed.append((m, n))
        if was_stolen:
            stolen.append((pair_domains[p], m, n))
        else:
            nl = queue.C // queue.D
            assert (
                pair_domains[p] * nl <= n < (pair_domains[p] + 1) * nl or queue.D == 1
            )
    return claimed, stolen


@pytest.mark.parametrize("nclusters", [18, 20, 36])  # L1, L2, shared L1 with S = 2
@pytest.mark.parametrize("num_m_blocks", [1, 3, 11, 12, 40])
@pytest.mark.parametrize("split", [(53, 53), (80, 26)])
def test_claims_cover_every_task_exactly_once_and_stealing_follows_the_threshold(
    nclusters, num_m_blocks, split
):
    rng = random.Random(nclusters * 1000 + num_m_blocks)
    domains = [0] * split[0] + [1] * split[1]
    q = QueueModel(2, nclusters, num_m_blocks)
    claimed, stolen = drain(q, domains, rng)
    assert sorted(claimed) == sorted(
        (m, n) for m in range(num_m_blocks) for n in range(nclusters)
    )
    enabled = num_m_blocks * nclusters >= NUM_MIN_STEAL_WAVES * (NUM_SMS // 2)
    if not enabled:
        assert not stolen
    # one domain only: with stealing disabled the other queue is NEVER drained (the hang of an imbalanced map); with stealing everything completes
    q1 = QueueModel(2, nclusters, num_m_blocks)
    claimed1, stolen1 = drain(q1, [0] * 106, random.Random(7))
    if enabled:
        assert sorted(claimed1) == sorted(
            (m, n) for m in range(num_m_blocks) for n in range(nclusters)
        )
        assert stolen1
    else:
        nl = nclusters // 2
        assert sorted(claimed1) == sorted(
            (m, n) for m in range(num_m_blocks) for n in range(nl)
        )  # domain 1's N half is never claimed


@pytest.mark.parametrize("pool_block", [0, 1, 7, 100])
def test_l2_wait_threshold(pool_block):
    cfg = H.get_sm107_kernel_config(
        4,
        384,
        3840,
        1024,
        6,
        5120,
        2304,
        1,
        num_sms=NUM_SMS,
        activation_clamp=10.0,
        per_die=True,
        localized_weights=True,
    )
    assert cfg.num_locality_domains == 2
    assert cfg.num_l1_clusters_per_domain == 9 and cfg.num_l2_clusters_per_domain == 10
    assert (
        cfg.num_min_steal_waves == NUM_MIN_STEAL_WAVES
        and cfg.domain_counter_stride == 1
    )
    need = (pool_block + 1) * (
        cfg.num_l1_clusters // cfg.num_locality_domains
    )  # scheduler_perdie.wait_l1_claimed
    assert need == (pool_block + 1) * 9
    # the single-domain config keeps the single-queue numbers
    cfg1 = H.get_sm107_kernel_config(
        4, 384, 3840, 1024, 6, 5120, 2304, 1, num_sms=NUM_SMS, activation_clamp=10.0
    )
    assert cfg1.num_locality_domains == 1 and cfg1.num_l1_clusters_per_domain == 18


@pytest.mark.parametrize("ranks,E", [(4, 384), (8, 384), (1, 32)])
@pytest.mark.parametrize("kind", ["routed_l1", "routed_l2"])
def test_weight_geometry_and_slice_stride_match_host(ranks, E, kind):
    cfg = H.get_sm107_kernel_config(
        ranks,
        E,
        1920,
        256,
        6,
        5120,
        2304,
        0,
        num_sms=NUM_SMS,
        activation_clamp=10.0,
        per_die=True,
        localized_weights=True,
    )
    rows, row_bytes = perdie.weight_geometry(cfg, kind)
    spec = H.get_perdie_weight_tma_spec(cfg, kind)
    k_elems, host_rows, host_row_bytes, *_ = H._perdie_weight_geometry(cfg, kind)
    assert (rows, row_bytes) == (host_rows, host_row_bytes)
    assert spec.global_dims == (k_elems, rows, 2) and spec.global_strides_bytes == (
        row_bytes,
        perdie.domain_stride_bytes(cfg, kind),
    )
    assert k_elems // 2 == row_bytes  # packed FP4: 2 elements per byte
    assert (
        perdie.domain_stride_bytes(cfg, kind) % H.LOCALITY_ALLOC_GRANULARITY == 0
        and perdie.domain_stride_bytes(cfg, kind) >= rows * row_bytes
    )
    assert (
        spec.box_dims == (128, 128, 1) and spec.swizzle_bytes == 64
    )  # 128 fp4 = 64 B x 128 rows x 1 (the port's UINT8 map: box (64 B, 128, 1))


def _kernel_tile_coords(local_expert, n_block, shape_n):
    """Pure-Python twin of ``perdie.b_tile_coords`` (K:934-936) in units of 128-row tiles."""
    half_n = shape_n // 2
    n_in_matrix = n_block * 128
    half_idx = 1 if n_in_matrix >= half_n else 0
    half_row = local_expert * half_n + n_in_matrix - half_idx * half_n
    assert half_row % 128 == 0
    return half_row // 128, half_idx


@pytest.mark.parametrize("shape_n", [4608, 5120])
@pytest.mark.parametrize("experts", [1, 8, 96])
def test_tile_coords_match_host_reference(shape_n, experts):
    for e in range(experts):
        for n_cluster in range(shape_n // 256):
            for r in range(2):  # the two CTAs of a cluster
                n_block = 2 * n_cluster + r
                k_idx, half_row, half_idx = H.get_perdie_weight_tile_coords(
                    e, n_block, 0, shape_n
                )
                tile_row, dom = _kernel_tile_coords(e, n_block, shape_n)
                assert (tile_row * 128, dom, k_idx) == (half_row, half_idx, 0)
                # both CTAs of a cluster are in the same half, and the half is the one of the cluster's domain (n_cluster in [d*C/2, (d+1)*C/2))
                assert dom == (1 if n_cluster >= shape_n // 512 else 0)
