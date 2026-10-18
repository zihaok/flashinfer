# SPDX-License-Identifier: Apache-2.0
# Port of DeepGEMM (MIT License, Copyright (c) 2025 DeepSeek; see licenses/LICENSE.deepgemm.txt) sm107_fp8_fp4_mega_moe.cuh / test_mega_moe.py + bench.py (benchmark protocol);.
# PTX helper patterns derived from FlashInfer's vendored NVIDIA CuTe-DSL MegaMoE (BSD-3-Clause).
"""Smoke tests for the SM107 DeepGEMM-vs-CuteDSL A/B benchmark harness (CPU helpers + a 1-GPU EP1 DeepGEMM run).

Run: ``pytest tests/experimental/test_sm107_dg_bench_smoke.py`` (CPU tests always; GPU tests need a Rubin GPU, cc (10, 7), and
the ``deep_gemm`` module importable, otherwise they skip).
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

HARNESS = (
    Path(__file__).resolve().parents[2]
    / "benchmarks"
    / "moe_ep"
    / "backends"
    / "mega"
    / "kernel"
    / "sm107"
    / "bench_moe_ep_sm107_deepgemm_cutedsl_mega.py"
)


@pytest.fixture(scope="module")
def bench():
    spec = importlib.util.spec_from_file_location("bench_mega_moe_ab", HARNESS)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["bench_mega_moe_ab"] = mod
    spec.loader.exec_module(mod)
    return mod


def _is_rubin_with_deep_gemm() -> bool:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
        return False
    return importlib.util.find_spec("deep_gemm") is not None


needs_rubin = pytest.mark.skipif(
    not _is_rubin_with_deep_gemm(), reason="needs a Rubin (cc 10.7) GPU and deep_gemm"
)


# ---------------------------------------------------------------------------------------------------------------------
# CPU: statistics, schema helpers, M lists, config parsing, perf formulas
# ---------------------------------------------------------------------------------------------------------------------
def test_percentile_and_primary_statistic(bench):
    assert bench.pct([1, 2, 3, 4], 50) == 2.5
    assert bench.pct([5], 10) == 5
    # two ranks, four iterations: max over ranks per iteration is [3, 4, 3, 9] -> median 3.5
    samples = [[1.0, 4.0, 3.0, 9.0], [3.0, 2.0, 1.0, 2.0]]
    st = bench.summarize_samples(samples)
    assert st["median_of_max_us"] == 3.5
    assert st["rank_median_us"] == [3.5, 2.0]
    assert st["max_of_max_us"] == 9.0
    st2 = bench.summarize_samples([[1.0, 2.0, 3.0, 4.0]], n_windows=2)
    assert st2["window_medians_of_max_us"] == [1.5, 3.5]


def test_perf_model_matches_deepgemm_formulas(bench):
    # DEP4, M=1024, S=1, 219 us: formulas of tests/test_mega_moe.py:385-418 (numbers recomputed by hand below)
    H, I, S, K = 5120, 2304, 1, 6
    nrecv, ntouch, M, t = 6000, 96, 1024, 219.0
    p = bench.perf_model(M, nrecv, ntouch, H, I, S, K, t)
    flops = 2 * nrecv * H * I * 3 + 2 * M * H * I * S * 3
    assert math.isclose(p["tflops"], flops / 1e12 / (t * 1e-6), rel_tol=1e-12)
    t_red = M * H * 2 * (1 + K) / 6.5e12
    assert math.isclose(p["t_reduction_us"], t_red * 1e6, rel_tol=1e-12)
    assert p["overlap_tflops"] > p["tflops"]
    p0 = bench.perf_model(M, nrecv, ntouch, H, I, 0, K, t)
    assert p0["flops"] == 2 * nrecv * H * I * 3
    assert bench.perf_model(0, 0, 0, H, I, 0, K, 100.0)["tflops"] == 0.0


def test_parse_config_text_on_deepgemm_lines(bench):
    text = (
        "MegaMoEConfig(num_ranks=4, num_experts=384, hidden=5120, intermediate_hidden=2304, num_max_tokens_per_rank=9600, "
        "num_tokens=1024, num_topk=6): MegaMoEConfig(block_m=128, block_n=128, block_k=128, load_block_m=64, load_block_n=128, "
        "store_block_m_l1=32, store_block_m_l2=32, sf_block_m=128, sf_block_n=128, num_ring_tokens=30720, num_sf_ring_tokens=491520, "
        "swizzle_acts_mode=128, swizzle_weights_mode=128, num_stages=7, smem_size=219540, num_dispatch_threads=128, "
        "num_non_epilogue_threads=128, num_epilogue_threads=256, num_bytes_per_pull=5120)\n"
        "SM107 MegaMoE: stages=8, smem=217508\nunrelated line\n"
    )
    cfg = bench.parse_config_text(text)
    assert (cfg["block_m"], cfg["num_ring_tokens"], cfg["num_sf_ring_tokens"]) == (
        128,
        30720,
        491520,
    )
    assert (cfg["stages"], cfg["smem"]) == (
        8,
        217508,
    )  # the shipped SM107 values, not the SM100-sized num_stages=7 of line 1
    assert (cfg["cfg_num_stages"], cfg["cfg_smem_size"]) == (7, 219540)
    assert cfg["key_num_ranks"] == 4 and len(cfg["lines"]) == 2


def test_regime_helpers(bench):
    def bm(
        m,
    ):  # synthetic: 16 up to 160, 32 up to 384, 64 up to 903, 128 after (like DEP4 for the first four regimes)
        return 16 if m <= 160 else 32 if m <= 384 else 64 if m <= 903 else 128

    segs = bench.regime_segments(bm, 1200)
    assert segs == [(1, 160, 16), (161, 384, 32), (385, 903, 64), (904, 1200, 128)]
    assert bench.boundary_points(segs) == [160, 161, 384, 385, 903, 904]
    assert bench.regime_representatives(segs) == [80, 272, 644, 1052]


def test_parse_m_tokens(bench):
    def presets(name):
        return {
            "core": [1, 8, 64, 8192],
            "serving": [1280],
            "boundary": [160, 161],
            "regimes": [80],
            "perf": [1, 64, 80],
        }[name]

    pts = bench.parse_m_tokens(
        "core,serving,1000,64/1024/1024/64,rand512,64", presets, t_user=4096
    )
    labels = [p.label for p in pts]
    assert labels == [
        "1",
        "8",
        "64",
        "1280",
        "1000",
        "64/1024/1024/64",
        "rand512",
    ]  # 8192 > T_max dropped, duplicate 64 dropped
    assert pts[5].m_vec(world=8, t_user=4096, seed=0) == [
        64,
        1024,
        1024,
        64,
        64,
        1024,
        1024,
        64,
    ]
    mv = pts[6].m_vec(world=4, t_user=4096, seed=0)
    assert len(mv) == 4 and all(3584 <= m <= 4096 for m in mv)
    assert mv == pts[6].m_vec(
        world=4, t_user=4096, seed=0
    )  # deterministic: every rank derives the same vector
    with pytest.raises(ValueError):
        bench.parse_m_tokens("9999", presets, t_user=4096)


def _stock_shared_sf_layout(
    sf: torch.Tensor, block_m: int, num_max_sf_tokens: int
) -> torch.Tensor:
    """tests/test_mega_moe.py:36-50 verbatim (reference for the vectorised version)."""
    num_tokens, packed_sf_k = sf.shape
    align = lambda x, y: (x + y - 1) // y * y  # noqa: E731
    aligned_block_m = align(block_m, 128)
    num_m_blocks = (num_tokens + block_m - 1) // block_m
    result = torch.empty_strided(
        (num_max_sf_tokens, packed_sf_k),
        (1, num_max_sf_tokens),
        dtype=sf.dtype,
        device=sf.device,
    )
    result.zero_()
    for block_idx in range(num_m_blocks):
        num_block_tokens = min(block_m, num_tokens - block_idx * block_m)
        for m_idx in range(num_block_tokens):
            transposed_m_idx = (
                (m_idx // 128) * 128 + (m_idx % 32) * 4 + (m_idx % 128) // 32
            )
            result[block_idx * aligned_block_m + transposed_m_idx].copy_(
                sf[block_idx * block_m + m_idx]
            )
    return result


@pytest.mark.parametrize(
    "m,block_m",
    [(1, 16), (17, 16), (200, 32), (300, 128), (500, 192), (1000, 240), (0, 64)],
)
def test_vectorised_shared_sf_layout_is_bit_identical_to_stock(bench, m, block_m):
    torch.manual_seed(1234 + m)
    k4 = 40
    sf = torch.randint(-(2**31), 2**31 - 1, (m, k4), dtype=torch.int32)
    num_max = (
        math.ceil(max(m, 1) / 8)
    ) * 128 * 4 + 1024  # comfortably above the needed rows
    ref = _stock_shared_sf_layout(sf, block_m, num_max)
    got = bench.shared_sf_layout(sf, block_m, num_max)
    assert got.shape == ref.shape and got.stride() == ref.stride()
    assert torch.equal(got, ref)
    # the index helper used for documentation agrees with the tensor path
    if m:
        rows = bench.shared_sf_row_index(m, block_m)
        assert torch.equal(got[rows], sf)


# ---------------------------------------------------------------------------------------------------------------------
# GPU: profiler sample extraction + event timer on a trivial kernel (no deep_gemm needed)
# ---------------------------------------------------------------------------------------------------------------------
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
def test_bench_samples_returns_one_sample_per_iteration(bench):
    x = torch.zeros(1 << 20, device="cuda")
    # discover the full kernel name of the elementwise kernel, then use it as the (full-name) filter
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA]
    ) as prof:
        x.add_(1.0)
        torch.cuda.synchronize()
    names = [
        e.name for e in prof.events() if getattr(e.device_type, "name", "") == "CUDA"
    ]
    assert names, "profiler returned no CUDA kernel events"
    filt = names[0]
    res = bench.bench_samples(
        {"a": lambda: x.add_(1.0)}, {"a": filt}, num_tests=6, barrier=None, flush=False
    )
    assert res["ok"], res["reason"]
    assert len(res["samples"]["a"]) == 6 and all(s > 0 for s in res["samples"]["a"])
    # A/A on the same filter: events are assigned to labels by launch order
    res2 = bench.bench_samples(
        {"a": lambda: x.add_(1.0), "b": lambda: x.add_(2.0)},
        {"a": filt, "b": filt},
        num_tests=4,
        barrier=None,
        flush=False,
    )
    assert res2["ok"], res2["reason"]
    assert len(res2["samples"]["a"]) == 4 and len(res2["samples"]["b"]) == 4
    ev = bench.bench_events(
        {"a": lambda: x.add_(1.0)}, num_tests=5, barrier=None, flush=False
    )
    assert len(ev["a"]) == 5 and all(v > 0 for v in ev["a"])
    # a wrong filter is reported, not silently accepted
    bad = bench.bench_samples(
        {"a": lambda: x.add_(1.0)},
        {"a": "no_such_kernel_name"},
        num_tests=3,
        barrier=None,
        flush=False,
    )
    assert not bad["ok"] and "expected 3 kernel events" in bad["reason"]


def test_stress_m_sequence_is_deterministic_and_rank_independent(bench):
    cyc = [8192, 1, 4096, 2]
    assert [bench.stress_m_at(cyc, False, 0, i) for i in range(6)] == [
        8192,
        1,
        4096,
        2,
        8192,
        1,
    ]
    a = [bench.stress_m_at(cyc, True, 3, i) for i in range(16)]
    assert a == [
        bench.stress_m_at(cyc, True, 3, i) for i in range(16)
    ]  # same permutation every call (and on every rank)
    assert sorted(a[:4]) == sorted(cyc) and sorted(a[4:8]) == sorted(
        cyc
    )  # every round is a permutation of the cycle
    assert a != [
        bench.stress_m_at(cyc, True, 4, i) for i in range(16)
    ]  # a different seed gives a different sequence


def test_hot_topk_idx_patterns(bench):
    idx = bench.hot_topk_idx("hot", 5, 384, 6, rank=3, world=32, device="cpu")
    assert idx.shape == (5, 6) and bool(
        (idx == torch.arange(6)).all()
    )  # every token of every rank -> experts 0..topk-1
    idx = bench.hot_topk_idx(
        "hot_spread", 4, 384, 6, rank=30, world=32, device="cpu"
    )  # 12 experts per rank
    assert idx[0].tolist() == [
        360,
        372,
        0,
        12,
        24,
        36,
    ]  # the first expert of ranks 30, 31, 0, 1, 2, 3
    idx = bench.hot_topk_idx(
        "hot_spread", 2, 384, 6, rank=3, world=4, device="cpu"
    )  # fewer ranks than topk: the wrap offset keeps the experts distinct
    assert (
        len(set(idx[0].tolist())) == 6 and int(idx.min()) >= 0 and int(idx.max()) < 384
    )


def test_parse_stress_cycle_accepts_ints_vectors_and_rand(bench):
    cyc = bench.parse_stress_cycle("8192,1,0/1/300/8192,rand512,9000", 4, 8192, 0)
    toks = [k for k, _ in cyc]
    assert (
        toks[:3] == [8192, 1, "0/1/300/8192"] and "rand512" in toks and 9000 not in toks
    )  # an entry above T_max is dropped
    d = dict(cyc)
    assert (
        d[8192] == [8192] * 4
        and d["0/1/300/8192"] == [0, 1, 300, 8192]
        and len(d["rand512"]) == 4
    )
    seq = [bench.stress_m_at(toks, True, 1, i) for i in range(2 * len(toks))]
    assert sorted(map(str, seq[: len(toks)])) == sorted(
        map(str, toks)
    )  # every round of the shuffled cycle is a permutation of the tokens


# ---------------------------------------------------------------------------------------------------------------------
# 1 Rubin GPU: the whole harness end to end at EP1 (DeepGEMM local buffer), through a subprocess (spawn needs __main__)
# ---------------------------------------------------------------------------------------------------------------------
@needs_rubin
def test_end_to_end_ep1_deepgemm(tmp_path):
    out = tmp_path / "ep1.jsonl"
    env = dict(
        os.environ,
        DG_MEGA_MOE_SM107="1",
        WORLD_SIZE="1",
        RANK="0",
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT="8391",
    )
    cmd = [
        sys.executable,
        str(HARNESS),
        "--mode",
        "bench",
        "--impl",
        "deepgemm",
        "--num-processes",
        "1",
        "--num-shared-experts",
        "1",
        "--num-experts",
        "96",
        "--num-max-tokens-per-rank",
        "1920",
        "--m-list",
        "64,512",
        "--weight-init",
        "fast",
        "--num-tests",
        "8",
        "--out",
        str(out),
    ]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=1800)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    rows = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    pts = [r for r in rows if r["kind"] == "point"]
    assert [r["M_label"] for r in pts] == ["64", "512"]
    for r in pts:
        assert (
            r["valid"]
            and r["dep"] == 1
            and r["n_iters"] == 8
            and len(r["samples_us"]) == 1
            and len(r["samples_us"][0]) == 8
        )
        assert (
            r["stat"]["median_of_max_us"] > 0
            and r["sm107"] == 1
            and r["variant"] == "B0"
        )
        assert r["config"].get("stages") in (
            6,
            7,
            8,
            11,
            12,
            13,
        )  # SM107 MegaMoE line captured
        assert r["y_check"]["finite"]
        # the extraction agrees with DeepGEMM's own bench_kineto table parse (3-decimal us rounding)
        assert abs(r["stock_mean_us"] - r["stat"]["rank_mean_us"][0]) < 0.01


@needs_rubin
def test_stress_ep1_records_progress_and_done(tmp_path):
    out = tmp_path / "stress.jsonl"
    env = dict(
        os.environ,
        DG_MEGA_MOE_SM107="1",
        WORLD_SIZE="1",
        RANK="0",
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT="8393",
    )
    cmd = [
        sys.executable,
        str(HARNESS),
        "--mode",
        "stress",
        "--impl",
        "deepgemm",
        "--num-processes",
        "1",
        "--num-shared-experts",
        "0",
        "--num-experts",
        "96",
        "--num-max-tokens-per-rank",
        "1920",
        "--stress-m-list",
        "64,512,1",
        "--stress-per-m",
        "5",
        "--stress-launches",
        "40",
        "--stress-log-every",
        "2",
        "--stress-barrier",
        "allreduce",
        "--weight-init",
        "fast",
        "--out",
        str(out),
    ]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=1800)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    rows = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    done = [r for r in rows if r["kind"] == "stress_done"]
    assert (
        len(done) == 1
        and done[0]["launches"] == 40
        and done[0]["segments"] == 8
        and done[0]["cycle"] == [64, 512, 1]
    )
    prog = [r for r in rows if r["kind"] == "progress" and r["mode"] == "stress"]
    assert prog and prog[-1]["launch"] == 40


@needs_rubin
def test_graphrace_ep1_runs_graph_and_eager_without_mismatch(tmp_path):
    # EP1 has no peers, so the stale-combine-tag hazard cannot show: this guards the code path (capture, restaging, bitwise comparison, tag trace) only
    out = tmp_path / "graphrace.jsonl"
    env = dict(
        os.environ,
        DG_MEGA_MOE_SM107="1",
        WORLD_SIZE="1",
        RANK="0",
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT="8394",
    )
    cmd = [
        sys.executable,
        str(HARNESS),
        "--mode",
        "graphrace",
        "--impl",
        "deepgemm",
        "--num-processes",
        "1",
        "--num-shared-experts",
        "0",
        "--num-experts",
        "96",
        "--num-max-tokens-per-rank",
        "1920",
        "--graphrace-m-list",
        "64",
        "--graphrace-replays",
        "24",
        "--graphrace-sets",
        "2",
        "--weight-init",
        "fast",
        "--out",
        str(out),
    ]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=1800)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    rows = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    rec = [r for r in rows if r["kind"] == "graphrace"]
    assert sorted(r["graph"] for r in rec) == [False, True]
    for r in rec:
        assert (
            r["n_replays"] == 24
            and r["n_mismatch_launches_sum"] == 0
            and len(r["tag_trace_first8_per_rank"][0]) == 8
        )


@needs_rubin
def test_hostcost_ep1_deepgemm_reports_positive_us_per_launch(tmp_path):
    out = tmp_path / "hostcost.jsonl"
    env = dict(
        os.environ,
        DG_MEGA_MOE_SM107="1",
        WORLD_SIZE="1",
        RANK="0",
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT="8395",
    )
    cmd = [
        sys.executable,
        str(HARNESS),
        "--mode",
        "hostcost",
        "--impl",
        "deepgemm",
        "--num-processes",
        "1",
        "--num-shared-experts",
        "0",
        "--num-experts",
        "96",
        "--num-max-tokens-per-rank",
        "1920",
        "--m-list",
        "64,1024",
        "--hostcost-launches",
        "8",
        "--hostcost-reps",
        "3",
        "--weight-init",
        "fast",
        "--out",
        str(out),
    ]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=1800)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    rows = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    rec = [r for r in rows if r["kind"] == "hostcost"]
    assert sorted(r["M_label"] for r in rec) == ["1024", "64"]
    for r in rec:
        assert (
            r["impl"] == "deepgemm"
            and r["n_launches"] == 8
            and r["reps"] == 3
            and len(r["reps_us_rank0"]) == 3
        )
        assert 0.0 < r["us_per_launch_median"] < 5000.0


@needs_rubin
def test_graphlayers_ep1_deepgemm_records_per_launch_stats(tmp_path):
    out = tmp_path / "graphlayers.jsonl"
    env = dict(
        os.environ,
        DG_MEGA_MOE_SM107="1",
        WORLD_SIZE="1",
        RANK="0",
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT="8396",
    )
    cmd = [
        sys.executable,
        str(HARNESS),
        "--mode",
        "graphlayers",
        "--impl",
        "deepgemm",
        "--num-processes",
        "1",
        "--num-shared-experts",
        "0",
        "--num-experts",
        "96",
        "--num-max-tokens-per-rank",
        "1920",
        "--m-list",
        "64",
        "--graph-layers-list",
        "1,3",
        "--windows",
        "1",
        "--num-tests",
        "6",
        "--weight-init",
        "fast",
        "--out",
        str(out),
    ]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=1800)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    rows = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    rec = {r["layers"]: r for r in rows if r["kind"] == "graphlayers"}
    assert sorted(rec) == [1, 3]
    for L, r in rec.items():
        assert (
            r["valid"]
            and r["n_replays"] == 6
            and len(r["node_stat"]) == L
            and r["bit_equal_vs_first_impl"]
        )
        assert (
            len(r["samples_us"][0]) == 6 * L
            and r["stat"]["median_of_max_us"] > 0
            and r["replay_per_layer_stat"]["median_of_max_us"] > 0
        )
