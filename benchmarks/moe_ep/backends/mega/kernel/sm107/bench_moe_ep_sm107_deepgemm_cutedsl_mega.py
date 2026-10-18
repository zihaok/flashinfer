#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Benchmark protocol derived from DeepGEMM (MIT License, Copyright (c) 2025 DeepSeek; see licenses/LICENSE.deepgemm.txt) tests/test_mega_moe.py and deep_gemm/testing/bench.py;
# launch/PTX helper patterns derived from FlashInfer's vendored NVIDIA CuTe-DSL MegaMoE (BSD-3-Clause).
"""A/B benchmark + correctness harness ("bench_mega_moe_ab") for the SM107 (Rubin) MegaMoE kernels.

Kernels (adapters): ``deepgemm`` (DeepGEMM 3ec0eb4, the oracle/baseline), ``deepgemm2`` (a second, independent DeepGEMM
instance: A/A noise runs and self-tests of the A/B machinery), ``port`` (the CuteDSL port, DeepGEMM-compatible API: ``flashinfer.experimental.sm107_deepgemm_megamoe.compat``), ``fi_existing`` (FlashInfer's existing SM107 CuteDSL
MegaMoE backend; not implemented in this revision).  ``--impl both`` == ``deepgemm,port`` (interleaved).

One process per GPU (``torch.multiprocessing.spawn`` + ``deep_gemm.utils.dist.init_dist``): run one copy per node with
``WORLD_SIZE`` = number of NODES, ``RANK`` = node index, ``MASTER_ADDR``/``MASTER_PORT`` (DeepGEMM convention) and
``--num-processes 4``.  DEP = nodes x 4.  Default shape = DeepSeek-V4.1-Flash (H=5120, I=2304, E=384, top-k 6, clamp 10).

Modes (``--mode``)
  bench    time the kernel(s) over an M list (default).  Protocol = DeepGEMM's ``bench_kineto``: per iteration 256 MB L2
           flush, ``torch.cuda._sleep(2e7)``, cross-rank barrier, ONE kernel launch; 2 x ``--num-tests`` iterations under
           ``torch.profiler`` (wait 0, warmup 1, active 1: only the second loop is recorded); kernel durations are read from
           the profiler events by FULL kernel-name filter and kept PER ITERATION PER RANK.  Primary statistic = median over
           iterations of the max over ranks.  A CUDA-event pass (no profiler) is the cross-check.  ``--exec graph`` replays a
           CUDA graph instead of launching eagerly.
  configs  untimed: one launch per BLOCK_M regime with ``DG_PRINT_CONFIGS=1`` captured at file-descriptor level; the
           ``MegaMoEConfig(...)`` and ``SM107 MegaMoE: stages=, smem=`` lines go to the JSONL (kind="config").
  check    kernel-vs-kernel ``y`` / stats comparison (bit equality, max abs diff, calc_diff) on identical inputs.
  gate0    DeepGEMM-vs-DeepGEMM run-to-run bit-reproducibility with injected rank skew and permuted token order.
  graphlayers L = --graph-layers-list back-to-back launches (same workspace) in ONE CUDA graph, replayed --windows x --num-tests times: per-launch kernel durations (profiler; pooled and per node), replay time / L,
           bitwise y / stats comparison of the impls; rows kind="graphlayers" (multi-layer graphs, L >= 2: with L = 1 every replay is the same graph node, i.e. an identical %gridid).
  hostcost host (CPU) cost of one EAGER launch per impl (wall time of the launch call; back-to-back launches queued behind a GPU sleep); rows kind="hostcost" (added with the L2_256B tensor-map change).

JSONL schema ("mega_moe_ab/1"); one JSON object per line, written by global rank 0 only:
  kind="run"     once per invocation: args, env (torch/DSL/driver/SHAs/clocks), per-rank host + GPU uuid, ClusterUUID.
  kind="point"   one timed point = (impl, shape, DEP, T_max, M vector, S, exec mode, PDL, arch, per-die, pass):
      impl, variant, kernel_filter, shape{hidden,intermediate,num_experts,topk,S,clamp,fast_math}, dep, num_nodes,
      t_max_user, t_max_aligned, M (rank-0 M), m_vec (all ranks), m_uniform, block_m (per rank, via
      get_block_m_for_mega_moe), config{block_m,stages,smem,num_ring_tokens,num_sf_ring_tokens,...,lines}, mode
      (eager|graph), pdl, arch, sm107, perdie, weights_localized, routing, masked_ratio, weight_init, seed, pass,
      n_windows, n_iters, valid, samples_us [rank][iteration] (profiler kernel durations), samples_ev_us (CUDA events),
      stat{median_of_max_us (PRIMARY), mean_of_max_us, p10/p90_of_max_us, min/max/std_of_max_us, window_medians_of_max_us,
      rank_mean_p50/p10/p90_us, rank_median_us[], rank_mean_us[]}, ev_stat{...}, stock_mean_us[rank],
      perf{num_recv_tokens[], num_touched_experts[], tflops_rank[], overlap_tflops_rank[], hbm_gbs_rank[],
      nvlink_gbs_rank[], tflops_mean, t_reduction_us}, y_check{finite, abs_max, sum}, clocks, job_id, ts.
  kind="config"  one launch with DG_PRINT_CONFIGS=1 (see above), per rank-0 plus the block_m of every rank.
  kind="check"   kernel-vs-kernel comparison for one (S, M); kind="gate0" Gate-0 record per (S, masked, M).
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import faulthandler
import hashlib
import json
import math
import os
import platform
import random
import re
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import zlib
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist

SCHEMA = "mega_moe_ab/1"
CORE_M = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 3072, 4096, 8192]
SERVING_M = [1280, 1536, 2304, 2520]  # serving-scale points
DG_SM107_FILTER = "sm107_fp8_fp4_mega_moe_impl"
DG_SM100_FILTER = "sm100_fp8_fp4_mega_moe_impl"
# The DSL names the kernel after its Python class:
# profiler events read "kernel_cutlass_kernel_flashinferexperimentalsm107_deepgemm_megamoekernelkernelDeepGemmSm10...", unique to the port (never the bare "mega_moe")
PORT_FILTER_DEFAULT = (
    "kernel_cutlass_kernel_flashinferexperimentalsm107_deepgemm_megamoe"
)
PORT_MODULE_DEFAULT = "flashinfer.experimental.sm107_deepgemm_megamoe.compat"
L2_FLUSH_BYTES = int(256e6)
SLEEP_CYCLES = int(2e7)  # ~10 ms, as bench_kineto


# ======================================================================================================================
# 1. Pure-python helpers (no GPU, no deep_gemm): statistics, M lists, config-line parsing, DeepGEMM's perf formulas
# ======================================================================================================================
def pct(xs: Sequence[float], q: float) -> float:
    """Linear-interpolation percentile (numpy default), q in [0, 100]."""
    s = sorted(xs)
    if not s:
        return float("nan")
    k = (len(s) - 1) * q / 100.0
    lo, hi = int(math.floor(k)), int(math.ceil(k))
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def summarize_samples(
    samples: Sequence[Sequence[float]], n_windows: int = 1
) -> Dict[str, Any]:
    """samples[rank][iteration] in us.  PRIMARY statistic = median over iterations of the max over ranks."""
    R, N = len(samples), len(samples[0])
    mx = [max(samples[r][i] for r in range(R)) for i in range(N)]
    mn = [min(samples[r][i] for r in range(R)) for i in range(N)]
    mean_rank = [sum(samples[r][i] for r in range(R)) / R for i in range(N)]
    out: Dict[str, Any] = {
        "median_of_max_us": statistics.median(mx),
        "median_of_min_us": statistics.median(
            mn
        ),  # the last-arriving rank: ~ kernel time without launch-skew wait
        "mean_of_max_us": statistics.fmean(mx),
        "p10_of_max_us": pct(mx, 10),
        "p90_of_max_us": pct(mx, 90),
        "min_of_max_us": min(mx),
        "max_of_max_us": max(mx),
        "std_of_max_us": statistics.pstdev(mx) if N > 1 else 0.0,
        "rank_mean_p50_us": pct(mean_rank, 50),
        "rank_mean_p10_us": pct(mean_rank, 10),
        "rank_mean_p90_us": pct(mean_rank, 90),
        "rank_median_us": [statistics.median(samples[r]) for r in range(R)],
        "rank_mean_us": [statistics.fmean(samples[r]) for r in range(R)],
    }
    if n_windows > 1 and N % n_windows == 0:
        w = N // n_windows
        out["window_medians_of_max_us"] = [
            statistics.median(mx[k * w : (k + 1) * w]) for k in range(n_windows)
        ]
    return out


def align(x: int, y: int) -> int:
    return (x + y - 1) // y * y


def perf_model(
    M: int,
    num_recv: int,
    num_touched: int,
    hidden: int,
    inter: int,
    S: int,
    topk: int,
    elapsed_us: float,
    mma_type: str = "fp8xfp4",
) -> Dict[str, float]:
    """DeepGEMM's formulas (tests/test_mega_moe.py:385-418), for ONE rank with its own M / received tokens."""
    shared_i = inter * S
    routed_flops = 2 * num_recv * hidden * inter * 3
    shared_flops = 2 * M * hidden * shared_i * 3
    flops = routed_flops + shared_flops
    act = 2 if mma_type == "bf16xbf16" else 1
    wrouted = 2 if mma_type == "bf16xbf16" else (1 if mma_type == "fp8xfp8" else 0.5)
    wshared = 2 if mma_type == "bf16xbf16" else 1
    routed_bytes = (
        num_touched * inter * 2 * hidden * wrouted
        + num_touched * hidden * inter * wrouted
        + num_recv * hidden * act
        + num_recv * inter * act
        + num_recv * inter * act
        + num_recv * hidden * 2
    )
    shared_bytes = (
        0
        if S == 0
        else (
            shared_i * 2 * hidden * wshared
            + hidden * shared_i * wshared
            + M * hidden * act
            + M * shared_i * act
            + M * shared_i * act
            + M * hidden * 2
        )
    )
    nvl_bytes = num_recv * hidden * 3
    t_red = (
        M * hidden * 2 * (1 + topk) / 6.5e12
    )  # seconds, DeepGEMM's serial-combine approximation
    el = elapsed_us * 1e-6

    def sdiv(a, b):
        return float("nan") if b == 0 else a / b

    tflops = sdiv(flops / 1e12, el)
    approx = sdiv(el, el - t_red)
    return {
        "tflops": tflops,
        "overlap_tflops": tflops * approx,
        "hbm_gbs": sdiv((routed_bytes + shared_bytes) / 1e9, el) * approx,
        "nvlink_gbs": sdiv(nvl_bytes / 1e9, el) * approx,
        "t_reduction_us": t_red * 1e6,
        "flops": flops,
    }


def regime_segments(
    block_m_fn: Callable[[int], int], m_max: int
) -> List[Tuple[int, int, int]]:
    """Maximal M ranges with constant BLOCK_M: [(first_M, last_M, block_m), ...] for M in [1, m_max]."""
    segs: List[Tuple[int, int, int]] = []
    prev, start = None, 1
    for m in range(1, m_max + 1):
        b = block_m_fn(m)
        if b != prev:
            if prev is not None:
                segs.append((start, m - 1, prev))
            start, prev = m, b
    segs.append((start, m_max, prev))
    return segs


def boundary_points(segs: Sequence[Tuple[int, int, int]]) -> List[int]:
    """Regime-boundary pairs (t-1 | t)"""
    pts: List[int] = []
    for s, _e, _b in segs[1:]:
        pts += [s - 1, s]
    return pts


def regime_representatives(segs: Sequence[Tuple[int, int, int]]) -> List[int]:
    """One M per distinct BLOCK_M: the middle of its widest segment."""
    best: Dict[int, Tuple[int, int]] = {}
    for s, e, b in segs:
        if b not in best or (e - s) > (best[b][1] - best[b][0]):
            best[b] = (s, e)
    return sorted((s + e) // 2 for (s, e) in best.values())


@dataclass(frozen=True)
class Point:
    """One M specification: uniform M, an explicit per-rank vector, or DeepGEMM-style random removal."""

    label: str
    m: Optional[int] = None
    vec: Optional[Tuple[int, ...]] = None
    rand_removed: Optional[int] = None

    def m_vec(self, world: int, t_user: int, seed: int) -> List[int]:
        if self.m is not None:
            return [self.m] * world
        if self.vec is not None:
            return [self.vec[r % len(self.vec)] for r in range(world)]
        # tests/test_mega_moe.py:86 : num_tokens = max(0, T_max - random.randint(0, removed)) with random.seed(rank)
        return [
            max(
                0,
                t_user
                - random.Random(seed * 1000 + r).randint(0, self.rand_removed or 0),
            )
            for r in range(world)
        ]


def hot_topk_idx(
    kind: str,
    t: int,
    num_experts: int,
    topk: int,
    rank: int,
    world: int,
    device: Any = "cuda",
) -> "torch.Tensor":
    """Adversarial routing for the hot-expert coverage runs.  ``hot``: every token of every rank goes to the SAME experts 0..topk-1 (one rank receives
    world x M tokens for each of them).  ``hot_spread``: one hot expert per rank: the tokens of rank s go to the first expert of the ranks s, s+1, ... (mod world)
    (offset by the wrap count when topk > world so the experts of one token stay distinct), i.e. every rank is the hot destination of some source rank."""
    j = torch.arange(topk, device=device)
    if kind == "hot":
        ids = j
    elif kind == "hot_spread":
        epr = num_experts // world
        ids = ((rank + j) % world) * epr + (j // world)
    else:
        raise ValueError(kind)
    return ids.unsqueeze(0).expand(t, topk).contiguous().to(torch.int64)


def parse_stress_cycle(
    spec: str, world: int, t_user: int, seed: int
) -> List[Tuple[Any, List[int]]]:
    """``--stress-m-list`` -> [(token, per-rank M vector)]: a plain integer = the same M on every rank (token stays an int, as before); ``a/b/c`` = per-rank vector (tiled over the ranks as in --m-list);
    ``randR`` = DeepGEMM-style random removal.  Entries whose largest per-rank M exceeds T_max are dropped."""
    out: List[Tuple[Any, List[int]]] = []
    for tok in [t.strip() for t in spec.split(",") if t.strip()]:
        if re.fullmatch(r"-?\d+", tok):
            pt, key = Point(tok, m=int(tok)), int(tok)
        elif "/" in tok:
            pt, key = Point(tok, vec=tuple(int(v) for v in tok.split("/"))), tok
        elif tok.startswith("rand"):
            pt, key = Point(tok, rand_removed=int(tok[4:])), tok
        else:
            raise ValueError(f"bad --stress-m-list entry {tok!r}")
        mv = pt.m_vec(world, t_user, seed)
        if max(mv) <= t_user:
            out.append((key, mv))
    return out


def parse_m_tokens(
    spec: str, preset_fn: Callable[[str], List[int]], t_user: int
) -> List[Point]:
    """'core,serving,1000,64/1024/1024/64,rand512' -> [Point]; presets resolved by ``preset_fn`` (needs block_m for some)."""
    pts: List[Point] = []
    seen = set()
    for tok in [t.strip() for t in spec.split(",") if t.strip()]:
        if tok in (
            "core",
            "serving",
            "boundary",
            "regimes",
            "perf",
            "regmax",
            "regmaxall",
        ):
            for m in preset_fn(tok):
                if m <= t_user and ("u", m) not in seen:
                    seen.add(("u", m))
                    pts.append(Point(str(m), m=m))
        elif tok.startswith("rand"):
            pts.append(Point(tok, rand_removed=int(tok[4:])))
        elif "/" in tok:
            pts.append(Point(tok, vec=tuple(int(v) for v in tok.split("/"))))
        else:
            m = int(tok)
            if m > t_user:
                raise ValueError(f"M={m} exceeds --num-max-tokens-per-rank {t_user}")
            if ("u", m) not in seen:
                seen.add(("u", m))
                pts.append(Point(tok, m=m))
    return pts


_KV = re.compile(r"([A-Za-z_0-9]+)=([-0-9.]+)")


def parse_config_text(text: str) -> Dict[str, Any]:
    """Parse DG_PRINT_CONFIGS output (HEUR:255-265 MegaMoEConfig line, IMPL:142 'SM107 MegaMoE:' line)."""
    cfg: Dict[str, Any] = {
        "lines": [
            ln
            for ln in text.splitlines()
            if ln.startswith(("MegaMoEConfig(", "SM107 MegaMoE:"))
        ]
    }
    for ln in cfg["lines"]:
        if ln.startswith("SM107 MegaMoE:"):
            kv = dict((k, int(v)) for k, v in _KV.findall(ln))
            cfg["stages"], cfg["smem"] = kv.get("stages"), kv.get("smem")
        elif ln.startswith("MegaMoEConfig("):
            head, _, tail = ln.partition("): MegaMoEConfig(")
            for k, v in _KV.findall(tail):
                cfg["cfg_" + k] = int(v) if re.fullmatch(r"-?\d+", v) else float(v)
            for k, v in _KV.findall(head):
                cfg["key_" + k] = int(v)
    if "cfg_block_m" in cfg:
        cfg["block_m"] = cfg["cfg_block_m"]
        cfg["num_ring_tokens"] = cfg.get("cfg_num_ring_tokens")
        cfg["num_sf_ring_tokens"] = cfg.get("cfg_num_sf_ring_tokens")
    return cfg


def shared_sf_row_index(m: int, block_m: int) -> List[int]:
    """Destination row of token t in DeepGEMM's shared-expert SF layout (tests/test_mega_moe.py:36-50), vectorisable form."""
    aligned = align(block_m, 128)
    out = []
    for t in range(m):
        blk, mi = divmod(t, block_m)
        out.append(blk * aligned + (mi // 128) * 128 + (mi % 32) * 4 + (mi % 128) // 32)
    return out


# ======================================================================================================================
# 2. Torch-level helpers: fd-level stdout capture, watchdog, L2 flush / barrier, profiler + CUDA-event timers, graph
# ======================================================================================================================
@contextlib.contextmanager
def capture_fd_stdout():
    """Capture everything written to file descriptor 1 (the C++ ``std::cout`` prints of DeepGEMM included)."""
    sys.stdout.flush()
    with contextlib.suppress(Exception):  # best effort, C stdio flush is optional
        ctypes.CDLL(None).fflush(None)
    tmp = tempfile.TemporaryFile(mode="w+b")  # noqa: SIM115 - closed explicitly in the finally block
    saved = os.dup(1)
    os.dup2(tmp.fileno(), 1)
    box: List[str] = []
    try:
        yield box
    finally:
        sys.stdout.flush()
        with contextlib.suppress(Exception):
            ctypes.CDLL(None).fflush(None)
        os.dup2(saved, 1)
        os.close(saved)
        tmp.seek(0)
        box.append(tmp.read().decode(errors="replace"))
        tmp.close()


class Watchdog:
    """Python-thread watchdog for hung persistent kernels (a lost rank leaves the others spinning in NVLink barriers).

    On timeout: dump all Python stacks (stderr), call ``dump_fn`` (the harness dumps the head of every symmetric buffer through a
    fresh non-blocking CUDA stream, i.e. WITHOUT waiting for the stuck kernel), then ``os._exit(1)``.  The main thread blocks in
    ``torch.cuda.synchronize()`` (GIL released), so this thread keeps running."""

    def __init__(
        self,
        default_s: float,
        dump_fn: Optional[Callable[[], None]] = None,
        post_wait_s: float = 0.0,
    ):
        self.default_s = default_s
        self.dump_fn = dump_fn
        self.post_wait_s = post_wait_s  # after the dump keep the hung process alive this long (lets DeepGEMM's own 300 s device timeout print its location)
        self._deadline: Optional[float] = None
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None

    def _loop(self) -> None:
        while True:
            time.sleep(0.5)
            with self._lock:
                dl = self._deadline
            if dl is not None and time.monotonic() > dl:
                self._fire()

    def _fire(self) -> None:
        sys.stderr.write(
            f"\nWATCHDOG: no progress for the armed timeout (default {self.default_s}s); dumping stacks and state\n"
        )
        faulthandler.dump_traceback(all_threads=True)
        try:
            if self.dump_fn is not None:
                self.dump_fn()
        except Exception:  # noqa: BLE001 - the state dump is best effort, the exit below must happen
            traceback.print_exc()
        sys.stderr.flush()
        if self.post_wait_s > 0:
            sys.stderr.write(
                f"WATCHDOG: state dumped; waiting {self.post_wait_s}s for the in-kernel timeout message before exiting\n"
            )
            sys.stderr.flush()
            time.sleep(self.post_wait_s)
        os._exit(1)

    def arm(self, seconds: Optional[float] = None) -> None:
        with self._lock:
            self._deadline = time.monotonic() + (seconds or self.default_s)
        if self._thread is None:
            self._thread = threading.Thread(
                target=self._loop, daemon=True, name="harn-watchdog"
            )
            self._thread.start()

    def disarm(self) -> None:
        with self._lock:
            self._deadline = None


def _kernel_events(
    prof, name_substrings: Sequence[str]
) -> List[Tuple[float, float, str]]:
    """(start_us, duration_us, name) of CUDA kernel events whose name contains one of the substrings, time-sorted."""
    out: List[Tuple[float, float, str]] = []
    for e in prof.events():
        dt = getattr(e, "device_type", None)
        if getattr(dt, "name", str(dt)) != "CUDA":
            continue
        nm = e.name
        if any(s in nm for s in name_substrings):
            out.append((float(e.time_range.start), float(e.device_time), nm))
    out.sort()
    return out


def _kernel_events_from_trace(
    prof, name_substrings: Sequence[str]
) -> List[Tuple[float, float, str]]:
    """Fallback: parse the chrome trace (kernel category events)."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "trace.json")
        prof.export_chrome_trace(path)
        with open(path) as f:
            doc = json.load(f)
    out = [
        (float(ev["ts"]), float(ev["dur"]), ev["name"])
        for ev in doc.get("traceEvents", [])
        if ev.get("cat") == "kernel"
        and any(s in ev.get("name", "") for s in name_substrings)
    ]
    out.sort()
    return out


def _stock_mean_us(prof, name: str) -> Optional[float]:
    """DeepGEMM's own table parse (bench_kineto), for a cross-check of the sample extraction (needs one unique row)."""
    lines = (
        prof.key_averages()
        .table(sort_by="cuda_time_total", max_name_column_width=100)
        .split("\n")
    )
    hits = [ln for ln in lines if name in ln]
    if len(hits) != 1:
        return None
    units = {
        "ms": 1e3,
        "us": 1e6,
    }  # same unit handling as bench_kineto: seconds = value / scale
    ts = hits[0].split()[-2]
    for u, scale in units.items():
        if u in ts:
            return float(ts.replace(u, "")) / scale * 1e6  # mean in us
    return None


def bench_samples(
    calls: "Dict[str, Callable[[], Any]]",
    filters: Dict[str, str],
    num_tests: int = 30,
    barrier: Optional[Callable[[], Any]] = None,
    flush: bool = True,
    trace_path: Optional[str] = None,
    stock_check: bool = True,
    layers: int = 1,
) -> Dict[str, Any]:
    """DeepGEMM's ``bench_kineto`` protocol, but returning PER-ITERATION samples (us) for several kernels.

    calls   : label -> callable launching exactly one kernel of that label (inputs staged / y allocated beforehand).
    filters : label -> FULL kernel-name substring.  Labels may share a filter (A/A runs): events are then assigned to
              labels by launch order (single stream => strict order).  Every launch must give exactly one event.
    Per iteration (labels in alternating order for A/B): L2 flush -> sleep -> barrier -> launch.  2 loops of ``num_tests``
    under ``torch.profiler`` (wait 0, warmup 1, active 1 => only the 2nd loop is recorded).
    ``layers`` > 1: every call is a CUDA-graph replay of ``layers`` back-to-back kernel nodes (the same workspace); each replay yields ``layers`` events, assigned to the label in node order:
    ``samples`` then holds ALL per-launch durations (iteration-major) and ``node[label][j]`` the durations of node j.
    Returns {"samples": {label: [us]}, "node": {label: [[us] per node]}, "ok": bool, "reason": str, "stock_mean_us": {label: float|None}, "names": [...]}.
    """
    labels = list(calls)
    flush_buf = torch.empty(L2_FLUSH_BYTES // 4, dtype=torch.int, device="cuda")
    for lab in labels:  # stock: "fn() once, for auto-tuning kernels with prints"
        calls[lab]()
    torch.cuda.synchronize()
    sched = torch.profiler.schedule(wait=0, warmup=1, active=1, repeat=1)
    prof = torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
        schedule=sched,
        acc_events=True,
    )
    order: List[str] = []
    with prof:
        for loop in range(2):
            for it in range(num_tests):
                seq = labels if it % 2 == 0 else labels[::-1]
                for lab in seq:
                    if flush:
                        flush_buf.zero_()
                    if barrier is not None:
                        torch.cuda._sleep(
                            SLEEP_CYCLES
                        )  # large kernel + barrier: removes unbalanced CPU launch overhead
                        barrier()
                    calls[lab]()
                    if loop == 1:
                        order.append(lab)
            torch.cuda.synchronize()
            prof.step()
    subs = sorted(set(filters.values()))
    events = _kernel_events(prof, subs)
    if not events:
        events = _kernel_events_from_trace(prof, subs)
    res: Dict[str, Any] = {
        "samples": {lab: [] for lab in labels},
        "node": {lab: [[] for _ in range(layers)] for lab in labels},
        "ok": True,
        "reason": "",
        "stock_mean_us": {},
        "names": sorted({e[2] for e in events})[:4],
    }
    if len(events) != len(order) * layers:
        seen = {}
        for e in prof.events():
            if getattr(getattr(e, "device_type", None), "name", "") == "CUDA":
                seen[e.name[:90]] = seen.get(e.name[:90], 0) + 1
        top = sorted(seen.items(), key=lambda kv: -kv[1])[:6]
        res.update(
            ok=False,
            reason=f"expected {len(order) * layers} kernel events for filters {subs}, got {len(events)}; "
            f"CUDA kernels in window: {top}",
        )
        return res
    for idx, lab in enumerate(order):
        for j in range(layers):
            _s, dur, nm = events[idx * layers + j]
            if filters[lab] not in nm:
                res.update(
                    ok=False,
                    reason=f"event order mismatch: label {lab} filter {filters[lab]} vs event {nm[:100]}",
                )
                return res
            res["samples"][lab].append(dur)
            res["node"][lab][j].append(dur)
    if stock_check and len(labels) == 1:
        try:
            res["stock_mean_us"][labels[0]] = _stock_mean_us(prof, filters[labels[0]])
        except Exception:  # noqa: BLE001 - diagnostic only
            res["stock_mean_us"][labels[0]] = None
    if trace_path:
        prof.export_chrome_trace(trace_path)
    del prof
    return res


def bench_events(
    calls: "Dict[str, Callable[[], Any]]",
    num_tests: int = 30,
    barrier: Optional[Callable[[], Any]] = None,
    flush: bool = True,
) -> Dict[str, List[float]]:
    """Cross-check timer WITHOUT the profiler: CUDA events around each launch (same flush/sleep/barrier)."""
    labels = list(calls)
    flush_buf = torch.empty(L2_FLUSH_BYTES // 4, dtype=torch.int, device="cuda")
    for lab in labels:
        calls[lab]()
    torch.cuda.synchronize()
    pairs: List[Tuple[str, Any, Any]] = []
    for it in range(num_tests):
        seq = labels if it % 2 == 0 else labels[::-1]
        for lab in seq:
            if flush:
                flush_buf.zero_()
            if barrier is not None:
                torch.cuda._sleep(SLEEP_CYCLES)
                barrier()
            s, e = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            s.record()
            calls[lab]()
            e.record()
            pairs.append((lab, s, e))
    torch.cuda.synchronize()
    out: Dict[str, List[float]] = {lab: [] for lab in labels}
    for lab, s, e in pairs:
        out[lab].append(s.elapsed_time(e) * 1e3)
    return out


def capture_graph(fn: Callable[[], Any]):
    """Capture one launch of ``fn`` into a CUDA graph (fn must already have run once: JIT, lazy init)."""
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    torch.cuda.synchronize()
    return g


def gather_rows(group, local: torch.Tensor) -> torch.Tensor:
    """all_gather equal-length 1-D tensors -> [world, n]."""
    outs = [torch.empty_like(local) for _ in range(dist.get_world_size(group))]
    dist.all_gather(outs, local, group=group)
    return torch.stack(outs)


# ======================================================================================================================
# 3. Problem: per-rank inputs and weights, built ONCE per (shape, T_max) and sliced per M
# ======================================================================================================================
def _dg():
    import deep_gemm  # noqa: PLC0415 - lazy: the smoke test imports this module without deep_gemm

    return deep_gemm


def _cast_fp8_for_mega_moe(x: torch.Tensor):
    """tests/test_mega_moe.py:53-59 (activations / shared weights): fp8 + packed UE8M0 SF (+ TMA-aligned SF copy)."""
    dg = _dg()
    from deep_gemm.utils import per_token_cast_to_fp8  # noqa: PLC0415

    x_fp8, x_sf = per_token_cast_to_fp8(
        x, use_ue8m0=True, gran_k=32, use_packed_ue8m0=True
    )
    mn, packed_sf_k = x_sf.shape
    x_sf_tma = torch.empty_strided(
        (mn, packed_sf_k), (1, align(mn, 4)), dtype=x_sf.dtype, device=x_sf.device
    )
    x_sf_tma.copy_(x_sf)
    del dg
    return x_fp8, x_sf, x_sf_tma


class Problem:
    """Per-rank problem data.  Same seeds for every kernel; one Problem is shared by all adapters of a process."""

    def __init__(self, a: argparse.Namespace, rank: int, world: int, t_user: int):
        self.a, self.rank, self.world, self.t_user = a, rank, world, t_user
        self.epr = a.num_experts // world
        self.H, self.I, self.E, self.K = (
            a.hidden,
            a.intermediate_hidden,
            a.num_experts,
            a.num_topk,
        )
        base = a.seed * 1_000_003 + 12345
        self.gen = torch.Generator(device="cuda")
        self.gen.manual_seed(base + 7919 * rank)
        self.wgen = torch.Generator(device="cuda")
        self.wgen.manual_seed(base + 104729 * rank + 1)
        self.popularity_seed = (
            base + 99
        )  # rank-independent: heavy experts coincide across ranks
        from deep_gemm.utils import per_token_cast_to_fp8  # noqa: PLC0415

        x = torch.randn(
            (t_user, self.H), dtype=torch.bfloat16, device="cuda", generator=self.gen
        )
        self.x_fp8, self.x_sf = per_token_cast_to_fp8(
            x, use_ue8m0=True, gran_k=32, use_packed_ue8m0=True
        )
        del x
        self.topk_w, self.topk_idx = self._routing(a.routing, a.masked_ratio)
        self.stats_init = torch.randint(
            0, 100, (self.epr,), dtype=torch.int, device="cuda", generator=self.wgen
        )
        self.l1 = self._routed_weights(self.I * 2, self.H)
        self.l2 = self._routed_weights(self.H, self.I)
        self.sl1 = self.sl2 = None  # shared-expert weights are created lazily (S >= 1)

    # -- routing -----------------------------------------------------------------------------------------------------
    def _routing(self, kind: str, masked_ratio: float):
        T, E, K = self.t_user, self.E, self.K
        if (
            kind == "gaussian"
        ):  # DeepGEMM test: topk of randn scores, scores as routing weights
            scores = torch.randn(
                (T, E), dtype=torch.float, device="cuda", generator=self.gen
            )
            w, idx = torch.topk(scores, K, dim=-1, largest=True, sorted=False)
        elif (
            kind == "power_law"
        ):  # Zipf-popularity routing (FlashInfer bench), Gumbel top-k, shared popularity
            g2 = torch.Generator(device="cuda")
            g2.manual_seed(self.popularity_seed)
            pop = torch.randperm(E, device="cuda", generator=g2)
            freq = 1.0 / (
                torch.arange(1, E + 1, device="cuda", dtype=torch.float64)
                ** self.a.routing_alpha
            )
            probs = torch.empty(E, dtype=torch.float64, device="cuda")
            probs[pop] = freq / freq.sum()
            u = torch.rand(
                (T, E), device="cuda", dtype=torch.float64, generator=self.gen
            )
            gumbel = -torch.log(-torch.log(u.clamp_min(1e-30)) + 1e-30)
            idx = (
                (torch.log(probs.clamp_min(1e-30)).unsqueeze(0) + gumbel)
                .topk(K, dim=-1)
                .indices
            )
            w = torch.randn(
                (T, E), dtype=torch.float, device="cuda", generator=self.gen
            ).gather(1, idx)
        elif kind in (
            "hot",
            "hot_spread",
        ):  # adversarial expert imbalance (see hot_topk_idx)
            idx = hot_topk_idx(kind, T, E, K, self.rank, self.world, "cuda")
            w = torch.randn(
                (T, K), dtype=torch.float, device="cuda", generator=self.gen
            )
        else:
            raise ValueError(f"unknown routing {kind}")
        if masked_ratio > 0:  # tests/test_mega_moe.py:144-147
            rm = torch.rand(
                idx.shape, dtype=torch.float, device="cuda", generator=self.gen
            )
            idx = idx.masked_fill(rm < masked_ratio, -1)
            w = w.masked_fill(idx < 0, 0)
        masked_ranks = {
            int(x)
            for x in str(getattr(self.a, "masked_ranks", "") or "").split(",")
            if x.strip()
        }
        if (
            self.rank in masked_ranks
        ):  # an ALL-MASKED rank (every route -1: the rank dispatches nothing but still takes part in every barrier and receives tokens)
            idx = torch.full_like(idx, -1)
            w = torch.zeros_like(w)
        return w, idx

    def with_masked_ratio(self, ratio: float) -> "Problem":
        """Shallow copy with fresh routing (same activations / weights)."""
        import copy

        p = copy.copy(self)
        p.topk_w, p.topk_idx = self._routing(self.a.routing, ratio)
        return p

    # -- weights -----------------------------------------------------------------------------------------------------
    def _fp4_pools(self):
        """Empirical FP4-byte / UE8M0-SF pools from a real randn cast, for the 'fast' weight init."""
        from deep_gemm.utils import per_token_cast_to_fp4  # noqa: PLC0415

        bf = torch.randn(
            (512, 5120), dtype=torch.bfloat16, device="cuda", generator=self.wgen
        )
        w, sf = per_token_cast_to_fp4(bf, use_ue8m0=True, gran_k=32)
        return w.flatten(), sf.flatten()

    def _routed_weights(self, n: int, k: int):
        """(w int8 [g, n, k/2], sf int32 in the kernel's required layout) for the local experts."""
        dg = _dg()
        from deep_gemm.utils import per_token_cast_to_fp4  # noqa: PLC0415

        g = self.epr
        w = torch.empty((g, n, k // 2), device="cuda", dtype=torch.int8)
        sf = torch.empty((g, n, k // 32), device="cuda", dtype=torch.float)
        if (
            self.a.weight_init == "cast"
        ):  # same procedure as the stock test (randn bf16 -> per-token FP4 cast)
            for i in range(g):
                bf = torch.randn(
                    (n, k), dtype=torch.bfloat16, device="cuda", generator=self.wgen
                )
                w[i], sf[i] = per_token_cast_to_fp4(bf, use_ue8m0=True, gran_k=32)
        else:  # "fast": bytes / SFs sampled from the empirical distribution of a real cast
            pw, psf = self._fp4_pools()
            for i in range(g):
                w[i] = pw[
                    torch.randint(
                        0, pw.numel(), (n, k // 2), device="cuda", generator=self.wgen
                    )
                ]
                sf[i] = psf[
                    torch.randint(
                        0, psf.numel(), (n, k // 32), device="cuda", generator=self.wgen
                    )
                ]
        sf = dg.transform_sf_into_required_layout(sf, n, k, (1, 32), g)
        return w, sf

    def ensure_shared(self, S: int) -> None:
        if S > 0 and self.sl1 is None:
            si = self.I * S
            l1 = torch.randn(
                (si * 2, self.H),
                dtype=torch.bfloat16,
                device="cuda",
                generator=self.wgen,
            )
            l2 = torch.randn(
                (self.H, si), dtype=torch.bfloat16, device="cuda", generator=self.wgen
            )
            self.sl1 = _cast_fp8_for_mega_moe(l1)[0::2]
            self.sl2 = _cast_fp8_for_mega_moe(l2)[0::2]

    def slice(self, m: int) -> "Inputs":
        return Inputs(self.x_fp8[:m], self.x_sf[:m], self.topk_idx[:m], self.topk_w[:m])


@dataclass
class Inputs:
    x_fp8: torch.Tensor
    x_sf: torch.Tensor
    topk_idx: torch.Tensor
    topk_w: torch.Tensor

    @property
    def m(self) -> int:
        return int(self.x_fp8.shape[0])

    def permuted(self, perm: torch.Tensor) -> "Inputs":
        return Inputs(
            self.x_fp8[perm], self.x_sf[perm], self.topk_idx[perm], self.topk_w[perm]
        )


def shared_sf_layout(
    x_sf: torch.Tensor, block_m: int, num_max_sf_tokens: int
) -> torch.Tensor:
    """Vectorised ``_to_shared_mega_moe_sf_layout`` of tests/test_mega_moe.py:36-50 (bit-identical result)."""
    m, k4 = x_sf.shape
    out = torch.empty_strided(
        (num_max_sf_tokens, k4),
        (1, num_max_sf_tokens),
        dtype=x_sf.dtype,
        device=x_sf.device,
    )
    out.zero_()
    if m > 0:
        t = torch.arange(m, device=x_sf.device)
        blk, mi = t // block_m, t % block_m
        dst = (
            blk * align(block_m, 128)
            + (mi // 128) * 128
            + (mi % 32) * 4
            + (mi % 128) // 32
        )
        out.index_copy_(0, dst, x_sf)
    return out


# ======================================================================================================================
# 4. Kernel adapters (DeepGEMM-compatible API surface: deepgemm, deepgemm2, port)
# ======================================================================================================================
class ModuleAdapter:
    """Adapter for any module exposing the DeepGEMM mega-MoE API."""

    def __init__(
        self, label: str, mod_name: str, kernel_filter: str, a: argparse.Namespace
    ):
        self.label, self.mod_name, self.kernel_filter, self.a = (
            label,
            mod_name,
            kernel_filter,
            a,
        )
        self.mod = None
        self.buffer = None
        self.S = 0
        self.l1 = self.l2 = self.sl1 = self.sl2 = None
        self.weights_localized = False

    # lifecycle ------------------------------------------------------------------------------------------------------
    def import_module(self):
        import importlib

        self.mod = importlib.import_module(self.mod_name)
        if self.label == "port" and (
            getattr(self.a, "port_tag_mode", "")
            or getattr(self.a, "port_regs", "")
            or getattr(self.a, "port_faithful_regs", False)
        ):
            # knobs of the port's JIT layer that the DeepGEMM-compatible API does not expose: install a kernel factory (frontend.set_kernel_factory)
            from flashinfer.experimental.sm107_deepgemm_megamoe import frontend, jit  # noqa: PLC0415

            kw: Dict[str, Any] = {}
            if self.a.port_tag_mode:
                kw["tag_mode"] = self.a.port_tag_mode
            if self.a.port_regs:
                kw["regs_override"] = tuple(int(v) for v in self.a.port_regs.split(","))
            if getattr(self.a, "port_faithful_regs", False):
                assert not self.a.port_regs, (
                    "--port-faithful-regs and --port-regs are mutually exclusive"
                )
                kw["faithful_regs"] = (
                    True  # DeepGEMM's own setmaxnreg split (96/88/160 at E/R > 64, else 48/40/208); needs a package with KernelBuild.faithful_regs
                )
            frontend.set_kernel_factory(
                lambda cfg: jit.get_or_compile_kernel(cfg, **kw)
            )
        return self.mod

    def build(self, group, prob: Problem, S: int) -> None:
        a = self.a
        mod = self.mod or self.import_module()
        self.S, self.R = S, dist.get_world_size(group)
        prob.ensure_shared(S)
        self.buffer = mod.get_symm_buffer_for_mega_moe(
            group,
            a.num_experts,
            a.num_max_tokens_per_rank,
            a.num_topk,
            a.hidden,
            a.intermediate_hidden,
            num_shared_experts=S,
            mma_type=a.mma_type,
            activation=a.activation,
        )
        self.l1, self.l2 = mod.transform_weights_for_mega_moe(
            prob.l1, prob.l2, activation=a.activation
        )
        self.sl1 = self.sl2 = None
        if S > 0:
            self.sl1, self.sl2 = mod.transform_weights_for_mega_moe(
                prob.sl1, prob.sl2, activation=a.activation
            )
        self.weights_localized = False
        if self.label.startswith("deepgemm") and a.perdie != "0":
            # B1: the SM107 native localizer; NEVER deep_gemm.localize / --use-localization (MLOPart).
            from deep_gemm.mega import sm107_locality as L  # noqa: PLC0415

            L.get_balanced_sm_locality_domains()  # once, outside graph capture
            sl1, sl2 = (self.sl1, self.sl2) if a.perdie == "all" else (None, None)
            l1, l2, ls1, ls2 = L.localize_mega_moe_weights(self.l1, self.l2, sl1, sl2)
            self.l1, self.l2 = l1, l2
            if a.perdie == "all" and S > 0:
                self.sl1, self.sl2 = ls1, ls2
            try:
                self.weights_localized = bool(
                    torch.ops.deep_gemm.sm107_is_localized(self.l1[0])
                )
            except Exception:  # noqa: BLE001 - recorded as False
                self.weights_localized = False
        elif self.label == "port" and a.perdie != "0":
            # B1 for the port: the port's own locality helpers localize the routed weights exactly like DeepGEMM's; the package's frontend
            # then selects the per-die kernel variant automatically from the die-local weights (no env var).  Shared 3D maps ("all") are not implemented in the port.
            if a.perdie == "all":
                raise NotImplementedError(
                    "--perdie all (die-local shared weights) is not implemented in the port; use --perdie routed"
                )
            from flashinfer.experimental.sm107_deepgemm_megamoe import locality as PL  # noqa: PLC0415

            PL.get_balanced_sm_locality_domains()  # once, outside graph capture
            l1, l2, _, _ = PL.localize_mega_moe_weights(self.l1, self.l2, None, None)
            self.l1, self.l2 = l1, l2
            self.weights_localized = bool(PL.weights_localized(self.l1))

    def close(self) -> None:
        if self.buffer is not None:
            self.buffer.destroy()
        self.buffer = self.l1 = self.l2 = self.sl1 = self.sl2 = None

    # per-M ----------------------------------------------------------------------------------------------------------
    @property
    def t_aligned(self) -> int:
        return int(self.buffer.num_max_tokens_per_rank)

    def block_m(self, m: int) -> int:
        a = self.a
        return int(
            self.mod.get_block_m_for_mega_moe(
                self.R, a.num_experts, self.t_aligned, m, a.num_topk, a.mma_type
            )
        )

    def stage(self, inp: Inputs) -> None:
        """Copy inputs into the symmetric buffer (NOT timed). Shared-expert SF layout is vectorised."""
        buf, m = self.buffer, inp.m
        if m > 0:
            buf.x[:m].copy_(inp.x_fp8)
            buf.x_sf[:m].copy_(inp.x_sf)
            buf.topk_idx[:m].copy_(inp.topk_idx)
            buf.topk_weights[:m].copy_(inp.topk_w)
        if self.S > 0:
            buf.shared_l1_acts_sf.copy_(
                shared_sf_layout(
                    inp.x_sf, self.block_m(m), buf.shared_l1_acts_sf.shape[0]
                )
            )

    def launch_fn(
        self, y: torch.Tensor, stats: Optional[torch.Tensor]
    ) -> Callable[[], Any]:
        a = self.a
        kw = dict(
            y=y,
            l1_weights=self.l1,
            l2_weights=self.l2,
            sym_buffer=self.buffer,
            cumulative_local_expert_recv_stats=stats,
            activation=a.activation,
            activation_clamp=a.activation_clamp,
            fast_math=bool(a.fast_math),
            activation_alpha=a.activation_alpha,
            activation_beta=a.activation_beta,
        )
        if self.S > 0:
            kw.update(shared_l1_weights=self.sl1, shared_l2_weights=self.sl2)
        fn = self.mod.fp8_fp4_mega_moe
        return lambda: fn(**kw)

    def effective_filter(self) -> str:
        if (
            self.label.startswith("deepgemm")
            and os.environ.get("DG_MEGA_MOE_SM107", "1") == "0"
        ):
            return DG_SM100_FILTER  # B2: the SM100 kernel on Rubin
        return self.kernel_filter


def variant_label(a: argparse.Namespace, S: int) -> str:
    """Variant id: B0 (R0, S=1), B0n (S=0), B1/B1n (per-die), B2/B2n (SM100 kernel on Rubin); explicit --variant wins."""
    if a.variant != "auto":
        return a.variant + (
            "n" if S == 0 else ""
        )  # explicit label X -> X (S=1) / Xn (S=0)
    # the variant id names the CONFIGURATION (R0/R1/SM100, S); the kernel is in the row's `impl` field, so that rows of different
    # kernels with the same configuration pair up in the A/B ratio tables
    base = (
        "B2"
        if os.environ.get("DG_MEGA_MOE_SM107", "1") == "0"
        else ("B1" if a.perdie != "0" else "B0")
    )
    return base + ("n" if S == 0 else "")


def make_adapters(a: argparse.Namespace) -> List[ModuleAdapter]:
    impls = {"both": "deepgemm,port"}.get(a.impl, a.impl).split(",")
    out = []
    for name in impls:
        if name == "deepgemm":
            out.append(
                ModuleAdapter(
                    "deepgemm", "deep_gemm", a.kernel_filter_dg or DG_SM107_FILTER, a
                )
            )
        elif name in (
            "deepgemm2",
            "deepgemm3",
            "deepgemm4",
        ):  # extra independent DeepGEMM instances (A/A noise, placement effects)
            out.append(
                ModuleAdapter(
                    name, "deep_gemm", a.kernel_filter_dg or DG_SM107_FILTER, a
                )
            )
        elif name == "port":
            out.append(
                ModuleAdapter(
                    "port",
                    a.port_module,
                    a.kernel_filter_port or PORT_FILTER_DEFAULT,
                    a,
                )
            )
        elif name == "fi_existing":
            raise NotImplementedError(
                "fi_existing adapter (B3) is not implemented in this revision"
            )
        else:
            raise ValueError(f"unknown impl {name}")
    return out


# ======================================================================================================================
# 5. Context, env metadata, JSONL
# ======================================================================================================================
class Ctx:
    def __init__(self, a: argparse.Namespace, local_rank: int, nlocal: int):
        from deep_gemm.utils.dist import init_dist  # noqa: PLC0415

        self.a, self.local_rank, self.nlocal = a, local_rank, nlocal
        self.rank, self.R, self.group = init_dist(local_rank, nlocal)
        self.num_nodes = int(os.environ.get("WORLD_SIZE", 1))
        self.node_rank = int(os.environ.get("RANK", 0))
        self.is_root = self.rank == 0
        self.epr = a.num_experts // self.R
        self.t_user = a.num_max_tokens_per_rank
        self.hang_context: Dict[str, Any] = {}
        self.hang_buffers: List[Tuple[str, Any]] = []
        self.watchdog = Watchdog(
            a.timeout, dump_fn=self.dump_hang_state, post_wait_s=a.post_dump_wait
        )
        dg = _dg()
        dg.use_deterministic_algorithms(
            True
        )  # as the stock test (no effect on MegaMoE, kept for identical process state)
        dg.set_pdl(bool(a.pdl))
        torch.manual_seed(a.seed * 1000 + self.rank)
        random.seed(a.seed * 1000 + self.rank)
        self._bar = torch.zeros(1, device="cuda")
        self._ep = None
        self._barrier_fn = lambda: dist.all_reduce(self._bar, group=self.group)
        self.barrier_kind = "allreduce"
        self.out_path = a.out if self.is_root else None
        self.env = self._collect_env()

    # barrier used inside timed loops.  The stock test uses DeepEP's ``ep_buffer.barrier`` when deep_ep imports and
    # ``dist.all_reduce`` otherwise.  An NCCL all_reduce completes in rank order, which makes the earliest-leaving rank start its
    # kernel a few us before the others (visible as a monotonic per-rank ramp of ~1.3 us/rank in the per-rank medians), so the
    # harness offers lower-skew barriers: 'symm' = torch symmetric-memory handle barrier (device-side, spin on signal pads),
    # 'deepep' = the stock test's barrier, 'allreduce' = the fallback
    def barrier(self) -> None:
        self._barrier_fn()

    def setup_barrier(self, kind: str, ad: Optional["ModuleAdapter"] = None) -> str:
        self.close_barrier()
        a = self.a
        if kind in ("auto", "symm"):
            h = (
                getattr(ad.buffer, "handle", None)
                if ad is not None and ad.buffer is not None
                else None
            )
            if h is not None and hasattr(h, "barrier"):
                self._barrier_fn = lambda: h.barrier(0)
                self.barrier_kind = "symm"
                return self.barrier_kind
            kind = "allreduce"
        if kind == "deepep":
            import deep_ep  # noqa: PLC0415

            buf_type = getattr(deep_ep, "EPBuffer", None) or deep_ep.ElasticBuffer
            self._ep = buf_type(
                self.group,
                num_max_tokens_per_rank=a.num_max_tokens_per_rank,
                hidden=a.hidden,
                num_topk=a.num_topk,
                use_fp8_dispatch=True,
                explicitly_destroy=True,
                allow_multiple_reduction=False,
                num_gpu_timeout_secs=10,
                num_cpu_timeout_secs=30,
            )
            kw = (
                {"wait_comm_stream": False}
                if hasattr(deep_ep, "EPBuffer")
                else {"use_comm_stream": False}
            )
            ep = self._ep
            self._barrier_fn = lambda: ep.barrier(**kw)
            self.barrier_kind = "deepep"
            return self.barrier_kind
        self._barrier_fn = lambda: dist.all_reduce(self._bar, group=self.group)
        self.barrier_kind = "allreduce"
        return self.barrier_kind

    def close_barrier(self) -> None:
        if getattr(self, "_ep", None) is not None:
            with contextlib.suppress(Exception):  # best effort
                self._ep.destroy()
            self._ep = None
        self._barrier_fn = lambda: dist.all_reduce(self._bar, group=self.group)
        self.barrier_kind = "allreduce"

    def log(self, msg: str) -> None:
        if self.is_root:
            print(msg, flush=True)

    def dump_hang_state(self) -> None:
        """Called by the watchdog thread of a HUNG process: write the context and the first ``--hang-dump-bytes`` bytes of every symmetric
        buffer (MegaMoESignals counters / flags / barrier state, source indices) of THIS rank to ``--hang-dump-dir``.  The copy goes through
        a new non-blocking stream: it does not wait for the spinning kernel (copy engines are idle)."""
        d = self.a.hang_dump_dir
        if not d:
            return
        os.makedirs(d, exist_ok=True)
        torch.cuda.set_device(self.local_rank)
        tag = f"{self.env.get('job_id')}_rank{self.rank:03d}"
        meta = {
            "rank": self.rank,
            "world": self.R,
            "host": platform.node(),
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "context": self.hang_context,
            "args": {k: v for k, v in vars(self.a).items() if not k.startswith("_")},
        }
        prog = getattr(self, "progress_dev", None)
        if (
            prog is not None
        ):  # stress mode: number of launches this rank has COMPLETED (a side-stream copy does not wait for the stuck kernel)
            try:
                side = torch.cuda.Stream()
                with torch.cuda.stream(side):
                    pv = int(prog.to("cpu").item())
                side.synchronize()
                meta["device_progress_completed_launches"] = pv
            except Exception:  # noqa: BLE001 - best effort
                meta["device_progress_completed_launches"] = None
        with open(os.path.join(d, f"hang_{tag}.json"), "w") as f:
            json.dump(meta, f, default=_jsonable)
        stream = torch.cuda.Stream()
        for label, buf in self.hang_buffers:
            t = getattr(buf, "buffer", None)
            if t is None:
                continue
            n = min(int(t.numel()), int(self.a.hang_dump_bytes))
            with torch.cuda.stream(stream):
                host = t[:n].to("cpu")
            stream.synchronize()
            host.numpy().tofile(os.path.join(d, f"hang_{tag}_{label}_head.bin"))
        sys.stderr.write(f"hang state dumped to {d} (tag {tag})\n")

    def write(self, row: Dict[str, Any]) -> None:
        if not self.is_root or not self.out_path:
            return
        os.makedirs(os.path.dirname(os.path.abspath(self.out_path)), exist_ok=True)
        with open(self.out_path, "a") as f:
            f.write(json.dumps(row, default=_jsonable) + "\n")

    def clocks(self) -> Dict[str, Any]:
        try:
            return {
                "sm_mhz": torch.cuda.clock_rate(),
                "power_mw": torch.cuda.power_draw(),
                "temp_c": torch.cuda.temperature(),
            }
        except Exception:  # noqa: BLE001
            return {}

    def _collect_env(self) -> Dict[str, Any]:
        p = torch.cuda.get_device_properties(torch.cuda.current_device())
        env: Dict[str, Any] = {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "python": platform.python_version(),
            "host": platform.node(),
            "gpu": p.name,
            "multi_processor_count": p.multi_processor_count,
            "cc": list(torch.cuda.get_device_capability()),
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "nodelist": os.environ.get("SLURM_JOB_NODELIST"),
            "image": os.environ.get("SM107_BENCH_IMAGE"),
            "deepgemm_sha": os.environ.get("SM107_BENCH_DEEPGEMM_SHA", "3ec0eb4"),
            "flashinfer_sha": os.environ.get("SM107_BENCH_FLASHINFER_SHA", "3184bbea"),
            "env": {
                k: v
                for k, v in os.environ.items()
                if k.startswith(("DG_", "CUTE_DSL", "NCCL_", "TORCH_SYMMMEM"))
            },
        }
        try:
            import importlib.metadata as md

            for n in ("nvidia-cutlass-dsl", "flashinfer-python", "apache-tvm-ffi"):
                with contextlib.suppress(Exception):
                    env[n] = md.version(n)
        except Exception:  # noqa: BLE001
            pass
        try:
            import deep_gemm

            env["deep_gemm"] = getattr(deep_gemm, "__version__", None)
            env["deep_gemm_file"] = deep_gemm.__file__
        except Exception:  # noqa: BLE001
            pass
        with contextlib.suppress(Exception):
            import pynvml

            pynvml.nvmlInit()
            env["driver"] = pynvml.nvmlSystemGetDriverVersion()
            if isinstance(env["driver"], bytes):
                env["driver"] = env["driver"].decode()
        env["port_tree_hash"] = _port_tree_hash()
        with (
            contextlib.suppress(Exception),
            open(os.path.abspath(__file__), "rb") as fh,
        ):
            env["harness_hash"] = hashlib.sha256(fh.read()).hexdigest()[:12]
        return env


def _jsonable(o):
    if isinstance(o, (torch.Tensor,)):
        return o.tolist()
    if isinstance(o, (set, tuple)):
        return list(o)
    return str(o)


def _port_tree_hash() -> Optional[str]:
    """Content hash of the port package (traceability of which port revision a row measured)."""
    try:
        import importlib.util

        spec = importlib.util.find_spec(
            "flashinfer.experimental.sm107_deepgemm_megamoe"
        )
        if spec is None or not spec.submodule_search_locations:
            return None
        h = hashlib.sha256()
        root = list(spec.submodule_search_locations)[0]
        for dp, dn, fn in sorted(os.walk(root)):
            dn[:] = sorted(d for d in dn if d != "__pycache__")
            for f in sorted(fn):
                if f.endswith(".py"):
                    with open(os.path.join(dp, f), "rb") as fh:
                        h.update(f.encode() + fh.read())
        return h.hexdigest()[:16]
    except Exception:  # noqa: BLE001
        return None


def cluster_info() -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    try:
        txt = subprocess.run(
            ["nvidia-smi", "-q"], capture_output=True, text=True, timeout=60
        ).stdout
        out["cluster_uuid"] = sorted(set(re.findall(r"ClusterUUID\s*:\s*(\S+)", txt)))
        out["clique_id"] = sorted(set(re.findall(r"CliqueId\s*:\s*(\S+)", txt)))
    except Exception:  # noqa: BLE001
        pass
    return out


# ======================================================================================================================
# 6. Modes
# ======================================================================================================================
def mpoint_inputs(
    ctx: Ctx, prob: Problem, ad: ModuleAdapter, pt: Point
) -> Tuple[List[int], int, Inputs, int]:
    """(m_vec, my M, my staged inputs, my block_m) of one point."""
    a = ctx.a
    mv = pt.m_vec(ctx.R, ctx.t_user, a.seed)
    m = mv[ctx.rank]
    return mv, m, prob.slice(m), ad.block_m(m)


def count_recv(ctx: Ctx, inp: Inputs) -> Tuple[int, int]:
    """(num_recv_tokens, num_touched_experts) of this rank (tests/test_mega_moe.py:370-374, 392)."""
    from deep_gemm.utils.dist import uneven_all_gather  # noqa: PLC0415

    gathered = uneven_all_gather(inp.topk_idx, group=ctx.group)
    lo, hi = ctx.rank * ctx.epr, (ctx.rank + 1) * ctx.epr
    mine = gathered[(gathered >= lo) & (gathered < hi)]
    return int(mine.numel()), int(torch.unique(mine).numel())


def build_adapters(
    ctx: Ctx, prob: Problem, S: int, order: Optional[Sequence[int]] = None
) -> List[ModuleAdapter]:
    """Build the adapters (buffer + weights) in the given allocation ORDER (default: as listed in --impl).  The order matters for A/B:
    the instance allocated second runs up to ~3% faster at tiny M on Rubin (placement of the latency-critical workspace head)."""
    ads = make_adapters(ctx.a)
    if order is not None:
        ads = [ads[i] for i in order]
    for ad in ads:
        ctx.watchdog.arm(ctx.a.jit_timeout)
        ad.import_module()
        ad.build(ctx.group, prob, S)
        ctx.watchdog.disarm()
    ctx.setup_barrier(ctx.a.barrier, ads[0])
    ctx.hang_buffers = [(ad.label, ad.buffer) for ad in ads]
    return ads


def resolve_points(ctx: Ctx, ad: ModuleAdapter) -> List[Point]:
    a = ctx.a
    cache: Dict[str, Any] = {}

    def segs() -> List[Tuple[int, int, int]]:
        if "segs" not in cache:
            cache["segs"] = regime_segments(ad.block_m, ctx.t_user)
        return cache["segs"]

    def preset(name: str) -> List[int]:
        if name == "core":
            return list(CORE_M)
        if name == "serving":
            return list(SERVING_M)
        if name == "boundary":
            return boundary_points(segs())
        if name == "regimes":
            return regime_representatives(segs())
        if name == "perf":
            return sorted(
                set(CORE_M) | set(SERVING_M) | set(regime_representatives(segs()))
            )
        if (
            name == "regmax"
        ):  # the vLLM warmup list: the LARGEST M of each distinct BLOCK_M
            best: Dict[int, int] = {}
            for _s, e, b in segs():
                best[b] = max(best.get(b, 0), e)
            return sorted(best.values())
        if (
            name == "regmaxall"
        ):  # the largest M of EVERY constant-BLOCK_M segment (every island, e.g. the BLOCK_M 128 islands at DEP>=8)
            return sorted({e for (_s, e, _b) in segs()})
        raise KeyError(name)

    return parse_m_tokens(a.m_list or a.m_preset, preset, ctx.t_user)


def _timing_window(
    ctx: Ctx,
    calls: Dict[str, Callable],
    filters: Dict[str, str],
    n_windows: int,
    trace_path=None,
):
    """n_windows profiler windows (bench_kineto protocol each) -> concatenated per-label samples + stock means."""
    a = ctx.a
    acc: Dict[str, List[float]] = {lab: [] for lab in calls}
    stock: Dict[str, Any] = {}
    for w in range(n_windows):
        res = bench_samples(
            calls,
            filters,
            num_tests=a.num_tests,
            barrier=ctx.barrier,
            flush=not a.no_flush,
            trace_path=trace_path if w == 0 else None,
        )
        if not res["ok"]:
            return None, res["reason"], stock
        for lab in calls:
            acc[lab] += res["samples"][lab]
        stock.update(res["stock_mean_us"])
    return acc, "", stock


def _load_resume_state(path: str) -> Tuple[List[Any], List[Any]]:
    """Points already measured (valid rows) and points that were started but never finished (hung/crashed)."""
    done: List[Any] = []
    started: List[Any] = []
    if path and os.path.exists(path):
        with open(path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("kind") == "point" and r.get("valid", True):
                    done.append(
                        [
                            r["shape"]["S"],
                            r["M_label"],
                            r["pass"],
                            r["t_max_user"],
                            r["impl"],
                            r["mode"],
                        ]
                    )
                elif r.get("kind") == "start":
                    started.append([r["S"], r["M_label"], r["pass"], r["t_max_user"]])
    return done, started


def run_bench(ctx: Ctx) -> None:
    a = ctx.a
    shape = {
        "hidden": a.hidden,
        "intermediate": a.intermediate_hidden,
        "num_experts": a.num_experts,
        "topk": a.num_topk,
        "clamp": a.activation_clamp,
        "fast_math": a.fast_math,
    }
    prob = Problem(a, ctx.rank, ctx.R, ctx.t_user)
    dev_hosts = _gather_hosts(ctx)
    ctx.write(
        {
            "schema": SCHEMA,
            "kind": "run",
            "mode": "bench",
            "args": vars(a),
            "env": ctx.env,
            "hosts": dev_hosts,
            "cluster": cluster_info(),
            "dep": ctx.R,
            "num_nodes": ctx.num_nodes,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
    )
    n_impl = len(make_adapters(a))
    orders: List[Optional[List[int]]] = [None]
    if a.ab_swap and n_impl > 1:
        orders = [list(range(n_impl)), list(reversed(range(n_impl)))]
    for S, alloc_order in [
        (int(s), o) for s in str(a.num_shared_experts).split(",") for o in orders
    ]:
        ads = build_adapters(ctx, prob, S, alloc_order)
        ctx.alloc_order = [x.label for x in ads]
        ctx.log(
            f"== DEP{ctx.R} S={S} T_max={ctx.t_user}->{ads[0].t_aligned} impls={[x.label for x in ads]} "
            f"sm107={os.environ.get('DG_MEGA_MOE_SM107')} perdie={a.perdie} pdl={_dg().get_pdl()} "
            f"weights_localized={[x.weights_localized for x in ads]} barrier={ctx.barrier_kind}"
        )
        points = resolve_points(ctx, ads[0])
        ctx.log(f"   points ({len(points)}): {[p.label for p in points]}")
        cfg_cache: Dict[Tuple[str, int], Dict[str, Any]] = {}
        # --resume (set by the retry loop of the campaign scripts after a hang): skip finished points; points that were started but
        # never finished are the ones that hung -> recorded as kind="hang" and skipped (rank 0 decides, everybody follows)
        state = [_load_resume_state(a.out) if (a.resume and ctx.is_root) else ([], [])]
        if a.resume:
            dist.broadcast_object_list(state, src=0, group=ctx.group)
        done_set = {tuple(x) for x in state[0][0]}
        started_set = {tuple(x) for x in state[0][1]}
        modes_all = ["eager", "graph"] if a.exec == "both" else [a.exec]
        for ps in range(a.passes):
            for pt in points:
                key = (S, pt.label, ps, ctx.t_user)
                if a.resume and all(
                    (S, pt.label, ps, ctx.t_user, x.label, m) in done_set
                    for x in ads
                    for m in modes_all
                ):
                    continue
                if (
                    a.resume
                    and key in started_set
                    and not any(k[:4] == key for k in done_set)
                ):
                    ctx.log(
                        f"   resume: skipping HUNG point S={S} M={pt.label} pass {ps}"
                    )
                    ctx.write(
                        {
                            "schema": SCHEMA,
                            "kind": "hang",
                            "S": S,
                            "M_label": pt.label,
                            "pass": ps,
                            "dep": ctx.R,
                            "t_max_user": ctx.t_user,
                            "impl": [x.label for x in ads],
                            "job_id": ctx.env.get("job_id"),
                            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                        }
                    )
                    continue
                try:
                    _bench_point(ctx, prob, ads, pt, S, shape, ps, cfg_cache)
                except Exception:  # noqa: BLE001 - keep sweeping, but make the failure visible
                    ctx.log(
                        f"!! point {pt.label} failed on rank {ctx.rank}:\n{traceback.format_exc()}"
                    )
                    raise
        ctx.close_barrier()
        for ad in ads:
            ad.close()
        torch.cuda.empty_cache()


def _gather_hosts(ctx: Ctx) -> List[Dict[str, Any]]:
    objs = [None] * ctx.R
    dist.all_gather_object(
        objs,
        {
            "rank": ctx.rank,
            "host": platform.node(),
            "gpu_uuid": str(
                torch.cuda.get_device_properties(torch.cuda.current_device()).uuid
            ),
        },
        group=ctx.group,
    )
    return objs  # type: ignore[return-value]


def _bench_point(
    ctx: Ctx,
    prob: Problem,
    ads: List[ModuleAdapter],
    pt: Point,
    S: int,
    shape: Dict[str, Any],
    ps: int,
    cfg_cache: Dict[Tuple[str, int], Dict[str, Any]],
) -> None:
    a = ctx.a
    ctx.watchdog.arm()
    ctx.write(
        {
            "schema": SCHEMA,
            "kind": "start",
            "S": S,
            "M_label": pt.label,
            "pass": ps,
            "t_max_user": ctx.t_user,
            "dep": ctx.R,
            "impl": [x.label for x in ads],
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
    )
    ctx.hang_context = {
        "mode": "bench",
        "S": S,
        "M": pt.label,
        "pass": ps,
        "t_max_user": ctx.t_user,
        "alloc_order": getattr(ctx, "alloc_order", None),
    }
    mv, m, inp, _ = mpoint_inputs(ctx, prob, ads[0], pt)
    nrecv, ntouch = count_recv(ctx, inp)
    y_by, stats_by, calls, filters, block_ms = {}, {}, {}, {}, {}
    for ad in ads:
        bm = ad.block_m(m)
        block_ms[ad.label] = bm
        ad.stage(inp)
        y = torch.empty((m, a.hidden), dtype=torch.bfloat16, device="cuda")
        stats = prob.stats_init.clone() if not a.no_stats else None
        y_by[ad.label], stats_by[ad.label] = y, stats
        fn = ad.launch_fn(y, stats)
        # untimed first launch (JIT, lazy init); DG_PRINT_CONFIGS only here, never in timed loops
        ckey = (ad.label, bm)
        first_new = ckey not in cfg_cache
        ctx.watchdog.arm(a.jit_timeout if first_new else None)
        if first_new and a.capture_config and ad.label.startswith("deepgemm"):
            os.environ["DG_PRINT_CONFIGS"] = "1"
            try:
                with capture_fd_stdout() as box:
                    fn()
                    torch.cuda.synchronize()
            finally:
                os.environ.pop("DG_PRINT_CONFIGS", None)
            cfg_cache[ckey] = parse_config_text(box[0] if box else "")
        else:
            fn()
            torch.cuda.synchronize()
            cfg_cache.setdefault(ckey, {})
        ctx.watchdog.arm()
        calls[ad.label] = fn
        filters[ad.label] = ad.effective_filter()
    # sanity of the output (finite, non-trivial)
    ychk = {}
    for lab, y in y_by.items():
        yf = y.float() if y.numel() else torch.zeros(1, device="cuda")
        ychk[lab] = {
            "finite": bool(torch.isfinite(yf).all().item()),
            "abs_max": float(yf.abs().max().item()),
            "sum": float(yf.sum().item()),
        }
    modes = ["eager", "graph"] if a.exec == "both" else [a.exec]
    for mode in modes:
        use_calls = calls
        if mode == "graph":
            use_calls = {lab: _GraphCall(fn) for lab, fn in calls.items()}
        samples, reason, stock = _timing_window(
            ctx,
            use_calls,
            filters,
            a.windows,
            trace_path=(
                f"{a.trace_dir}/mega_{ctx.rank}_M{pt.label.replace('/', '-')}.json"
                if a.trace_dir
                else None
            ),
        )
        ev = (
            None
            if a.no_event_check
            else bench_events(
                use_calls,
                num_tests=a.num_tests,
                barrier=ctx.barrier,
                flush=not a.no_flush,
            )
        )
        info = torch.tensor(
            [m, block_ms[ads[0].label], nrecv, ntouch]
            + [ctx.clocks().get("sm_mhz", 0), ctx.clocks().get("power_mw", 0) / 1000.0],
            dtype=torch.float64,
            device="cuda",
        )
        ginfo = gather_rows(ctx.group, info).cpu().tolist()
        for ad in ads:
            lab = ad.label
            ok = samples is not None
            n_iters = a.num_tests * a.windows
            loc = torch.tensor(
                samples[lab] if ok else [float("nan")] * n_iters,
                dtype=torch.float64,
                device="cuda",
            )
            allv = gather_rows(ctx.group, loc).cpu().tolist()
            evall = None
            if ev is not None:
                evall = (
                    gather_rows(
                        ctx.group,
                        torch.tensor(ev[lab], dtype=torch.float64, device="cuda"),
                    )
                    .cpu()
                    .tolist()
                )
            valid = all(not math.isnan(v) for r in allv for v in r)
            row: Dict[str, Any] = {
                "schema": SCHEMA,
                "kind": "point",
                "impl": lab,
                "variant": variant_label(a, S),
                "kernel_filter": filters[lab],
                "shape": {**shape, "S": S},
                "dep": ctx.R,
                "num_nodes": ctx.num_nodes,
                "t_max_user": ctx.t_user,
                "t_max_aligned": ads[0].t_aligned,
                "M": mv[0],
                "M_label": pt.label,
                "m_vec": mv,
                "m_uniform": len(set(mv)) == 1,
                "block_m": [int(r[1]) for r in ginfo],
                "config": cfg_cache.get((lab, block_ms[lab]), {}),
                "mode": mode,
                "barrier": ctx.barrier_kind,
                "alloc_order": getattr(ctx, "alloc_order", None),
                "pdl": int(_dg().get_pdl()),
                "arch": os.environ.get("DG_MEGA_MOE_SM107_ARCH") or "100f",
                "sm107": int(os.environ.get("DG_MEGA_MOE_SM107", "0") or 0),
                "perdie": a.perdie,
                "weights_localized": ad.weights_localized,
                "routing": a.routing,
                "masked_ratio": a.masked_ratio,
                "weight_init": a.weight_init,
                "seed": a.seed,
                "pass": ps,
                "n_windows": a.windows,
                "n_iters": n_iters,
                "valid": valid,
                "invalid_reason": "" if valid else reason,
                "y_check": ychk[lab],
                "clocks_after": [{"sm_mhz": r[4], "power_w": r[5]} for r in ginfo],
                "job_id": ctx.env.get("job_id"),
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            if valid:
                row["samples_us"] = allv
                row["stat"] = summarize_samples(allv, a.windows)
                row["stock_mean_us"] = stock.get(lab)
                if evall is not None:
                    row["samples_ev_us"] = evall
                    row["ev_stat"] = summarize_samples(evall)
                perf = []
                for r in range(ctx.R):
                    mr, nr, nt = int(ginfo[r][0]), int(ginfo[r][2]), int(ginfo[r][3])
                    perf.append(
                        perf_model(
                            mr,
                            nr,
                            nt,
                            a.hidden,
                            a.intermediate_hidden,
                            S,
                            a.num_topk,
                            row["stat"]["rank_mean_us"][r],
                            a.mma_type,
                        )
                    )
                row["perf"] = {
                    "num_recv_tokens": [int(r[2]) for r in ginfo],
                    "num_touched_experts": [int(r[3]) for r in ginfo],
                    "tflops_rank": [p["tflops"] for p in perf],
                    "overlap_tflops_rank": [p["overlap_tflops"] for p in perf],
                    "hbm_gbs_rank": [p["hbm_gbs"] for p in perf],
                    "nvlink_gbs_rank": [p["nvlink_gbs"] for p in perf],
                    "tflops_mean": statistics.fmean(p["tflops"] for p in perf),
                    "t_reduction_us": perf[0]["t_reduction_us"],
                }
                st = row["stat"]
                ctx.log(
                    f"[DEP{ctx.R} {lab} S={S} {mode} M={pt.label} bm={sorted(set(row['block_m']))} p{ps}] "
                    f"median-of-max {st['median_of_max_us']:8.2f} us (p10 {st['p10_of_max_us']:.1f} p90 {st['p90_of_max_us']:.1f}) | "
                    f"rank-median [{', '.join(f'{v:.1f}' for v in st['rank_median_us'][:4])}{'...' if ctx.R > 4 else ''}] | "
                    + (
                        f"ev {row['ev_stat']['median_of_max_us']:.1f} | "
                        if evall is not None
                        else ""
                    )
                    + f"{row['perf']['tflops_mean']:.0f} TFLOPS | stats={'y' if stats_by[lab] is not None else 'n'}"
                )
            else:
                ctx.log(
                    f"[DEP{ctx.R} {lab} S={S} {mode} M={pt.label}] INVALID: {reason}"
                )
            ctx.write(row)
        del use_calls
    ctx.watchdog.disarm()


class _GraphCall:
    def __init__(self, fn: Callable[[], Any]):
        self.g = capture_graph(fn)

    def __call__(self):
        self.g.replay()


def run_configs(ctx: Ctx) -> None:
    """Untimed config-printing pass: one launch per BLOCK_M regime with DG_PRINT_CONFIGS=1 captured at fd level."""
    a = ctx.a
    prob = Problem(a, ctx.rank, ctx.R, ctx.t_user)
    ctx.write(
        {
            "schema": SCHEMA,
            "kind": "run",
            "mode": "configs",
            "args": vars(a),
            "env": ctx.env,
            "dep": ctx.R,
            "cluster": cluster_info(),
            "hosts": _gather_hosts(ctx),
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
    )
    for si, S in enumerate([int(s) for s in str(a.num_shared_experts).split(",")]):
        ads = build_adapters(ctx, prob, S)
        ad = ads[0]
        segs = regime_segments(ad.block_m, ctx.t_user)
        reps = (
            regime_representatives(segs)
            if not a.m_list
            else [int(x) for x in a.m_list.split(",")]
        )
        ctx.log(
            f"== configs DEP{ctx.R} S={S} T_max={ctx.t_user}->{ad.t_aligned} regimes={segs}"
        )
        for m in reps:
            # DeepGEMM prints the MegaMoEConfig line once per (shape, T_max, num_tokens) key per process: shift M by the S index so
            # that the line is printed again for the later S values (mid-regime M, so the BLOCK_M does not change)
            m = m + si
            ctx.watchdog.arm(a.jit_timeout)
            inp = prob.slice(m)
            ad.stage(inp)
            y = torch.empty((m, a.hidden), dtype=torch.bfloat16, device="cuda")
            fn = ad.launch_fn(y, prob.stats_init.clone())
            os.environ["DG_PRINT_CONFIGS"] = "1"
            try:
                with capture_fd_stdout() as box:
                    fn()
                    torch.cuda.synchronize()
            finally:
                os.environ.pop("DG_PRINT_CONFIGS", None)
            cfg = parse_config_text(box[0] if box else "")
            bm = ad.block_m(m)
            info = torch.tensor(
                [m, bm, cfg.get("stages", -1), cfg.get("smem", -1)],
                dtype=torch.float64,
                device="cuda",
            )
            g = gather_rows(ctx.group, info).cpu().tolist()
            ctx.write(
                {
                    "schema": SCHEMA,
                    "kind": "config",
                    "impl": ad.label,
                    "dep": ctx.R,
                    "num_nodes": ctx.num_nodes,
                    "S": S,
                    "t_max_user": ctx.t_user,
                    "t_max_aligned": ad.t_aligned,
                    "M": m,
                    "block_m": bm,
                    "config": cfg,
                    "all_ranks_block_m_stages_smem": g,
                    "sm107": os.environ.get("DG_MEGA_MOE_SM107"),
                    "perdie": a.perdie,
                    "regimes": segs,
                    "job_id": ctx.env.get("job_id"),
                    "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                }
            )
            ctx.log(
                f"   M={m:5d} block_m={bm:3d} stages={cfg.get('stages')} smem={cfg.get('smem')} ring={cfg.get('num_ring_tokens')} "
                f"sf_ring={cfg.get('num_sf_ring_tokens')}"
            )
            ctx.watchdog.disarm()
        ctx.close_barrier()
        for x in ads:
            x.close()


# ----------------------------------------------------------------------------------------------------------------------
# check: kernel-vs-kernel comparison on identical inputs
# ----------------------------------------------------------------------------------------------------------------------
def compare_y(ya: torch.Tensor, yb: torch.Tensor) -> Dict[str, Any]:
    """Bit-level comparison of two bf16 outputs of the same shape (this rank)."""
    from deep_gemm.testing import calc_diff  # noqa: PLC0415

    if ya.numel() == 0:
        return {
            "bit_equal": True,
            "torch_equal": True,
            "n_diff_elems": 0,
            "n_diff_rows": 0,
            "max_abs": 0.0,
            "calc_diff": 0.0,
        }
    d = ya.view(torch.int16) != yb.view(torch.int16)
    diff = (ya.float() - yb.float()).abs()
    diff = torch.where(torch.isnan(diff), torch.full_like(diff, float("inf")), diff)
    return {
        "bit_equal": not bool(d.any().item()),
        "torch_equal": bool(torch.equal(ya, yb)),
        "n_diff_elems": int(d.sum().item()),
        "n_diff_rows": int(d.any(dim=1).sum().item()),
        "max_abs": float(diff.max().item()),
        "calc_diff": float(calc_diff(ya, yb)),
        "a_finite": bool(torch.isfinite(ya.float()).all().item()),
        "b_finite": bool(torch.isfinite(yb.float()).all().item()),
    }


def run_check(ctx: Ctx) -> None:
    a = ctx.a
    prob = Problem(a, ctx.rank, ctx.R, ctx.t_user)
    ctx.write(
        {
            "schema": SCHEMA,
            "kind": "run",
            "mode": "check",
            "args": vars(a),
            "env": ctx.env,
            "dep": ctx.R,
            "hosts": _gather_hosts(ctx),
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
    )
    for S in [int(s) for s in str(a.num_shared_experts).split(",")]:
        ads = build_adapters(ctx, prob, S)
        assert len(ads) >= 2, (
            "--mode check needs two impls (e.g. --impl deepgemm,deepgemm2 or --impl both)"
        )
        points = resolve_points(ctx, ads[0])
        for pt in points:
            ctx.watchdog.arm(a.jit_timeout)
            mv, m, inp, bm = mpoint_inputs(ctx, prob, ads[0], pt)
            ys, sts = {}, {}
            for ad in ads:
                ad.stage(inp)
                y = torch.full((m, a.hidden), 0, dtype=torch.bfloat16, device="cuda")
                st = prob.stats_init.clone()
                ad.launch_fn(y, st)()
                torch.cuda.synchronize()
                ys[ad.label], sts[ad.label] = y, st
            ref = ads[0].label
            for ad in ads[1:]:
                cmp_ = compare_y(ys[ref], ys[ad.label])
                cmp_["stats_equal"] = bool(torch.equal(sts[ref], sts[ad.label]))
                keys = [
                    "bit_equal",
                    "n_diff_elems",
                    "n_diff_rows",
                    "max_abs",
                    "calc_diff",
                    "stats_equal",
                ]
                loc = torch.tensor(
                    [float(cmp_[k]) for k in keys], dtype=torch.float64, device="cuda"
                )
                allr = gather_rows(ctx.group, loc).cpu().tolist()
                row = {
                    "schema": SCHEMA,
                    "kind": "check",
                    "impl_a": ref,
                    "impl_b": ad.label,
                    "S": S,
                    "dep": ctx.R,
                    "M": pt.label,
                    "m_vec": mv,
                    "block_m": bm,
                    "keys": keys,
                    "per_rank": allr,
                    "all_bit_equal": all(r[0] == 1.0 for r in allr),
                    "all_stats_equal": all(r[5] == 1.0 for r in allr),
                    "max_abs": max(r[3] for r in allr),
                    "n_diff_elems": sum(r[1] for r in allr),
                    "job_id": ctx.env.get("job_id"),
                    "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                }
                ctx.write(row)
                ctx.log(
                    f"[check DEP{ctx.R} S={S} M={pt.label} bm={bm}] {ref} vs {ad.label}: bit_equal={row['all_bit_equal']} "
                    f"stats_equal={row['all_stats_equal']} n_diff_elems={int(row['n_diff_elems'])} max_abs={row['max_abs']:.3g}"
                )
            ctx.watchdog.disarm()
        ctx.close_barrier()
        for x in ads:
            x.close()


# ----------------------------------------------------------------------------------------------------------------------
# Gate 0: DeepGEMM-vs-DeepGEMM run-to-run bit-reproducibility
# ----------------------------------------------------------------------------------------------------------------------
def run_gate0(ctx: Ctx) -> None:
    """Re-launch the same inputs many times with injected rank skew / L2 state / permuted token order; compare ``y``
    (bitwise) and the recv stats against the first launch.  Records per (S, masked, M) counts and where they differ."""
    a = ctx.a
    prob0 = Problem(a, ctx.rank, ctx.R, ctx.t_user)
    ctx.write(
        {
            "schema": SCHEMA,
            "kind": "run",
            "mode": "gate0",
            "args": vars(a),
            "env": ctx.env,
            "dep": ctx.R,
            "hosts": _gather_hosts(ctx),
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
    )
    g0_pts = parse_m_tokens(
        a.gate0_m_list, lambda name: [], ctx.t_user
    )  # integers, a/b/c per-rank vectors, randR
    m_list = [(pt.label, pt.m_vec(ctx.R, ctx.t_user, a.seed)) for pt in g0_pts]
    flush_buf = torch.empty(L2_FLUSH_BYTES // 4, dtype=torch.int, device="cuda")
    for S in [int(s) for s in str(a.num_shared_experts).split(",")]:
        ads = build_adapters(ctx, prob0, S)
        ad = ads[0]
        for masked in [float(x) for x in a.gate0_masked.split(",")]:
            prob = (
                prob0 if masked == a.masked_ratio else prob0.with_masked_ratio(masked)
            )
            for label, mv_ in m_list:
                m = mv_[ctx.rank]
                ctx.watchdog.arm(a.jit_timeout)
                rec = _gate0_one(
                    ctx, ad, prob, S, masked, m, flush_buf, label=label, m_vec=mv_
                )
                ctx.write(rec)
                ctx.log(
                    f"[gate0 DEP{ctx.R} S={S} masked={masked} M={label} bm={rec['block_m']}] launches={rec['n_launches']} "
                    f"bit_equal={rec['n_bit_equal']} mismatch_launches={rec['n_mismatch_launches']} "
                    f"perm_bit_equal={rec['perm']['n_bit_equal']}/{rec['perm']['n_launches']} rows_ever_diff={rec['rows_ever_diff_total']} "
                    f"max_abs={rec['max_abs_diff']:.3g} stats_mismatch={rec['n_stats_mismatch']} neg_control_detected={rec['neg_control_detected']}"
                )
                ctx.watchdog.disarm()
        ctx.close_barrier()
        for x in ads:
            x.close()


def _gate0_one(
    ctx: Ctx,
    ad: ModuleAdapter,
    prob: Problem,
    S: int,
    masked: float,
    m: int,
    flush_buf: torch.Tensor,
    label: Optional[str] = None,
    m_vec: Optional[List[int]] = None,
) -> Dict[str, Any]:
    a = ctx.a
    R = ctx.R
    plan_key = (
        m
        if (label is None or re.fullmatch(r"\d+", label))
        else zlib.crc32(label.encode())
    )  # identical on all ranks (a per-rank vector has different local M)
    inp = prob.slice(m)
    bm = ad.block_m(m)
    ad.stage(inp)
    y = torch.empty((m, a.hidden), dtype=torch.bfloat16, device="cuda")
    stats = prob.stats_init.clone()
    fn = ad.launch_fn(y, stats)
    # reference launch (no skew)
    stats.copy_(prob.stats_init)
    fn()
    torch.cuda.synchronize()
    ref = y.clone()
    ref_stats = stats.clone()
    ref16 = ref.view(torch.int16)
    poison = torch.full_like(
        ref16, 0x7FC1
    )  # a bf16 NaN pattern: rows the kernel fails to write are caught
    plan = random.Random(
        a.seed * 7 + plan_key * 31 + int(masked * 1000)
    )  # identical on all ranks
    n = a.gate0_launches
    n_mismatch = n_stats_mismatch = n_bit_equal = n_torch_equal = 0
    rows_hit = torch.zeros(max(m, 1), dtype=torch.int32, device="cuda")
    max_abs = torch.zeros((), dtype=torch.float32, device="cuda")
    examples: List[Dict[str, Any]] = []
    elems_total = 0
    skew_hist: Dict[str, int] = {}
    ctx.hang_context = {
        "mode": "gate0",
        "S": S,
        "masked": masked,
        "M": m,
        "block_m": bm,
        "t_max_user": ctx.t_user,
        "n": n,
        "launch": -1,
        "alloc_order": getattr(ctx, "alloc_order", None),
    }
    for i in range(n):
        ctx.watchdog.arm()  # per-launch progress: a hang is detected --timeout seconds after the last completed launch
        ctx.hang_context["launch"] = i
        if i % 250 == 0:
            ctx.write(
                {
                    "schema": SCHEMA,
                    "kind": "progress",
                    "mode": "gate0",
                    "S": S,
                    "masked": masked,
                    "M": m,
                    "launch": i,
                    "n": n,
                    "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                }
            )
        delayed = i % R if (i % 3 != 2) else plan.randrange(R)
        cyc = plan.choice([0, 20_000, 200_000, 1_000_000, 4_000_000])
        do_flush = plan.random() < 0.5
        ctx.barrier()
        if do_flush:
            flush_buf.zero_()
        if ctx.rank == delayed and cyc:
            torch.cuda._sleep(
                cyc
            )  # this rank launches late: dispatch arrival / slot orders differ between launches
        y.view(torch.int16).copy_(poison)
        stats.copy_(prob.stats_init)
        if not (
            a.inject_hang == i and ctx.rank == 1
        ):  # --inject-hang N: rank 1 skips launch N -> the others spin (tests the hang dump)
            fn()
        d = y.view(torch.int16) != ref16
        any_d = bool(d.any().item()) if m > 0 else False
        st_eq = bool(torch.equal(stats, ref_stats))
        n_stats_mismatch += 0 if st_eq else 1
        if any_d:
            n_mismatch += 1
            elems_total += int(d.sum().item())
            rows_hit += d.any(dim=1).to(torch.int32)
            df = (y.float() - ref.float()).abs()
            max_abs = torch.maximum(
                max_abs,
                torch.where(
                    torch.isnan(df), torch.full_like(df, float("inf")), df
                ).max(),
            )
            if len(examples) < 8:
                r_, c_ = [int(v) for v in d.nonzero()[0].tolist()]
                examples.append(
                    {
                        "launch": i,
                        "row": r_,
                        "col": c_,
                        "ref": float(ref[r_, c_].float()),
                        "got": float(y[r_, c_].float()),
                        "n_rows": int(d.any(dim=1).sum().item()),
                        "n_elems": int(d.sum().item()),
                        "delayed_rank": delayed,
                        "skew_cycles": cyc,
                        "flushed": do_flush,
                    }
                )
        else:
            n_bit_equal += 1
        n_torch_equal += int(torch.equal(y, ref))
        skew_hist[str(cyc)] = skew_hist.get(str(cyc), 0) + 1
    torch.cuda.synchronize()
    # permuted token order: same logical inputs, un-permute outputs before comparing
    pn = a.gate0_perm_launches
    perm_bit_equal = perm_stats_mismatch = 0
    perm_rows = torch.zeros(max(m, 1), dtype=torch.int32, device="cuda")
    perm_elems = 0
    perm_max_abs = torch.zeros((), dtype=torch.float32, device="cuda")
    pgen = torch.Generator(device="cuda")
    ctx.hang_context["stage"] = "perm"
    for i in range(pn):
        ctx.watchdog.arm()
        ctx.hang_context["launch"] = i
        pgen.manual_seed(a.seed * 131 + i * 7 + m)
        perm = (
            torch.randperm(m, device="cuda", generator=pgen)
            if m > 0
            else torch.zeros(0, dtype=torch.long, device="cuda")
        )
        pinp = inp.permuted(perm)
        ad.stage(pinp)
        yp = torch.empty_like(y)
        stats.copy_(prob.stats_init)
        ctx.barrier()
        ad.launch_fn(yp, stats)()
        yu = torch.empty_like(yp)
        if m > 0:
            yu.index_copy_(0, perm, yp)  # row perm[j] <- yp[j]
        d = yu.view(torch.int16) != ref16
        anyd = bool(d.any().item()) if m > 0 else False
        perm_stats_mismatch += 0 if bool(torch.equal(stats, ref_stats)) else 1
        if anyd:
            perm_elems += int(d.sum().item())
            perm_rows += d.any(dim=1).to(torch.int32)
            df = (yu.float() - ref.float()).abs()
            perm_max_abs = torch.maximum(
                perm_max_abs,
                torch.where(
                    torch.isnan(df), torch.full_like(df, float("inf")), df
                ).max(),
            )
        else:
            perm_bit_equal += 1
    # negative control (the test must be able to FAIL): perturb the routing weights of the first tokens (x1.01) on every rank,
    # launch, and require that y differs from the reference; then restore the staged inputs
    neg_diff_elems = 0
    nt = min(m, 8)
    if nt > 0:
        ad.buffer.topk_weights[:nt].mul_(1.01)
    # the launch is COLLECTIVE: a rank without tokens (per-rank M vector with a 0 entry) must take part too, only the perturbation is skipped there
    yn = torch.empty_like(y)
    stats.copy_(prob.stats_init)
    ctx.barrier()
    ad.launch_fn(yn, stats)()
    if m > 0:
        neg_diff_elems = int((yn.view(torch.int16) != ref16).sum().item())
    ad.stage(inp)  # restore
    torch.cuda.synchronize()
    ref_f = ref.float()
    loc = torch.tensor(
        [
            n,
            n_bit_equal,
            n_torch_equal,
            n_mismatch,
            n_stats_mismatch,
            elems_total,
            float(max_abs.item()),
            int((rows_hit > 0).sum().item()),
            pn,
            perm_bit_equal,
            perm_stats_mismatch,
            perm_elems,
            float(perm_max_abs.item()),
            int((perm_rows > 0).sum().item()),
            float(torch.isfinite(ref_f).all().item()),
            float(ref_f.abs().max().item() if m > 0 else 0.0),
            neg_diff_elems,
        ],
        dtype=torch.float64,
        device="cuda",
    )
    allr = gather_rows(ctx.group, loc).cpu().tolist()
    ex_all: List[Any] = [None] * R
    dist.all_gather_object(ex_all, examples, group=ctx.group)
    rec = {
        "schema": SCHEMA,
        "kind": "gate0",
        "impl": ad.label,
        "S": S,
        "masked_ratio": masked,
        "dep": R,
        "M": m,
        "M_label": label if label is not None else str(m),
        "m_vec": list(m_vec) if m_vec is not None else [m] * R,
        "t_max_user": ctx.t_user,
        "t_max_aligned": ad.t_aligned,
        "barrier": ctx.barrier_kind,
        "n_perm_launches": pn,
        "block_m": bm,
        "n_launches": n,
        "keys": [
            "n",
            "n_bit_equal",
            "n_torch_equal",
            "n_mismatch_launches",
            "n_stats_mismatch",
            "n_diff_elems",
            "max_abs",
            "rows_ever_diff",
            "perm_n",
            "perm_bit_equal",
            "perm_stats_mismatch",
            "perm_diff_elems",
            "perm_max_abs",
            "perm_rows_ever_diff",
            "ref_finite",
            "ref_abs_max",
            "neg_control_diff_elems",
        ],
        "per_rank": allr,
        "neg_control_detected": bool(sum(r[16] for r in allr) > 0)
        if any(v > 0 for v in (m_vec if m_vec is not None else [m] * R))
        else None,
        "n_bit_equal": int(min(r[1] for r in allr)),
        "n_mismatch_launches": int(max(r[3] for r in allr)),
        "n_stats_mismatch": int(sum(r[4] for r in allr)),
        "max_abs_diff": max(r[6] for r in allr),
        "rows_ever_diff_total": int(sum(r[7] for r in allr)),
        "diff_elems_total": int(sum(r[5] for r in allr)),
        "perm": {
            "n_launches": int(allr[0][8]),
            "n_bit_equal": int(min(r[9] for r in allr)),
            "stats_mismatch": int(sum(r[10] for r in allr)),
            "diff_elems_total": int(sum(r[11] for r in allr)),
            "max_abs_diff": max(r[12] for r in allr),
            "rows_ever_diff_total": int(sum(r[13] for r in allr)),
        },
        "all_runs_bit_equal": all(r[3] == 0 for r in allr)
        and all(r[9] == r[8] for r in allr),
        "examples": ex_all,
        "skew_hist": skew_hist,
        "job_id": ctx.env.get("job_id"),
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    return rec


# ----------------------------------------------------------------------------------------------------------------------
# Graph-replay race test: DeepGEMM tags its combine-readiness handshake with ``%gridid + 1`` and never resets it.
# ``%gridid`` is unique per DIRECT launch but identical for every replay of the same CUDA-graph kernel node, so the flag left by replay k equals the tag
# of replay k+1: a rank may start combining before its peers wrote the CURRENT launch's L2 rows (stale rows).  The input data must change between
# replays (else stale == correct) and one rank must be late (rank skew) to open the window.  Compare every replay (graph) / launch (eager control) bitwise
# with a reference computed by an eager launch on the same inputs.
# ----------------------------------------------------------------------------------------------------------------------
def run_graphrace(ctx: Ctx) -> None:
    a = ctx.a
    R = ctx.R
    prob = Problem(a, ctx.rank, R, ctx.t_user)
    ctx.write(
        {
            "schema": SCHEMA,
            "kind": "run",
            "mode": "graphrace",
            "args": vars(a),
            "env": ctx.env,
            "dep": R,
            "hosts": _gather_hosts(ctx),
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
    )
    m_list = [int(x) for x in a.graphrace_m_list.split(",")]
    n = a.graphrace_replays
    nsets = max(
        1, a.graphrace_sets
    )  # 1 = control: identical inputs every replay (stale == fresh, no mismatch expected even with stale flags)
    skew_choices = [0, 20_000, 200_000, 1_000_000, 4_000_000]
    for S in [int(s) for s in str(a.num_shared_experts).split(",")]:
        ads = build_adapters(ctx, prob, S)
        ad = ads[0]
        for m in m_list:
            ctx.watchdog.arm(a.jit_timeout)
            bm = ad.block_m(m)
            base = prob.slice(m)
            pgen = torch.Generator(device="cuda")
            sets = []
            for k in range(
                nsets
            ):  # different token orders: same multiset of tokens, different row -> output mapping, so stale rows differ from fresh ones
                pgen.manual_seed(a.seed * 977 + k * 31 + m)
                perm = (
                    torch.randperm(m, device="cuda", generator=pgen)
                    if m > 0
                    else torch.zeros(0, dtype=torch.long, device="cuda")
                )
                sets.append(base.permuted(perm))
            y = torch.empty((m, a.hidden), dtype=torch.bfloat16, device="cuda")
            stats = prob.stats_init.clone()
            fn = ad.launch_fn(y, stats)
            refs = []
            for k in range(
                nsets
            ):  # references by DIRECT (eager) launches, each rank on its own inputs
                dist.barrier(group=ctx.group)
                ad.stage(sets[k])
                torch.cuda.synchronize()
                dist.barrier(group=ctx.group)
                fn()
                torch.cuda.synchronize()
                refs.append(y.clone())
            # a second eager reference per set (is the eager result itself stable under skew?  Gate 0 says yes; this keeps the control self-contained)
            for k in range(nsets):
                ad.stage(sets[k])
                torch.cuda.synchronize()
                dist.barrier(group=ctx.group)
                fn()
                torch.cuda.synchronize()
                assert torch.equal(y, refs[k]), (
                    "eager reference not reproducible (Gate 0 violated)"
                )
            # capture AFTER the warm launches (JIT done); the graph reads/writes the same symmetric buffer / y / stats
            g = capture_graph(fn)
            plan = random.Random(a.seed * 7919 + m + 13 * S)  # identical on every rank
            for use_graph in [True, False] if a.graphrace_control else [True]:
                n_bad = 0
                rows_bad_total = 0
                by_skew: Dict[str, List[int]] = {
                    str(c): [0, 0] for c in skew_choices
                }  # skew cycles -> [launches, mismatches]
                examples: List[Dict[str, Any]] = []
                poison = torch.full_like(y.view(torch.int16), 0x7FC1)
                tag_trace: List[
                    List[List[int]]
                ] = []  # [replay][0 = combine_ready_grid_idx, 1 = peer_grid_idx][peer] of THIS rank, first 8 launches (MegaMoESignals offsets 128 / 768)
                ctx.hang_context = {
                    "mode": "graphrace",
                    "S": S,
                    "M": m,
                    "block_m": bm,
                    "graph": use_graph,
                    "n": n,
                    "launch": -1,
                    "t_max_user": ctx.t_user,
                }
                for i in range(n):
                    ctx.watchdog.arm()
                    ctx.hang_context["launch"] = i
                    k = plan.randrange(nsets)
                    delayed = i % R if (i % 3 != 2) else plan.randrange(R)
                    cyc = plan.choice(skew_choices)
                    dist.barrier(
                        group=ctx.group
                    )  # every rank idle: remote pulls of the previous launch are done before the inputs are restaged
                    ad.stage(sets[k])
                    y.view(torch.int16).copy_(poison)
                    stats.copy_(prob.stats_init)
                    ctx.barrier()
                    if ctx.rank == delayed and cyc:
                        torch.cuda._sleep(cyc)
                    if use_graph:
                        g.replay()
                    else:
                        fn()
                    d = y.view(torch.int16) != refs[k].view(torch.int16)
                    bad = bool(d.any().item()) if m > 0 else False
                    if i < 8:
                        raw = ad.buffer.buffer
                        tag_trace.append(
                            [
                                raw[128 : 128 + 8 * R].view(torch.int64).tolist(),
                                raw[768 : 768 + 8 * R].view(torch.int64).tolist(),
                            ]
                        )
                    by_skew[str(cyc)][0] += 1
                    if bad:
                        n_bad += 1
                        by_skew[str(cyc)][1] += 1
                        nr = int(d.any(dim=1).sum().item())
                        rows_bad_total += nr
                        if len(examples) < 6:
                            r_, c_ = [int(v) for v in d.nonzero()[0].tolist()]
                            examples.append(
                                {
                                    "launch": i,
                                    "set": k,
                                    "rows_bad": nr,
                                    "first_row": r_,
                                    "first_col": c_,
                                    "delayed_rank": delayed,
                                    "skew_cycles": cyc,
                                    "got_poison": bool(
                                        (y.view(torch.int16)[r_, c_] == 0x7FC1).item()
                                    ),
                                }
                            )
                torch.cuda.synchronize()
                loc = torch.tensor(
                    [n, n_bad, rows_bad_total], dtype=torch.float64, device="cuda"
                )
                allr = gather_rows(ctx.group, loc).cpu().tolist()
                ex_all: List[Any] = [None] * R
                dist.all_gather_object(ex_all, examples, group=ctx.group)
                tt_all: List[Any] = [None] * R
                dist.all_gather_object(tt_all, tag_trace, group=ctx.group)
                rec = {
                    "schema": SCHEMA,
                    "kind": "graphrace",
                    "tag_trace_first8_per_rank": tt_all,
                    "impl": ad.label,
                    "graph": use_graph,
                    "S": S,
                    "M": m,
                    "block_m": bm,
                    "dep": R,
                    "t_max_user": ctx.t_user,
                    "t_max_aligned": ad.t_aligned,
                    "n_replays": n,
                    "n_sets": nsets,
                    "per_rank": allr,
                    "keys": ["n", "n_mismatch_launches", "rows_mismatch_total"],
                    "n_mismatch_launches_max_rank": int(max(r[1] for r in allr)),
                    "n_mismatch_launches_sum": int(sum(r[1] for r in allr)),
                    "by_skew_cycles_rank": by_skew,
                    "examples": ex_all,
                    "barrier": ctx.barrier_kind,
                    "job_id": ctx.env.get("job_id"),
                    "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                }
                ctx.write(rec)
                ctx.log(
                    f"[graphrace DEP{R} S={S} M={m} bm={bm} {'GRAPH' if use_graph else 'eager'}] launches={n} mismatching launches per rank={[int(r[1]) for r in allr]} "
                    f"rows={[int(r[2]) for r in allr]} by_skew(rank0)={by_skew}"
                )
            ctx.watchdog.disarm()
            del g
        ctx.close_barrier()
        for x in ads:
            x.close()


# ----------------------------------------------------------------------------------------------------------------------
# Stress: hang hunting (DEP32, S=0 hang seen after ~3k launches of one buffer): back-to-back launches over an M sequence on ONE buffer
# with a launch-exact context for the watchdog dump (segment / launch range / M / device-side progress counter)
# ----------------------------------------------------------------------------------------------------------------------
def stress_m_at(cycle: Sequence[Any], shuffle: bool, seed: int, seg: int) -> Any:
    """M token of segment ``seg`` (a segment = ``--stress-per-m`` launches at one M; the token is an int, or a per-rank vector string ``a/b/c`` / ``randR``).  The cycle is permuted per
    round when ``shuffle`` (the same permutation on every rank: a private RNG, not the per-rank global one)."""
    rnd, pos = divmod(seg, len(cycle))
    if not shuffle:
        return cycle[pos]
    order = list(range(len(cycle)))
    random.Random(seed * 1009 + rnd).shuffle(order)
    return cycle[order[pos]]


def run_stress(ctx: Ctx) -> None:
    a = ctx.a
    prob = Problem(a, ctx.rank, ctx.R, ctx.t_user)
    ctx.write(
        {
            "schema": SCHEMA,
            "kind": "run",
            "mode": "stress",
            "args": vars(a),
            "env": ctx.env,
            "dep": ctx.R,
            "hosts": _gather_hosts(ctx),
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
    )
    cyc_pts = parse_stress_cycle(
        a.stress_m_list, ctx.R, ctx.t_user, a.seed
    )  # [(token, per-rank M vector)]: integers = same M on every rank, a/b/c = per-rank vector, randR
    assert cyc_pts, (
        f"--stress-m-list has no entry with every per-rank M <= T_max ({ctx.t_user})"
    )
    cycle = [k for k, _ in cyc_pts]
    cyc_mv = {
        k: mv for k, mv in cyc_pts
    }  # token -> per-rank M vector (this rank uses cyc_mv[token][rank])
    flush_buf = (
        torch.empty(L2_FLUSH_BYTES // 4, dtype=torch.int, device="cuda")
        if a.stress_flush
        else None
    )
    progress = torch.zeros(
        1, dtype=torch.int64, device="cuda"
    )  # device-side count of COMPLETED launches (written after each launch, stream-ordered)
    ctx.progress_dev = progress
    plan = random.Random(a.seed * 17 + ctx.rank)  # per-rank skew generator

    def host_barrier() -> None:
        torch.cuda.synchronize()
        dist.barrier(group=ctx.group)

    def pre_launch_barrier() -> None:
        if a.stress_barrier == "symm":
            ctx.barrier()
        elif a.stress_barrier == "allreduce":
            dist.all_reduce(ctx._bar, group=ctx.group)

    for S in [int(s) for s in str(a.num_shared_experts).split(",")]:
        ads = build_adapters(ctx, prob, S)
        ad = ads[0]
        if a.stress_barrier == "symm" and ctx.barrier_kind != "symm":
            ctx.log(
                f"!! --stress-barrier symm requested but the barrier kind is {ctx.barrier_kind}"
            )
        cache: Dict[
            Any, Tuple[Callable[[], Any], torch.Tensor, torch.Tensor, Inputs, int]
        ] = {}

        def get(tok):
            if tok not in cache:
                m = cyc_mv[tok][ctx.rank]
                y = torch.empty((m, a.hidden), dtype=torch.bfloat16, device="cuda")
                stats = prob.stats_init.clone()
                cache[tok] = (
                    ad.launch_fn(y, stats),
                    y,
                    stats,
                    prob.slice(m),
                    ad.block_m(m),
                )
            return cache[tok]

        ctx.log(
            f"== STRESS DEP{ctx.R} S={S} T_max={ctx.t_user}->{ad.t_aligned} cycle={cycle} per_m={a.stress_per_m} launches={a.stress_launches} "
            f"barrier={a.stress_barrier} flush={a.stress_flush} sleep={a.stress_sleep_cycles} skew={a.stress_skew_cycles} shuffle={a.stress_shuffle} "
            f"sm107={os.environ.get('DG_MEGA_MOE_SM107')} barrier_kind={ctx.barrier_kind}"
        )
        # pre-warm: one launch per distinct M token (JIT of every BLOCK_M variant outside the stress loop; first launch lazy init)
        for m in dict.fromkeys(cycle):
            fn, y, stats, inp, bm = get(m)
            ctx.watchdog.arm(a.jit_timeout)
            ctx.hang_context = {
                "mode": "stress",
                "stage": "warm",
                "S": S,
                "M": m,
                "block_m": bm,
                "t_max_user": ctx.t_user,
            }
            host_barrier()
            ad.stage(inp)
            host_barrier()
            fn()
            torch.cuda.synchronize()
        cur_m: Optional[Any] = None
        launch = seg = 0
        t0 = last = time.time()
        launches_last = 0
        n_total = int(a.stress_launches)
        while launch < n_total:
            m = stress_m_at(cycle, a.stress_shuffle, a.seed, seg)
            fn, y, stats, inp, bm = get(m)
            cnt = min(int(a.stress_per_m), n_total - launch)
            ctx.watchdog.arm()
            ctx.hang_context = {
                "mode": "stress",
                "S": S,
                "segment": seg,
                "M": m,
                "prev_M": cur_m,
                "block_m": bm,
                "launch_first": launch,
                "launch_last": launch + cnt - 1,
                "per_m": a.stress_per_m,
                "t_max_user": ctx.t_user,
                "barrier": a.stress_barrier,
                "flush": a.stress_flush,
                "sleep_cycles": a.stress_sleep_cycles,
                "skew_cycles": a.stress_skew_cycles,
                "cycle": cycle,
                "shuffle": a.stress_shuffle,
                "seed": a.seed,
                "sm107": os.environ.get("DG_MEGA_MOE_SM107"),
            }
            if (
                m != cur_m
            ):  # every rank is idle before the symmetric inputs are restaged (remote pulls of the previous launch are done)
                host_barrier()
                ad.stage(inp)
                host_barrier()
                cur_m = m
            for j in range(cnt):
                pre_launch_barrier()
                if flush_buf is not None:
                    flush_buf.zero_()
                if a.stress_sleep_cycles:
                    torch.cuda._sleep(a.stress_sleep_cycles)
                if a.stress_skew_cycles:
                    torch.cuda._sleep(plan.randrange(a.stress_skew_cycles + 1))
                if not (
                    a.inject_hang == launch + j and ctx.rank == 1
                ):  # --inject-hang N: rank 1 skips launch N (tests the dump)
                    fn()
                progress.fill_(launch + j + 1)
            torch.cuda.synchronize()
            launch += cnt
            seg += 1
            if seg % max(1, a.stress_log_every) == 0 or launch >= n_total:
                now = time.time()
                rate = (launch - launches_last) / max(now - last, 1e-9)
                last, launches_last = now, launch
                ctx.log(
                    f"[stress DEP{ctx.R} S={S}] launches {launch}/{n_total} seg {seg} M={m} bm={bm} elapsed {now - t0:.0f}s rate {rate:.0f}/s"
                )
                ctx.write(
                    {
                        "schema": SCHEMA,
                        "kind": "progress",
                        "mode": "stress",
                        "S": S,
                        "launch": launch,
                        "n": n_total,
                        "segment": seg,
                        "M": m,
                        "elapsed_s": round(now - t0, 1),
                        "rate_per_s": round(rate, 1),
                        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    }
                )
        ctx.watchdog.disarm()
        ctx.write(
            {
                "schema": SCHEMA,
                "kind": "stress_done",
                "S": S,
                "dep": ctx.R,
                "t_max_user": ctx.t_user,
                "launches": n_total,
                "segments": seg,
                "elapsed_s": round(time.time() - t0, 1),
                "cycle": cycle,
                "per_m": a.stress_per_m,
                "barrier": a.stress_barrier,
                "flush": a.stress_flush,
                "sleep_cycles": a.stress_sleep_cycles,
                "skew_cycles": a.stress_skew_cycles,
                "shuffle": a.stress_shuffle,
                "job_id": ctx.env.get("job_id"),
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
        )
        host_barrier()
        ctx.close_barrier()
        for x in ads:
            x.close()
        torch.cuda.empty_cache()


def run_graphlayers(ctx: Ctx) -> None:
    """L >= 1 back-to-back launches of the SAME kernel on the SAME workspace captured in ONE CUDA graph (vLLM's full-graph pattern: L MoE layers sharing one symmetric buffer), replayed
    ``--windows`` x ``--num-tests`` times per cell (multi-layer graph experiment).  With L = 1 every replay is the same graph node (identical ``%gridid``, so DeepGEMM's combine-ready flags of the
    previous replay already match its tag); with L >= 2 the predecessor of every node on the workspace is a different node (honest waits).  Per (S, M, L) cell and impl: per-launch kernel
    durations (profiler; pooled over the L nodes = ``stat``, and per node = ``node_stat``; median of max over ranks), replay time / L (CUDA events, ``replay_per_layer_stat``) and a bitwise
    comparison of y and the recv stats between the impls after one replay from identical initial stats.  Use ``--impl deepgemm,port`` once per ``--port-tag-mode`` (epoch / gridid).
    Rows ``kind="graphlayers"``."""
    a = ctx.a
    prob = Problem(a, ctx.rank, ctx.R, ctx.t_user)
    shape = {
        "hidden": a.hidden,
        "intermediate": a.intermediate_hidden,
        "num_experts": a.num_experts,
        "topk": a.num_topk,
        "clamp": a.activation_clamp,
        "fast_math": a.fast_math,
    }
    ctx.write(
        {
            "schema": SCHEMA,
            "kind": "run",
            "mode": "graphlayers",
            "args": vars(a),
            "env": ctx.env,
            "hosts": _gather_hosts(ctx),
            "dep": ctx.R,
            "num_nodes": ctx.num_nodes,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
    )
    Ls = [int(x) for x in str(a.graph_layers_list).split(",") if x.strip()]
    n_impl = len(make_adapters(a))
    orders: List[Optional[List[int]]] = [None]
    if a.ab_swap and n_impl > 1:
        orders = [list(range(n_impl)), list(reversed(range(n_impl)))]
    n_iters = a.num_tests * a.windows
    for S, order in [
        (int(s), o) for s in str(a.num_shared_experts).split(",") for o in orders
    ]:
        ads = build_adapters(ctx, prob, S, order)
        ctx.alloc_order = [x.label for x in ads]
        points = resolve_points(ctx, ads[0])
        ctx.log(
            f"== GRAPHLAYERS DEP{ctx.R} S={S} T_max={ctx.t_user}->{ads[0].t_aligned} impls={[x.label for x in ads]} tag={a.port_tag_mode or 'default'} L={Ls} points={[p.label for p in points]} replays={n_iters}"
        )
        for pt in points:
            ctx.watchdog.arm(a.jit_timeout)
            mv, m, inp, _ = mpoint_inputs(ctx, prob, ads[0], pt)
            fns, ys, sts, filters, bms = {}, {}, {}, {}, {}
            for ad in ads:
                bms[ad.label] = ad.block_m(m)
                ad.stage(inp)
                y = torch.empty((m, a.hidden), dtype=torch.bfloat16, device="cuda")
                stats = prob.stats_init.clone() if not a.no_stats else None
                fn = ad.launch_fn(y, stats)
                fn()  # untimed first launch (JIT, lazy init)
                torch.cuda.synchronize()
                fns[ad.label], ys[ad.label], sts[ad.label], filters[ad.label] = (
                    fn,
                    y,
                    stats,
                    ad.effective_filter(),
                )
            labs = list(fns)
            for L in Ls:
                ctx.watchdog.arm(a.jit_timeout)
                graphs = {
                    lab: capture_graph(
                        (lambda f=fns[lab], n=L: [f() for _ in range(n)])
                    )
                    for lab in labs
                }
                calls = {lab: (lambda g=g: g.replay()) for lab, g in graphs.items()}
                # bitwise comparison after ONE replay of each impl from identical initial stats (all L launches see the same inputs)
                outs = {}
                for lab in labs:
                    if sts[lab] is not None:
                        sts[lab].copy_(prob.stats_init)
                    ctx.barrier()
                    graphs[lab].replay()
                    torch.cuda.synchronize()
                    outs[lab] = (
                        ys[lab].clone(),
                        sts[lab].clone() if sts[lab] is not None else None,
                    )
                ref = labs[0]
                eq = []
                for lab in labs[1:]:
                    y_eq = bool(
                        torch.equal(
                            outs[ref][0].view(torch.int16),
                            outs[lab][0].view(torch.int16),
                        )
                    )
                    s_eq = (
                        True
                        if outs[ref][1] is None
                        else bool(torch.equal(outs[ref][1], outs[lab][1]))
                    )
                    eq.append(1.0 if (y_eq and s_eq) else 0.0)
                eq_all = (
                    gather_rows(
                        ctx.group,
                        torch.tensor(eq or [1.0], dtype=torch.float64, device="cuda"),
                    )
                    .cpu()
                    .tolist()
                )
                bit_equal = all(v == 1.0 for r in eq_all for v in r)
                # timing: n_windows profiler windows (bench_kineto protocol per window) + a CUDA-event pass for the replay time / L
                acc = {lab: [] for lab in labs}
                acc_node = {lab: [[] for _ in range(L)] for lab in labs}
                ok, reason = True, ""
                for _w in range(a.windows):
                    res = bench_samples(
                        calls,
                        filters,
                        num_tests=a.num_tests,
                        barrier=ctx.barrier,
                        flush=not a.no_flush,
                        layers=L,
                    )
                    if not res["ok"]:
                        ok, reason = False, res["reason"]
                        break
                    for lab in labs:
                        acc[lab] += res["samples"][lab]
                        for j in range(L):
                            acc_node[lab][j] += res["node"][lab][j]
                ev = (
                    bench_events(
                        calls,
                        num_tests=n_iters,
                        barrier=ctx.barrier,
                        flush=not a.no_flush,
                    )
                    if ok
                    else None
                )
                for lab in labs:
                    row: Dict[str, Any] = {
                        "schema": SCHEMA,
                        "kind": "graphlayers",
                        "impl": lab,
                        "variant": variant_label(a, S),
                        "shape": {**shape, "S": S},
                        "dep": ctx.R,
                        "num_nodes": ctx.num_nodes,
                        "t_max_user": ctx.t_user,
                        "t_max_aligned": ads[0].t_aligned,
                        "M": mv[0],
                        "M_label": pt.label,
                        "m_vec": mv,
                        "block_m": bms[lab],
                        "layers": L,
                        "port_tag_mode": a.port_tag_mode or "default",
                        "alloc_order": getattr(ctx, "alloc_order", None),
                        "mode": "graph",
                        "barrier": ctx.barrier_kind,
                        "n_windows": a.windows,
                        "n_replays": n_iters,
                        "bit_equal_vs_first_impl": bit_equal,
                        "valid": ok,
                        "invalid_reason": reason,
                        "job_id": ctx.env.get("job_id"),
                        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    }
                    if ok:
                        allv = (
                            gather_rows(
                                ctx.group,
                                torch.tensor(
                                    acc[lab], dtype=torch.float64, device="cuda"
                                ),
                            )
                            .cpu()
                            .tolist()
                        )
                        node_all = [
                            gather_rows(
                                ctx.group,
                                torch.tensor(
                                    acc_node[lab][j], dtype=torch.float64, device="cuda"
                                ),
                            )
                            .cpu()
                            .tolist()
                            for j in range(L)
                        ]
                        evv = (
                            gather_rows(
                                ctx.group,
                                torch.tensor(
                                    ev[lab], dtype=torch.float64, device="cuda"
                                ),
                            )
                            .cpu()
                            .tolist()
                        )
                        row["samples_us"] = allv
                        row["stat"] = summarize_samples(allv, a.windows)
                        row["node_stat"] = [
                            summarize_samples(node_all[j], a.windows) for j in range(L)
                        ]
                        row["replay_per_layer_stat"] = summarize_samples(
                            [[v / L for v in r] for r in evv], 1
                        )
                        ctx.log(
                            f"[graphlayers DEP{ctx.R} {lab} S={S} M={pt.label} bm={bms[lab]} L={L}] per-launch median-of-max {row['stat']['median_of_max_us']:8.2f} us "
                            f"(nodes {', '.join(format(x['median_of_max_us'], '.1f') for x in row['node_stat'])}) | replay/L {row['replay_per_layer_stat']['median_of_max_us']:.1f} us | bit_equal={bit_equal}"
                        )
                    else:
                        ctx.log(
                            f"[graphlayers DEP{ctx.R} {lab} S={S} M={pt.label} L={L}] INVALID: {reason}"
                        )
                    ctx.write(row)
                del graphs, calls, outs
            del fns, ys, sts
        ctx.close_barrier()
        for x in ads:
            x.close()
        torch.cuda.empty_cache()
    ctx.watchdog.disarm()


def run_hostcost(ctx: Ctx) -> None:
    """Host (CPU) cost of one EAGER launch per impl (added with the L2_256B weight-map change: the port encodes 2-4 extra tensor maps on the host at every launch at BLOCK_M >= 128):
    wall time of ``launch_fn()`` per launch.  ``--hostcost-launches`` back-to-back launches are queued behind one long GPU sleep (the host never waits for the device), median over
    ``--hostcost-reps`` repetitions (impls alternate); all ranks time their own host loop at the same time, as in a real eager run.  Rows ``kind="hostcost"`` (per-rank medians)."""
    a = ctx.a
    prob = Problem(a, ctx.rank, ctx.R, ctx.t_user)
    ctx.write(
        {
            "schema": SCHEMA,
            "kind": "run",
            "mode": "hostcost",
            "args": vars(a),
            "env": ctx.env,
            "dep": ctx.R,
            "hosts": _gather_hosts(ctx),
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
    )
    n, reps = int(a.hostcost_launches), int(a.hostcost_reps)
    for S in [int(s) for s in str(a.num_shared_experts).split(",")]:
        ads = build_adapters(ctx, prob, S, None)
        points = resolve_points(ctx, ads[0])
        ctx.log(
            f"== HOSTCOST DEP{ctx.R} S={S} T_max={ctx.t_user}->{ads[0].t_aligned} impls={[x.label for x in ads]} points={[p.label for p in points]} launches={n} reps={reps}"
        )
        for pt in points:
            ctx.watchdog.arm(a.jit_timeout)
            mv, m, inp, _ = mpoint_inputs(ctx, prob, ads[0], pt)
            fns: Dict[
                str, Tuple[Callable[[], Any], torch.Tensor, Optional[torch.Tensor]]
            ] = {}
            bms: Dict[str, int] = {}
            for ad in ads:
                bms[ad.label] = ad.block_m(m)
                ad.stage(inp)
                y = torch.empty((m, a.hidden), dtype=torch.bfloat16, device="cuda")
                stats = prob.stats_init.clone() if not a.no_stats else None
                fn = ad.launch_fn(y, stats)
                fn()  # untimed first launch (JIT, lazy init)
                torch.cuda.synchronize()
                fns[ad.label] = (fn, y, stats)
            ctx.watchdog.arm()
            labs = list(fns)
            res: Dict[str, List[float]] = {lab: [] for lab in labs}
            for rep in range(reps):
                for lab in labs if rep % 2 == 0 else labs[::-1]:
                    fn = fns[lab][0]
                    torch.cuda.synchronize()
                    ctx.barrier()  # stream-ordered: lines the ranks' GPUs up before the sleep kernel
                    torch.cuda._sleep(int(a.hostcost_sleep_cycles))
                    t0 = time.perf_counter_ns()
                    for _ in range(n):
                        fn()
                    t1 = time.perf_counter_ns()
                    torch.cuda.synchronize()
                    res[lab].append((t1 - t0) / n / 1e3)
            for lab in labs:
                loc = torch.tensor(
                    [statistics.median(res[lab]), min(res[lab]), max(res[lab])],
                    dtype=torch.float64,
                    device="cuda",
                )
                g = gather_rows(ctx.group, loc).cpu().tolist()
                row = {
                    "schema": SCHEMA,
                    "kind": "hostcost",
                    "impl": lab,
                    "S": S,
                    "M_label": pt.label,
                    "M": mv[0],
                    "m_vec": mv,
                    "block_m": bms[lab],
                    "dep": ctx.R,
                    "t_max_user": ctx.t_user,
                    "n_launches": n,
                    "reps": reps,
                    "perdie": a.perdie,
                    "us_per_launch_median_rank": [r[0] for r in g],
                    "us_per_launch_min_rank": [r[1] for r in g],
                    "us_per_launch_max_rank": [r[2] for r in g],
                    "us_per_launch_median": statistics.median(r[0] for r in g),
                    "reps_us_rank0": res[lab],
                    "job_id": ctx.env.get("job_id"),
                    "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                }
                ctx.write(row)
                ctx.log(
                    f"[hostcost DEP{ctx.R} {lab} S={S} M={pt.label} bm={bms[lab]}] {row['us_per_launch_median']:.1f} us/launch (median of rank medians; ranks "
                    f"{', '.join(f'{r[0]:.1f}' for r in g[:4])}{'...' if ctx.R > 4 else ''}; min {min(r[1] for r in g):.1f} max {max(r[2] for r in g):.1f})"
                )
            del fns
        ctx.watchdog.disarm()
        ctx.close_barrier()
        for x in ads:
            x.close()
        torch.cuda.empty_cache()


# ======================================================================================================================
# 7. Entry point
# ======================================================================================================================
def _install_port_pkg_dir(a: argparse.Namespace) -> None:
    """Put a PRIVATE snapshot of the port package (a directory CONTAINING ``sm107_deepgemm_megamoe/``) in front of the shared overlay
    (the other roles' sync scripts rewrite the overlay while campaigns run); recorded via ``port_tree_hash``."""
    d = getattr(a, "port_pkg_dir", "")
    if not d:
        return
    import flashinfer.experimental as _e  # noqa: PLC0415 - the image's flashinfer; the package dir is added to its __path__

    if d not in _e.__path__:
        _e.__path__.insert(0, d)


def worker(local_rank: int, nlocal: int, a: argparse.Namespace) -> None:
    _install_port_pkg_dir(a)
    ctx = Ctx(a, local_rank, nlocal)
    try:
        {
            "bench": run_bench,
            "configs": run_configs,
            "check": run_check,
            "gate0": run_gate0,
            "stress": run_stress,
            "graphrace": run_graphrace,
            "hostcost": run_hostcost,
            "graphlayers": run_graphlayers,
        }[a.mode](ctx)
        torch.cuda.synchronize()
        dist.barrier()
    finally:
        ctx.watchdog.disarm()
    dist.destroy_process_group()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--mode",
        choices=(
            "bench",
            "configs",
            "check",
            "gate0",
            "stress",
            "graphrace",
            "hostcost",
            "graphlayers",
        ),
        default="bench",
    )
    p.add_argument(
        "--impl",
        default="deepgemm",
        help="deepgemm | deepgemm2 | port | fi_existing | both (=deepgemm,port) | comma list",
    )
    p.add_argument(
        "--variant",
        default="auto",
        help="label stored in every row; 'auto' = B0 (S=1) / B0n (S=0) [B1/B1n with --perdie, B2/B2n with --sm107 0]",
    )
    p.add_argument("--num-processes", type=int, default=4)
    # shape (defaults = DeepSeek-V4.1-Flash)
    p.add_argument("--hidden", type=int, default=5120)
    p.add_argument("--intermediate-hidden", type=int, default=2304)
    p.add_argument("--num-experts", type=int, default=384)
    p.add_argument("--num-topk", type=int, default=6)
    p.add_argument(
        "--num-shared-experts",
        default="1",
        help="S, or a comma list (e.g. 1,0) swept in that order",
    )
    p.add_argument("--activation", default="swiglu")
    p.add_argument("--activation-clamp", type=float, default=10.0)
    p.add_argument("--activation-alpha", type=float, default=1.0)
    p.add_argument("--activation-beta", type=float, default=0.0)
    p.add_argument("--fast-math", type=int, default=1)
    p.add_argument("--mma-type", default="fp8xfp4", choices=("fp8xfp4",))
    p.add_argument(
        "--num-max-tokens-per-rank",
        type=int,
        default=8192,
        help="user T_max (DeepGEMM aligns up to 1920)",
    )
    # points
    p.add_argument(
        "--m-list",
        default="",
        help="comma list of M | presets core,boundary,serving,regimes,perf,regmax (largest M per BLOCK_M),regmaxall (largest M of every BLOCK_M segment) | a/b/c/d per-rank vector | randR",
    )
    p.add_argument("--m-preset", default="core")
    p.add_argument(
        "--passes",
        type=int,
        default=1,
        help="repeat the whole M list (independent windows; sigma per point)",
    )
    # data
    p.add_argument(
        "--routing",
        choices=("gaussian", "power_law", "hot", "hot_spread"),
        default="gaussian",
        help="hot = every token to experts 0..topk-1; hot_spread = one hot expert per rank",
    )
    p.add_argument("--routing-alpha", type=float, default=1.0)
    p.add_argument("--masked-ratio", type=float, default=0.0)
    p.add_argument(
        "--masked-ranks",
        default="",
        help="comma list of global ranks whose routes are ALL masked (topk_idx = -1 for every token), e.g. 3",
    )
    p.add_argument("--weight-init", choices=("cast", "fast"), default="cast")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--no-stats",
        action="store_true",
        help="pass cumulative_local_expert_recv_stats=None",
    )
    # timing
    p.add_argument("--num-tests", type=int, default=30)
    p.add_argument(
        "--windows",
        type=int,
        default=1,
        help="independent profiler windows per point (samples concatenated)",
    )
    p.add_argument("--exec", choices=("eager", "graph", "both"), default="eager")
    p.add_argument("--pdl", type=int, default=0, choices=(0, 1))
    p.add_argument(
        "--barrier",
        choices=("auto", "symm", "deepep", "allreduce"),
        default="auto",
        help="cross-rank barrier inside the timed loops (auto = symmetric-memory handle barrier if available)",
    )
    p.add_argument(
        "--ab-swap",
        action="store_true",
        help="with 2+ impls: run the whole sweep twice, allocating the kernels' buffers/weights in swapped order "
        "(averages out the allocation-order placement effect, ~3%% at tiny M)",
    )
    p.add_argument("--no-flush", action="store_true")
    p.add_argument("--no-event-check", action="store_true")
    p.add_argument("--no-config-capture", dest="capture_config", action="store_false")
    p.add_argument("--trace-dir", default="")
    # DeepGEMM env-level knobs recorded in rows
    p.add_argument(
        "--sm107", default=None, help="DG_MEGA_MOE_SM107 (default: keep env, else 1)"
    )
    p.add_argument(
        "--perdie",
        default="0",
        choices=("0", "routed", "all"),
        help="B1: DG_MEGA_MOE_SM107_PERDIE=1 + SM107 localizer",
    )
    p.add_argument(
        "--arch",
        default="",
        help="DG_MEGA_MOE_SM107_ARCH for the DeepGEMM JIT (e.g. 107a: arch-specific SASS); default 100f",
    )
    p.add_argument("--kernel-filter-dg", default="")
    p.add_argument("--kernel-filter-port", default="")
    p.add_argument("--port-module", default=PORT_MODULE_DEFAULT)
    p.add_argument(
        "--port-pkg-dir",
        default="",
        help="private snapshot dir containing sm107_deepgemm_megamoe/ (inserted into flashinfer.experimental.__path__)",
    )
    p.add_argument(
        "--port-tag-mode",
        default="",
        choices=("", "gridid", "epoch"),
        help="tag_mode of the port's combine-readiness handshake (jit.get_or_compile_kernel; default gridid = DeepGEMM, epoch = unique under CUDA-graph replay)",
    )
    p.add_argument(
        "--port-regs",
        default="",
        help="setmaxnreg split of the port kernel 'dispatch,non_epilogue,epilogue' (regs_override), e.g. 56,40,200; default = the DeepGEMM split",
    )
    p.add_argument(
        "--port-faithful-regs",
        action="store_true",
        help="faithful_regs=True: the port uses DeepGEMM's own setmaxnreg split (96/88/160 at experts-per-rank > 64, else 48/40/208) instead of its product default 96/88/160 at every EP size",
    )
    # gate0
    p.add_argument("--gate0-m-list", default="64,256,512,1024,2048,3072,4096")
    p.add_argument("--gate0-launches", type=int, default=200)
    p.add_argument("--gate0-perm-launches", type=int, default=20)
    p.add_argument("--gate0-masked", default="0.0,0.1")
    # graphrace: CUDA-graph replay vs the %gridid combine-readiness tag
    p.add_argument("--graphrace-m-list", default="64,1024")
    p.add_argument("--graphrace-replays", type=int, default=2000)
    p.add_argument(
        "--graphrace-sets",
        type=int,
        default=4,
        help="distinct token orders staged between replays (the data must change, else stale == correct)",
    )
    p.add_argument(
        "--graphrace-control",
        type=int,
        default=1,
        help="also run the same loop with direct (eager) launches",
    )
    # stress (hang hunting): back-to-back launches over an M sequence on one buffer
    p.add_argument(
        "--stress-launches",
        type=int,
        default=20000,
        help="launches per S on one buffer",
    )
    p.add_argument(
        "--stress-m-list",
        default="8192,1,4096,2,3072,605,1024,64,2048,1280,128,16",
        help="M cycle (entries > T_max are dropped)",
    )
    p.add_argument(
        "--stress-per-m",
        type=int,
        default=100,
        help="launches per segment (one M); a host sync + watchdog re-arm per segment",
    )
    p.add_argument(
        "--stress-barrier",
        choices=("none", "symm", "allreduce"),
        default="none",
        help="barrier before every launch (none = launches back to back)",
    )
    p.add_argument(
        "--stress-flush",
        action="store_true",
        help="256 MB L2 flush before every launch (bench protocol)",
    )
    p.add_argument(
        "--stress-sleep-cycles",
        type=int,
        default=0,
        help="torch.cuda._sleep cycles before every launch (bench protocol: 20000000)",
    )
    p.add_argument(
        "--stress-skew-cycles",
        type=int,
        default=0,
        help="random per-rank extra delay (cycles, uniform) before every launch",
    )
    p.add_argument(
        "--stress-shuffle",
        action="store_true",
        help="permute the M cycle every round (same permutation on all ranks)",
    )
    p.add_argument(
        "--stress-log-every",
        type=int,
        default=50,
        help="progress line / row every this many segments",
    )
    # graphlayers: L back-to-back launches in ONE CUDA graph
    p.add_argument(
        "--graph-layers-list",
        default="1,2,4",
        help="graphlayers mode: numbers of MoE launches captured back to back on the same workspace in one CUDA graph",
    )
    # hostcost: CPU time of one eager launch per impl
    p.add_argument(
        "--hostcost-launches",
        type=int,
        default=100,
        help="back-to-back launches per timed repetition (queued behind a GPU sleep)",
    )
    p.add_argument(
        "--hostcost-reps",
        type=int,
        default=9,
        help="repetitions per (impl, M); the median is reported",
    )
    p.add_argument(
        "--hostcost-sleep-cycles",
        type=int,
        default=400_000_000,
        help="torch.cuda._sleep cycles issued before each timed loop (~0.17 s at 2.4 GHz: longer than the host loop)",
    )
    # output / safety
    p.add_argument("--out", default="")
    p.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="watchdog per point (s): a hung persistent kernel is killed with a stack dump",
    )
    p.add_argument(
        "--jit-timeout",
        type=float,
        default=1500.0,
        help="watchdog for first launches (JIT) (s)",
    )
    p.add_argument(
        "--hang-dump-dir",
        default="",
        help="on a watchdog timeout write the head of every symmetric buffer + context of each rank here",
    )
    p.add_argument(
        "--hang-dump-bytes",
        type=int,
        default=48_000_000,
        help="bytes of each symmetric buffer to dump (MegaMoESignals = 21,140,864 B)",
    )
    p.add_argument(
        "--post-dump-wait",
        type=float,
        default=0.0,
        help="seconds to keep a hung process alive after the state dump (DeepGEMM's kernel-side timeouts fire after ~300 s and printf their location)",
    )
    p.add_argument(
        "--inject-hang",
        type=int,
        default=-1,
        help="TEST ONLY (gate0): rank 1 skips the launch with this index, the other ranks hang",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="skip points already in --out; skip (and mark kind=hang) points started but not finished",
    )
    return p


def main(argv: Optional[Sequence[str]] = None) -> None:
    a = build_parser().parse_args(argv)
    if a.sm107 is not None:
        os.environ["DG_MEGA_MOE_SM107"] = a.sm107
    os.environ.setdefault("DG_MEGA_MOE_SM107", "1")
    if a.arch:
        os.environ["DG_MEGA_MOE_SM107_ARCH"] = a.arch
    if a.perdie != "0":
        os.environ["DG_MEGA_MOE_SM107_PERDIE"] = "1"
        assert os.environ["DG_MEGA_MOE_SM107"] == "1", "per-die needs the SM107 kernel"
    else:
        os.environ.pop("DG_MEGA_MOE_SM107_PERDIE", None)
    os.environ.pop(
        "DG_PRINT_CONFIGS", None
    )  # never on in timed runs; toggled only around single untimed launches
    if not a.out:
        a.out = f"mega_ab_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
    torch.multiprocessing.spawn(
        worker, args=(a.num_processes, a), nprocs=a.num_processes
    )


if __name__ == "__main__":
    main()
