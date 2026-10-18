# SPDX-License-Identifier: Apache-2.0
# Port of DeepGEMM (MIT License, Copyright (c) 2025 DeepSeek; see licenses/LICENSE.deepgemm.txt) sm107_fp8_fp4_mega_moe.cuh combine-readiness handshake (K:445-470, K:1690-1730).
"""CUDA-graph replay of the combine-readiness handshake (``%gridid`` tag vs the port's launch-unique ``epoch`` tag).

DeepGEMM tags the combine-ready flags with ``%gridid + 1`` of the RECEIVING rank's launch and never resets them.  ``%gridid`` is identical for every replay of one CUDA-graph kernel node, so when the previous
launch on a workspace was the same graph node (back-to-back replays of ONE graph) the flag already holds the expected value and the combine wait passes before the peers finished writing their L2 results: the combine reads
the PREVIOUS replay's rows.  Replays with identical inputs cannot see it (stale == fresh); inputs that change between replays (new tokens each step) do.  ``_sm107_dg_graph_hazard.py`` runs the adversarial protocol
(synchronize + barrier + synchronize before every input copy; eager DeepGEMM reference of the same input set; eager / identical-input / two-graph / replay+eager controls; production-like async enqueue; vLLM-like bucket graphs).

* ``epoch`` (``tag_mode="epoch"``, ``(nvl_barrier_counter >> 1) + 1``): must be bit-exact in every experiment (hard assertion).
* ``gridid`` (faithful DeepGEMM semantics): the hazard is documented; the test asserts the property we want (exact) and is ``xfail`` (non-strict) because the faithful tag is known to fail on back-to-back replays of one graph.

Run: ``pytest tests/experimental/test_sm107_dg_dispatch_combine_graph_replay.py`` (2 SM107 GPUs).
"""

from __future__ import annotations

import json
import os
import socket
import sys
import tempfile

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
NGPU = torch.cuda.device_count() if torch.cuda.is_available() else 0


def _is_sm107() -> bool:
    return (
        NGPU > 0
        and torch.cuda.get_device_capability(0) == (10, 7)
        and torch.cuda.get_device_properties(0).multi_processor_count == 212
    )


pytestmark = [
    pytest.mark.skipif(
        not _is_sm107() or NGPU < 2, reason="needs 2 SM107 GPUs (cc 10.7, 212 SMs)"
    )
]


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _run(kernels: str, exps: str, ms: str = "300/300", n: int = 120, ranks: int = 2):
    pytest.importorskip("deep_gemm")
    import _sm107_dg_graph_hazard

    with tempfile.TemporaryDirectory() as td:
        js = os.path.join(td, "res.json")
        env = dict(
            G_KERNELS=kernels,
            G_EXPS=exps,
            G_MS=ms,
            G_N=str(n),
            G_E="32",
            G_JSON=js,
            R_RANKS=str(ranks),
            MASTER_ADDR="127.0.0.1",
            MASTER_PORT=str(_free_port()),
            DG_MEGA_MOE_SM107="1",
            G_POISON=os.environ.get("G_POISON", "1"),
        )  # poison ON by default: every kernel-written region of the symmetric buffer(s) is filled with 0xFF before every launch / replay of both kernels (strict protocol)
        saved = {k: os.environ.get(k) for k in env}
        try:
            os.environ.update(env)
            torch.multiprocessing.spawn(
                _sm107_dg_graph_hazard.worker, args=(ranks,), nprocs=ranks, join=True
            )
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        with open(js) as f:
            return json.load(f)


def _wrong(results, kernel, names=None):
    return {
        (r["kernel"], r["exp"], r["skew"]): r["wrong"]
        for r in results
        if r["kernel"] == kernel and (names is None or r["exp"] in names)
    }


def _bad(cells):
    return {k: v for k, v in cells.items() if any(v)}


ALL_EXPS = "eager:0,graph:0,graph:500000,same:0,two:500000,mix:500000,async:0,buckets:0,twonode:500000,twobuf:500000"
POISONED = os.environ.get("G_POISON", "1") not in ("", "0")
# With the symmetric buffer poisoned before every replay (default), "identical inputs" no longer hides a stale-flag read: the combine of the faithful tag consumes the 0xFF-filled rows instead of the previous replay's identical
# rows, so `same` joins the hazard experiments for the faithful tag (the epoch tag stays exact in every experiment, poisoned or not).
CONTROL_EXPS = (
    {"eager", "two", "mix", "mix/eager", "twonode", "twonode/2"}
    | (set() if POISONED else {"same"})
)  # the previous launch on the workspace is never the same graph node: the faithful tag is exact here
HAZARD_EXPS = (
    {"graph", "async", "twobuf", "twobuf/2"} | ({"same"} if POISONED else set())
)  # the same graph node repeats on a workspace (with changing inputs, or with poisoned rows)


@pytest.fixture(scope="module")
def balanced():
    return _run(
        "port-gridid,port-epoch", ALL_EXPS
    )  # one spawn: both tag modes x every experiment, Ms = [300, 300]


@pytest.fixture(scope="module")
def imbalanced():
    return _run("port-epoch", "graph:0,async:0,buckets:0", ms="1500/200")


def test_epoch_tag_is_exact_under_graph_replay_with_changing_inputs(balanced):
    # one graph replayed with alternating inputs (strict + production-like async enqueue), vLLM-like buckets, replay+eager, two graphs, two nodes in one graph, two workspaces in one graph; skewed and not
    cells = _wrong(balanced, "port-epoch")
    assert cells and not _bad(cells), f"epoch tag produced stale results: {_bad(cells)}"


def test_epoch_tag_is_exact_with_imbalanced_ranks(imbalanced):
    cells = _wrong(imbalanced, "port-epoch")
    assert cells and not _bad(cells), f"epoch tag produced stale results: {_bad(cells)}"


def test_controls_are_exact_for_the_faithful_gridid_tag(balanced):
    # eager launches (unique %gridid), (unpoisoned only: identical inputs), two alternating graph nodes, replay + eager, two nodes sharing one workspace: the flag never equals the new tag
    cells = _wrong(balanced, "port-gridid", CONTROL_EXPS)
    assert cells and not _bad(cells), (
        f"controls failed (harness artifact?): {_bad(cells)}"
    )


@pytest.mark.xfail(
    strict=False,
    reason="faithful DeepGEMM tag: %gridid is identical for every replay of one graph node -> stale combine on back-to-back replays with changing inputs (documented, DeepGEMM stock has it too)",
)
def test_gridid_tag_is_exact_under_graph_replay_with_changing_inputs(balanced):
    cells = _wrong(balanced, "port-gridid", HAZARD_EXPS)
    assert cells and not _bad(cells), (
        f"gridid tag produced stale results (documented hazard): {_bad(cells)}"
    )
