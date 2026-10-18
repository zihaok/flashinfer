# SPDX-License-Identifier: Apache-2.0
# Port of DeepGEMM (MIT License, Copyright (c) 2025 DeepSeek; see licenses/LICENSE.deepgemm.txt) sm107_fp8_fp4_mega_moe.cuh / moe_ep layer tests.
"""The SM107 mega kernel behind ``MoEEpLayer`` (Tier 2) with the REAL kernel, EP1, compared bit for bit with FlashInfer's ``sm100_fp8_fp4_bf16_deepgemm`` backend
(stock DeepGEMM, ``DG_MEGA_MOE_SM107=1`` selects its Rubin kernel) on the same canonical ``MoEWeightPack``.

Every other backend test replaces the kernel by a recorder, so staging (BF16 -> MXFP8, top-k dtype conversion), the output slicing of caller-provided buffers,
``layer.warmup()`` and CUDA-graph capture / replay through the layer had no shipped end-to-end check.  Here a layer is built from the config class resolved
lazily by ``flashinfer.moe_ep`` (the backend module is loaded by the layer, not by the test), and:

* plain routing at several token counts (every BLOCK_M family; the token count goes up AND down in one workspace), bit-exact with the stock-DeepGEMM backend;
* masked routing (top-k slots of -1 / weight 0, fully masked rows), bit-exact with the stock-DeepGEMM backend;
* a caller-provided ``output`` larger than the live token count: the live rows are written, the other rows are untouched;
* ``layer.warmup()`` + CUDA-graph capture of ``layer.forward`` + replays with changed inputs, bit-exact with the eager results;
* a layer that was never warmed up refuses a capture with the clear error of ``validate_capture_ready``;
* (third test) ``layer.warmup()`` with ONE token (what vLLM's adapter does) leaves every BLOCK_M variant of the layer's geometry built: a CUDA-graph capture at a token
  count of another BLOCK_M family needs no compilation inside the capture;
* (second test) the same with die-local routed weights (``per_die=True``): bit-identical to the plain layer; skipped on a device without two locality domains.

The scenario runs in a fresh interpreter (``python <this file> --worker ...``): it needs an NCCL default process group, which the other tests of a pytest session
(gloo EP1 groups) would have claimed first.  The worker sets the single-process ``torch.distributed`` environment itself, at run time and only in its own process.
Needs one SM107 GPU with 212 SMs and the ``deep_gemm`` package.
"""

import importlib.util
import os
import pathlib
import subprocess
import sys

import pytest

E, H, I, TOPK, T_MAX = (
    32,
    5120,
    2304,
    6,
    1920,
)  # DeepSeek-V4.1-Flash shape (a geometry of heuristics.VALIDATED_GEOMETRIES), 32 experts at EP1
MS = (
    1,
    64,
    300,
    1900,
    64,
    1,
)  # BLOCK_M families of the heuristic; the token count also goes back down
MASKED_M = 300
GRAPH_M, GRAPH_REPLAYS = 64, 6
SKIP_EXIT_CODE = 77
PASS_LINE = "MOE_EP_LAYER_EP1 PASS"
BACKEND_MODULE = "flashinfer.experimental.sm107_deepgemm_megamoe.backend"


def _gate():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
        pytest.skip("needs a compute capability 10.7 (Rubin) GPU")
    if (
        torch.cuda.get_device_properties(
            torch.cuda.current_device()
        ).multi_processor_count
        != 212
    ):
        pytest.skip("the SM107 MegaMoE kernel is built for a 212-SM device")
    if importlib.util.find_spec("deep_gemm") is None:
        pytest.skip("needs the deep_gemm package (the reference backend)")
    spec = importlib.util.find_spec("flashinfer.moe_ep")
    if (
        spec is None
        or not spec.origin
        or "Sm107_Mxfp8_Mxfp4_Bf16_DeepgemmCutedsl_MegaMoeConfig"
        not in pathlib.Path(spec.origin).read_text(encoding="utf-8")
    ):
        pytest.skip(
            "needs the lazy-registry core patch of flashinfer.moe_ep (applied by the PR cut; the port tree and the snapshot overlay have no core files)"
        )


def _run_worker(*flags):
    repo_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )  # the tree this test belongs to: the child must import ITS flashinfer
    env = dict(
        os.environ, DG_MEGA_MOE_SM107="1", PYTHONDONTWRITEBYTECODE="1"
    )  # the child's environment only
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (repo_root, os.environ.get("PYTHONPATH")) if p
    )
    proc = subprocess.run(
        [sys.executable, "-W", "ignore", os.path.abspath(__file__), "--worker", *flags],
        capture_output=True,
        text=True,
        env=env,
        timeout=1800,
    )
    if proc.returncode == SKIP_EXIT_CODE:
        pytest.skip(proc.stdout.strip().splitlines()[-1])
    assert proc.returncode == 0 and PASS_LINE in proc.stdout, (
        f"worker failed (rc={proc.returncode})\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}"
    )
    return proc.stdout


def test_moe_ep_layer_matches_stock_deepgemm_backend_ep1():
    _gate()
    out = _run_worker()
    assert "plain: all token counts bit-exact" in out and "masked: bit-exact" in out
    assert "caller output: live rows written, other rows untouched" in out
    assert "graph: 0/" in out and "capture before warmup refused" in out


def test_moe_ep_layer_per_die_matches_plain_layer_ep1():
    _gate()
    out = _run_worker("--per-die-only")
    assert (
        "per-die: bit-identical to the plain layer" in out
        and "graph per-die: 0/" in out
    )


def test_moe_ep_layer_one_token_warmup_builds_every_block_m_variant_ep1():
    _gate()
    out = _run_worker("--warmup-only")
    assert "warmup: capture at another BLOCK_M family works" in out


# ----------------------------------------------------------------------------------------------------------------------------------------------------
# worker (runs in the child process)
# ----------------------------------------------------------------------------------------------------------------------------------------------------
def _worker(per_die_only: bool, warmup_only: bool = False) -> None:
    import socket
    import zlib

    import torch
    import torch.distributed as dist

    def log(*a):
        print(*a, flush=True)

    def same(a, b):
        return a.shape == b.shape and torch.equal(
            a.contiguous().view(torch.int16), b.contiguous().view(torch.int16)
        )

    if (
        "RANK" not in os.environ
    ):  # single-process torch.distributed environment (the layer bootstraps NCCL from it)
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        os.environ.update(
            RANK="0",
            WORLD_SIZE="1",
            LOCAL_RANK="0",
            MASTER_ADDR="127.0.0.1",
            MASTER_PORT=str(port),
        )
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)

    import flashinfer.moe_ep as moe_ep

    assert BACKEND_MODULE not in sys.modules, (
        "importing flashinfer.moe_ep must not import the backend"
    )
    config_cls = (
        moe_ep.Sm107_Mxfp8_Mxfp4_Bf16_DeepgemmCutedsl_MegaMoeConfig
    )  # resolved lazily (PEP 562)
    assert BACKEND_MODULE not in sys.modules, (
        "resolving the config must not import the backend"
    )
    from flashinfer.moe_ep import (
        BootstrapConfig,
        FleetParams,
        MegaConfig,
        MoEEpLayer,
        MoEEpTensors,
        MoEWeightPack,
        Sm100_Fp8_Fp4_Bf16_Deepgemm_MegaMoeConfig,
        ensure_moe_ep_cuda_device,
    )

    bootstrap = BootstrapConfig(world_size=1, rank=0)
    ensure_moe_ep_cuda_device(bootstrap)
    fleet = FleetParams(num_experts=E, max_tokens_per_rank=T_MAX, token_hidden_size=H)

    if per_die_only:
        from flashinfer.experimental.sm107_deepgemm_megamoe import locality

        if not locality.is_localization_available():
            log(
                "SKIP: the device does not report two locality domains (die-local weights unavailable)"
            )
            sys.exit(SKIP_EXIT_CODE)

    g = torch.Generator(device=device).manual_seed(17)
    w13 = torch.randint(
        0, 256, (E, 2 * I, H // 2), dtype=torch.uint8, device=device, generator=g
    )  # packed MXFP4
    w2 = torch.randint(
        0, 256, (E, H, I // 2), dtype=torch.uint8, device=device, generator=g
    )
    s13 = torch.randint(
        116, 123, (E, 2 * I, H // 32), dtype=torch.uint8, device=device, generator=g
    )  # UE8M0 scales 2^-11 .. 2^-5
    s2 = torch.randint(
        116, 123, (E, H, I // 32), dtype=torch.uint8, device=device, generator=g
    )

    def make_layer(megakernel, fleet_params=None):
        pack = MoEWeightPack(w13=w13, w2=w2, w13_scale=s13, w2_scale=s2)
        return MoEEpLayer(
            bootstrap=bootstrap,
            fleet_params=fleet_params or fleet,
            weights=pack,
            backend=MegaConfig(
                megakernel=megakernel, quantize_input=True, preprocess_weights=True
            ),
        )

    port_cfg = config_cls(intermediate_size=I, top_k=TOPK, gate_up_clamp=10.0)
    port = make_layer(port_cfg)
    assert BACKEND_MODULE in sys.modules, "MoEEpLayer must load the backend by name"
    layers = {"port": port}
    if warmup_only:
        _warmup_scenario(
            port, make_layer, port_cfg, FleetParams, MoEEpTensors, device, E, H
        )
        port.destroy()
        dist.destroy_process_group()
        log(PASS_LINE)
        return
    if per_die_only:
        layers["perdie"] = make_layer(
            config_cls(
                intermediate_size=I, top_k=TOPK, gate_up_clamp=10.0, per_die=True
            )
        )
    else:
        layers["dgbe"] = make_layer(
            Sm100_Fp8_Fp4_Bf16_Deepgemm_MegaMoeConfig(
                intermediate_size=I, top_k=TOPK, activation_clamp=10.0, fast_math=True
            )
        )
    torch.cuda.synchronize()
    log("layers:", {k: v._kernel.kernel_name() for k, v in layers.items()})

    def inputs(m, tag, masked=False):
        gi = torch.Generator(device=device).manual_seed(
            zlib.crc32(f"{m}-{tag}".encode())
        )
        hs = torch.randn((m, H), dtype=torch.bfloat16, device=device, generator=gi)
        tw, ti = torch.topk(
            torch.randn((m, E), device=device, generator=gi), TOPK, dim=-1
        )
        ids, wts = ti.to(torch.int64), tw.softmax(-1).float()
        if masked:  # masked top-k slots (-1 / weight 0), two leading and the last row fully masked
            drop = torch.rand((m, TOPK), device=device, generator=gi) < 0.3
            drop[:2] = True
            drop[-1] = True
            ids = ids.masked_fill(drop, -1)
            wts = wts.masked_fill(drop, 0.0)
        return MoEEpTensors(hidden_states=hs, topk_ids=ids, topk_weights=wts)

    other = "dgbe" if "dgbe" in layers else "perdie"

    # ---- plain routing, every token count (the count also goes back down in the same workspace)
    for m in MS:
        t = inputs(m, "plain")
        y_port = layers["port"].forward(t).clone()
        y_other = layers[other].forward(t).clone()
        torch.cuda.synchronize()
        assert y_port.shape == (m, H) and bool(torch.isfinite(y_port.float()).all()), (
            f"M={m}: port output shape / finiteness"
        )
        assert same(y_port, y_other), (
            f"M={m}: {'port' if other == 'dgbe' else 'perdie'} differs from {other}"
        )
    log(
        "per-die: bit-identical to the plain layer"
        if per_die_only
        else "plain: all token counts bit-exact with the stock-DeepGEMM backend",
        MS,
    )

    # ---- masked routing (not for the per-die run: the plain run already covers the staging)
    if not per_die_only:
        t = inputs(MASKED_M, "masked", masked=True)
        y_port = layers["port"].forward(t).clone()
        y_dg = layers["dgbe"].forward(t).clone()
        torch.cuda.synchronize()
        assert bool(torch.isfinite(y_port.float()).all())
        assert same(y_port, y_dg), (
            "masked routing: port differs from the stock-DeepGEMM backend"
        )
        log("masked: bit-exact with the stock-DeepGEMM backend")

        # ---- caller-provided output larger than the live token count
        t = inputs(MASKED_M, "plain")
        eager = layers["port"].forward(t).clone()
        out = torch.full(
            (MASKED_M + 7, H), float("nan"), dtype=torch.bfloat16, device=device
        )
        y = layers["port"].forward(
            MoEEpTensors(
                hidden_states=t.hidden_states,
                topk_ids=t.topk_ids,
                topk_weights=t.topk_weights,
                output=out,
            )
        )
        torch.cuda.synchronize()
        assert y.data_ptr() == out.data_ptr(), (
            "the layer must return the caller's output tensor"
        )
        assert same(out[:MASKED_M], eager), (
            "caller output: live rows differ from the owned-output forward"
        )
        assert bool(torch.isnan(out[MASKED_M:].float()).all()), (
            "caller output: rows beyond the live token count were written"
        )
        log("caller output: live rows written, other rows untouched")

    # ---- warmup + CUDA graph capture / replay of layer.forward
    layer = layers["port"] if not per_die_only else layers["perdie"]
    sets = [inputs(GRAPH_M, f"graph{i}") for i in range(GRAPH_REPLAYS)]
    refs = [layer.forward(t).clone() for t in sets]  # eager results
    t_static = MoEEpTensors(
        hidden_states=sets[0].hidden_states.clone(),
        topk_ids=sets[0].topk_ids.clone(),
        topk_weights=sets[0].topk_weights.clone(),
    )
    layer.warmup(t_static)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        y_view = layer.forward(t_static, return_workspace_view=True)
    torch.cuda.synchronize()
    bad = []
    for i, t in enumerate(sets):
        t_static.hidden_states.copy_(t.hidden_states)
        t_static.topk_ids.copy_(t.topk_ids)
        t_static.topk_weights.copy_(t.topk_weights)
        graph.replay()
        torch.cuda.synchronize()
        if not same(y_view, refs[i]):
            bad.append(i)
    log(
        f"graph{' per-die' if per_die_only else ''}: {len(bad)}/{GRAPH_REPLAYS} replays differ from the eager results"
    )
    assert not bad, f"replays {bad} differ from the eager results"
    del graph

    # ---- a layer that never launched refuses a capture (use a geometry no warmed layer shares: the pooled workspace and the built variants are shared otherwise)
    if not per_die_only:
        fresh = make_layer(
            port_cfg,
            FleetParams(
                num_experts=E, max_tokens_per_rank=2 * T_MAX, token_hidden_size=H
            ),
        )
        fresh._ensure_workspace()
        t7 = inputs(7, "neg")
        message = "no error"
        try:
            g2 = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g2):
                fresh.forward(t7, return_workspace_view=True)
        except Exception as exc:  # noqa: BLE001
            message = f"{type(exc).__name__}: {exc}"
        torch.cuda.synchronize()
        assert "not warmed for CUDA graph capture" in message, message
        log("capture before warmup refused:", message[:120])
        layers["fresh"] = fresh

    for lyr in layers.values():
        lyr.destroy()
    dist.destroy_process_group()
    log(PASS_LINE)


def _warmup_scenario(
    port,
    make_layer,
    port_cfg,
    fleet_params_cls,
    tensors_cls,
    device,
    num_experts,
    hidden,
):
    """A one-token ``layer.warmup()`` followed by a capture at a token count of another BLOCK_M family (the vLLM adapter's warm-up and capture sizes)."""
    import torch

    from flashinfer.experimental.sm107_deepgemm_megamoe import heuristics, jit

    t_max = (
        2 * T_MAX
    )  # a geometry of its own: nothing is built or launched for it yet (the layer `port` of the caller was only constructed)
    layer = make_layer(
        port_cfg,
        fleet_params_cls(
            num_experts=num_experts, max_tokens_per_rank=t_max, token_hidden_size=hidden
        ),
    )
    big = 1900
    bm_small = heuristics.get_block_m_for_mega_moe(
        1, num_experts, t_max, 1, TOPK, "fp8xfp4"
    )
    bm_big = heuristics.get_block_m_for_mega_moe(
        1, num_experts, t_max, big, TOPK, "fp8xfp4"
    )
    assert bm_small != bm_big, "the test needs two BLOCK_M families"

    def inputs(m, seed):
        gi = torch.Generator(device=device).manual_seed(seed)
        hs = torch.randn((m, hidden), dtype=torch.bfloat16, device=device, generator=gi)
        tw, ti = torch.topk(
            torch.randn((m, num_experts), device=device, generator=gi), TOPK, dim=-1
        )
        return tensors_cls(
            hidden_states=hs,
            topk_ids=ti.to(torch.int64),
            topk_weights=tw.softmax(-1).float(),
        )

    layer.warmup(
        inputs(1, 1)
    )  # ONE token: the BLOCK_M family of M = 1 only is launched
    torch.cuda.synchronize()
    names = jit.compiled_variant_names()
    assert any(f"bm{bm_big}" in n for n in names), (
        f"after a one-token warmup the BLOCK_M={bm_big} variant is not built (built: {sorted(names)})"
    )
    t_static = inputs(big, 2)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(
        graph
    ):  # raises "compiling kernel ... cannot run during CUDA graph capture" when the variant was not built
        y_view = layer.forward(t_static, return_workspace_view=True)
    torch.cuda.synchronize()
    graph.replay()
    torch.cuda.synchronize()
    replayed = y_view.clone()
    eager = layer.forward(t_static).clone()
    torch.cuda.synchronize()
    assert replayed.shape == eager.shape and torch.equal(
        replayed.contiguous().view(torch.int16), eager.contiguous().view(torch.int16)
    ), "replay differs from the eager result"
    print(
        "warmup: capture at another BLOCK_M family works",
        {"BLOCK_M of the warmup": bm_small, "BLOCK_M of the capture": bm_big},
        flush=True,
    )
    del graph
    layer.destroy()


if __name__ == "__main__":
    if "--worker" in sys.argv:
        _worker(
            per_die_only="--per-die-only" in sys.argv,
            warmup_only="--warmup-only" in sys.argv,
        )
