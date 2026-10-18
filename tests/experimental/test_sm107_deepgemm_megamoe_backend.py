# SPDX-License-Identifier: Apache-2.0
# Port of DeepGEMM (MIT License, Copyright (c) 2025 DeepSeek; see licenses/LICENSE.deepgemm.txt) sm107_fp8_fp4_mega_moe.cuh / backend tests.
"""moe_ep (Tier 2) backend of the SM107 DeepGEMM-style MegaMoE: registration, config validation, weight preprocessing (bit-exact
against DeepGEMM's layout functions), workspace lifecycle and the stage -> compute path, with a stub kernel factory (no kernel is
compiled or launched; one Rubin GPU, EP1)."""

import warnings

import pytest

torch = pytest.importorskip("torch")

from flashinfer.experimental.sm107_deepgemm_megamoe import config as cfgmod  # noqa: E402

_RUBIN = torch.cuda.is_available() and torch.cuda.get_device_capability() == (10, 7)
pytestmark = [
    pytest.mark.arch_rubin,
    pytest.mark.skipif(not _RUBIN, reason="needs an SM107 (Rubin, cc 10.7) device"),
]

E, H, I, TOPK, T = 8, 1024, 512, 2, 1920


def _ep1_group():
    import atexit
    import socket

    import torch.distributed as dist

    if not dist.is_initialized():
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        dist.init_process_group(
            "gloo", init_method=f"tcp://127.0.0.1:{port}", rank=0, world_size=1
        )
        atexit.register(lambda: dist.is_initialized() and dist.destroy_process_group())
    return dist.new_group([0])


def _make_config(**kw):
    kw.setdefault(
        "allow_unvalidated_geometry", True
    )  # the tiny stub-kernel geometry of this file has never run on a GPU (see test_validate_init_admits_only_validated_geometries)
    return cfgmod.Sm107_Mxfp8_Mxfp4_Bf16_DeepgemmCutedsl_MegaMoeConfig(
        intermediate_size=I, top_k=TOPK, gate_up_clamp=10.0, **kw
    )


def test_config_validation_and_aliases():
    c = _make_config()
    assert (
        c.kernel_name
        == cfgmod.MOE_EP_KERNEL_NAME
        == "sm107_mxfp8_mxfp4_bf16_deepgemm_cutedsl"
        and c.resolved_clamp == 10.0
    )
    assert (
        cfgmod.Sm107_Mxfp8_Mxfp4_Bf16_DeepgemmCutedsl_MegaMoeConfig(
            intermediate_size=I, top_k=1, activation_clamp=6.0
        ).resolved_clamp
        == 6.0
    )
    for bad in (dict(activation="situ"), dict(combine_dtype="nvfp4"), dict(knobs={"a": 1}), dict(num_shared_experts=1), dict(apply_topk_in_fc1=False),
                dict(block_m_override=96), dict(activation_clamp=7.0), dict(gate_up_clamp=-1.0, activation_clamp=None)):  # fmt: skip
        kw = dict(intermediate_size=I, top_k=TOPK, gate_up_clamp=10.0)
        kw.update(bad)
        with pytest.raises(ValueError):
            cfgmod.Sm107_Mxfp8_Mxfp4_Bf16_DeepgemmCutedsl_MegaMoeConfig(**kw)
    with pytest.raises(ValueError):
        cfgmod.Sm107_Mxfp8_Mxfp4_Bf16_DeepgemmCutedsl_MegaMoeConfig(
            intermediate_size=100, top_k=1
        )
    assert (
        _make_config(per_die=True).per_die is True and _make_config().per_die is False
    )  # per-die (routed weights die-local) is accepted


def test_registration_by_name_after_import():
    from flashinfer.experimental.sm107_deepgemm_megamoe import backend
    from flashinfer.moe_ep.core.kernel.registry import (
        create_mega_kernel,
        is_mega_kernel_config,
    )

    cfg = _make_config()
    assert is_mega_kernel_config(cfg)
    kernel = create_mega_kernel(cfg)
    assert (
        isinstance(kernel, backend.Sm107DeepgemmCutedslMegaKernelBackend)
        and kernel.kernel_name() == cfg.kernel_name
    )
    assert kernel.supports_output_view is True


def _canonical_pack(gen, dtype_scale=torch.uint8):
    from flashinfer.moe_ep.weights import MoEWeightPack

    w13 = torch.randint(
        0, 256, (E, 2 * I, H // 2), generator=gen, dtype=torch.uint8
    ).cuda()
    w2 = torch.randint(0, 256, (E, H, I // 2), generator=gen, dtype=torch.uint8).cuda()
    s13 = torch.randint(
        100, 135, (E, 2 * I, H // 32), generator=gen, dtype=torch.uint8
    ).cuda()
    s2 = torch.randint(
        100, 135, (E, H, I // 32), generator=gen, dtype=torch.uint8
    ).cuda()
    if dtype_scale is not torch.uint8:
        s13, s2 = s13.view(dtype_scale), s2.view(dtype_scale)
    return MoEWeightPack(w13, w2, s13, s2)


def test_preprocess_weights_bit_exact_vs_deepgemm_layout():
    deep_gemm = pytest.importorskip("deep_gemm")
    from flashinfer.experimental.sm107_deepgemm_megamoe import backend

    gen = torch.Generator().manual_seed(7)
    for scale_dtype in (torch.uint8, torch.float8_e8m0fnu):
        pack = _canonical_pack(gen, scale_dtype)
        mine = backend.preprocess_canonical_weights(
            pack, intermediate_size=I, hidden_size=H
        )
        # reference: DeepGEMM's own layout functions on the same canonical data
        sf = lambda t, n, k: deep_gemm.transform_sf_into_required_layout(
            (t.view(torch.uint8).to(torch.int32) << 23).view(torch.float32),
            n,
            k,
            (1, 32),
            E,
        )  # noqa: E731
        l1 = (pack.w13.view(torch.int8), sf(pack.w13_scale, 2 * I, H))
        l2 = (pack.w2.view(torch.int8), sf(pack.w2_scale, H, I))
        ref = deep_gemm.transform_weights_for_mega_moe(l1, l2)
        for a, b in zip(mine, ref, strict=False):
            for x, y in zip(a, b, strict=False):
                assert (
                    x.dtype == y.dtype
                    and x.shape == y.shape
                    and x.stride() == y.stride()
                    and torch.equal(x, y)
                )
    from flashinfer.moe_ep.weights import MoEWeightPack
    from flashinfer.moe_ep.core.validation.common import MoEEpConfigError

    with pytest.raises(MoEEpConfigError):  # an unquantized pack is rejected
        backend.preprocess_canonical_weights(
            MoEWeightPack(
                torch.zeros(E, 2 * I, H, dtype=torch.bfloat16, device="cuda"),
                torch.zeros(E, H, I, dtype=torch.bfloat16, device="cuda"),
            ),
            intermediate_size=I,
            hidden_size=H,
        )
    with pytest.raises(MoEEpConfigError):  # wrong scale shape
        bad = _canonical_pack(gen)
        bad = MoEWeightPack(
            bad.w13, bad.w2, bad.w13_scale[:, :, :-1].contiguous(), bad.w2_scale
        )
        backend.preprocess_canonical_weights(bad, intermediate_size=I, hidden_size=H)


def test_backend_lifecycle_ep1_with_stub_kernel():
    from flashinfer.experimental.sm107_deepgemm_megamoe import backend, frontend
    from flashinfer.moe_ep.config import BootstrapConfig, FleetParams
    from flashinfer.moe_ep.core.validation.common import MoEEpConfigError
    from flashinfer.moe_ep.tensors import MoEEpTensors

    group = _ep1_group()
    cfg = _make_config()
    kernel = backend.Sm107DeepgemmCutedslMegaKernelBackend(cfg)
    bootstrap = BootstrapConfig(
        world_size=1,
        rank=0,
        auto_bootstrap=False,
        device=torch.cuda.current_device(),
        process_group=group,
    )
    fleet = FleetParams(num_experts=E, max_tokens_per_rank=T, token_hidden_size=H)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        kernel.bind_ep_bootstrap(bootstrap)
        kernel.validate_init(bootstrap, fleet)
        kernel.validate_init(bootstrap, fleet)  # warns once per (api, backend)
    assert sum("experimental" in str(x.message).lower() for x in w) == 1
    with pytest.raises(MoEEpConfigError):  # hidden size not a multiple of 128
        kernel.validate_init(
            bootstrap,
            FleetParams(num_experts=E, max_tokens_per_rank=T, token_hidden_size=1000),
        )

    gen = torch.Generator().manual_seed(1)
    transformed = kernel.preprocess_weights(_canonical_pack(gen), fleet)
    kernel.validate_transformed_weights(transformed, bootstrap, fleet)
    with pytest.raises(MoEEpConfigError):
        kernel.validate_transformed_weights(
            (transformed[0], (transformed[1][0].float(), transformed[1][1])),
            bootstrap,
            fleet,
        )

    calls = []
    frontend.set_kernel_factory(lambda c: (lambda *a: calls.append((c, a))))
    try:
        ws = kernel.prepare_workspace(
            bootstrap, fleet
        )  # pooled (the key captures geometry + EP identity)
        ws2 = kernel.prepare_workspace(bootstrap, fleet)
        assert ws is ws2  # one workspace is shared by layers of equal geometry
        m = 130
        t = MoEEpTensors(
            hidden_states=torch.randn((m, H), dtype=torch.bfloat16, device="cuda"),
            topk_ids=torch.randint(0, E, (m, TOPK), dtype=torch.int64, device="cuda"),
            topk_weights=torch.rand((m, TOPK), device="cuda"),
        )
        kernel.validate_forward(t, fleet, quantize_input=True)
        for bad_t, kw in ((MoEEpTensors(t.hidden_states.float(), t.topk_ids, t.topk_weights), {}),
                          (MoEEpTensors(t.hidden_states, t.topk_ids[:, :1], t.topk_weights[:, :1]), {}),
                          (t, dict(quantize_input=False))):  # fmt: skip
            with pytest.raises(MoEEpConfigError):
                kernel.validate_forward(
                    bad_t, fleet, **dict(dict(quantize_input=True), **kw)
                )
        kernel.set_compile_tokens_per_rank(ws, None)
        kernel.stage_inputs(t, ws, quantize_input=True)
        out_view = kernel.compute(ws, transformed, output=None)
        cfg_used, args = calls[-1]
        assert out_view.data_ptr() == kernel.workspace_output(
            ws
        ).data_ptr() and out_view.shape == (m, H)
        assert (
            cfg_used.num_experts == E
            and cfg_used.hidden == H
            and cfg_used.clamp == 10.0
            and cfg_used.block_m in (16, 32, 64, 128, 192, 240)
        )
        assert (
            args[0].shape == (m, H) and args[2] == m and args[3] is ws.buffer
        )  # (y, stats, num_tokens, sym_buffer, ...)
        assert args[4] is transformed[0][0] and args[7] is transformed[1][1]
        # staged data really sits in the symmetric buffer
        assert torch.equal(ws.buffer.topk_idx[:m], t.topk_ids)
        # caller-provided output
        user_out = torch.empty((m + 5, H), dtype=torch.bfloat16, device="cuda")
        assert (
            kernel.compute(ws, transformed, output=user_out) is user_out
            and calls[-1][1][0].data_ptr() == user_out.data_ptr()
        )
        assert calls[-1][1][0].shape == (m, H)  # the live rows only
        # precompile hook
        built = []
        frontend.set_kernel_factory(
            lambda c: built.append(c.block_m) or (lambda *a: None)
        )
        assert tuple(built) == () and kernel.precompile_variants(fleet) == tuple(built)
        kernel.destroy(ws)  # pooled: first release keeps the buffer alive
        assert ws.buffer is not None
        kernel.destroy(ws2)
        assert ws.buffer is None and ws.output is None
    finally:
        frontend.set_kernel_factory(None)


def test_backend_lifecycle_per_die_with_stub_kernel():
    """``per_die=True``: routed weights are die-local (``preprocess_weights``), validation expects them, the kernel variant is the per-die one
    and the launch carries the SM -> locality-domain map (``sm_locality_domains=``)."""
    from flashinfer.experimental.sm107_deepgemm_megamoe import (
        backend,
        frontend,
        locality,
    )
    from flashinfer.moe_ep.config import BootstrapConfig, FleetParams
    from flashinfer.moe_ep.core.validation.common import MoEEpConfigError
    from flashinfer.moe_ep.tensors import MoEEpTensors

    if not locality.is_localization_available():
        pytest.skip("needs a compute capability 10.7 device with 2 locality domains")
    group = _ep1_group()
    kernel = backend.Sm107DeepgemmCutedslMegaKernelBackend(_make_config(per_die=True))
    bootstrap = BootstrapConfig(
        world_size=1,
        rank=0,
        auto_bootstrap=False,
        device=torch.cuda.current_device(),
        process_group=group,
    )
    fleet = FleetParams(num_experts=E, max_tokens_per_rank=T, token_hidden_size=H)
    kernel.bind_ep_bootstrap(bootstrap)
    kernel.validate_init(bootstrap, fleet)
    gen = torch.Generator().manual_seed(11)
    pack = _canonical_pack(gen)
    transformed = kernel.preprocess_weights(pack, fleet)
    plain = backend.preprocess_canonical_weights(
        pack, intermediate_size=I, hidden_size=H
    )
    assert (
        locality.weights_localized(transformed[0])
        and locality.weights_localized(transformed[1])
        and not locality.weights_localized(plain[0])
    )
    assert tuple(transformed[0][0].shape) == (2, E, I, H // 2) and tuple(
        transformed[1][0].shape
    ) == (2, E, H // 2, I // 2)
    assert torch.equal(
        transformed[0][0].movedim(0, 1).flatten(1, 2), plain[0][0]
    ) and torch.equal(transformed[1][1], plain[1][1])
    kernel.validate_transformed_weights(transformed, bootstrap, fleet)
    with pytest.raises(MoEEpConfigError):  # plain weights are not die-local
        kernel.validate_transformed_weights(plain, bootstrap, fleet)
    with pytest.raises(
        MoEEpConfigError
    ):  # ... and the non-per-die backend rejects die-local weights
        backend.validate_transformed_weights(
            transformed, intermediate_size=I, hidden_size=H, world_size=1, num_experts=E
        )

    calls = []
    frontend.set_kernel_factory(lambda c: (lambda *a, **kw: calls.append((c, a, kw))))
    try:
        ws = kernel.prepare_workspace(bootstrap, fleet)
        m = 130
        t = MoEEpTensors(
            hidden_states=torch.randn((m, H), dtype=torch.bfloat16, device="cuda"),
            topk_ids=torch.randint(0, E, (m, TOPK), dtype=torch.int64, device="cuda"),
            topk_weights=torch.rand((m, TOPK), device="cuda"),
        )
        kernel.validate_forward(t, fleet, quantize_input=True)
        kernel.stage_inputs(t, ws, quantize_input=True)
        kernel.compute(ws, transformed, output=None)
        cfg_used, args, kw = calls[-1]
        assert (
            cfg_used.num_locality_domains,
            cfg_used.localized_weights,
            cfg_used.localized_shared_weights,
            cfg_used.per_die,
        ) == (2, True, False, True)
        assert (
            set(kw) == {"sm_locality_domains"}
            and kw["sm_locality_domains"].data_ptr()
            == locality.get_balanced_sm_locality_domains().data_ptr()
        )
        assert args[4] is transformed[0][0] and args[6] is transformed[1][0]
        # precompile: every variant is the per-die one
        built = []
        frontend.set_kernel_factory(
            lambda c: built.append(
                (c.block_m, c.num_locality_domains, c.localized_weights)
            )
            or (lambda *a, **kw: None)
        )
        bms = kernel.precompile_variants(fleet)
        assert (
            tuple(b[0] for b in built) == tuple(bms)
            and built
            and all(b[1:] == (2, True) for b in built)
        )
        # capture readiness of the per-die variants: they are registered under the per-die key, the SM map was probed by the launch above
        assert frontend.built_block_ms(
            1,
            E,
            T,
            TOPK,
            H,
            I,
            0,
            activation_clamp=10.0,
            num_sms=ws.buffer.num_sms,
            per_die=True,
            localized_weights=True,
        ) == frozenset(bms)
        assert (
            frontend.built_block_ms(
                1, E, T, TOPK, H, I, 0, activation_clamp=10.0, num_sms=ws.buffer.num_sms
            )
            == frozenset()
        )
        kernel.validate_capture_ready(ws, transformed)
        locality.destroy()  # forget the SM map: capture would have to probe it
        with pytest.raises(RuntimeError, match="SM -> locality-domain map"):
            kernel.validate_capture_ready(ws, transformed)
        locality.get_balanced_sm_locality_domains()
        kernel.destroy(ws)
    finally:
        frontend.set_kernel_factory(None)


def test_validate_capture_ready_with_stub_kernel():
    """``validate_capture_ready`` (called by the layer before staging while a CUDA graph is being captured) fails early, host-only, when lazy work
    (staging / kernel module load, kernel compile, per-die SM map) would run inside the capture; it passes once the workspace is warmed."""
    from flashinfer.experimental.sm107_deepgemm_megamoe import (
        backend,
        frontend,
        heuristics,
    )
    from flashinfer.moe_ep.config import BootstrapConfig, FleetParams
    from flashinfer.moe_ep.tensors import MoEEpTensors

    group = _ep1_group()
    kernel = backend.Sm107DeepgemmCutedslMegaKernelBackend(_make_config())
    bootstrap = BootstrapConfig(
        world_size=1,
        rank=0,
        auto_bootstrap=False,
        device=torch.cuda.current_device(),
        process_group=group,
    )
    fleet = FleetParams(num_experts=E, max_tokens_per_rank=T, token_hidden_size=H)
    kernel.bind_ep_bootstrap(bootstrap)
    transformed = kernel.preprocess_weights(
        _canonical_pack(torch.Generator().manual_seed(5)), fleet
    )
    frontend.set_kernel_factory(lambda c: (lambda *a, **kw: None))
    try:
        ws = kernel.prepare_workspace(bootstrap, fleet)
        with pytest.raises(
            RuntimeError, match="not warmed for CUDA graph capture"
        ) as exc:
            kernel.validate_capture_ready(ws, transformed)
        assert (
            "no eager forward" in str(exc.value)
            and "no kernel variant" in str(exc.value)
            and "warmup" in str(exc.value)
        )
        # variants built (but nothing launched yet): still not warmed
        built = kernel.precompile_variants(fleet)
        assert built and frontend.built_block_ms(
            1, E, T, TOPK, H, I, 0, activation_clamp=10.0, num_sms=ws.buffer.num_sms
        ) == frozenset(built)
        with pytest.raises(RuntimeError, match="no eager forward"):
            kernel.validate_capture_ready(ws, transformed)
        # one eager forward
        m = 130
        t = MoEEpTensors(
            hidden_states=torch.randn((m, H), dtype=torch.bfloat16, device="cuda"),
            topk_ids=torch.randint(0, E, (m, TOPK), dtype=torch.int64, device="cuda"),
            topk_weights=torch.rand((m, TOPK), device="cuda"),
        )
        kernel.stage_inputs(t, ws, quantize_input=True)
        kernel.compute(ws, transformed, output=None)
        kernel.validate_capture_ready(ws, transformed)  # warmed
        # a token bucket whose variant was not built
        bm_small = heuristics.get_block_m_for_mega_moe(1, E, T, 1, TOPK, "fp8xfp4")
        bm_big = heuristics.get_block_m_for_mega_moe(1, E, T, 1500, TOPK, "fp8xfp4")
        assert bm_small != bm_big
        frontend.set_kernel_factory(
            lambda c: (lambda *a, **kw: None)
        )  # forgets the built set
        frontend.precompile_mega_moe_variants(
            1,
            E,
            T,
            TOPK,
            H,
            I,
            0,
            activation_clamp=10.0,
            num_sms=ws.buffer.num_sms,
            block_ms=(bm_small,),
        )
        kernel.set_compile_tokens_per_rank(ws, 1)
        kernel.validate_capture_ready(ws, transformed)  # the bucket's variant is built
        kernel.set_compile_tokens_per_rank(ws, 1500)
        with pytest.raises(RuntimeError, match=f"BLOCK_M={bm_big}"):
            kernel.validate_capture_ready(ws, transformed)
        kernel.destroy(ws)
        with pytest.raises(RuntimeError, match="destroyed"):
            kernel.validate_capture_ready(ws, transformed)
    finally:
        frontend.set_kernel_factory(None)


def test_config_kernel_option_fields():
    assert (
        _make_config().tag_mode == "epoch"
        and _make_config().regs_override is None
        and _make_config().timeout_sec == 300.0
    )
    assert (
        _make_config().faithful_regs is False
        and _make_config(faithful_regs=True).faithful_regs is True
    )
    c = _make_config(
        tag_mode="gridid",
        regs_override=(56, 40, 200),
        timeout_sec=5.0,
        block_m_override=64,
        stages_cap=4,
    )
    assert (c.tag_mode, c.regs_override, c.timeout_sec) == (
        "gridid",
        (56, 40, 200),
        5.0,
    )
    assert (
        cfgmod.Sm107_Mxfp8_Mxfp4_Bf16_DeepgemmCutedsl_MegaMoeConfig(
            intermediate_size=I, top_k=TOPK
        ).allow_unvalidated_geometry
        is False
    )
    for bad in (
        dict(tag_mode="tag"),
        dict(regs_override=(56, 40)),
        dict(regs_override=(56, 40, 201)),
        dict(timeout_sec=0.0),
        dict(faithful_regs=True, regs_override=(56, 40, 200)),
    ):  # DeepGEMM's own split and an explicit split exclude each other
        with pytest.raises(ValueError):
            _make_config(**bad)


def test_validate_init_admits_only_validated_geometries():
    from flashinfer.experimental.sm107_deepgemm_megamoe import backend
    from flashinfer.moe_ep.config import BootstrapConfig, FleetParams
    from flashinfer.moe_ep.core.validation.common import MoEEpConfigError

    group = _ep1_group()
    bootstrap = BootstrapConfig(
        world_size=1,
        rank=0,
        auto_bootstrap=False,
        device=torch.cuda.current_device(),
        process_group=group,
    )
    # the tiny test geometry derives (it is what the stub-kernel tests run) but never ran on a GPU: refused unless the config opts out
    strict = backend.Sm107DeepgemmCutedslMegaKernelBackend(
        _make_config(allow_unvalidated_geometry=False)
    )
    strict.bind_ep_bootstrap(bootstrap)
    with pytest.raises(MoEEpConfigError, match="not been validated"):
        strict.validate_init(
            bootstrap,
            FleetParams(num_experts=E, max_tokens_per_rank=T, token_hidden_size=H),
        )
    backend.Sm107DeepgemmCutedslMegaKernelBackend(_make_config()).validate_init(
        bootstrap,
        FleetParams(num_experts=E, max_tokens_per_rank=T, token_hidden_size=H),
    )
    # the validated V4.1 shape is admitted
    v41 = cfgmod.Sm107_Mxfp8_Mxfp4_Bf16_DeepgemmCutedsl_MegaMoeConfig(
        intermediate_size=2304, top_k=6, gate_up_clamp=10.0
    )
    k = backend.Sm107DeepgemmCutedslMegaKernelBackend(v41)
    k.bind_ep_bootstrap(bootstrap)
    k.validate_init(
        bootstrap,
        FleetParams(num_experts=384, max_tokens_per_rank=8192, token_hidden_size=5120),
    )


def test_backend_options_carry_faithful_regs():
    from flashinfer.experimental.sm107_deepgemm_megamoe import backend

    opts = backend.Sm107DeepgemmCutedslMegaKernelBackend(
        _make_config(faithful_regs=True)
    )._options()
    assert opts.faithful_regs is True and opts.regs_override is None
    assert (
        backend.Sm107DeepgemmCutedslMegaKernelBackend(_make_config())
        ._options()
        .faithful_regs
        is False
    )


def test_preprocess_accepts_float32_ue8m0_scales_like_the_sm100_deepgemm_backend():
    """The canonical pack of FlashInfer's ``sm100_fp8_fp4_bf16_deepgemm`` backend (fp32 block scales from ``per_token_cast_to_fp4(use_ue8m0=True)``)
    feeds this backend unchanged: exponent bytes, bit-identical to the uint8 form."""
    from flashinfer.experimental.sm107_deepgemm_megamoe import backend
    from flashinfer.moe_ep.core.validation.common import MoEEpConfigError
    from flashinfer.moe_ep.weights import MoEWeightPack

    gen = torch.Generator().manual_seed(21)
    pack = _canonical_pack(gen)
    as_f32 = lambda t: (t.to(torch.int32) << 23).view(torch.float32)  # noqa: E731
    pack32 = MoEWeightPack(
        pack.w13, pack.w2, as_f32(pack.w13_scale), as_f32(pack.w2_scale)
    )
    a = backend.preprocess_canonical_weights(pack, intermediate_size=I, hidden_size=H)
    b = backend.preprocess_canonical_weights(pack32, intermediate_size=I, hidden_size=H)
    for x, y in zip((*a[0], *a[1]), (*b[0], *b[1]), strict=False):
        assert x.dtype == y.dtype and x.stride() == y.stride() and torch.equal(x, y)
    bad = pack32.w13_scale.clone()
    bad.view(torch.int32)[0, 0, 0] |= 1  # not an exact power of two
    with pytest.raises(MoEEpConfigError, match="powers of two"):
        backend.preprocess_canonical_weights(
            MoEWeightPack(pack.w13, pack.w2, bad, pack32.w2_scale),
            intermediate_size=I,
            hidden_size=H,
        )
    with pytest.raises(MoEEpConfigError, match="powers of two"):
        backend.preprocess_canonical_weights(
            MoEWeightPack(pack.w13, pack.w2, -pack32.w13_scale, pack32.w2_scale),
            intermediate_size=I,
            hidden_size=H,
        )


def test_backend_passes_its_kernel_options_to_the_jit_layer(monkeypatch):
    from flashinfer.experimental.sm107_deepgemm_megamoe import backend, frontend, jit
    from flashinfer.moe_ep.config import BootstrapConfig, FleetParams
    from flashinfer.moe_ep.tensors import MoEEpTensors

    group = _ep1_group()
    kernel = backend.Sm107DeepgemmCutedslMegaKernelBackend(
        _make_config(
            tag_mode="gridid",
            regs_override=(56, 40, 200),
            timeout_sec=9.0,
            block_m_override=64,
        )
    )
    bootstrap = BootstrapConfig(
        world_size=1,
        rank=0,
        auto_bootstrap=False,
        device=torch.cuda.current_device(),
        process_group=group,
    )
    fleet = FleetParams(num_experts=E, max_tokens_per_rank=T, token_hidden_size=H)
    kernel.bind_ep_bootstrap(bootstrap)
    transformed = kernel.preprocess_weights(
        _canonical_pack(torch.Generator().manual_seed(2)), fleet
    )
    calls = []
    monkeypatch.setattr(
        jit,
        "get_or_compile_kernel",
        lambda cfg, **kw: calls.append((cfg, kw)) or (lambda *a, **k: None),
    )
    frontend.set_kernel_factory(None)
    try:
        assert kernel.precompile_variants(fleet) == (
            64,
        )  # the forced BLOCK_M is the only variant
        ws = kernel.prepare_workspace(bootstrap, fleet)
        m = 5
        t = MoEEpTensors(hidden_states=torch.randn((m, H), dtype=torch.bfloat16, device="cuda"), topk_ids=torch.randint(0, E, (m, TOPK), dtype=torch.int64, device="cuda"),
                         topk_weights=torch.rand((m, TOPK), device="cuda"))  # fmt: skip
        kernel.stage_inputs(t, ws, quantize_input=True)
        kernel.compute(ws, transformed, output=None)
        cfg, kw = calls[-1]
        assert cfg.block_m == 64 and kw == dict(
            tag_mode="gridid",
            regs_override=(56, 40, 200),
            timeout_sec=9.0,
            faithful_regs=False,
        )
        assert (
            frontend.get_kernel_options() == frontend.KernelOptions()
        )  # the backend's options are scoped to its own launches
        with pytest.raises(
            RuntimeError, match="tag_mode='gridid'"
        ):  # graph capture is refused under DeepGEMM's %gridid tag
            kernel.validate_capture_ready(ws, transformed)
        kernel.destroy(ws)
    finally:
        frontend.set_kernel_factory(None)


# ---------------------------------------------------------------------------------------------------------------------
# RC2: refusals instead of silent drops, first-launch precompile, per-call variant check while capturing
# ---------------------------------------------------------------------------------------------------------------------
def test_preprocess_refuses_packs_with_fields_it_does_not_consume():
    """A fused-shared-expert pack (a ``PrequantizedMoEWeights`` subclass with ``shared_*`` tensors) must not pass with its shared tensors dropped."""
    import dataclasses

    from flashinfer.experimental.sm107_deepgemm_megamoe import backend
    from flashinfer.moe_ep.core.validation.common import MoEEpConfigError
    from flashinfer.moe_ep.weights import PrequantizedMoEWeights

    @dataclasses.dataclass(frozen=True)
    class _PackWithSharedExpert(PrequantizedMoEWeights):
        shared_w13: torch.Tensor
        shared_w2: torch.Tensor

        def __new__(
            cls, *args, **kwargs
        ):  # MoEWeightPack.__new__ has the fixed four-tensor signature
            return object.__new__(cls)

    z = torch.zeros(1, dtype=torch.uint8)
    with pytest.raises(
        MoEEpConfigError, match="fused shared experts are not supported"
    ):
        backend.preprocess_canonical_weights(
            _PackWithSharedExpert(
                w13=z, w2=z, w13_scale=z, w2_scale=z, shared_w13=z, shared_w2=z
            ),
            intermediate_size=I,
            hidden_size=H,
        )
    # extra fields that are None carry nothing: the pack goes on to the normal shape checks
    with pytest.raises(MoEEpConfigError, match="must have shape"):
        backend.preprocess_canonical_weights(
            _PackWithSharedExpert(
                w13=z, w2=z, w13_scale=z, w2_scale=z, shared_w13=None, shared_w2=None
            ),
            intermediate_size=I,
            hidden_size=H,
        )


def test_backend_refuses_nvfp4_scalars_and_graph_capture_under_the_gridid_tag():
    from flashinfer.experimental.sm107_deepgemm_megamoe import backend, frontend
    from flashinfer.moe_ep.config import BootstrapConfig, FleetParams
    from flashinfer.moe_ep.core.validation.common import MoEEpConfigError
    from flashinfer.moe_ep.tensors import MoEEpTensors

    group = _ep1_group()
    bootstrap = BootstrapConfig(
        world_size=1,
        rank=0,
        auto_bootstrap=False,
        device=torch.cuda.current_device(),
        process_group=group,
    )
    fleet = FleetParams(num_experts=E, max_tokens_per_rank=T, token_hidden_size=H)
    m = 16
    t = MoEEpTensors(
        hidden_states=torch.randn((m, H), dtype=torch.bfloat16, device="cuda"),
        topk_ids=torch.randint(0, E, (m, TOPK), dtype=torch.int64, device="cuda"),
        topk_weights=torch.rand((m, TOPK), device="cuda"),
    )
    kernel = backend.Sm107DeepgemmCutedslMegaKernelBackend(_make_config())
    kernel.bind_ep_bootstrap(bootstrap)
    kernel.validate_forward(t, fleet, quantize_input=True)
    for name in (
        "fc1_alpha",
        "fc2_alpha",
        "fc1_norm_const",
    ):  # NVFP4-path scalars: refused, not ignored
        bad = MoEEpTensors(
            t.hidden_states,
            t.topk_ids,
            t.topk_weights,
            **{name: torch.ones(1, device="cuda")},
        )
        with pytest.raises(MoEEpConfigError, match=name):
            kernel.validate_forward(bad, fleet, quantize_input=True)
    # tag_mode="gridid": warned at construction, graph capture refused (DeepGEMM's %gridid tag is stale under replay)
    with pytest.warns(UserWarning, match="gridid"):
        cfg = _make_config(tag_mode="gridid")
    gridid = backend.Sm107DeepgemmCutedslMegaKernelBackend(cfg)
    gridid.bind_ep_bootstrap(bootstrap)
    frontend.set_kernel_factory(lambda c: (lambda *a, **kw: None))
    try:
        ws = gridid.prepare_workspace(bootstrap, fleet)
        transformed = gridid.preprocess_weights(
            _canonical_pack(torch.Generator().manual_seed(3)), fleet
        )
        with pytest.raises(RuntimeError, match="tag_mode='gridid'"):
            gridid.validate_capture_ready(ws, transformed)
        gridid.destroy(ws)
    finally:
        frontend.set_kernel_factory(None)


def test_first_launch_builds_every_block_m_variant_and_capture_checks_the_live_variant(
    monkeypatch,
):
    """The first launch of a workspace (``MoEEpMegaLayer.warmup()``) builds every BLOCK_M variant of the geometry once (idempotent); while a CUDA graph is
    captured, ``stage_inputs`` (where the live token count is known) refuses a token count whose variant is not built, and accepts the others."""
    from flashinfer.experimental.sm107_deepgemm_megamoe import (
        backend,
        frontend,
        heuristics,
    )
    from flashinfer.moe_ep.config import BootstrapConfig, FleetParams
    from flashinfer.moe_ep.tensors import MoEEpTensors

    group = _ep1_group()
    bootstrap = BootstrapConfig(
        world_size=1,
        rank=0,
        auto_bootstrap=False,
        device=torch.cuda.current_device(),
        process_group=group,
    )
    fleet = FleetParams(num_experts=E, max_tokens_per_rank=T, token_hidden_size=H)

    def tensors(m):
        return MoEEpTensors(
            hidden_states=torch.randn((m, H), dtype=torch.bfloat16, device="cuda"),
            topk_ids=torch.randint(0, E, (m, TOPK), dtype=torch.int64, device="cuda"),
            topk_weights=torch.rand((m, TOPK), device="cuda"),
        )

    expected = set(
        heuristics.get_block_m_variants(
            1, E, TOPK, heuristics.align(T, heuristics.TOKEN_ALIGNMENT)
        )
    )
    bm_small = heuristics.get_block_m_for_mega_moe(1, E, T, 1, TOPK, "fp8xfp4")
    bm_big = heuristics.get_block_m_for_mega_moe(1, E, T, 1500, TOPK, "fp8xfp4")
    assert len(expected) > 1 and bm_small != bm_big

    built = []
    frontend.set_kernel_factory(
        lambda c: built.append(c.block_m) or (lambda *a, **kw: None)
    )
    try:
        kernel = backend.Sm107DeepgemmCutedslMegaKernelBackend(_make_config())
        kernel.bind_ep_bootstrap(bootstrap)
        transformed = kernel.preprocess_weights(
            _canonical_pack(torch.Generator().manual_seed(11)), fleet
        )
        ws = kernel.prepare_workspace(bootstrap, fleet)
        kernel.stage_inputs(tensors(1), ws, quantize_input=True)
        assert built == []  # nothing is built before the first launch
        kernel.compute(
            ws, transformed, output=None
        )  # a one-token warm-up (what vLLM does) ...
        assert sorted(built) == sorted(expected)  # ... builds EVERY variant, once
        n = len(built)
        kernel.stage_inputs(tensors(1500), ws, quantize_input=True)
        kernel.compute(ws, transformed, output=None)
        assert len(built) == n  # idempotent: later launches build nothing
        monkeypatch.setattr(
            torch.cuda, "is_current_stream_capturing", lambda: True
        )  # pretend a capture is active (the stub kernel launches nothing)
        for m in (1, 130, 1500, T):
            kernel.stage_inputs(
                tensors(m), ws, quantize_input=True
            )  # every variant exists: accepted at any token count
        monkeypatch.undo()  # the workspace release refuses to run during a capture
        kernel.destroy(ws)

        # opt-out: only the live variant is built; a capture at another token count is refused with the BLOCK_M named
        built.clear()
        frontend.set_kernel_factory(
            lambda c: built.append(c.block_m) or (lambda *a, **kw: None)
        )  # forgets the built set
        lazy = backend.Sm107DeepgemmCutedslMegaKernelBackend(
            _make_config(precompile_on_warmup=False)
        )
        lazy.bind_ep_bootstrap(bootstrap)
        ws = lazy.prepare_workspace(bootstrap, fleet)
        lazy.stage_inputs(tensors(1), ws, quantize_input=True)
        lazy.compute(ws, transformed, output=None)
        assert built == [bm_small]
        monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
        lazy.stage_inputs(tensors(1), ws, quantize_input=True)  # the built variant
        with pytest.raises(RuntimeError, match=f"BLOCK_M={bm_big}") as exc:
            lazy.stage_inputs(tensors(1500), ws, quantize_input=True)
        assert "1500 tokens" in str(
            exc.value
        ) and "cannot be compiled during the capture" in str(exc.value)
        monkeypatch.undo()
        lazy.destroy(ws)
    finally:
        frontend.set_kernel_factory(None)
