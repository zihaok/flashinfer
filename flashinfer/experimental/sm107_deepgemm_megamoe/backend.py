# SPDX-License-Identifier: Apache-2.0
# Port of DeepGEMM (MIT License, Copyright (c) 2025 DeepSeek; see licenses/LICENSE.deepgemm.txt) sm107_fp8_fp4_mega_moe.cuh / deep_gemm/mega/__init__.py; the backend structure follows FlashInfer's moe_ep SM107 CuTe-DSL mega backend (Apache-2.0).
"""moe_ep (Tier 2, "Route A") backend of the SM107 DeepGEMM-style MegaMoE.

``@register_mega_kernel("sm107_mxfp8_mxfp4_bf16_deepgemm_cutedsl")``: MXFP8 E4M3 activations x packed MXFP4 weights -> BF16 on
the NVLink symmetric heap (torch ``symm_mem``, no NVSHMEM), driven by ``MoEEpMegaLayer`` exactly like FlashInfer's other mega
kernels.  This module imports the moe_ep machinery, so it is NOT imported by the package ``__init__``: it is imported on demand
(``import flashinfer.experimental.sm107_deepgemm_megamoe.backend`` registers the kernel; the lazy registry of ``flashinfer.moe_ep``
(``moe_ep/core/kernel/registry.py``) makes ``create_mega_kernel`` do that by name).

Scope of this version: routed experts only (the canonical ``MoEWeightPack`` has no shared-expert tensors: a pack that carries extra
fields is refused instead of having them dropped; die-local routed weights are supported with ``per_die=True``), BF16 hidden states
staged inside the backend (``quantize_input=True``).  The kernel itself comes from ``jit.get_or_compile_kernel`` through
:func:`.frontend.fp8_fp4_mega_moe`.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Optional, Tuple

import torch

from flashinfer.api_logging import warn_experimental_backend_once
from flashinfer.moe_ep.config import BootstrapConfig, FleetParams
from flashinfer.moe_ep.core.kernel.base import MegaKernelBackend
from flashinfer.moe_ep.core.kernel.registry import register_mega_kernel
from flashinfer.moe_ep.core.validation.common import (
    MoEEpConfigError,
    validate_mega_arch_sm107,
    validate_mega_fleet_params,
)
from flashinfer.moe_ep.weights import MoEWeightPack, PrequantizedMoEWeights

from . import frontend, heuristics, staging
from .config import (
    MOE_EP_KERNEL_NAME,
    Sm107_Mxfp8_Mxfp4_Bf16_DeepgemmCutedsl_MegaMoeConfig,
)
from .weights import sf_to_required_layout, transform_weights_for_mega_moe
from .workspace import SymmBuffer

TransformedWeights = Tuple[
    Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]
]

# the fields of the canonical pre-quantized pack this backend consumes; any other non-None field (e.g. the shared_* tensors of a fused-shared-expert pack) would be dropped silently
_CONSUMED_PACK_FIELDS = frozenset({"w13", "w2", "w13_scale", "w2_scale"})


class Sm107DeepgemmWorkspace:
    """Backend workspace: the symmetric buffer plus the stable output tensor (``workspace_output``)."""

    def __init__(self, buffer: SymmBuffer, output: torch.Tensor) -> None:
        self.buffer: Optional[SymmBuffer] = buffer
        self.output: Optional[torch.Tensor] = output
        self._staged_tokens: Optional[int] = None
        self.compile_tokens_per_rank: Optional[int] = None
        self.eager_launches = 0  # kernel launches issued through this workspace (0: nothing is warmed yet)

    def note_staged_tokens(self, n: Optional[int]) -> None:
        self._staged_tokens = n

    def staged_tokens(self) -> Optional[int]:
        return self._staged_tokens

    def destroy(self) -> None:
        if self.buffer is not None:
            self.buffer.destroy()
        self.buffer = None
        self.output = None


def _u8(t: torch.Tensor) -> torch.Tensor:
    """Scale planes may be typed E8M0 or raw uint8: view as bytes."""
    return t if t.dtype == torch.uint8 else t.view(torch.uint8)


def _ue8m0_bytes(t: torch.Tensor, name: str) -> torch.Tensor:
    """UE8M0 scale bytes of a canonical scale plane: uint8 / ``float8_e8m0fnu`` as they are, ``float32`` block scales (the layout of
    FlashInfer's ``sm100_fp8_fp4_bf16_deepgemm`` backend and of ``per_token_cast_to_fp4(use_ue8m0=True)``) through their exponent byte;
    they must be exact positive powers of two."""
    if t.dtype != torch.float32:
        return _u8(t)
    bits = t.contiguous().view(torch.int32)
    if bool(((bits & 0x7FFFFF) != 0).any()) or bool((bits < 0).any()):
        raise MoEEpConfigError(
            f"{name}: float32 scales must be exact positive powers of two (UE8M0 values)"
        )
    return (bits >> 23).to(torch.uint8)


def preprocess_canonical_weights(
    weights: MoEWeightPack,
    *,
    intermediate_size: int,
    hidden_size: int,
    per_die: bool = False,
) -> TransformedWeights:
    """Canonical pre-quantized pack -> the kernel's DeepGEMM layout ``((w1, sf1), (w2, sf2))``.

    ``w13`` ``[E_l, 2I, H/2]`` packed E2M1 bytes (rows ``[gate (I); up (I)]``, low nibble = even K), ``w2`` ``[E_l, H, I/2]``,
    ``w13_scale`` ``[E_l, 2I, H/32]`` / ``w2_scale`` ``[E_l, H, I/32]`` E8M0 per 32 values (uint8, typed E8M0, or float32 UE8M0 values).  The L1 weights are a
    new tensor (the interleave needs a transient copy per layer); the L2 weights ALIAS the caller's ``w2`` unless ``per_die`` (the transform only re-lays out
    the scales), so the caller must keep ``w2`` alive and unchanged.  ``per_die``: the routed weights are additionally localized into the two locality domains
    (``(2, E_l, N/2, K/2)``, a copy; scales stay put; die-local routed weights, DeepGEMM's per-die mode).  A pack that carries fields this backend does not
    consume (fused shared experts: ``shared_*`` tensors) is refused: fused shared experts are not supported by this backend yet."""
    if not isinstance(weights, PrequantizedMoEWeights):
        raise MoEEpConfigError(
            "sm107_mxfp8_mxfp4_bf16_deepgemm_cutedsl needs a pre-quantized MoEWeightPack (packed MXFP4 + E8M0 scales)"
        )
    extra = [
        f.name
        for f in dataclasses.fields(weights)
        if f.name not in _CONSUMED_PACK_FIELDS
        and getattr(weights, f.name, None) is not None
    ]
    if extra:
        raise MoEEpConfigError(
            f"the weight pack carries fields this backend does not consume ({', '.join(extra)}): fused shared experts are not supported yet "
            "(use a plain MoEWeightPack and add the shared expert outside the kernel); refusing instead of dropping them silently"
        )
    w13, w2 = weights.w13, weights.w2
    s13, s2 = weights.w13_scale, weights.w2_scale
    e_l = w13.shape[0]
    expected = {
        "w13": (w13, (e_l, 2 * intermediate_size, hidden_size // 2)),
        "w2": (w2, (e_l, hidden_size, intermediate_size // 2)),
        "w13_scale": (s13, (e_l, 2 * intermediate_size, hidden_size // 32)),
        "w2_scale": (s2, (e_l, hidden_size, intermediate_size // 32)),
    }
    for name, (t, shape) in expected.items():
        if tuple(t.shape) != shape:
            raise MoEEpConfigError(
                f"{name} must have shape {shape}, got {tuple(t.shape)}"
            )
    for name, t in (("w13", w13), ("w2", w2)):
        if t.element_size() != 1 or not t.is_contiguous():
            raise MoEEpConfigError(
                f"{name} must be a contiguous 1-byte-per-element packed MXFP4 tensor, got {t.dtype}"
            )
    for name, t in (("w13_scale", s13), ("w2_scale", s2)):
        if t.element_size() != 1 and t.dtype != torch.float32:
            raise MoEEpConfigError(
                f"{name} must be 1-byte E8M0 scales (uint8 / float8_e8m0fnu) or float32 UE8M0 values, got {t.dtype}"
            )
    l1 = (
        w13.contiguous().view(torch.int8),
        sf_to_required_layout(
            _ue8m0_bytes(s13, "w13_scale"), 2 * intermediate_size, hidden_size, e_l
        ),
    )
    l2 = (
        w2.contiguous().view(torch.int8),
        sf_to_required_layout(
            _ue8m0_bytes(s2, "w2_scale"), hidden_size, intermediate_size, e_l
        ),
    )
    (w1, sf1), (w2, sf2) = transform_weights_for_mega_moe(l1, l2)
    if per_die:
        from .locality import localize

        w1, w2 = (
            localize(w1),
            localize(w2),
        )  # the transient plain copies are freed when the caller drops them
    return (w1, sf1), (w2, sf2)


def validate_transformed_weights(
    transformed: Any,
    *,
    intermediate_size: int,
    hidden_size: int,
    world_size: int,
    num_experts: int,
    per_die: bool = False,
) -> None:
    """Structure/dtype/stride checks of user-supplied kernel-ready weights (the 2x2 tuple structure is required by vLLM's aliasing)."""
    try:
        (w1, sf1), (w2, sf2) = transformed
    except (TypeError, ValueError) as exc:
        raise MoEEpConfigError(
            "transformed weights must be ((w1, sf1), (w2, sf2))"
        ) from exc
    e_l = num_experts // world_size
    try:
        if per_die:
            from .locality import is_canonical_localized

            for name, t, shape4 in (
                ("w1", w1, (2, e_l, intermediate_size, hidden_size // 2)),
                ("w2", w2, (2, e_l, hidden_size // 2, intermediate_size // 2)),
            ):
                assert (
                    tuple(t.shape) == shape4
                    and t.dtype == torch.int8
                    and is_canonical_localized(t)
                ), (
                    f"{name}: expected die-local int8 {shape4}, got {tuple(t.shape)} {t.dtype}"
                )
        else:
            for name, t, shape, dtype in (
                ("w1", w1, (e_l, 2 * intermediate_size, hidden_size // 2), torch.int8),
                ("w2", w2, (e_l, hidden_size, intermediate_size // 2), torch.int8),
            ):
                assert (
                    tuple(t.shape) == shape and t.dtype == dtype and t.is_contiguous()
                ), (
                    f"{name}: expected contiguous {dtype} {shape}, got {tuple(t.shape)} {t.dtype}"
                )
        frontend._check_sf_layout(sf1, 2 * intermediate_size, hidden_size, e_l, "w1_sf")
        frontend._check_sf_layout(sf2, hidden_size, intermediate_size, e_l, "w2_sf")
    except (AssertionError, NotImplementedError) as exc:
        raise MoEEpConfigError(str(exc)) from exc


@register_mega_kernel(MOE_EP_KERNEL_NAME)
class Sm107DeepgemmCutedslMegaKernelBackend(MegaKernelBackend):
    """Fused Rubin MegaMoE with DeepGEMM's kernel mapping (MXFP8 x MXFP4 -> BF16)."""

    supports_output_view = (
        True  # compute(output=None) returns a view of the workspace output
    )

    def __init__(
        self, config: Sm107_Mxfp8_Mxfp4_Bf16_DeepgemmCutedsl_MegaMoeConfig
    ) -> None:
        super().__init__(config)
        self._kernel_config = config
        self._options_memo: Optional[tuple] = (
            None  # (source field values, the KernelOptions built from them)
        )

    @classmethod
    def kernel_name(cls) -> str:
        return MOE_EP_KERNEL_NAME

    # ---- init-time validation (host only, before any collective runtime setup)
    def _options(self) -> "frontend.KernelOptions":
        """Kernel build options of this config (``tag_mode`` / ``regs_override`` / ``timeout_sec`` and the DeepGEMM tuning knobs)."""
        k = self._kernel_config
        source = (
            k.tag_mode,
            k.regs_override,
            k.timeout_sec,
            k.faithful_regs,
            k.block_m_override,
            k.launch_smem_capacity,
            k.stages_cap,
        )
        memo = self._options_memo
        if (
            memo is not None and memo[0] == source
        ):  # one object per option set: the launch path keys its validated-call records on it, and it is built (validated) once
            return memo[1]
        options = frontend.KernelOptions(
            tag_mode=k.tag_mode, regs_override=k.regs_override, timeout_sec=k.timeout_sec, faithful_regs=k.faithful_regs, block_m=k.block_m_override,
            smem_capacity=k.launch_smem_capacity, max_stages=k.stages_cap,
        )  # fmt: skip
        self._options_memo = (source, options)
        return options

    def validate_init(
        self, bootstrap: BootstrapConfig, fleet_params: FleetParams
    ) -> None:
        warn_experimental_backend_once("MoEEpLayer", self.kernel_name())
        validate_mega_arch_sm107()
        self._ensure_ep_bootstrap(bootstrap)
        k = self._kernel_config
        if k.per_die:
            from . import locality

            if torch.cuda.is_available() and not locality.is_localization_available():
                raise MoEEpConfigError(
                    "per_die=True needs a compute capability 10.7 device that reports exactly 2 locality domains"
                )
        validate_mega_fleet_params(
            fleet_params,
            bootstrap.world_size,
            intermediate_size=k.intermediate_size,
            top_k=k.top_k,
            alignment=128,
        )
        if (
            fleet_params.num_experts // bootstrap.world_size < 1
            or k.top_k > fleet_params.num_experts
        ):
            raise MoEEpConfigError(
                "need top_k <= num_experts and at least one expert per rank"
            )
        # Only geometries that have run on GPUs (heuristics.VALIDATED_GEOMETRIES) unless the config opts out (kernel development)
        reason = heuristics.get_mega_moe_unsupported_reason(
            bootstrap.world_size, fleet_params.num_experts, fleet_params.max_tokens_per_rank, k.top_k, fleet_params.token_hidden_size, k.intermediate_size,
            k.num_shared_experts, num_sms=heuristics.resolve_num_sms() if torch.cuda.is_available() else 212, fast_math=k.fast_math,
            allow_unvalidated=k.allow_unvalidated_geometry,
        )  # fmt: skip
        if reason is not None:
            raise MoEEpConfigError(
                f"unsupported MegaMoE geometry for the SM107 kernel: {reason}"
            )
        # Every kernel variant must be derivable (the host heuristics assert shape constraints such as the combine overlay size)
        try:
            for bm in heuristics.get_block_m_variants(
                bootstrap.world_size,
                fleet_params.num_experts,
                k.top_k,
                fleet_params.max_tokens_per_rank,
            ):
                heuristics.get_sm107_kernel_config(
                    bootstrap.world_size, fleet_params.num_experts, fleet_params.max_tokens_per_rank, 0, k.top_k,
                    fleet_params.token_hidden_size, k.intermediate_size, k.num_shared_experts,
                    num_sms=heuristics.resolve_num_sms() if torch.cuda.is_available() else 212,
                    activation_clamp=k.resolved_clamp, fast_math=k.fast_math, block_m=bm, per_die=k.per_die, localized_weights=k.per_die,
                )  # fmt: skip
        except AssertionError as exc:
            raise MoEEpConfigError(
                f"unsupported MegaMoE geometry for the SM107 kernel: {exc}"
            ) from exc

    # ---- weights
    def preprocess_weights(
        self, weights: MoEWeightPack, fleet_params: FleetParams
    ) -> TransformedWeights:
        return preprocess_canonical_weights(
            weights, intermediate_size=self._kernel_config.intermediate_size, hidden_size=fleet_params.token_hidden_size,
            per_die=self._kernel_config.per_die,
        )  # fmt: skip

    def validate_transformed_weights(
        self,
        transformed_weights: Any,
        bootstrap: BootstrapConfig,
        fleet_params: FleetParams,
    ) -> None:
        validate_transformed_weights(
            transformed_weights, intermediate_size=self._kernel_config.intermediate_size, hidden_size=fleet_params.token_hidden_size,
            world_size=self.ep_world_size, num_experts=fleet_params.num_experts, per_die=self._kernel_config.per_die,
        )  # fmt: skip

    # ---- workspace (symmetric buffer; collective)
    def _allocate_workspace(self, fleet_params: FleetParams) -> Sm107DeepgemmWorkspace:
        k = self._kernel_config
        buf = SymmBuffer(
            self.ep_comm_group, fleet_params.num_experts, fleet_params.max_tokens_per_rank, k.top_k,
            fleet_params.token_hidden_size, k.intermediate_size, num_shared_experts=k.num_shared_experts,
        )  # fmt: skip
        out = torch.empty(
            (buf.num_max_tokens_per_rank, fleet_params.token_hidden_size),
            dtype=torch.bfloat16,
            device="cuda",
        )
        return Sm107DeepgemmWorkspace(buf, out)

    def _workspace_pool_key(self, fleet_params: FleetParams):
        """Everything baked into the workspace (geometry, EP identity, shared experts, ...); the kernel variant is per call."""
        k = self._kernel_config
        return (
            self.kernel_name(), torch.cuda.current_device(), self.ep_rank, self.ep_world_size, id(self._ep_comm_group),
            fleet_params.num_experts, fleet_params.max_tokens_per_rank, k.top_k, fleet_params.token_hidden_size,
            k.intermediate_size, k.num_shared_experts,
        )  # fmt: skip

    def workspace_output(self, workspace: Sm107DeepgemmWorkspace) -> torch.Tensor:
        assert workspace.output is not None, "workspace was destroyed"
        return workspace.output

    def set_compile_tokens_per_rank(
        self, workspace: Sm107DeepgemmWorkspace, compile_tokens_per_rank: Optional[int]
    ) -> None:
        # The BLOCK_M variant follows the live token count of each call (as in DeepGEMM); a bucket is only recorded.
        workspace.compile_tokens_per_rank = compile_tokens_per_rank

    def precompile_variants(self, fleet_params: FleetParams) -> Tuple[int, ...]:
        """Build every BLOCK_M variant a rank can select (call before CUDA graph capture; collective-free)."""
        k = self._kernel_config
        return frontend.precompile_mega_moe_variants(
            self.ep_world_size, fleet_params.num_experts, fleet_params.max_tokens_per_rank, k.top_k, fleet_params.token_hidden_size,
            k.intermediate_size, k.num_shared_experts, activation_clamp=k.resolved_clamp, fast_math=k.fast_math,
            per_die=k.per_die, localized_weights=k.per_die, kernel_options=self._options(),
        )  # fmt: skip

    # ---- per-iteration
    def validate_forward(
        self, t, fleet_params: FleetParams, *, quantize_input: bool
    ) -> None:
        """Metadata-only validation (no device sync)."""
        k = self._kernel_config
        hs, ids, w = t.hidden_states, t.topk_ids, t.topk_weights
        if not quantize_input:
            raise MoEEpConfigError(
                "pre-quantized activations (quantize_input=False) are not supported by this backend yet"
            )
        if (
            hs.ndim != 2
            or hs.shape[1] != fleet_params.token_hidden_size
            or hs.shape[0] > fleet_params.max_tokens_per_rank
        ):
            raise MoEEpConfigError(
                f"hidden_states must be [<= {fleet_params.max_tokens_per_rank}, {fleet_params.token_hidden_size}], got {tuple(hs.shape)}"
            )
        if hs.dtype != torch.bfloat16:
            raise MoEEpConfigError(f"hidden_states must be bfloat16, got {hs.dtype}")
        if ids.shape != (hs.shape[0], k.top_k) or w.shape != ids.shape:
            raise MoEEpConfigError(
                f"topk_ids and topk_weights must have shape ({hs.shape[0]}, {k.top_k})"
            )
        if ids.dtype not in (torch.int32, torch.int64) or not w.is_floating_point():
            raise MoEEpConfigError(
                "topk_ids must be int32/int64 and topk_weights floating point"
            )
        for name in ("fc1_alpha", "fc2_alpha", "fc1_norm_const"):
            if getattr(t, name, None) is not None:
                raise MoEEpConfigError(
                    f"MoEEpTensors.{name} belongs to the NVFP4 path and is not used by this backend (alpha / beta / clamp are config fields); refusing instead of ignoring it"
                )

    def _require_variant_built(
        self, workspace: Sm107DeepgemmWorkspace, num_tokens: int
    ) -> None:
        """While a CUDA graph is captured: the kernel variant of THIS invocation (BLOCK_M of the live token count) must exist already; compiling inside a capture is
        an error.  Called from ``stage_inputs``, where the live token count is known (``validate_capture_ready`` runs before the layer records the bucket)."""
        buf = workspace.buffer
        assert buf is not None, "workspace was destroyed"
        k = self._kernel_config
        bm = self._options().block_m or heuristics.get_block_m_for_mega_moe(
            buf.group.size(), buf.num_experts, buf.num_max_tokens_per_rank, num_tokens, buf.num_topk, "fp8xfp4"
        )  # fmt: skip
        built = frontend.built_block_ms(
            buf.group.size(), buf.num_experts, buf.num_max_tokens_per_rank, buf.num_topk, buf.hidden, buf.intermediate_hidden, buf.num_shared_experts,
            activation_clamp=k.resolved_clamp, fast_math=k.fast_math, num_sms=buf.num_sms, per_die=k.per_die, localized_weights=k.per_die,
            kernel_options=self._options(),
        )  # fmt: skip
        if bm not in built:
            raise RuntimeError(
                f"CUDA graph capture at {num_tokens} tokens needs the BLOCK_M={bm} kernel variant, which is not built (built: {sorted(built)}); it cannot be compiled during "
                "the capture. Run one eager forward at this token count, or build every variant first (layer.warmup() does it with precompile_on_warmup=True, or call "
                "precompile_mega_moe_variants on every rank) before capturing."
            )

    def _precompile_workspace_variants(self, workspace: Sm107DeepgemmWorkspace) -> None:
        """Build every BLOCK_M variant of this workspace's geometry (host only, collective-free; ~1-1.5 s per variant cold, ~0.1 s from the disk cache)."""
        buf = workspace.buffer
        assert buf is not None, "workspace was destroyed"
        k = self._kernel_config
        frontend.precompile_mega_moe_variants(
            buf.group.size(), buf.num_experts, buf.num_max_tokens_per_rank, buf.num_topk, buf.hidden, buf.intermediate_hidden, buf.num_shared_experts,
            activation_clamp=k.resolved_clamp, fast_math=k.fast_math, num_sms=buf.num_sms, per_die=k.per_die, localized_weights=k.per_die,
            kernel_options=self._options(),
        )  # fmt: skip

    def stage_inputs(
        self, t, workspace: Sm107DeepgemmWorkspace, *, quantize_input: bool
    ) -> None:
        assert workspace.buffer is not None, "workspace was destroyed"
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            self._require_variant_built(workspace, int(t.hidden_states.shape[0]))
        staging.stage_inputs(
            workspace.buffer,
            t.hidden_states,
            t.topk_ids,
            t.topk_weights,
            block_m=self._options().block_m or None,
        )
        workspace.note_staged_tokens(t.hidden_states.shape[0])

    def compute(
        self,
        workspace: Sm107DeepgemmWorkspace,
        transformed_weights: TransformedWeights,
        *,
        output: Optional[torch.Tensor],
    ) -> torch.Tensor:
        assert workspace.buffer is not None and workspace.output is not None, (
            "workspace was destroyed"
        )
        if (
            workspace.eager_launches == 0
            and self._kernel_config.precompile_on_warmup
            and not (
                torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()
            )
        ):
            self._precompile_workspace_variants(
                workspace
            )  # the first launch of a workspace (layer.warmup()) builds every BLOCK_M variant: later token counts never compile
        num_tokens = workspace.staged_tokens()
        if num_tokens is None:
            if output is None:
                raise RuntimeError(
                    "compute() called before stage_inputs(); no token count is staged."
                )
            num_tokens = int(output.shape[0])
        dst = output if output is not None else workspace.output
        y = dst[:num_tokens]
        k = self._kernel_config
        with frontend.kernel_options_scope(self._options()):
            frontend.fp8_fp4_mega_moe(
                y, transformed_weights[0], transformed_weights[1], workspace.buffer,
                activation_clamp=k.resolved_clamp, fast_math=k.fast_math,
            )  # fmt: skip
        workspace.eager_launches += 1
        return output if output is not None else y

    def validate_capture_ready(
        self, workspace: Sm107DeepgemmWorkspace, transformed_weights: Any
    ) -> None:
        """Called by the layer before staging when a CUDA graph capture is active: fail early (host only, no device work) when lazy work would
        run INSIDE the capture: the Triton staging kernel / CuteDSL module not loaded yet, no kernel variant of this geometry built (or, when the
        workspace knows its token bucket, that variant), the per-die SM -> locality-domain map not probed."""
        buf = workspace.buffer
        if buf is None:
            raise RuntimeError("workspace was destroyed")
        k = self._kernel_config
        if k.tag_mode == "gridid":
            raise RuntimeError(
                "CUDA graph capture is refused with tag_mode='gridid': DeepGEMM's %gridid combine tag is identical for every replay of one graph node and returns "
                "stale results when the inputs change between replays; use tag_mode='epoch' (the default)"
            )
        missing = []
        if workspace.eager_launches == 0:
            missing.append(
                "no eager forward has run on this workspace (staging / kernel modules are loaded lazily)"
            )
        per_die = False
        if k.per_die:
            from . import locality

            per_die = locality.weights_localized(transformed_weights[0])
            if not locality.has_balanced_sm_locality_domains():
                missing.append(
                    "the SM -> locality-domain map is not probed (locality.get_balanced_sm_locality_domains())"
                )
        built = frontend.built_block_ms(
            buf.group.size(), buf.num_experts, buf.num_max_tokens_per_rank, buf.num_topk, buf.hidden, buf.intermediate_hidden, buf.num_shared_experts,
            activation_clamp=k.resolved_clamp, fast_math=k.fast_math, num_sms=buf.num_sms, per_die=per_die, localized_weights=per_die,
            kernel_options=self._options(),
        )  # fmt: skip
        if not built:
            missing.append(
                "no kernel variant of this geometry is built (run layer.warmup(...), or precompile_mega_moe_variants on every rank)"
            )
        elif workspace.compile_tokens_per_rank is not None:
            bm = self._options().block_m or heuristics.get_block_m_for_mega_moe(
                buf.group.size(), buf.num_experts, buf.num_max_tokens_per_rank, workspace.compile_tokens_per_rank, buf.num_topk, "fp8xfp4"
            )  # fmt: skip
            if bm not in built:
                missing.append(
                    f"the BLOCK_M={bm} variant (token bucket {workspace.compile_tokens_per_rank}) is not built; built: {sorted(built)}"
                )
        if missing:
            raise RuntimeError(
                "MegaMoE workspace is not warmed for CUDA graph capture: "
                + "; ".join(missing)
                + ". Call layer.warmup(...) on all EP ranks before capturing (the first launch of a workspace builds every BLOCK_M variant with precompile_on_warmup=True)."
            )

    def _forget_workspace_state(self, workspace: Sm107DeepgemmWorkspace) -> None:
        workspace.note_staged_tokens(None)
