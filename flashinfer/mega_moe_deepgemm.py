"""
Copyright (c) 2026 by FlashInfer team.

The entry points follow the API of DeepGEMM's MegaMoE (MIT License, Copyright (c) 2025 DeepSeek; see licenses/LICENSE.deepgemm.txt).

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# Public, DeepGEMM-compatible entry points of the experimental SM107 (Rubin) MegaMoE port
# (MXFP8 activations x MXFP4 weights -> BF16, fused dispatch / L1 / SwiGLU / L2 / combine over symmetric memory).
#
# Policy (flashinfer/experimental/README.md): public experimental APIs live in core and only validate and hand off; the
# backend lives in ``flashinfer.experimental.sm107_deepgemm_megamoe`` and is reached through function-local imports, so
# importing this module never imports the experimental package (nor torch.distributed symmetric memory, cutlass, nvshmem).
#
# Same names, signatures and semantics as ``deep_gemm.get_symm_buffer_for_mega_moe`` / ``fp8_fp4_mega_moe`` /
# ``transform_weights_for_mega_moe`` / ``get_block_m_for_mega_moe`` so that a DeepGEMM-based caller (vLLM's native
# MegaMoE path) can switch the library with a few lines.  ``SymmBuffer`` is a class: it is created by
# ``get_symm_buffer_for_mega_moe`` and re-exported lazily below.

from __future__ import annotations

from typing import Any, Optional, Tuple, Union

from .api_logging import flashinfer_experimental_api

_FEATURE = "SM107 DeepGEMM-compatible MegaMoE"


@flashinfer_experimental_api(feature=_FEATURE)
def get_symm_buffer_for_mega_moe(
    group,
    num_experts: int,
    num_max_tokens_per_rank: int,
    num_topk: int,
    hidden: int,
    intermediate_hidden: int,
    num_shared_experts: int = 0,
    use_fp8_dispatch: Union[bool, None] = None,
    mma_type: str = "fp8xfp4",
    activation: str = "swiglu",
):
    """Allocate (collectively) the symmetric buffer of the MegaMoE; returns a ``SymmBuffer``.

    ``num_max_tokens_per_rank`` is aligned up to 1920; the aligned value is ``SymmBuffer.num_max_tokens_per_rank``.
    The byte layout is bit-identical to DeepGEMM's.
    """
    from .experimental.sm107_deepgemm_megamoe.frontend import require_sm107
    from .experimental.sm107_deepgemm_megamoe.workspace import (
        get_symm_buffer_for_mega_moe as _impl,
    )

    require_sm107()
    return _impl(
        group, num_experts, num_max_tokens_per_rank, num_topk, hidden, intermediate_hidden,
        num_shared_experts, use_fp8_dispatch, mma_type, activation,
    )  # fmt: skip


@flashinfer_experimental_api(feature=_FEATURE)
def transform_weights_for_mega_moe(l1_weights, l2_weights, activation: str = "swiglu"):
    """Gate/up 8-row interleave of the L1 weights + scale factors and the UTCCP 4x32 SF transpose (pure torch, bit-exact
    with ``deep_gemm.transform_weights_for_mega_moe``)."""
    from .experimental.sm107_deepgemm_megamoe.weights import (
        transform_weights_for_mega_moe as _impl,
    )

    return _impl(l1_weights, l2_weights, activation)


@flashinfer_experimental_api(feature=_FEATURE)
def fp8_fp4_mega_moe(
    y,
    l1_weights: Tuple[Any, Any],
    l2_weights: Tuple[Any, Any],
    sym_buffer,
    shared_l1_weights: Optional[Tuple[Any, Any]] = None,
    shared_l2_weights: Optional[Tuple[Any, Any]] = None,
    cumulative_local_expert_recv_stats=None,
    recipe: Tuple[int, int, int] = (1, 1, 32),
    activation: str = "swiglu",
    activation_clamp: Optional[float] = None,
    fast_math: bool = True,
    activation_alpha: float = 1.0,
    activation_beta: float = 0.0,
):
    """Fused MegaMoE for one rank (DeepGEMM-compatible ``fp8_fp4_mega_moe``).  The BLOCK_M kernel variant is selected from
    ``y.size(0)`` of this call."""
    from .experimental.sm107_deepgemm_megamoe.frontend import fp8_fp4_mega_moe as _impl

    return _impl(
        y, l1_weights, l2_weights, sym_buffer, shared_l1_weights, shared_l2_weights, cumulative_local_expert_recv_stats,
        recipe, activation, activation_clamp, fast_math, activation_alpha, activation_beta,
    )  # fmt: skip


@flashinfer_experimental_api(feature=_FEATURE)
def get_block_m_for_mega_moe(
    num_ranks: int,
    num_experts: int,
    num_max_tokens_per_rank: int,
    num_tokens: int,
    num_topk: int,
    mma_type: str = "fp8xfp4",
) -> int:
    """BLOCK_M the kernel uses for ``num_tokens`` tokens on the calling rank (bit-exact ``deep_gemm.get_block_m_for_mega_moe``; a forced BLOCK_M, ``set_kernel_options(block_m=...)`` like DeepGEMM's ``DG_MEGA_MOE_SM107_BLOCK_M``, wins)."""
    from .experimental.sm107_deepgemm_megamoe.frontend import (
        get_block_m_for_mega_moe as _impl,
    )

    return _impl(
        num_ranks, num_experts, num_max_tokens_per_rank, num_tokens, num_topk, mma_type
    )


@flashinfer_experimental_api(feature=_FEATURE)
def get_token_alignment_for_mega_moe() -> int:
    """Alignment (1920 = lcm of the candidate BLOCK_M) applied to ``num_max_tokens_per_rank``."""
    from .experimental.sm107_deepgemm_megamoe.heuristics import (
        get_token_alignment_for_mega_moe as _impl,
    )

    return _impl()


@flashinfer_experimental_api(feature=_FEATURE)
def get_unsupported_reason_for_mega_moe(
    num_ranks: int,
    num_experts: int,
    num_max_tokens_per_rank: int,
    num_topk: int,
    hidden: int,
    intermediate_hidden: int,
    num_shared_experts: int = 0,
    mma_type: str = "fp8xfp4",
    activation: str = "swiglu",
    *,
    fast_math: bool = True,
    localized_shared_weights: bool = False,
    allow_unvalidated: bool = False,
) -> Optional[str]:
    """``None`` if the SM107 port can serve a MegaMoE call of this geometry, else the reason (host only).  Lets a DeepGEMM-API caller fall through to DeepGEMM
    per call.  Admitted: ``fp8xfp4`` + ``swiglu`` (``fast_math`` True and False), at most one merged shared expert, die-local routed (not shared) weights, at most 32
    ranks, a geometry that has run on GPUs (``heuristics.VALIDATED_GEOMETRIES``, with at least 12 experts per rank; ``allow_unvalidated=True`` skips that check) and
    every BLOCK_M variant derivable on this device.  Total: it returns a reason for any input and never raises."""
    from .experimental.sm107_deepgemm_megamoe.heuristics import (
        get_mega_moe_unsupported_reason as _impl,
    )

    return _impl(
        num_ranks, num_experts, num_max_tokens_per_rank, num_topk, hidden, intermediate_hidden, num_shared_experts, mma_type, activation,
        fast_math=fast_math, localized_shared_weights=localized_shared_weights, allow_unvalidated=allow_unvalidated,
    )  # fmt: skip


# ---- input staging and warmup (what a DeepGEMM user writes by hand around the three calls above)
@flashinfer_experimental_api(feature=_FEATURE)
def stage_inputs_for_mega_moe(
    sym_buffer, hidden_states, topk_ids, topk_weights, *, is_padding=None, block_m=None
) -> int:
    """Stage one call's inputs into ``sym_buffer``: quantize the BF16 ``hidden_states`` to MXFP8 with packed UE8M0 scales, write the top-k routing and, with a
    fused shared expert, the shared-expert scale layout of THIS call's BLOCK_M.  Returns the BLOCK_M used (``hidden_states.size(0)`` is the call's token count).
    Bit-exact with DeepGEMM's ``per_token_cast_to_fp8`` and with vLLM's ``prepare_megamoe_inputs``; one Triton launch."""
    from .experimental.sm107_deepgemm_megamoe.staging import stage_inputs as _impl

    return _impl(
        sym_buffer,
        hidden_states,
        topk_ids,
        topk_weights,
        is_padding=is_padding,
        block_m=block_m,
    )


@flashinfer_experimental_api(feature=_FEATURE)
def precompile_mega_moe_variants(
    num_ranks: int,
    num_experts: int,
    num_max_tokens_per_rank: int,
    num_topk: int,
    hidden: int,
    intermediate_hidden: int,
    num_shared_experts: int = 0,
    *,
    activation_clamp: Optional[float] = None,
    activation_alpha: float = 1.0,
    activation_beta: float = 0.0,
    fast_math: bool = True,
    per_die: bool = False,
    localized_weights: bool = False,
    localized_shared_weights: bool = False,
):
    """Build (compile or load from the disk cache) every BLOCK_M kernel variant a rank can select for ``M in [0, num_max_tokens_per_rank]``; call at engine
    init, on every rank, before any CUDA-graph capture (compiling inside a capture is an error).  Returns the BLOCK_M values built."""
    from .experimental.sm107_deepgemm_megamoe.frontend import (
        precompile_mega_moe_variants as _impl,
    )

    return _impl(
        num_ranks, num_experts, num_max_tokens_per_rank, num_topk, hidden, intermediate_hidden, num_shared_experts,
        activation_clamp=activation_clamp, activation_alpha=activation_alpha, activation_beta=activation_beta, fast_math=fast_math,
        per_die=per_die, localized_weights=localized_weights, localized_shared_weights=localized_shared_weights,
    )  # fmt: skip


# ---- kernel build options (explicit arguments instead of DeepGEMM's DG_MEGA_MOE_SM107_* environment variables)
@flashinfer_experimental_api(feature=_FEATURE)
def set_kernel_options(options=None, **kwargs):
    """Set the process-wide kernel build options, e.g. ``set_kernel_options(regs_override=(56, 40, 200))``: ``tag_mode`` (``"epoch"`` default, exact under
    CUDA-graph replay, or DeepGEMM's ``"gridid"``), ``regs_override`` (explicit ``setmaxnreg`` split; the default is the spill-free 96/88/160 at every EP size), ``faithful_regs`` (DeepGEMM's own
    split, for parity runs), ``timeout_sec``, ``block_m`` (force one variant), ``smem_capacity``, ``max_stages``.  Call before the first launch / precompile; no argument restores the defaults.  Returns the active ``KernelOptions``."""
    from .experimental.sm107_deepgemm_megamoe.frontend import (
        set_kernel_options as _impl,
    )

    return _impl(options, **kwargs)


@flashinfer_experimental_api(feature=_FEATURE)
def get_kernel_options():
    """The active ``KernelOptions`` (see :func:`set_kernel_options`)."""
    from .experimental.sm107_deepgemm_megamoe.frontend import (
        get_kernel_options as _impl,
    )

    return _impl()


# ---- per-die (Rubin locality-domain) helpers: pure Python on cuda-python 13.4, no ``deep_gemm`` import (README, "Per-die execution")
@flashinfer_experimental_api(feature=_FEATURE)
def is_localization_available() -> bool:
    """Whether this GPU is a compute capability 10.7 device that reports exactly two locality domains (die-local weights / per-die scheduling)."""
    from .experimental.sm107_deepgemm_megamoe.locality import (
        is_localization_available as _impl,
    )

    return _impl()


@flashinfer_experimental_api(feature=_FEATURE)
def get_balanced_sm_locality_domains():
    """Per-SM locality domain map (``uint8[num_sms]`` on the device, indexed by the physical ``%smid``, balanced to equal TPC counts), probed once
    and cached.  Call it OUTSIDE CUDA graph capture; the per-die kernel launch reuses the cached tensor."""
    from .experimental.sm107_deepgemm_megamoe.locality import (
        get_balanced_sm_locality_domains as _impl,
    )

    return _impl()


@flashinfer_experimental_api(feature=_FEATURE)
def localize(t, dim: int = -2):
    """``(*B, N, K)`` -> ``(2, *B, N / 2, K)``: a new allocation whose slice ``d`` is homed in locality domain ``d`` (one copy)."""
    from .experimental.sm107_deepgemm_megamoe.locality import localize as _impl

    return _impl(t, dim)


@flashinfer_experimental_api(feature=_FEATURE)
def localize_mega_moe_weights(l1, l2, shared_l1=None, shared_l2=None):
    """Localize transformed MegaMoE ``(weights, sf)`` tuples (scale factors stay where they are).  The production recipe localizes the routed pair only;
    die-local SHARED weights are not implemented and are refused at launch (keep the shared pair plain).  Call after
    ``transform_weights_for_mega_moe`` and before the first launch / CUDA graph capture; ``fp8_fp4_mega_moe`` then selects the per-die
    kernel variant by itself."""
    from .experimental.sm107_deepgemm_megamoe.locality import (
        localize_mega_moe_weights as _impl,
    )

    return _impl(l1, l2, shared_l1, shared_l2)


def __getattr__(
    name: str,
) -> Any:  # PEP 562: lazy re-export of the class (cannot carry the decorator)
    if name == "SymmBuffer":
        from .experimental.sm107_deepgemm_megamoe.workspace import SymmBuffer

        return SymmBuffer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "SymmBuffer",  # noqa: F822 (resolved lazily by the module __getattr__)
    "fp8_fp4_mega_moe",
    "get_balanced_sm_locality_domains",
    "get_block_m_for_mega_moe",
    "get_kernel_options",
    "get_symm_buffer_for_mega_moe",
    "get_token_alignment_for_mega_moe",
    "get_unsupported_reason_for_mega_moe",
    "is_localization_available",
    "localize",
    "localize_mega_moe_weights",
    "precompile_mega_moe_variants",
    "set_kernel_options",
    "stage_inputs_for_mega_moe",
    "transform_weights_for_mega_moe",
]
