# SPDX-License-Identifier: Apache-2.0
# Port of DeepGEMM (MIT License, Copyright (c) 2025 DeepSeek; see licenses/LICENSE.deepgemm.txt) sm107_fp8_fp4_mega_moe.cuh / DG_MEGA_MOE_SM107_BLOCK_M override tests.
"""A forced BLOCK_M reaches every public entry point, exactly like DeepGEMM's ``DG_MEGA_MOE_SM107_BLOCK_M``.

In DeepGEMM the override is read by ONE function that both the Python helper ``get_block_m_for_mega_moe`` and the launch go through, so a caller that stages the
fused shared expert's scale rows (``shared_l1_acts_sf``, laid out per BLOCK_M) with the helper's answer stays consistent with the kernel.  In the port the forced value is
``KernelOptions.block_m`` (``set_kernel_options(block_m=...)``): the launch, ``get_block_m_for_mega_moe`` and the default path of ``stage_inputs_for_mega_moe`` (no
``block_m=``) must all follow it.  Before the fix the launch used the forced BLOCK_M while the helper and the staging still answered with the heuristic one: the shared-expert
GEMM read scale rows of another layout (finite, silently wrong output).

EP1, V4.1 shape (H=5120, I=2304, top-6) with 32 experts, one merged FP8 shared expert (S=1), BLOCK_M forced to 128.  The token counts are chosen so that the heuristic BLOCK_M
differs from the forced one (100 -> 32, 300 -> 64, 1000 -> 240; at 500 tokens the heuristic is 128 itself and would test nothing).  The port's output must be bit-identical to
DeepGEMM's own kernel run under ``DG_MEGA_MOE_SM107_BLOCK_M=128`` on the same staged operands.

Needs one Rubin GPU (compute capability 10.7, 212 SMs) and ``deep_gemm`` (skipped otherwise).  Run: ``pytest tests/experimental/test_sm107_deepgemm_megamoe_forced_block_m.py -q``.
"""

import socket
import types

import pytest

torch = pytest.importorskip("torch")

E, TOPK, H, I, T_MAX = 32, 6, 5120, 2304, 1920
FORCED_BLOCK_M = 128
TOKEN_COUNTS = (100, 300, 1000)


def _rubin_212_sms() -> bool:
    return (
        torch.cuda.is_available()
        and torch.cuda.get_device_capability() == (10, 7)
        and torch.cuda.get_device_properties(
            torch.cuda.current_device()
        ).multi_processor_count
        == 212
    )


pytestmark = pytest.mark.skipif(
    not _rubin_212_sms(),
    reason="needs a compute capability 10.7 (Rubin) device with 212 SMs",
)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _random_mxfp4_weights(sf_to_required_layout, num_experts, n, k, gen):
    """Packed FP4 weights ``(E, n, k/2)`` int8 and UE8M0 scales in the kernel's MN-major int32 layout (random but valid values)."""
    w = torch.randint(
        -128,
        128,
        (num_experts, n, k // 2),
        dtype=torch.int8,
        device="cuda",
        generator=gen,
    )
    exp = torch.randint(
        116,
        123,
        (num_experts, n, k // 32),
        dtype=torch.int32,
        device="cuda",
        generator=gen,
    )  # scales 2^-11 .. 2^-5
    return w, sf_to_required_layout((exp << 23).view(torch.float32), n, k, num_experts)


def _random_fp8_weights(sf_to_required_layout, n, k, gen):
    """FP8 E4M3 shared-expert weights ``(n, k)`` and UE8M0 scales (2^-8) in the kernel's layout."""
    w = (torch.randn((n, k), device="cuda", generator=gen) * 0.05).to(
        torch.float8_e4m3fn
    )
    return w, sf_to_required_layout(
        torch.full((n, k // 32), 2.0**-8, device="cuda"), n, k
    )


@pytest.fixture(scope="module")
def env():
    mp = pytest.MonkeyPatch()
    mp.setenv(
        "DG_MEGA_MOE_SM107", "1"
    )  # DeepGEMM's SM107 MegaMoE kernel is opt-in (undone at teardown; nothing is written at import time)
    owns_group = False
    mega = port_buf = dg_buf = None
    try:
        deep_gemm = pytest.importorskip("deep_gemm")
        import torch.distributed as dist

        import flashinfer.mega_moe_deepgemm as mega
        from flashinfer.experimental.sm107_deepgemm_megamoe.weights import (
            sf_to_required_layout,
        )

        owns_group = not dist.is_initialized()
        if owns_group:
            dist.init_process_group(
                "nccl",
                init_method=f"tcp://127.0.0.1:{_free_port()}",
                world_size=1,
                rank=0,
                device_id=torch.device("cuda", torch.cuda.current_device()),
            )
        group = dist.new_group([0], backend="nccl")
        gen = torch.Generator(device="cuda").manual_seed(1234)
        l1, l2 = mega.transform_weights_for_mega_moe(
            _random_mxfp4_weights(sf_to_required_layout, E, 2 * I, H, gen),
            _random_mxfp4_weights(sf_to_required_layout, E, H, I, gen),
        )
        shared = mega.transform_weights_for_mega_moe(
            _random_fp8_weights(sf_to_required_layout, 2 * I, H, gen),
            _random_fp8_weights(sf_to_required_layout, H, I, gen),
        )  # one merged expert (S=1)
        port_buf = mega.get_symm_buffer_for_mega_moe(
            group, E, T_MAX, TOPK, H, I, num_shared_experts=1
        )
        dg_buf = deep_gemm.get_symm_buffer_for_mega_moe(
            group,
            E,
            T_MAX,
            TOPK,
            H,
            I,
            num_shared_experts=1,
            mma_type="fp8xfp4",
            activation="swiglu",
        )
        yield types.SimpleNamespace(
            deep_gemm=deep_gemm,
            mega=mega,
            group=group,
            l1=l1,
            l2=l2,
            s1=shared[0],
            s2=shared[1],
            port_buf=port_buf,
            dg_buf=dg_buf,
        )
    finally:
        if mega is not None:
            mega.set_kernel_options()  # never leave a forced BLOCK_M behind
        if port_buf is not None:
            port_buf.destroy()
        del dg_buf
        if owns_group:
            import torch.distributed as dist

            dist.destroy_process_group()
        mp.undo()


def _inputs(num_tokens):
    gen = torch.Generator(device="cuda").manual_seed(7000 + num_tokens)
    x = torch.randn((num_tokens, H), dtype=torch.bfloat16, device="cuda", generator=gen)
    scores = torch.randn(
        (num_tokens, E), dtype=torch.float, device="cuda", generator=gen
    )
    topk_weights, topk_ids = torch.topk(scores, TOPK, dim=-1, sorted=False)
    return x, topk_ids, topk_weights.softmax(-1)


def _run_port(env, x):
    y = torch.empty((x.size(0), H), dtype=torch.bfloat16, device="cuda")
    env.mega.fp8_fp4_mega_moe(
        y,
        env.l1,
        env.l2,
        env.port_buf,
        shared_l1_weights=env.s1,
        shared_l2_weights=env.s2,
        activation_clamp=10.0,
    )
    torch.cuda.synchronize()
    return y


@pytest.mark.parametrize("num_tokens", TOKEN_COUNTS)
def test_forced_block_m_follows_get_block_m_and_staging_and_matches_deepgemm(
    env, monkeypatch, num_tokens
):
    from flashinfer.experimental.sm107_deepgemm_megamoe import heuristics, staging

    mega, deep_gemm = env.mega, env.deep_gemm
    heuristic = heuristics.get_block_m_for_mega_moe(
        1, E, T_MAX, num_tokens, TOPK, "fp8xfp4"
    )
    assert heuristic != FORCED_BLOCK_M, (
        "pick a token count whose heuristic BLOCK_M differs from the forced one"
    )
    x, topk_ids, topk_weights = _inputs(num_tokens)
    assert (
        mega.get_block_m_for_mega_moe(1, E, T_MAX, num_tokens, TOPK) == heuristic
    )  # nothing forced: the heuristic
    try:
        mega.set_kernel_options(block_m=FORCED_BLOCK_M)
        assert (
            mega.get_block_m_for_mega_moe(1, E, T_MAX, num_tokens, TOPK)
            == FORCED_BLOCK_M
        )  # the helper follows the forced value, as DeepGEMM's does
        block_m = mega.stage_inputs_for_mega_moe(
            env.port_buf, x, topk_ids, topk_weights
        )  # the public default path: no block_m=
        assert block_m == FORCED_BLOCK_M
        x_sf = staging.per_token_cast_to_fp8(x)[1]
        rows = staging.get_shared_sf_rows(num_tokens, FORCED_BLOCK_M, "cuda")
        assert torch.equal(env.port_buf.shared_l1_acts_sf[rows], x_sf), (
            "the shared-expert scale rows are not in the forced BLOCK_M layout"
        )
        y = _run_port(env, x)  # the launch runs the forced variant
    finally:
        mega.set_kernel_options()
    assert (
        mega.get_block_m_for_mega_moe(1, E, T_MAX, num_tokens, TOPK) == heuristic
    )  # restored
    assert not torch.isnan(y.float()).any()

    # DeepGEMM's own kernel under DG_MEGA_MOE_SM107_BLOCK_M on the same staged operands (its helper must agree with the forced value as well)
    monkeypatch.setenv("DG_MEGA_MOE_SM107_BLOCK_M", str(FORCED_BLOCK_M))
    if hasattr(deep_gemm, "get_block_m_for_mega_moe"):
        assert (
            deep_gemm.get_block_m_for_mega_moe(1, E, T_MAX, num_tokens, TOPK, "fp8xfp4")
            == FORCED_BLOCK_M
        )
    src, dst = (
        env.port_buf,
        env.dg_buf,
    )  # the symmetric buffers are byte-layout compatible
    for name in ("x", "x_sf", "topk_idx", "topk_weights"):
        getattr(dst, name)[:num_tokens].copy_(getattr(src, name)[:num_tokens])
    dst.shared_l1_acts_sf.copy_(src.shared_l1_acts_sf)
    y_ref = torch.empty_like(y)
    deep_gemm.fp8_fp4_mega_moe(
        y_ref,
        env.l1,
        env.l2,
        dst,
        shared_l1_weights=env.s1,
        shared_l2_weights=env.s2,
        activation_clamp=10.0,
    )
    torch.cuda.synchronize()
    assert not torch.isnan(y_ref.float()).any()
    assert torch.equal(y.view(torch.int16), y_ref.view(torch.int16)), (
        "the port's output differs from DeepGEMM's under the same forced BLOCK_M"
    )


def test_staging_with_the_heuristic_block_m_disagrees_with_a_forced_kernel(env):
    """Control (independent of the fix): the shared scale rows staged in the heuristic BLOCK_M layout and a kernel forced to another BLOCK_M give a different
    (silently wrong) output than the rows staged in the forced layout, which is what the default path of ``stage_inputs_for_mega_moe`` produced before it followed
    the forced value."""
    from flashinfer.experimental.sm107_deepgemm_megamoe import heuristics

    mega = env.mega
    differs = []
    for num_tokens in TOKEN_COUNTS:
        x, topk_ids, topk_weights = _inputs(num_tokens)
        heuristic = heuristics.get_block_m_for_mega_moe(
            1, E, T_MAX, num_tokens, TOPK, "fp8xfp4"
        )
        try:
            mega.set_kernel_options(block_m=FORCED_BLOCK_M)
            mega.stage_inputs_for_mega_moe(
                env.port_buf, x, topk_ids, topk_weights, block_m=heuristic
            )  # the old default: the heuristic layout
            y_wrong = _run_port(env, x)
            mega.stage_inputs_for_mega_moe(
                env.port_buf, x, topk_ids, topk_weights, block_m=FORCED_BLOCK_M
            )  # the layout the forced kernel reads
            y_right = _run_port(env, x)
        finally:
            mega.set_kernel_options()
        differs.append(
            not torch.equal(y_wrong.view(torch.int16), y_right.view(torch.int16))
        )
    assert any(differs), (
        "staging in a BLOCK_M layout the kernel does not use must change the output"
    )
