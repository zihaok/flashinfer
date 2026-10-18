# SPDX-License-Identifier: Apache-2.0
# Port of DeepGEMM (MIT License, Copyright (c) 2025 DeepSeek; see licenses/LICENSE.deepgemm.txt) sm107_fp8_fp4_mega_moe.cuh / staging tests.
"""Input staging of the SM107 MegaMoE (BF16 -> MXFP8 + packed UE8M0 SF + routing + shared-expert SF layout).

Bit-exact against DeepGEMM's ``per_token_cast_to_fp8(use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)`` and against the loop
implementation of the shared-expert SF layout of DeepGEMM's own test (``tests/test_mega_moe.py:36-49``); when vLLM's Triton
kernel ``prepare_megamoe_inputs`` is importable the torch fallback is also compared with it bit for bit.
"""

import importlib

import pytest

torch = pytest.importorskip("torch")

from flashinfer.experimental.sm107_deepgemm_megamoe import staging as ST  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _bits(t):
    return (
        t.contiguous().view(torch.uint8)
        if t.dtype == torch.float8_e4m3fn
        else t.contiguous()
    )


def _inputs(m, h, seed, scale=1.0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = (torch.randn((m, h), generator=g, device="cuda") * scale).to(torch.bfloat16)
    return x


@pytest.mark.parametrize(
    "m,h,scale",
    [
        (1, 5120, 1.0),
        (7, 5120, 3e-3),
        (64, 4096, 1.0),
        (257, 5120, 100.0),
        (33, 128, 1e-6),
        (16, 5120, 5e4),
    ],
)
def test_per_token_cast_bit_exact_vs_deepgemm(m, h, scale):
    dg_math = pytest.importorskip("deep_gemm.utils.math")
    x = _inputs(m, h, seed=m + h, scale=scale)
    x[0, :32] = 0  # an all-zero group exercises the 1e-4 amax clamp
    if m > 1:
        x[1, 5] = 448.0 * 2.0**10  # amax landing exactly on a power of two times 448
    mine = ST.per_token_cast_to_fp8(x, use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)
    ref = dg_math.per_token_cast_to_fp8(
        x, use_ue8m0=True, gran_k=32, use_packed_ue8m0=True
    )
    assert (
        mine[0].dtype == ref[0].dtype == torch.float8_e4m3fn
        and mine[1].dtype == ref[1].dtype == torch.int32
    )
    assert mine[0].shape == ref[0].shape and mine[1].shape == ref[1].shape == (
        m,
        h // 128,
    )
    assert torch.equal(_bits(mine[0]), _bits(ref[0]))
    assert torch.equal(mine[1], ref[1])
    # unpacked / non-ue8m0 variants share the implementation: spot check
    a = ST.per_token_cast_to_fp8(x, use_ue8m0=False, gran_k=128, use_packed_ue8m0=False)
    b = dg_math.per_token_cast_to_fp8(
        x, use_ue8m0=False, gran_k=128, use_packed_ue8m0=False
    )
    assert torch.equal(_bits(a[0]), _bits(b[0])) and torch.equal(a[1], b[1])


def test_ue8m0_semantics_first_principles():
    # scale = 2^ceil(log2(amax / 448)), amax clamped >= 1e-4, exponent clamped to [1, 254]; byte j of word w = group 4w + j
    h = 256
    x = torch.zeros((2, h), dtype=torch.bfloat16, device="cuda")
    x[0, 0] = 448.0  # group 0: amax/448 = 1            -> 2^0  -> exponent 127
    x[0, 32] = 450.0  # group 1: just above (bf16 grid) -> 2^1  -> exponent 128
    x[0, 64] = 224.0  # group 2: 0.5                    -> 2^-1 -> exponent 126
    # group 3 all zero: clamp(1e-4) / 448 = 2.2e-7 -> ceil to 2^-22 -> exponent 105
    x[1, :] = 1e-30  # every group clamps to 1e-4 -> exponent 105
    fp8, sf = ST.per_token_cast_to_fp8(x)
    assert [(int(sf[0, 0]) >> (8 * j)) & 0xFF for j in range(4)] == [127, 128, 126, 105]
    assert [(int(sf[1, 0]) >> (8 * j)) & 0xFF for j in range(4)] == [105] * 4 and int(
        sf[1, 1]
    ) == 0x69696969
    assert (
        fp8[0, 0].float().item() == 448.0 and fp8[0, 64].float().item() == 448.0
    )  # amax / scale lands on the E4M3 maximum


def test_shared_sf_layout_matches_deepgemm_test_helper():
    """Reference = the loop of DeepGEMM's tests/test_mega_moe.py::_to_shared_mega_moe_sf_layout."""
    from flashinfer.experimental.sm107_deepgemm_megamoe.heuristics import align

    def reference(sf, block_m, num_max_sf_tokens):
        num_tokens, packed_sf_k = sf.shape
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
                transposed = (
                    (m_idx // 128) * 128 + (m_idx % 32) * 4 + (m_idx % 128) // 32
                )
                result[block_idx * aligned_block_m + transposed].copy_(
                    sf[block_idx * block_m + m_idx]
                )
        return result

    gen = torch.Generator(device="cuda").manual_seed(0)
    for bm in (16, 32, 64, 128, 192, 240):
        for m in (1, 17, 100, 389, 1000):
            sf = torch.randint(
                -(2**31),
                2**31 - 1,
                (m, 40),
                generator=gen,
                dtype=torch.int32,
                device="cuda",
            )
            n_s = -(-1920 // 8) * 128
            mine = ST.to_shared_mega_moe_sf_layout(sf, bm, n_s)
            ref = reference(sf, bm, n_s)
            assert mine.stride() == ref.stride() == (1, n_s) and torch.equal(
                mine, ref
            ), (bm, m)
    # write_shared_sf into an existing view only touches the rows of this call
    dst = torch.empty_strided(
        (n_s, 40), (1, n_s), dtype=torch.int32, device="cuda"
    ).fill_(7)
    sf = torch.arange(5 * 40, dtype=torch.int32, device="cuda").view(5, 40)
    ST.write_shared_sf(dst, sf, 128)
    rows = ST.get_shared_sf_rows(5, 128, "cuda")
    assert torch.equal(dst[rows], sf) and (dst.sum() - dst[rows].sum()).item() == 7 * (
        n_s * 40 - 5 * 40
    )
    with pytest.raises(AssertionError):
        ST.write_shared_sf(
            torch.zeros((8, 40), dtype=torch.int32, device="cuda"), sf, 128
        )  # insufficient rows


def test_prepare_inputs_routing_padding_and_shapes():
    m, h, k = 37, 5120, 6
    x = _inputs(m, h, seed=3)
    g = torch.Generator(device="cuda").manual_seed(4)
    ids = torch.randint(
        0, 384, (m, k), generator=g, dtype=torch.int32, device="cuda"
    )  # int32 routing input -> int64 output
    w = torch.rand((m, k), generator=g, device="cuda", dtype=torch.float32)
    pad = torch.zeros(m, dtype=torch.bool, device="cuda")
    pad[[0, 5, 36]] = True
    T = 1920
    x_out = torch.zeros((T, h), dtype=torch.float8_e4m3fn, device="cuda")
    sf_out = torch.zeros((T, h // 128), dtype=torch.int32, device="cuda")
    idx_out = torch.full((T, k), 12345, dtype=torch.int64, device="cuda")
    w_out = torch.full((T, k), 9.0, dtype=torch.float32, device="cuda")
    ST.prepare_megamoe_inputs_torch(
        x, w, ids, x_out, sf_out, idx_out, w_out, is_padding=pad
    )
    fp8, sf = ST.per_token_cast_to_fp8(x)
    assert torch.equal(_bits(x_out[:m]), _bits(fp8)) and torch.equal(sf_out[:m], sf)
    exp_idx = ids.to(torch.int64).clone()
    exp_idx[pad] = -1
    exp_w = w.clone()
    exp_w[pad] = 0
    assert torch.equal(idx_out[:m], exp_idx) and torch.equal(w_out[:m], exp_w)
    assert (idx_out[m:] == 12345).all() and (
        w_out[m:] == 9.0
    ).all()  # rows beyond num_tokens untouched
    with pytest.raises(ValueError):
        ST.prepare_megamoe_inputs(
            x, w, ids, x_out, sf_out, idx_out, w_out, shared_x_sf=sf_out
        )  # shared SF without block_m
    with pytest.raises(ValueError):
        ST.prepare_megamoe_inputs(
            x[:, :100], w, ids, x_out, sf_out, idx_out, w_out
        )  # hidden % 128
    ST.prepare_megamoe_inputs(
        x[:0], w[:0], ids[:0], x_out, sf_out, idx_out, w_out
    )  # empty batch is a no-op


def _run_staging(fn, x, w, ids, pad, with_shared, bm, h=5120, k=6, T=1920):
    n_s = -(-T // 8) * 128
    x_out = torch.zeros((T, h), dtype=torch.float8_e4m3fn, device="cuda")
    sf_out = torch.zeros((T, h // 128), dtype=torch.int32, device="cuda")
    idx_out = torch.zeros((T, k), dtype=torch.int64, device="cuda")
    w_out = torch.zeros((T, k), dtype=torch.float32, device="cuda")
    shared = (
        torch.empty_strided(
            (n_s, h // 128), (1, n_s), dtype=torch.int32, device="cuda"
        ).zero_()
        if with_shared
        else None
    )
    kw = dict(shared_x_sf=shared, shared_block_m=bm) if with_shared else {}
    if pad is not None:
        kw["is_padding"] = pad
    fn(x, w, ids, x_out, sf_out, idx_out, w_out, **kw)
    return _bits(x_out), sf_out, idx_out, w_out, shared


@pytest.mark.parametrize(
    "m,with_shared,with_pad",
    [
        (1, True, False),
        (7, False, True),
        (63, True, True),
        (64, True, False),
        (300, False, True),
        (1000, True, True),
    ],
)
def test_triton_staging_bit_exact_vs_torch_deepgemm_and_vllm(m, with_shared, with_pad):
    pytest.importorskip("triton")
    from flashinfer.experimental.sm107_deepgemm_megamoe.staging_triton import (
        prepare_megamoe_inputs_triton,
    )

    h, k = 5120, 6
    x = _inputs(m, h, seed=m)
    g = torch.Generator(device="cuda").manual_seed(m)
    ids = torch.randint(0, 384, (m, k), generator=g, dtype=torch.int32, device="cuda")
    w = torch.rand((m, k), generator=g, device="cuda", dtype=torch.float32)
    pad = (torch.rand(m, generator=g, device="cuda") < 0.2) if with_pad else None
    bm = 128
    ref = _run_staging(ST.prepare_megamoe_inputs_torch, x, w, ids, pad, with_shared, bm)
    outs = {"triton": _run_staging(prepare_megamoe_inputs_triton, x, w, ids, pad, with_shared, bm),
            "default": _run_staging(ST.prepare_megamoe_inputs, x, w, ids, pad, with_shared, bm)}  # fmt: skip
    for modname in (
        "vllm.models.deepseek_v4.nvidia.ops.prepare_megamoe",
        "vllm.model_executor.layers.fused_moe.prepare_megamoe",
    ):
        try:  # the module moved between vLLM revisions (user fork vs the image's vLLM)
            outs["vllm"] = _run_staging(
                importlib.import_module(modname).prepare_megamoe_inputs,
                x,
                w,
                ids,
                pad,
                with_shared,
                bm,
            )
            break
        except ImportError:
            continue
    for name, out in outs.items():
        for a, b in zip(ref, out, strict=False):
            assert (a is None and b is None) or torch.equal(a, b), name
    # the reference itself is DeepGEMM's cast
    dg_math = pytest.importorskip("deep_gemm.utils.math")
    fp8, sf = dg_math.per_token_cast_to_fp8(
        x, use_ue8m0=True, gran_k=32, use_packed_ue8m0=True
    )
    assert torch.equal(ref[0][:m], _bits(fp8)) and torch.equal(ref[1][:m], sf)
    if with_pad:
        assert (ref[2][:m][pad] == -1).all() and (ref[3][:m][pad] == 0).all()


def test_triton_staging_is_cuda_graph_capturable():
    pytest.importorskip("triton")
    from flashinfer.experimental.sm107_deepgemm_megamoe.staging_triton import (
        prepare_megamoe_inputs_triton,
    )

    m, h, k = 200, 5120, 6
    x = _inputs(m, h, seed=1)
    ids = torch.randint(0, 384, (m, k), dtype=torch.int64, device="cuda")
    w = torch.rand((m, k), device="cuda")
    x_out = torch.zeros((1920, h), dtype=torch.float8_e4m3fn, device="cuda")
    sf_out = torch.zeros((1920, h // 128), dtype=torch.int32, device="cuda")
    idx_out = torch.zeros((1920, k), dtype=torch.int64, device="cuda")
    w_out = torch.zeros((1920, k), dtype=torch.float32, device="cuda")
    prepare_megamoe_inputs_triton(
        x, w, ids, x_out, sf_out, idx_out, w_out
    )  # warm (compile)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        prepare_megamoe_inputs_triton(x, w, ids, x_out, sf_out, idx_out, w_out)
    x_out.zero_()
    sf_out.zero_()
    idx_out.zero_()  # noqa: E702
    graph.replay()
    torch.cuda.synchronize()
    fp8, sf = ST.per_token_cast_to_fp8(x)
    assert (
        torch.equal(_bits(x_out[:m]), _bits(fp8))
        and torch.equal(sf_out[:m], sf)
        and torch.equal(idx_out[:m], ids)
    )


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


def test_stage_inputs_into_symm_buffer_ep1():
    from flashinfer.experimental.sm107_deepgemm_megamoe.heuristics import (
        get_block_m_for_mega_moe,
    )
    from flashinfer.experimental.sm107_deepgemm_megamoe.workspace import SymmBuffer

    buf = SymmBuffer(_ep1_group(), 384, 1920, 6, 5120, 2304, num_shared_experts=1)
    m = 130
    x = _inputs(m, 5120, seed=9)
    g = torch.Generator(device="cuda").manual_seed(9)
    ids = torch.randint(0, 384, (m, 6), generator=g, dtype=torch.int64, device="cuda")
    w = torch.rand((m, 6), generator=g, device="cuda")
    bm = ST.stage_inputs(buf, x, ids, w)
    assert bm == get_block_m_for_mega_moe(1, 384, 1920, m, 6, "fp8xfp4")
    fp8, sf = ST.per_token_cast_to_fp8(x)
    assert torch.equal(_bits(buf.x[:m]), _bits(fp8)) and torch.equal(buf.x_sf[:m], sf)
    assert torch.equal(buf.topk_idx[:m], ids) and torch.equal(buf.topk_weights[:m], w)
    rows = ST.get_shared_sf_rows(m, bm, "cuda")
    assert torch.equal(buf.shared_l1_acts_sf[rows], sf)
    buf.destroy()
