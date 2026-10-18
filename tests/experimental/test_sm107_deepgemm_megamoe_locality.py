# SPDX-License-Identifier: Apache-2.0
# Port of DeepGEMM (MIT License, Copyright (c) 2025 DeepSeek; see licenses/LICENSE.deepgemm.txt) sm107_fp8_fp4_mega_moe.cuh / locality-domain tests (tests/test_locality_domain*.py).
"""Per-die (locality domain) host helpers of the SM107 MegaMoE: ``locality.py`` (pure Python on cuda-python 13.4, no ``deep_gemm``
import) and the per-die path of the frontend / Tier-2 backend.

``deep_gemm`` is used ONLY as an oracle (its native ``sm107_*`` ops allocate / classify / probe the same way): shapes, strides, homes,
the SM -> domain map and the localized values are compared with it.  GPU tests need a device that reports exactly 2 locality domains
(Rubin GPUs do) and skip otherwise.
"""

import gc

import pytest

torch = pytest.importorskip("torch")

from flashinfer.experimental.sm107_deepgemm_megamoe import locality as L  # noqa: E402

needs_domains = pytest.mark.skipif(
    not torch.cuda.is_available() or not L.is_localization_available(),
    reason="needs a compute capability 10.7 GPU with 2 locality domains",
)


def _dg_ops():
    pytest.importorskip("deep_gemm")
    return torch.ops.deep_gemm


def test_locality_module_does_not_import_deep_gemm():
    """The per-die host helpers are pure Python on cuda-python: importing them (and the frontend / compat that use them) never imports ``deep_gemm``."""
    import subprocess
    import sys

    code = ("import sys, flashinfer.experimental.sm107_deepgemm_megamoe.locality as L, flashinfer.experimental.sm107_deepgemm_megamoe.compat as c, "
            "flashinfer.experimental.sm107_deepgemm_megamoe.frontend as f; print('DG_IMPORTED', 'deep_gemm' in sys.modules)")  # fmt: skip
    proc = subprocess.run(
        [sys.executable, "-W", "ignore", "-c", code],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "DG_IMPORTED False" in proc.stdout, proc.stdout


# ---------------------------------------------------------------------------------------------------------------------
# CPU: the balancer (regression cases of DeepGEMM's tests/test_locality_domain_balance.py)
# ---------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("domain_one_tpcs", [0, 1, 51, 52, 53, 103, 104])
def test_balance_moves_only_surplus_tpcs(domain_one_tpcs):
    domains = torch.tensor(
        [1] * domain_one_tpcs + [0] * (104 - domain_one_tpcs), dtype=torch.uint8
    ).repeat_interleave(2)
    original = domains.clone()
    balanced = L.balance_sm_locality_domains(domains)
    assert torch.equal(domains, original)
    assert balanced.tolist().count(0) == 104 and balanced.tolist().count(1) == 104
    assert torch.equal(balanced[::2], balanced[1::2])
    assert int((balanced != original).sum()) == abs(domain_one_tpcs - 52) * 2


def test_balance_hecate_sized_map():
    # 106 TPCs (212 SMs): a 100/6 split moves exactly the 47 surplus TPCs, a balanced map is untouched
    domains = torch.tensor([1] * 100 + [0] * 6, dtype=torch.uint8).repeat_interleave(2)
    balanced = L.balance_sm_locality_domains(domains)
    assert (
        balanced.sum().item() * 2 == 212 and int((balanced != domains).sum()) == 2 * 47
    )
    even = (torch.arange(106) % 2).to(torch.uint8).repeat_interleave(2)
    assert torch.equal(L.balance_sm_locality_domains(even), even)


# ---------------------------------------------------------------------------------------------------------------------
# GPU: queries, allocation, localize
# ---------------------------------------------------------------------------------------------------------------------
@needs_domains
def test_queries_match_deepgemm():
    ops = _dg_ops()
    assert (
        L.get_num_locality_domains() == int(ops.sm107_get_num_locality_domains()) == 2
    )
    assert L.is_localization_available()
    g = L.get_granularity()
    assert g > 0 and g % (1 << 16) == 0  # 2 MiB on Rubin


@needs_domains
@pytest.mark.parametrize("sizes,dtype", [([4, 4608, 2560], torch.int8), ([2304, 5120], torch.float8_e4m3fn), ([3, 5120, 1152], torch.int8),
                                         ([7, 128], torch.bfloat16), ([1], torch.int32), ([5, 3, 11], torch.uint8)])  # fmt: skip
def test_empty_per_domain_matches_deepgemm(sizes, dtype):
    ops = _dg_ops()
    mine = L.empty_per_domain(sizes, dtype)
    ref = ops.sm107_empty_per_domain(sizes, dtype)
    assert mine.shape == ref.shape == (2, *sizes) and mine.dtype == ref.dtype == dtype
    assert (
        mine.stride() == ref.stride()
    )  # slices padded to the granularity, contiguous inside
    assert mine.device == ref.device and mine.is_cuda
    for t in (mine, ref):
        assert [
            L.get_locality_domain_of(t.data_ptr() + d * t.stride(0) * t.element_size())
            for d in range(2)
        ] == [0, 1]
        assert L.is_localized(t) and bool(ops.sm107_is_localized(t))
    # slice d is writable and independent of the other slice
    raw = mine if dtype == torch.uint8 else mine.view(torch.uint8)
    raw[0].fill_(0x11)
    raw[1].zero_()
    assert (
        int(raw[0].to(torch.int64).sum()) == 0x11 * raw[0].numel()
        and int(raw[1].to(torch.int64).sum()) == 0
    )


@needs_domains
@pytest.mark.parametrize("shape,dtype,dim", [((4, 4608, 2560), torch.int8, -2), ((3, 5120, 1152), torch.int8, -2), ((4608, 5120), torch.float8_e4m3fn, -2),
                                             ((4, 6144, 1792), torch.int8, 0), ((3, 512, 384), torch.bfloat16, -2)])  # fmt: skip
def test_localize_matches_plain_split(shape, dtype, dim):
    ops = _dg_ops()
    g = torch.Generator(device="cuda").manual_seed(sum(shape))
    elem = torch.empty((), dtype=dtype).element_size()
    # bytes < 0x70: no NaN encodings in e4m3fn / bf16, so a value-wise copy is bit-exact too
    plain = torch.randint(
        0,
        0x70,
        (*shape[:-1], shape[-1] * elem),
        generator=g,
        dtype=torch.uint8,
        device="cuda",
    ).view(dtype)
    localized = L.localize(plain, dim=dim)
    dim = dim % len(shape)
    assert (
        localized.shape == (2, *shape[:dim], shape[dim] // 2, *shape[dim + 1 :])
        and localized.dtype == dtype
    )
    assert all(localized[d].is_contiguous() for d in range(2))
    assert L.is_localized(localized) and bool(ops.sm107_is_localized(localized))
    # round trip: slice d == half d of the plain tensor along `dim` (bitwise)
    back = localized.movedim(0, dim).flatten(dim, dim + 1)
    assert torch.equal(back.view(torch.uint8), plain.view(torch.uint8))
    # DeepGEMM's own localize (same copy recipe on its native allocation) gives the same layout and bytes
    ref = ops.sm107_empty_per_domain(list(localized.shape[1:]), dtype)
    ref.copy_(plain.unflatten(dim, (2, -1)).movedim(dim, 0))
    assert localized.stride() == ref.stride()
    assert torch.equal(localized.view(torch.uint8), ref.view(torch.uint8))
    # a view of whole experts (plain copy between experts) stays consistent with the plain tensor
    if len(shape) == 3 and dim == 1:
        localized[:, 1].copy_(localized[:, 0])
        plain[1] = plain[0]
        assert torch.equal(
            localized.movedim(0, dim).flatten(dim, dim + 1).view(torch.uint8),
            plain.view(torch.uint8),
        )
    with pytest.raises(ValueError):
        L.localize(
            torch.zeros((3, 5), dtype=torch.int8, device="cuda"), dim=-1
        )  # odd extent does not split


@needs_domains
def test_is_localized_negative_and_view_cases():
    loc = L.localize(
        torch.randint(-100, 100, (4, 512, 256), dtype=torch.int8, device="cuda")
    )
    plain = torch.empty(loc.shape, dtype=torch.int8, device="cuda")
    assert (
        L.is_localized(torch.nn.Parameter(loc, requires_grad=False))
        and L.is_localized(loc[:, :1])
        and L.is_localized(loc[:, 1:3])
    )
    assert (
        not L.is_localized(plain)
        and not L.is_localized(loc[0])
        and not L.is_localized(plain[0])
        and not L.is_localized(loc[..., :8])
    )
    assert not L.is_localized(loc[:, 1:3, 10:])  # slices must stay contiguous
    assert not L.is_localized(torch.empty(2, device="cuda")) and not L.is_localized(
        loc.transpose(-1, -2)
    )
    assert (
        L.weights_localized((loc, torch.zeros(3)))
        and L.weights_localized(loc)
        and not L.weights_localized((plain, loc))
        and not L.weights_localized(None)
    )
    v = L.get_locality_domain_of(plain.data_ptr())
    assert v == -1, v  # ordinary allocations have no locality domain
    # canonical: the full-size tensor of localize() / empty_per_domain(); sub-views keep homes but not the padded slice stride the kernel bakes in
    big = L.localize(
        torch.randint(-100, 100, (4, 2048, 1024), dtype=torch.int8, device="cuda")
    )  # 4 MiB per slice = 2 granules
    assert (
        L.is_canonical_localized(loc)
        and L.is_canonical_localized(big)
        and L.is_canonical_localized(torch.nn.Parameter(big, requires_grad=False))
    )
    assert (
        not L.is_canonical_localized(big[:, :1])
        and not L.is_canonical_localized(big[:, 1:3])
        and not L.is_canonical_localized(plain)
    )  # padded to fewer granules
    assert L.is_localized(big[:, :1]) and big.stride(
        0
    ) * big.element_size() == L.get_slice_stride_bytes(
        big.shape[1:], big.element_size()
    )
    assert (
        L.get_slice_stride_bytes([1], 1)
        == L.get_granularity()
        == L.get_slice_stride_bytes([L.get_granularity()], 1)
        and L.get_slice_stride_bytes([L.get_granularity() + 1], 1)
        == 2 * L.get_granularity()
    )


@needs_domains
def test_localized_memory_is_released_with_the_last_view():
    torch.cuda.synchronize()
    gc.collect()
    free0 = torch.cuda.mem_get_info()[0]
    big = L.empty_per_domain([1 << 28], torch.uint8)  # 2 x 256 MiB
    view = big[:, :1024]
    free1 = torch.cuda.mem_get_info()[0]
    assert free0 - free1 >= (512 << 20) - (64 << 20)
    del big
    gc.collect()
    torch.cuda.synchronize()
    assert torch.cuda.mem_get_info()[0] <= free1 + (
        64 << 20
    )  # a live view keeps the reservation alive
    del view
    gc.collect()
    torch.cuda.synchronize()
    assert torch.cuda.mem_get_info()[0] >= free0 - (
        64 << 20
    )  # the last view released it


@needs_domains
def test_localize_mega_moe_weights_routed_only_scales_untouched():
    w1 = torch.randint(-100, 100, (4, 1024, 256), dtype=torch.int8, device="cuda")
    w2 = torch.randint(-100, 100, (4, 512, 128), dtype=torch.int8, device="cuda")
    sf1 = torch.zeros((4, 1024, 2), dtype=torch.int32, device="cuda")
    sf2 = torch.zeros((4, 512, 1), dtype=torch.int32, device="cuda")
    l1, l2, s1, s2 = L.localize_mega_moe_weights((w1, sf1), (w2, sf2))
    assert (
        s1 is None and s2 is None and l1[1] is sf1 and l2[1] is sf2
    )  # scale factors stay where they are
    assert (
        l1[0].shape == (2, 4, 512, 256)
        and l2[0].shape == (2, 4, 256, 128)
        and L.weights_localized(l1)
        and L.weights_localized(l2)
    )
    assert torch.equal(l1[0].movedim(0, 1).flatten(1, 2), w1)
    # shared pair (mode "1"): 2D weights -> (2, N/2, K)
    sw1 = torch.randn((512, 256), device="cuda").to(torch.float8_e4m3fn)
    _, _, ls1, _ = L.localize_mega_moe_weights(
        (w1, sf1), (w2, sf2), (sw1, sf1), (sw1.T.contiguous(), sf2)
    )
    assert ls1[0].shape == (2, sw1.shape[0] // 2, 256) and L.weights_localized(ls1)


# ---------------------------------------------------------------------------------------------------------------------
# GPU: SM -> locality domain map (latency probe)
# ---------------------------------------------------------------------------------------------------------------------
@needs_domains
@pytest.mark.filterwarnings("ignore:The CUDA Graph is empty")
def test_sm_locality_map_probe_matches_deepgemm_and_is_balanced():
    ops = _dg_ops()
    L.destroy()
    assert not L.has_balanced_sm_locality_domains()
    num_sms = torch.cuda.get_device_properties(
        torch.cuda.current_device()
    ).multi_processor_count
    raw = L.get_sm_locality_domains()
    balanced = L.get_balanced_sm_locality_domains()
    assert (
        L.has_balanced_sm_locality_domains()
        and balanced is L.get_balanced_sm_locality_domains()
    )  # cached
    for t in (raw, balanced):
        assert (
            t.dtype == torch.uint8
            and t.is_cuda
            and t.numel() == num_sms
            and t.is_contiguous()
        )
        tpcs = t.view(-1, L.NUM_SMS_PER_TPC)
        assert torch.equal(
            tpcs[:, 0], tpcs[:, 1]
        )  # the two SMs of a TPC share their locality domain
        assert 0 < int(t.sum()) < num_sms  # both domains present
    assert int(balanced.sum()) * 2 == num_sms  # equal TPC counts
    # the physical %smid -> domain map is a hardware property: DeepGEMM's own probe (native C++ kernel) finds the same one
    import deep_gemm.mega.sm107_locality as dg_locality

    reference = dg_locality.get_balanced_sm_locality_domains()
    assert torch.equal(reference.cpu(), balanced.cpu()), int(
        (reference != balanced).sum()
    )
    # destroy() drops the caches; re-probing gives the same map
    L.destroy()
    assert not L.has_balanced_sm_locality_domains()
    assert torch.equal(L.get_balanced_sm_locality_domains().cpu(), balanced.cpu())
    # the probe refuses to run inside a CUDA graph capture when the map is not cached yet
    L.destroy()
    g = torch.cuda.CUDAGraph()
    with (
        pytest.raises(L.LocalityDomainError, match="graph capture"),
        torch.cuda.graph(g),
    ):
        L.get_balanced_sm_locality_domains()
    torch.cuda.synchronize()
    L.get_balanced_sm_locality_domains()  # leave the cache warm for the following tests
    del ops


# ---------------------------------------------------------------------------------------------------------------------
# GPU: 3D weight tensor map geometry encodes on real die-local tensors
# ---------------------------------------------------------------------------------------------------------------------
def _encode_3d(drv, dtype_name, address, spec):
    u64, u32 = drv.cuuint64_t, drv.cuuint32_t
    dtype = {
        "16U4_ALIGN8B": drv.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B,
        "UINT8": drv.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_UINT8,
    }[dtype_name]
    swizzle = {64: drv.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_64B}[spec.swizzle_bytes]
    return drv.cuTensorMapEncodeTiled(
        dtype, 3, address, [u64(v) for v in spec.global_dims], [u64(v) for v in spec.global_strides_bytes], [u32(v) for v in spec.box_dims], [u32(1)] * 3,
        drv.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle, drv.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        drv.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
    )  # fmt: skip


@needs_domains
@pytest.mark.filterwarnings("ignore:.*is an experimental FlashInfer API")
def test_core_wrappers_and_compat_reexports_of_the_per_die_helpers():
    import flashinfer.mega_moe_deepgemm as core
    from flashinfer.experimental.sm107_deepgemm_megamoe import compat

    assert (
        core.is_localization_available() is True
        and compat.is_localization_available is L.is_localization_available
    )
    assert (
        core.get_balanced_sm_locality_domains() is L.get_balanced_sm_locality_domains()
    )
    x = torch.randint(-100, 100, (2, 8, 4), dtype=torch.int8, device="cuda")
    assert torch.equal(
        core.localize(x).movedim(0, -3).flatten(-3, -2), x
    ) and L.is_localized(core.localize(x))
    a, b, c, d = core.localize_mega_moe_weights((x, None), (x, None))
    assert (
        L.weights_localized(a)
        and L.weights_localized(b)
        and c is None
        and d is None
        and a[1] is None
    )
    for name in (
        "localize",
        "localize_mega_moe_weights",
        "get_balanced_sm_locality_domains",
        "is_localization_available",
    ):
        assert getattr(core, name).is_experimental and getattr(compat, name) is getattr(
            L, name
        )


@needs_domains
def test_perdie_weight_tma_geometry_encodes_with_the_driver():
    """The geometry handed to the kernel (``heuristics.get_perdie_weight_tma_spec``) must be accepted by ``cuTensorMapEncodeTiled`` for a REAL
    die-local weight tensor: 3D map ``(K, E_l * N/2 rows, 2 domains)``, strides ``(row bytes, domain-slice bytes)``, box ``(128, 128, 1)``."""
    import dataclasses

    from cuda.bindings import driver as drv

    from flashinfer.experimental.sm107_deepgemm_megamoe import heuristics as h

    H, I = 5120, 2304
    cfg = h.get_sm107_kernel_config(
        4,
        384,
        1920,
        100,
        6,
        H,
        I,
        1,
        num_sms=212,
        per_die=True,
        localized_weights=True,
        localized_shared_weights=True,
    )
    assert (
        cfg.num_experts_per_rank,
        cfg.num_locality_domains,
        cfg.localized_weights,
        cfg.localized_shared_weights,
    ) == (96, 2, True, True)
    # real die-local routed L1 weights of 3 experts: (2, 3, I, H/2) packed FP4 (int8); the spec is for E_l = 96, the encode check uses E_l = 3 rows
    w1 = L.empty_per_domain([3, I, H // 2], torch.int8)
    spec = h.get_perdie_weight_tma_spec(
        cfg,
        "routed_l1",
        row_stride_bytes=w1.stride(-2) * w1.element_size(),
        domain_stride_bytes=w1.stride(0) * w1.element_size(),
    )
    assert (
        spec.global_dims == (H, 96 * I, 2)
        and spec.box_dims == (128, 128, 1)
        and spec.swizzle_bytes == 64
        and spec.dtype == "16U4_ALIGN8B"
    )
    assert spec.global_strides_bytes == (H // 2, w1.stride(0))
    # defaults: the contiguous row bytes and the padded slice stride of the real allocation (compile-time constants of the kernel's tensor map)
    cfg3 = h.get_sm107_kernel_config(
        4,
        12,
        1920,
        100,
        6,
        H,
        I,
        1,
        num_sms=212,
        per_die=True,
        localized_weights=True,
        localized_shared_weights=True,
    )
    assert cfg3.num_experts_per_rank == 3
    assert h.get_perdie_domain_stride_bytes(cfg3, "routed_l1") == w1.stride(
        0
    ) * w1.element_size() and h.get_perdie_weight_tma_spec(
        cfg3, "routed_l1"
    ).global_strides_bytes == (H // 2, w1.stride(0))
    assert L.get_granularity() == h.LOCALITY_ALLOC_GRANULARITY
    small = dataclasses.replace(spec, global_dims=(H, 3 * I, 2))
    err = _encode_3d(drv, spec.dtype, w1.data_ptr(), small)[0]
    assert err == drv.CUresult.CUDA_SUCCESS, err
    # routed L2: (2, 3, H/2, I/2): K = I
    w2 = L.empty_per_domain([3, H // 2, I // 2], torch.int8)
    spec2 = h.get_perdie_weight_tma_spec(
        cfg,
        "routed_l2",
        row_stride_bytes=w2.stride(-2) * w2.element_size(),
        domain_stride_bytes=w2.stride(0) * w2.element_size(),
    )
    assert spec2.global_dims == (I, 96 * (H // 2), 2) and spec2.box_dims == (
        128,
        128,
        1,
    )
    err = _encode_3d(
        drv,
        spec2.dtype,
        w2.data_ptr(),
        dataclasses.replace(spec2, global_dims=(I, 3 * (H // 2), 2)),
    )[0]
    assert err == drv.CUresult.CUDA_SUCCESS, err
    # shared FP8 weights: L1 (2, I, H) -> half-K boxes (64, 128, 1), UINT8
    sw1 = L.empty_per_domain([I, H], torch.float8_e4m3fn)
    sspec = h.get_perdie_weight_tma_spec(
        cfg,
        "shared_l1",
        row_stride_bytes=sw1.stride(-2),
        domain_stride_bytes=sw1.stride(0),
    )
    assert (
        sspec.global_dims == (H, I, 2)
        and sspec.box_dims == (cfg.shared_block_k, 128, 1)
        and sspec.dtype == "UINT8"
    )
    err = _encode_3d(drv, sspec.dtype, sw1.data_ptr(), sspec)[0]
    assert err == drv.CUresult.CUDA_SUCCESS, err
    sw2 = L.empty_per_domain([H // 2, I], torch.float8_e4m3fn)
    sspec2 = h.get_perdie_weight_tma_spec(
        cfg,
        "shared_l2",
        row_stride_bytes=sw2.stride(-2),
        domain_stride_bytes=sw2.stride(0),
    )
    assert sspec2.global_dims == (I, H // 2, 2)
    err = _encode_3d(drv, sspec2.dtype, sw2.data_ptr(), sspec2)[0]
    assert err == drv.CUresult.CUDA_SUCCESS, err
    # the coordinates of the tile loads: (k, row-in-slice, domain) for the first / last N blocks of a matrix
    assert h.get_perdie_weight_tile_coords(5, 0, 3, 2 * I) == (3 * 128, 5 * I, 0)
    n_blocks = (2 * I) // 128
    assert h.get_perdie_weight_tile_coords(5, n_blocks - 1, 0, 2 * I) == (
        0,
        5 * I + (n_blocks - 1) * 128 - I,
        1,
    )


# ---------------------------------------------------------------------------------------------------------------------
# GPU: frontend per-die path (stub kernel) and the Tier-2 backend
# ---------------------------------------------------------------------------------------------------------------------
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


def _sf(g, n, k):
    k128 = -(-k // 128)
    n4 = -(-n // 4) * 4
    shape, strides = (
        ((g, n, k128), (k128 * n4, 1, n4)) if g is not None else ((n, k128), (1, n4))
    )
    return torch.empty_strided(shape, strides, dtype=torch.int32, device="cuda").zero_()


def _weights(E=8, H=1024, I=512, S=0, localize_routed=False, localize_shared=False):
    l1 = (
        torch.randint(-100, 100, (E, 2 * I, H // 2), dtype=torch.int8, device="cuda"),
        _sf(E, 2 * I, H),
    )
    l2 = (
        torch.randint(-100, 100, (E, H, I // 2), dtype=torch.int8, device="cuda"),
        _sf(E, H, I),
    )
    s1 = s2 = None
    if S:
        s1 = (
            torch.randn((2 * I * S, H), device="cuda").to(torch.float8_e4m3fn),
            _sf(None, 2 * I * S, H),
        )
        s2 = (
            torch.randn((H, I * S), device="cuda").to(torch.float8_e4m3fn),
            _sf(None, H, I * S),
        )
    if localize_routed or localize_shared:
        n1, n2, ns1, ns2 = L.localize_mega_moe_weights(
            l1, l2, s1 if localize_shared else None, s2 if localize_shared else None
        )
        l1, l2 = (n1, n2) if localize_routed else (l1, l2)
        s1, s2 = (ns1, ns2) if localize_shared else (s1, s2)
    return l1, l2, s1, s2


class _Recorder:
    def __init__(self):
        self.calls = []

    def factory(self, cfg):
        def runner(*args, **kwargs):
            self.calls.append((cfg, args, kwargs))

        return runner


@needs_domains
def test_die_local_shared_weights_are_refused_on_the_real_runner_path(monkeypatch):
    """No stub factory: the refusal happens in the host check, before ``jit.get_or_compile_kernel`` is reached (a launch would read the shared rows of slice 1 from the
    wrong addresses and silently return wrong results)."""
    from flashinfer.experimental.sm107_deepgemm_megamoe import compat, frontend, jit

    group = _ep1_group()
    buf = compat.SymmBuffer(group, 8, 1920, 2, 1024, 512, num_shared_experts=1)
    L.get_balanced_sm_locality_domains()
    frontend.set_kernel_factory(None)
    reached = []
    monkeypatch.setattr(
        jit,
        "get_or_compile_kernel",
        lambda cfg, **kw: reached.append(cfg) or (lambda *a, **k: None),
    )
    y = torch.empty((100, 1024), dtype=torch.bfloat16, device="cuda")
    try:
        for kwargs in (
            dict(localize_routed=True, localize_shared=True),
            dict(localize_routed=False, localize_shared=True),
        ):
            l1, l2, s1, s2 = _weights(S=1, **kwargs)
            assert s1[0].dim() == 3 and s2[0].dim() == 3
            with pytest.raises(NotImplementedError, match="die-local shared"):
                compat.fp8_fp4_mega_moe(y, l1, l2, buf, s1, s2, activation_clamp=10.0)
        with pytest.raises(NotImplementedError, match="die-local shared"):
            frontend.precompile_mega_moe_variants(
                1,
                8,
                1920,
                2,
                1024,
                512,
                1,
                per_die=True,
                localized_weights=True,
                localized_shared_weights=True,
            )
        assert reached == []
        l1, l2, s1, s2 = _weights(
            S=1, localize_routed=True
        )  # routed die-local + plain shared: the supported production recipe reaches the JIT layer
        compat.fp8_fp4_mega_moe(y, l1, l2, buf, s1, s2, activation_clamp=10.0)
        assert (
            len(reached) == 1
            and reached[0].localized_weights
            and not reached[0].localized_shared_weights
        )
    finally:
        frontend.set_kernel_factory(None)
        buf.destroy()


@needs_domains
def test_frontend_per_die_selection_and_sm_map_argument():
    from flashinfer.experimental.sm107_deepgemm_megamoe import compat, frontend

    group = _ep1_group()
    buf = compat.SymmBuffer(group, 8, 1920, 2, 1024, 512, num_shared_experts=1)
    domains = L.get_balanced_sm_locality_domains()
    rec = _Recorder()
    frontend.set_kernel_factory(rec.factory)
    y = torch.empty((100, 1024), dtype=torch.bfloat16, device="cuda")
    try:
        # plain weights -> plain kernel, no map argument
        l1, l2, s1, s2 = _weights(S=1)
        compat.fp8_fp4_mega_moe(y, l1, l2, buf, s1, s2, activation_clamp=10.0)
        cfg, args, kw = rec.calls[-1]
        assert (
            cfg.num_locality_domains,
            cfg.localized_weights,
            cfg.localized_shared_weights,
            cfg.per_die,
        ) == (1, False, False, False) and kw == {}
        # routed weights die-local (the production recipe): per-die kernel + the cached SM map
        l1, l2, s1, s2 = _weights(S=1, localize_routed=True)
        compat.fp8_fp4_mega_moe(y, l1, l2, buf, s1, s2, activation_clamp=10.0)
        cfg, args, kw = rec.calls[-1]
        assert (
            cfg.num_locality_domains,
            cfg.localized_weights,
            cfg.localized_shared_weights,
            cfg.per_die,
        ) == (2, True, False, True)
        assert (
            cfg.num_l1_clusters_per_domain,
            cfg.num_l2_clusters_per_domain,
            cfg.num_min_steal_waves,
            cfg.domain_counter_stride,
        ) == (cfg.num_l1_clusters // 2, cfg.num_l2_clusters // 2, 2, 1)
        assert (
            set(kw) == {"sm_locality_domains"}
            and kw["sm_locality_domains"] is domains
            and kw["sm_locality_domains"].data_ptr() == domains.data_ptr()
        )
        assert args[4] is l1[0] and args[4].dim() == 4 and args[6].dim() == 4
        # die-local shared experts as well: refused, the kernel's TMA-B only implements die-local routed weights
        n_calls = len(rec.calls)
        l1, l2, s1, s2 = _weights(S=1, localize_routed=True, localize_shared=True)
        with pytest.raises(NotImplementedError, match="die-local shared"):
            compat.fp8_fp4_mega_moe(y, l1, l2, buf, s1, s2, activation_clamp=10.0)
        assert len(rec.calls) == n_calls  # nothing reached the runner
        # forcing the mode: die-local weights with the plain scheduler, plain weights with per-die scheduling (DeepGEMM's "unlocalized per-die")
        l1, l2, s1, s2 = _weights(S=1, localize_routed=True)
        frontend.set_per_die_mode(False)
        compat.fp8_fp4_mega_moe(y, l1, l2, buf, s1, s2, activation_clamp=10.0)
        cfg, args, kw = rec.calls[-1]
        assert (cfg.num_locality_domains, cfg.localized_weights, cfg.per_die) == (
            1,
            True,
            False,
        ) and kw == {}
        l1, l2, s1, s2 = _weights(S=1)
        frontend.set_per_die_mode(True)
        compat.fp8_fp4_mega_moe(y, l1, l2, buf, s1, s2, activation_clamp=10.0)
        cfg, args, kw = rec.calls[-1]
        assert (cfg.num_locality_domains, cfg.localized_weights, cfg.per_die) == (
            2,
            False,
            True,
        ) and "sm_locality_domains" in kw
        frontend.set_per_die_mode(None)
        # a sub-view of die-local weights (same homes, other slice stride) is not what the kernel's 3D tensor map assumes
        l1b, l2b, _, _ = _weights(
            E=16, localize_routed=True
        )  # 4 MiB slices (2 granules): a 4-expert sub-view pads to 1 granule
        assert l1b[0].stride(0) * l1b[0].element_size() == 2 * L.get_granularity()
        with pytest.raises(AssertionError, match="die-local"):
            compat.fp8_fp4_mega_moe(y, (l1b[0][:, :4], l1b[1]), l2b, buf)
        # host asserts of die-local weights
        l1, l2, s1, s2 = _weights(S=1, localize_routed=True)
        with pytest.raises(AssertionError, match="both"):
            compat.fp8_fp4_mega_moe(
                y, l1, _weights(S=1)[1], buf, s1, s2
            )  # mixed die-local / plain L1, L2
        plain_l1 = _weights(S=1)[0]
        bad = (
            plain_l1[0].unsqueeze(0).expand(2, -1, -1, -1).contiguous(),
            plain_l1[1],
        )  # 4D but not die-local
        with pytest.raises(AssertionError, match="die-local"):
            compat.fp8_fp4_mega_moe(
                y,
                bad,
                (torch.zeros_like(bad[0][..., :256]).contiguous(), _sf(8, 1024, 512)),
                buf,
                s1,
                s2,
            )
    finally:
        frontend.set_per_die_mode(None)
        frontend.set_kernel_factory(None)
        buf.destroy()


@needs_domains
def test_frontend_passes_the_sm_map_pointer_through_the_foundation_runner():
    """Cross-role ABI: the ``sm_locality_domains`` keyword reaches the ``MegaMoeRunner`` pointer slot ``domains_ptr`` (kernel arg #14)."""
    try:
        from flashinfer.experimental.sm107_deepgemm_megamoe import compat, frontend, jit
    except ImportError as exc:  # pragma: no cover
        pytest.skip(f"jit layer not importable: {exc}")
    group = _ep1_group()
    buf = compat.SymmBuffer(group, 8, 1920, 2, 1024, 512)
    domains = L.get_balanced_sm_locality_domains()
    captured = []

    def factory(cfg):
        try:
            build = jit.make_build(cfg)
        except Exception as exc:  # pragma: no cover
            pytest.skip(f"KernelBuild not constructible here: {exc}")
        return jit.MegaMoeRunner(build, lambda *a: captured.append(a), 0.0, "fake")

    frontend.set_kernel_factory(factory)
    try:
        l1, l2, _, _ = _weights(localize_routed=True)
        compat.fp8_fp4_mega_moe(
            torch.empty((40, 1024), dtype=torch.bfloat16, device="cuda"),
            l1,
            l2,
            buf,
            activation_clamp=10.0,
        )
        (
            _y,
            _stats,
            _n,
            _sym,
            l1w,
            _l1s,
            l2w,
            _l2s,
            _a,
            _b,
            _c,
            _d,
            trace,
            domains_ptr,
            _stream,
        ) = captured[0]
        assert (
            domains_ptr == domains.data_ptr()
            and trace == 0
            and l1w == l1[0].data_ptr()
            and l2w == l2[0].data_ptr()
        )
        assert captured[0][3] == (buf.base_ptr, (0,), 0, 1)
        captured.clear()
        l1, l2, _, _ = _weights()
        compat.fp8_fp4_mega_moe(
            torch.empty((40, 1024), dtype=torch.bfloat16, device="cuda"),
            l1,
            l2,
            buf,
            activation_clamp=10.0,
        )
        assert captured[0][13] == 0  # plain weights: no map pointer
    finally:
        frontend.set_kernel_factory(None)
        buf.destroy()


@needs_domains
def test_backend_per_die_weights_and_validation():
    from flashinfer.experimental.sm107_deepgemm_megamoe import backend
    from flashinfer.experimental.sm107_deepgemm_megamoe import config as cfgmod
    from flashinfer.moe_ep.core.validation.common import MoEEpConfigError
    from flashinfer.moe_ep.weights import MoEWeightPack
    from flashinfer.experimental.sm107_deepgemm_megamoe.weights import (
        transform_weights_for_mega_moe,
    )

    E, H, I = 4, 1024, 512
    gen = torch.Generator().manual_seed(3)
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
    pack = MoEWeightPack(w13, w2, s13, s2)
    plain = backend.preprocess_canonical_weights(
        pack, intermediate_size=I, hidden_size=H
    )
    die = backend.preprocess_canonical_weights(
        pack, intermediate_size=I, hidden_size=H, per_die=True
    )
    assert (
        die[0][1] is not None
        and torch.equal(die[0][1], plain[0][1])
        and torch.equal(die[1][1], plain[1][1])
    )  # SF identical, not localized
    assert L.weights_localized(die[0]) and L.weights_localized(die[1])
    assert torch.equal(
        die[0][0].movedim(0, 1).flatten(1, 2), plain[0][0]
    ) and torch.equal(die[1][0].movedim(0, 1).flatten(1, 2), plain[1][0])
    backend.validate_transformed_weights(
        die,
        intermediate_size=I,
        hidden_size=H,
        world_size=1,
        num_experts=E,
        per_die=True,
    )
    backend.validate_transformed_weights(
        plain, intermediate_size=I, hidden_size=H, world_size=1, num_experts=E
    )
    with pytest.raises(MoEEpConfigError):
        backend.validate_transformed_weights(
            plain,
            intermediate_size=I,
            hidden_size=H,
            world_size=1,
            num_experts=E,
            per_die=True,
        )
    with pytest.raises(MoEEpConfigError):
        backend.validate_transformed_weights(
            die, intermediate_size=I, hidden_size=H, world_size=1, num_experts=E
        )
    # reference transform path: the same as the plain `transform_weights_for_mega_moe` composition
    ref = transform_weights_for_mega_moe(
        (plain[0][0], plain[0][1]), (plain[1][0], plain[1][1])
    )
    assert ref is not None
    assert (
        cfgmod.Sm107_Mxfp8_Mxfp4_Bf16_DeepgemmCutedsl_MegaMoeConfig(
            intermediate_size=I, top_k=2, per_die=True
        ).per_die
        is True
    )
