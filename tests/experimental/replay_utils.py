# SPDX-License-Identifier: Apache-2.0
# Port of DeepGEMM (MIT License, Copyright (c) 2025 DeepSeek; see licenses/LICENSE.deepgemm.txt) sm107_fp8_fp4_mega_moe.cuh / replay_utils (test helpers).
"""EP1 replay of DeepGEMM's post-dispatch state: the bit-exactness oracle of the GEMM core, reusable by the other kernel tests.

Strategy (BRIEF_G "Test strategy"):

1. run DeepGEMM's ``fp8_fp4_mega_moe`` at EP1 (one GPU, ``group.size() == 1``, S = 0) on a small problem whose pool blocks fit the ring (no wrap);
2. snapshot the whole symmetric buffer and ``y`` (DeepGEMM's cleanup has zeroed the signals again, but ``l1_token`` / ``l1_sf`` / ``l1_topk_weights`` /
   ``token_src_metadata`` and the results ``l2_token`` / ``l2_sf`` / ``combine_token`` are intact);
3. restore the snapshot, **re-stage what the cleanup zeroed** (``expert_recv_count_sum`` = ``(kNumSMs * kNumRanks) << 32 | count`` per expert,
   ``l1_full_count[ring block] = BLOCK_M`` for every used pool block = generation 0 "all tokens arrived"), zero the regions the GEMM part must produce
   (``l2_token``, ``l2_sf``, ``combine_token``);
4. launch the GEMM part (``mode_mask = MODE_GEMM``: scheduler + TMA-A/B + MMA + both epilogues; no dispatch / combine) on that buffer;
5. compare ``l2_token`` (valid rows), ``l2_sf`` (the UE8M0 bytes of valid tokens in the UTCCP-transposed layout) and ``combine_token[slot][token]`` rows
   (10240 B, only valid (token, slot) pairs) bit-exactly with the snapshot, and the torch ascending-slot FP32 reduce of the replayed ``combine_token``
   with DeepGEMM's ``y``.

Notes: the padding rows of a partial store block hold stale TMEM data (DeepGEMM writes them too): only valid rows/tokens are comparable.  Ring wrap
(generation >= 1) cannot be replayed from a snapshot (pool blocks > ring blocks -> the case is reported as ``skipped``).  The GEMM-only kernel leaves
the scheduler counters dirty (no cleanup warps): :func:`restore_clean_state` puts the buffer back into DeepGEMM's clean post-cleanup state (the snapshot),
which is REQUIRED before the next DeepGEMM call on the same buffer (it would hang on stale scheduler state otherwise).

The module also holds the CPU-side quantisation helpers and the Python model of the scheduler (``ref_task_sequence``) used by the unit tests.
"""

from __future__ import annotations

import dataclasses
import os
import types
from typing import Callable, List, Optional, Sequence, Tuple

import torch

FP4_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)  # E2M1 positive codes


# ---------------------------------------------------------------------------------------------------------------------
# CPU helpers (quantised-data construction, exact decode, layouts)
# ---------------------------------------------------------------------------------------------------------------------
def e8m0_scale(exp_bytes: torch.Tensor) -> torch.Tensor:
    """UE8M0 exponent byte -> 2^(e - 127) (float64)."""
    return torch.pow(2.0, exp_bytes.to(torch.float64) - 127.0)


def make_fp8_bytes(shape, gen: torch.Generator) -> torch.Tensor:
    """Random E4M3 bytes with exponents 5..8 (values 2^-2 .. 2^1), so that all products / partial sums are exact in FP32."""
    s = torch.randint(0, 2, shape, generator=gen, dtype=torch.int32)
    e = torch.randint(5, 9, shape, generator=gen, dtype=torch.int32)
    m = torch.randint(0, 8, shape, generator=gen, dtype=torch.int32)
    return ((s << 7) | (e << 3) | m).to(torch.uint8)


def decode_fp8(b: torch.Tensor) -> torch.Tensor:
    return b.view(torch.float8_e4m3fn).to(torch.float64)


def make_fp4_codes(shape, gen: torch.Generator) -> torch.Tensor:
    return torch.randint(0, 16, shape, generator=gen, dtype=torch.int32)


def decode_fp4(codes: torch.Tensor) -> torch.Tensor:
    mag = torch.tensor(FP4_MAGNITUDES, dtype=torch.float64)[(codes & 7).long()]
    return torch.where((codes & 8) != 0, -mag, mag)


def pack_fp4(codes: torch.Tensor) -> torch.Tensor:
    """(rows, K) codes -> (rows, K/2) bytes, low nibble = even k."""
    c = codes.to(torch.int32)
    return (c[:, 0::2] | (c[:, 1::2] << 4)).to(torch.uint8)


def pack_sf_words(exp_bytes: torch.Tensor) -> torch.Tensor:
    """(..., ngroups) uint8 -> (..., ngroups/4) int32, byte b of word w = group 4w + b (two's-complement wrap)."""
    e = exp_bytes.to(torch.int64)
    w = e[..., 0::4] | (e[..., 1::4] << 8) | (e[..., 2::4] << 16) | (e[..., 3::4] << 24)
    return ((w + (1 << 31)) % (1 << 32) - (1 << 31)).to(torch.int32)


def utccp_pos(idx: int, sf_block: int = 0) -> int:
    """K:184-188: position of token ``idx`` (inside a block) in the UTCCP-ordered SF block; ``sf_block`` is unused (kept for call-site symmetry)."""
    return (idx & ~127) + (idx & 31) * 4 + ((idx >> 5) & 3)


def ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def warmup_waves(
    num_total_m_blocks: int,
    num_clusters: int,
    num_l1_n_clusters: int,
    num_l2_n_clusters: int,
) -> int:
    """``get_num_l1_warmup_waves`` (SCH0): ``max(ceil(ceil(nc / l2n) * l1n / nc), ceil((l1n + (M - 1) * max(l1n - l2n, 0)) / nc) + 1)``."""
    nc, l1n, l2n, M = (
        num_clusters,
        num_l1_n_clusters,
        num_l2_n_clusters,
        num_total_m_blocks,
    )
    return max(
        ceil_div(ceil_div(nc, l2n) * l1n, nc),
        ceil_div(l1n + (M - 1) * max(l1n - l2n, 0), nc) + 1,
    )


def ref_task_sequence(
    counts: Sequence[int],
    block_m: int,
    num_l1_n_clusters: int,
    num_l2_n_clusters: int,
    num_clusters: int,
    shape_n1: int,
    shape_k1: int,
    shape_n2: int,
    shape_k2: int,
):
    """Python model of ``MegaMoESchedulerLD`` for ONE scheduler (one CTA pair; ``kNumLocalityDomains == 1``, S = 0): the published task sequence as tuples
    ``(phase, local_expert, m_block_in_expert, n_cluster, pool_block, valid_m, shape_n, shape_k)`` (phase 1 = Linear1, 2 = Linear2), plus the number of pool
    blocks.  Warm-up waves: ``max(ceil(ceil(nc / l2n) * l1n / nc), ceil((l1n + (M - 1) * max(l1n - l2n, 0)) / nc) + 1)`` capped by the total L1 waves."""
    DONE = -1
    blocks = []
    for e, c in enumerate(counts):
        for mb in range(ceil_div(c, block_m)):
            blocks.append((e, mb, min(c - mb * block_m, block_m)))
    M = len(blocks)
    l1n, l2n, nc = num_l1_n_clusters, num_l2_n_clusters, num_clusters
    total_l1_waves = ceil_div(M * l1n, nc)
    waves = min(warmup_waves(M, nc, l1n, l2n), total_l1_waves)
    l1c = l2c = 0
    out = []
    while True:
        if waves != DONE and waves != 0:
            waves -= 1
            idx = l1c
            l1c += 1
            if idx >= M * l1n:
                waves = DONE
                continue
            m, n = idx // l1n, idx % l1n
            e, mb, vm = blocks[m]
            out.append((1, e, mb, n, m, vm, shape_n1, shape_k1))
        else:
            idx = l2c
            l2c += 1
            if idx >= M * l2n:
                break
            if waves != DONE:
                waves = 1
            m, n = idx // l2n, idx % l2n
            if (
                l1c < (m + 1) * l1n
            ):  # the real scheduler would wait for the other CTA pairs; a single pair would spin forever
                raise AssertionError(
                    "single-pair reference would deadlock (L1 tasks of the block not yet claimed)"
                )
            e, mb, vm = blocks[m]
            out.append((2, e, mb, n, m, vm, shape_n2, shape_k2))
    return out, M


def ref_shared_tasks(
    num_tokens: int, block_m: int, shape_n: int, shape_k: int, phase: int
):
    """``shared_mainloop`` (SCH:274-288), one scheduler / one locality domain: the published shared tasks ``(phase, 0, m, n, m, valid_m, N, K)`` (phase 3 =
    SharedLinear1, 4 = SharedLinear2) in claim order (m-block major, N-cluster fastest; ``pool_block == m_block``)."""
    nclusters = shape_n // 256
    out = []
    for m in range(ceil_div(num_tokens, block_m)):
        valid_m = min(num_tokens - m * block_m, block_m)
        for n in range(nclusters):
            out.append((phase, 0, m, n, m, valid_m, shape_n, shape_k))
    return out


def shared_l2_is_early(
    num_tokens: int, block_m: int, shared_l2_shape_n: int, num_sms: int
) -> bool:
    """SCH:296: the shared L2 tasks go BEFORE the routed tasks when one wave fits the dispatch gap."""
    return ceil_div(num_tokens, block_m) * (shared_l2_shape_n // 256) <= num_sms // 2


def ref_full_task_sequence(
    counts: Sequence[int],
    block_m: int,
    num_l1_n_clusters: int,
    num_l2_n_clusters: int,
    num_clusters: int,
    shape_n1: int,
    shape_k1: int,
    shape_n2: int,
    shape_k2: int,
    num_tokens: int,
    num_sms: int,
    shared=None,
):
    """Published task sequence of one scheduler with S >= 0: ``shared = (shared_n1, shared_k1, shared_n2, shared_k2)`` or None.  Order (SCH:300-335):
    shared L1 -> [shared L2 if early] -> routed (warm-up, strict L2/L1 alternation) -> [shared L2 if late]."""
    routed, nblocks = ref_task_sequence(
        counts,
        block_m,
        num_l1_n_clusters,
        num_l2_n_clusters,
        num_clusters,
        shape_n1,
        shape_k1,
        shape_n2,
        shape_k2,
    )
    if shared is None:
        return routed, nblocks
    sn1, sk1, sn2, sk2 = shared
    l1 = ref_shared_tasks(num_tokens, block_m, sn1, sk1, 3)
    l2 = ref_shared_tasks(num_tokens, block_m, sn2, sk2, 4)
    if shared_l2_is_early(num_tokens, block_m, sn2, num_sms):
        return l1 + l2 + routed, nblocks
    return l1 + routed + l2, nblocks


# ---------------------------------------------------------------------------------------------------------------------
# EP1 replay against DeepGEMM
# ---------------------------------------------------------------------------------------------------------------------
@dataclasses.dataclass
class ReplayEnv:
    """One DeepGEMM symmetric buffer + transformed weights (reused by many cases of the same (E, topk, H, I, T_max))."""

    group: object
    buffer: object
    tl1: Tuple[torch.Tensor, torch.Tensor]
    tl2: Tuple[torch.Tensor, torch.Tensor]
    num_experts: int
    topk: int
    hidden: int
    intermediate: int
    num_max_tokens: int
    clamp: float
    num_shared_experts: int = 0
    shared: Optional[Tuple] = (
        None  # (transformed shared L1 weights (fp8, sf), transformed shared L2 weights) when num_shared_experts > 0
    )


@dataclasses.dataclass
class ReplayResult:
    skipped: bool = False
    passed: bool = False
    message: str = ""
    num_tokens: int = 0
    block_m: int = 0
    counts: Optional[List[int]] = None
    pool_blocks: int = 0
    ring_blocks: int = 0
    bad_l2_rows: int = 0
    bad_l2_sf_tokens: int = 0
    bad_combine_rows: int = 0
    bad_shared_rows: int = 0
    bad_shared_sf_tokens: int = 0
    bad_shared_combine_rows: int = 0
    dg_us: float = 0.0  # DeepGEMM full kernel, median per launch (perf_iters > 0)
    g_us: float = (
        0.0  # the replayed (GEMM-only) kernel, median per launch (perf_iters > 0)
    )
    shared_nonzero_rows: int = -1  # rows of the reference (DeepGEMM) shared L1 output that are non-zero (guards against a vacuous all-zero comparison)
    num_valid_rows: int = 0
    num_valid_pairs: int = 0
    y_equal: bool = False
    y_ref: Optional[torch.Tensor] = (
        None  # DeepGEMM's y of the case (only when ``keep_ref`` was requested; used to prove that a numerics knob changes DeepGEMM's output)
    )

    def __str__(self) -> str:
        if self.skipped:
            return f"SKIP ({self.message})"
        return (
            f"{'PASS' if self.passed else 'FAIL'} M={self.num_tokens} BM={self.block_m} pool_blocks={self.pool_blocks}/{self.ring_blocks}: "
            f"l2_token bad {self.bad_l2_rows}/{self.num_valid_rows}, l2_sf bad {self.bad_l2_sf_tokens}/{self.num_valid_rows}, "
            f"combine bad {self.bad_combine_rows}/{self.num_valid_pairs}, shared l2_token/sf/combine bad "
            f"{self.bad_shared_rows}/{self.bad_shared_sf_tokens}/{self.bad_shared_combine_rows} (non-zero reference rows {self.shared_nonzero_rows}/{self.num_tokens}), "
            f"y equal {self.y_equal}"
        )


def init_single_rank_group():
    """``init_dist(0, 1)`` of DeepGEMM's test utilities (random rendezvous port unless MASTER_PORT is set: a fixed port gives EADDRINUSE between runs)."""
    import torch.distributed as dist
    from deep_gemm.utils.dist import init_dist

    if dist.is_available() and dist.is_initialized():
        return dist.new_group([0])
    os.environ.setdefault(
        "MASTER_PORT", str(20000 + int.from_bytes(os.urandom(2), "little") % 20000)
    )
    prev = torch.get_default_device()  # init_dist sets the PROCESS-WIDE default device to cuda; restore it so that later tests of the same pytest process are unaffected
    group = init_dist(0, 1)[2]
    torch.set_default_device(None if prev.type == "cpu" else prev)
    return group


def _cast_fp8_for_mega_moe(x: torch.Tensor):
    """``tests/test_mega_moe.py::_cast_fp8_for_mega_moe``: (fp8 values, packed UE8M0 SF, SF in the TMA (MN-major, 4-aligned) layout)."""
    from deep_gemm.utils import per_token_cast_to_fp8

    x_fp8, x_sf = per_token_cast_to_fp8(
        x, use_ue8m0=True, gran_k=32, use_packed_ue8m0=True
    )
    mn, packed_sf_k = x_sf.shape
    x_sf_tma = torch.empty_strided(
        (mn, packed_sf_k), (1, -(-mn // 4) * 4), dtype=x_sf.dtype, device=x_sf.device
    )
    x_sf_tma.copy_(x_sf)
    return x_fp8, x_sf, x_sf_tma


def _to_shared_sf_layout(
    sf: torch.Tensor, block_m: int, num_max_sf_tokens: int
) -> torch.Tensor:
    """``tests/test_mega_moe.py::_to_shared_mega_moe_sf_layout``: per-token packed SF -> the shared L1 SF buffer layout (UTCCP-transposed inside every
    BLOCK_M block, blocks ``align(BLOCK_M, 128)`` apart, MN-major)."""
    num_tokens, packed_sf_k = sf.shape
    aligned_block_m = -(-block_m // 128) * 128
    num_m_blocks = (num_tokens + block_m - 1) // block_m
    result = torch.empty_strided(
        (num_max_sf_tokens, packed_sf_k),
        (1, num_max_sf_tokens),
        dtype=sf.dtype,
        device=sf.device,
    )
    result.zero_()
    if num_tokens == 0:
        return result
    m = torch.arange(num_tokens, device=sf.device)
    blk, mi = m // block_m, m % block_m
    dst = blk * aligned_block_m + (mi // 128) * 128 + (mi % 32) * 4 + (mi % 128) // 32
    result[dst] = sf[m]
    del num_m_blocks
    return result


def _copy_fp8_sf(dst: torch.Tensor, src: torch.Tensor, num_tokens: int) -> None:
    """``tests/test_mega_moe.py::_copy_fp8_sf``."""
    if num_tokens == 0:
        return
    if dst.shape == src.shape:
        dst.copy_(src)
        return
    dst[:num_tokens].copy_(src)
    if num_tokens < dst.shape[0]:
        dst[num_tokens:].copy_(src[-1:].expand(dst.shape[0] - num_tokens, -1))


def make_env(
    num_experts: int,
    num_max_tokens: int,
    topk: int = 6,
    hidden: int = 5120,
    intermediate: int = 2304,
    clamp: float = 10.0,
    seed: int = 0,
    group=None,
    num_shared_experts: int = 0,
) -> ReplayEnv:
    """DeepGEMM symmetric buffer (S = 0, MXFP8 x MXFP4, SwiGLU) and random N(0,1) weights quantised like DeepGEMM's own test."""
    import deep_gemm
    from deep_gemm.utils import per_token_cast_to_fp4

    if group is None:
        group = init_single_rank_group()
    torch.manual_seed(seed)
    dev = "cuda"
    buffer = deep_gemm.get_symm_buffer_for_mega_moe(
        group,
        num_experts,
        num_max_tokens,
        topk,
        hidden,
        intermediate,
        num_shared_experts=num_shared_experts,
        mma_type="fp8xfp4",
        activation="swiglu",
    )

    def cast_w(w):
        g, n, k = w.shape
        q = torch.empty((g, n, k // 2), device=dev, dtype=torch.int8)
        sf = torch.empty((g, n, k // 32), device=dev, dtype=torch.float)
        for i in range(g):
            q[i], sf[i] = per_token_cast_to_fp4(w[i], use_ue8m0=True, gran_k=32)
        return q, deep_gemm.transform_sf_into_required_layout(sf, n, k, (1, 32), g)

    l1 = cast_w(
        torch.randn(
            (num_experts, 2 * intermediate, hidden), dtype=torch.bfloat16, device=dev
        )
    )
    l2 = cast_w(
        torch.randn(
            (num_experts, hidden, intermediate), dtype=torch.bfloat16, device=dev
        )
    )
    tl1, tl2 = deep_gemm.transform_weights_for_mega_moe(l1, l2, activation="swiglu")
    shared = None
    if num_shared_experts > 0:
        IS = intermediate * num_shared_experts
        sh1 = _cast_fp8_for_mega_moe(
            torch.randn((IS * 2, hidden), dtype=torch.bfloat16, device=dev)
        )[0::2]  # (fp8 weights, TMA-layout SF)
        sh2 = _cast_fp8_for_mega_moe(
            torch.randn((hidden, IS), dtype=torch.bfloat16, device=dev)
        )[0::2]
        shared = deep_gemm.transform_weights_for_mega_moe(sh1, sh2, activation="swiglu")
    return ReplayEnv(
        group,
        buffer,
        tl1,
        tl2,
        num_experts,
        topk,
        hidden,
        intermediate,
        num_max_tokens,
        clamp,
        num_shared_experts,
        shared,
    )


def _layout_of(env: ReplayEnv, cfg):
    from flashinfer.experimental.sm107_deepgemm_megamoe.kernel import layout as L

    lay = L.buffer_layout(
        hidden=env.hidden,
        intermediate_hidden=env.intermediate,
        num_ranks=1,
        num_experts=env.num_experts,
        num_max_tokens_per_rank=cfg.num_max_tokens_per_rank,
        num_topk=env.topk,
        num_ring_tokens=cfg.num_ring_tokens,
        num_sf_ring_tokens=cfg.num_sf_ring_tokens,
        num_shared_experts=env.num_shared_experts,
    )
    base = env.buffer.buffer.data_ptr()
    assert lay.total_bytes == env.buffer.buffer.nbytes, (
        lay.total_bytes,
        env.buffer.buffer.nbytes,
    )
    assert env.buffer.l1_acts.data_ptr() - base == lay["l1_token"].offset
    assert env.buffer.l2_acts.data_ptr() - base == lay["l2_token"].offset
    return L, lay


def restore_clean_state(env: ReplayEnv, snapshot: torch.Tensor) -> None:
    """Put the symmetric buffer back into DeepGEMM's clean post-cleanup state (REQUIRED before the next DeepGEMM call after a GEMM-only launch)."""
    env.buffer.buffer.copy_(snapshot)
    torch.cuda.synchronize()


def default_launcher(
    env: ReplayEnv,
    cfg,
    y: torch.Tensor,
    num_tokens: int,
    *,
    mode_mask: Optional[int] = None,
    use_disk_cache: bool = True,
) -> Callable[[], None]:
    """Launcher of the port's kernel (``mode_mask`` defaults to MODE_GEMM) on the replayed buffer."""
    from flashinfer.experimental.sm107_deepgemm_megamoe import jit as sjit
    from flashinfer.experimental.sm107_deepgemm_megamoe.kernel import MODE_GEMM

    runner = sjit.get_or_compile_kernel(
        cfg,
        mode_mask=MODE_GEMM if mode_mask is None else mode_mask,
        use_disk_cache=use_disk_cache,
    )
    symb = types.SimpleNamespace(
        base_ptr=env.buffer.buffer.data_ptr(), peer_offsets=[0], group=env.group
    )
    if env.num_shared_experts > 0:
        (s1w, s1sf), (s2w, s2sf) = env.shared
        return lambda: runner(
            y,
            None,
            num_tokens,
            symb,
            env.tl1[0],
            env.tl1[1],
            env.tl2[0],
            env.tl2[1],
            s1w,
            s1sf,
            s2w,
            s2sf,
        )
    return lambda: runner(
        y, None, num_tokens, symb, env.tl1[0], env.tl1[1], env.tl2[0], env.tl2[1]
    )


NUMERICS_KEYS = ("clamp", "alpha", "beta", "fast_math")


def resolve_numerics(env: ReplayEnv, numerics: Optional[dict] = None) -> dict:
    """Activation numerics of one case: ``clamp`` (``None`` / ``inf``: no clamp, compiled out), ``alpha`` (sigmoid scale), ``beta`` (added to ``up`` after the clamp),
    ``fast_math``.  Defaults: the env's clamp (10.0), alpha 1.0, beta 0.0, fast_math True (DeepGEMM's / the port's defaults).  The SAME values go to
    ``deep_gemm.fp8_fp4_mega_moe`` (``activation_clamp`` / ``activation_alpha`` / ``activation_beta`` / ``fast_math``) and to ``get_sm107_kernel_config``."""
    n = dict(clamp=env.clamp, alpha=1.0, beta=0.0, fast_math=True)
    if numerics:
        unknown = set(numerics) - set(NUMERICS_KEYS)
        assert not unknown, (
            f"unknown numerics keys {sorted(unknown)} (valid: {NUMERICS_KEYS})"
        )
        n.update(numerics)
    return n


def _dg_numerics(n: dict) -> dict:
    return dict(
        activation_clamp=n["clamp"],
        activation_alpha=n["alpha"],
        activation_beta=n["beta"],
        fast_math=n["fast_math"],
    )


def _cfg_numerics(n: dict) -> dict:
    return dict(
        activation_clamp=n["clamp"],
        activation_alpha=n["alpha"],
        activation_beta=n["beta"],
        fast_math=n["fast_math"],
    )


def _scaled_randn_x(
    num_tokens: int, hidden: int, x_scale: float, dev: str
) -> torch.Tensor:
    """The N(0,1) BF16 activations of a case (``x_scale`` != 1 scales them AFTER the draw, so the random stream of the default scale is unchanged).  The L1 gate / up
    pre-activations are ~N(0, hidden * x_scale^2): scale 1 saturates every clamp (std ~ 72), 1/16 puts them at std ~ 4.5, 1/512 at std ~ 0.14."""
    x = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device=dev)
    return x if x_scale == 1.0 else (x * x_scale).to(torch.bfloat16)


#: regions of the symmetric buffer that the kernels WRITE before they READ: a DeepGEMM-then-port comparison on ONE reused buffer cannot
#: see a port write that lands at a wrong address (the right address still holds DeepGEMM's identical stale result), so these are filled with 0xFF before every port launch
POISON_REGIONS = (
    "shared_l2_token",
    "shared_l2_sf",
    "l1_token",
    "l1_sf",
    "l1_topk_weights",
    "l2_token",
    "l2_sf",
    "combine_token",
)
POISON_BYTE = 0xFF  # E4M3 NaN / BF16 NaN / UE8M0 255: any stale read poisons the result


def poison_regions(
    env: ReplayEnv, lay, names=POISON_REGIONS, value: int = POISON_BYTE
) -> None:
    """Stream-ordered fill of the named regions of the symmetric buffer (``names`` must be regions the kernel under test writes before it reads)."""
    raw = env.buffer.buffer.view(torch.uint8)
    for name in names:
        r = lay[name]
        if r.nbytes:
            raw[r.offset : r.offset + r.nbytes].fill_(value)


def apply_cfg_replace(cfg, cfg_replace: dict, use_disk_cache: bool):
    """``dataclasses.replace`` for the keys that are fields of the (frozen) config; keys that are NOT fields yet are set
    as plain attributes - such a config is not covered by the compile-cache digest, so the disk cache is switched off for it.  Returns ``(cfg, use_disk_cache)``."""
    fields = {f.name for f in dataclasses.fields(cfg)}
    known = {k: v for k, v in cfg_replace.items() if k in fields}
    extra = {k: v for k, v in cfg_replace.items() if k not in fields}
    if known:
        cfg = dataclasses.replace(cfg, **known)
    for k, v in extra.items():
        object.__setattr__(cfg, k, v)
    return cfg, use_disk_cache and not extra


def replay_case(
    env: ReplayEnv,
    num_tokens: int,
    block_m: int = 0,
    masked: float = 0.0,
    seed: int = 0,
    *,
    mode_mask: Optional[int] = None,
    launcher_factory: Optional[Callable] = None,
    use_disk_cache: bool = True,
    restore: bool = True,
    stress_launches: int = 0,
    stress_check_every: int = 50,
    perf_iters: int = 0,
    cfg_replace: Optional[dict] = None,
    numerics: Optional[dict] = None,
    x_scale: float = 1.0,
    keep_ref: bool = False,
) -> ReplayResult:
    """One replay case (see the module docstring).  ``block_m`` > 0 forces BLOCK_M on BOTH sides (``DG_MEGA_MOE_SM107_BLOCK_M`` is set for the DeepGEMM call,
    the port's config is built with ``block_m``); ``masked`` is the fraction of top-k slots set to -1.  Returns a :class:`ReplayResult`; on success (or when
    ``restore``) the buffer is left in DeepGEMM's clean state.  ``stress_launches`` > 0 additionally re-launches the kernel that many times on the same
    staged state (signals restaged, outputs zeroed before every launch) and requires the valid outputs to stay identical (no hang, no non-determinism).
    ``numerics`` (keys ``clamp`` / ``alpha`` / ``beta`` / ``fast_math``, see :func:`resolve_numerics`) selects the activation numerics of BOTH sides, ``x_scale`` the scale of the
    input activations (see :func:`_scaled_randn_x`)."""
    import deep_gemm
    from deep_gemm.utils import per_token_cast_to_fp8

    from flashinfer.experimental.sm107_deepgemm_megamoe.heuristics import (
        get_sm107_kernel_config,
    )

    M, E, topk, Hh, I = (
        num_tokens,
        env.num_experts,
        env.topk,
        env.hidden,
        env.intermediate,
    )
    S = env.num_shared_experts
    IS = I * S
    buffer = env.buffer
    dev = "cuda"
    num = resolve_numerics(env, numerics)
    torch.manual_seed(seed + M)
    x = _scaled_randn_x(M, Hh, x_scale, dev)
    scores = torch.randn((M, E), dtype=torch.float, device=dev)
    topk_weights, topk_idx = torch.topk(
        scores, topk, dim=-1, largest=True, sorted=False
    )
    if masked > 0:
        topk_idx.masked_fill_(torch.rand_like(topk_idx, dtype=torch.float) < masked, -1)
        topk_weights.masked_fill_(topk_idx < 0, 0)
    if M > 0:
        x_fp8, x_sf = per_token_cast_to_fp8(
            x, use_ue8m0=True, gran_k=32, use_packed_ue8m0=True
        )
        buffer.x[:M].copy_(x_fp8)
        buffer.x_sf[:M].copy_(x_sf)
        buffer.topk_idx[:M].copy_(topk_idx)
        buffer.topk_weights[:M].copy_(topk_weights)
    y = torch.empty((M, Hh), dtype=torch.bfloat16, device=dev)
    shared_kwargs = {}
    if S > 0:
        sf_block_m = block_m or deep_gemm.get_block_m_for_mega_moe(
            1, E, env.num_max_tokens, M, topk, "fp8xfp4"
        )
        if (
            M > 0
        ):  # host-side staging of the shared L1 token SF (stock test: `_to_shared_mega_moe_sf_layout` + `_copy_fp8_sf`)
            _copy_fp8_sf(
                buffer.shared_l1_acts_sf,
                _to_shared_sf_layout(
                    x_sf, sf_block_m, buffer.shared_l1_acts_sf.shape[0]
                ),
                M,
            )
        shared_kwargs = dict(
            shared_l1_weights=env.shared[0], shared_l2_weights=env.shared[1]
        )

    old_bm = os.environ.get("DG_MEGA_MOE_SM107_BLOCK_M")
    if block_m:
        os.environ["DG_MEGA_MOE_SM107_BLOCK_M"] = str(block_m)
    try:
        deep_gemm.fp8_fp4_mega_moe(
            y,
            env.tl1,
            env.tl2,
            buffer,
            activation="swiglu",
            **_dg_numerics(num),
            **shared_kwargs,
        )
        torch.cuda.synchronize()
    finally:
        if block_m:
            if old_bm is None:
                os.environ.pop("DG_MEGA_MOE_SM107_BLOCK_M", None)
            else:
                os.environ["DG_MEGA_MOE_SM107_BLOCK_M"] = old_bm
    snap = buffer.buffer.clone()
    y_ref = y.clone()

    cfg = get_sm107_kernel_config(
        1,
        E,
        env.num_max_tokens,
        M,
        topk,
        Hh,
        I,
        S,
        num_sms=212,
        block_m=block_m,
        **_cfg_numerics(num),
    )
    if cfg_replace:
        cfg, use_disk_cache = apply_cfg_replace(cfg, cfg_replace, use_disk_cache)
    BM, SFBM = cfg.block_m, cfg.sf_block_m
    L, lay = _layout_of(env, cfg)
    buf8 = buffer.buffer.view(torch.uint8)

    def region(name, dtype, shape):
        r = lay[name]
        return buf8[r.offset : r.offset + r.nbytes].view(dtype).view(*shape)

    flat = topk_idx.reshape(-1)
    counts = torch.bincount(flat[flat >= 0].long(), minlength=E).cpu().tolist()
    nblk = [ceil_div(c, BM) for c in counts]
    pool_off = [sum(nblk[:e]) for e in range(E)]
    total_blocks = sum(nblk)
    res = ReplayResult(
        num_tokens=M,
        block_m=BM,
        counts=counts,
        pool_blocks=total_blocks,
        ring_blocks=cfg.num_ring_blocks,
        y_ref=y_ref if keep_ref else None,
    )
    if total_blocks > cfg.num_ring_blocks:
        res.skipped = True
        res.message = f"pool blocks {total_blocks} > ring blocks {cfg.num_ring_blocks}: DeepGEMM wrapped the ring, no snapshot replay possible"
        return res

    # ---- restore + re-stage what DeepGEMM's cleanup zeroed
    buffer.buffer.copy_(snap)
    # the regions the GEMM part must produce are POISONED (0xFF), not zeroed: an output the kernel fails to write (or writes to a wrong address) can neither match a zero
    # reference (degenerate activations, e.g. clamp 0) nor hide behind DeepGEMM's stale snapshot data
    poison_regions(
        env,
        lay,
        ("l2_token", "l2_sf", "shared_l2_token", "shared_l2_sf", "combine_token"),
    )
    sig = buf8[: L.SIGNALS_SIZEOF]
    recv = sig[
        L.SIGNALS_EXPERT_RECV_COUNT_SUM : L.SIGNALS_EXPERT_RECV_COUNT_SUM + 8 * E
    ].view(torch.int64)
    for e in range(E):
        recv[e] = (cfg.num_sms * cfg.num_ranks << 32) | counts[e]
    l1_full = sig[
        L.SIGNALS_L1_FULL_COUNT : L.SIGNALS_L1_FULL_COUNT + 4 * cfg.num_ring_blocks
    ].view(torch.int32)
    l1_full[:total_blocks] = BM  # generation 0: BLOCK_M tokens arrived
    for nm, off, n, dt in (
        ("l1_empty", L.SIGNALS_L1_EMPTY_COUNT, 4 * cfg.num_ring_blocks, torch.int32),
        ("l2_mask", L.SIGNALS_L2_FULL_MASK, 8 * cfg.num_ring_blocks, torch.int64),
        ("l2_empty", L.SIGNALS_L2_EMPTY_COUNT, 4 * cfg.num_ring_blocks, torch.int32),
        ("l1_task", L.SIGNALS_L1_TASK_COUNT, 8, torch.int32),
        ("l2_task", L.SIGNALS_L2_TASK_COUNT, 8, torch.int32),
        ("shared_l1_task", L.SIGNALS_SHARED_L1_TASK_COUNT, 8, torch.int32),
        ("shared_l2_task", L.SIGNALS_SHARED_L2_TASK_COUNT, 8, torch.int32),
        ("shared_l2_full", L.SIGNALS_SHARED_L2_FULL_COUNT, 4 * 1024, torch.int32),
    ):
        assert int(sig[off : off + n].view(dt).abs().sum().item()) == 0, (
            f"{nm} not zero after DeepGEMM's cleanup"
        )
    torch.cuda.synchronize()
    staged_sig = buf8[
        : L.SIGNALS_SIZEOF
    ].clone()  # all state the GEMM-only kernel mutates besides its idempotent outputs

    # ---- the port's kernel
    if launcher_factory is None:
        launch = default_launcher(
            env, cfg, y, M, mode_mask=mode_mask, use_disk_cache=use_disk_cache
        )
    else:
        launch = launcher_factory(env, cfg, y, M)
    launch()
    torch.cuda.synchronize()

    # ---- compare
    snap8 = snap.view(torch.uint8)

    def sview(name, dtype, shape):
        r_ = lay[name]
        return snap8[r_.offset : r_.offset + r_.nbytes].view(dtype).view(*shape)

    rows, pos = [], []
    for e in range(E):
        for t in range(counts[e]):
            pb, tin = pool_off[e] + t // BM, t % BM
            rows.append(pb * BM + tin)
            pos.append(pb * SFBM + utccp_pos(tin))
    res.num_valid_rows = len(rows)
    if rows:
        rows_t = torch.tensor(rows, dtype=torch.long, device=dev)
        pos_t = torch.tensor(pos, dtype=torch.long, device=dev)
        mine_l2 = region("l2_token", torch.uint8, (cfg.num_ring_tokens, I))
        ref_l2 = sview("l2_token", torch.uint8, (cfg.num_ring_tokens, I))
        res.bad_l2_rows = int(
            (mine_l2[rows_t] != ref_l2[rows_t]).any(dim=1).sum().item()
        )
        mine_sf = region("l2_sf", torch.int32, (I // 128, cfg.num_sf_ring_tokens))
        ref_sf = sview("l2_sf", torch.int32, (I // 128, cfg.num_sf_ring_tokens))
        res.bad_l2_sf_tokens = int(
            (mine_sf[:, pos_t] != ref_sf[:, pos_t]).any(dim=0).sum().item()
        )

    nslots = topk + (1 if S > 0 else 0)
    cb_mine = region(
        "combine_token", torch.uint8, (nslots, cfg.num_max_tokens_per_rank, Hh * 2)
    )
    cb_ref = sview(
        "combine_token", torch.uint8, (nslots, cfg.num_max_tokens_per_rank, Hh * 2)
    )
    valid = (topk_idx >= 0).t()  # (topk, M)
    res.num_valid_pairs = int(valid.sum().item())
    if M > 0:
        eq = (cb_mine[:topk, :M] == cb_ref[:topk, :M]).all(dim=-1)  # (topk, M)
        res.bad_combine_rows = int((valid & ~eq).sum().item())
    if S > 0 and M > 0:
        # shared-expert results: shared L1 output (valid rows / SF tokens of the T local tokens) and the combine slot kNumTopk of EVERY token
        mine_sh = region(
            "shared_l2_token", torch.uint8, (cfg.num_max_tokens_per_rank, IS)
        )
        ref_sh = sview(
            "shared_l2_token", torch.uint8, (cfg.num_max_tokens_per_rank, IS)
        )
        res.bad_shared_rows = int((mine_sh[:M] != ref_sh[:M]).any(dim=1).sum().item())
        res.shared_nonzero_rows = int((ref_sh[:M] != 0).any(dim=1).sum().item())
        t = torch.arange(M, device=dev)
        spos = (
            (t // BM) * SFBM
            + (t % BM // 128) * 128
            + (t % BM % 32) * 4
            + (t % BM // 32) % 4
        )
        mine_ssf = region(
            "shared_l2_sf", torch.int32, (IS // 128, cfg.num_shared_sf_tokens)
        )
        ref_ssf = sview(
            "shared_l2_sf", torch.int32, (IS // 128, cfg.num_shared_sf_tokens)
        )
        res.bad_shared_sf_tokens = int(
            (mine_ssf[:, spos] != ref_ssf[:, spos]).any(dim=0).sum().item()
        )
        res.bad_shared_combine_rows = int(
            (cb_mine[topk, :M] != cb_ref[topk, :M]).any(dim=1).sum().item()
        )
    acc = torch.zeros((M, Hh), dtype=torch.float32, device=dev)
    cbm = cb_mine.view(torch.bfloat16).view(nslots, cfg.num_max_tokens_per_rank, Hh)
    vm = topk_idx >= 0
    for k in range(topk):
        rows_k = cbm[k, :M].float()
        acc += torch.where(vm[:, k : k + 1], rows_k, torch.zeros_like(rows_k))
    if S > 0:
        acc += cbm[topk, :M].float()  # the shared-expert slot is reduced last
    res.y_equal = bool(
        torch.equal(acc.to(torch.bfloat16).view(torch.int16), y_ref.view(torch.int16))
    )
    res.passed = (
        res.bad_l2_rows == 0
        and res.bad_l2_sf_tokens == 0
        and res.bad_combine_rows == 0
        and res.bad_shared_rows == 0
        and res.bad_shared_sf_tokens == 0
        and res.bad_shared_combine_rows == 0
        and res.y_equal
        and (S == 0 or M == 0 or res.shared_nonzero_rows == M)
    )
    if res.passed and stress_launches > 0:
        l2 = region("l2_token", torch.uint8, (cfg.num_ring_tokens, I))
        l2sf = region("l2_sf", torch.int32, (I // 128, cfg.num_sf_ring_tokens))
        ref_rows = l2[rows_t].clone() if rows else None
        ref_pos = l2sf[:, pos_t].clone() if rows else None
        ref_cb = cb_mine[:, :M].clone()
        valid_all = (
            valid
            if S == 0
            else torch.cat(
                [valid, torch.ones((1, M), dtype=torch.bool, device=dev)], dim=0
            )
        )  # (nslots, M)
        if S > 0:
            sh_tok = region(
                "shared_l2_token", torch.uint8, (cfg.num_max_tokens_per_rank, IS)
            )
            sh_sf = region(
                "shared_l2_sf", torch.int32, (IS // 128, cfg.num_shared_sf_tokens)
            )
            ref_sh_tok, ref_sh_sf = (
                sh_tok[:M].clone(),
                (sh_sf[:, spos].clone() if M > 0 else None),
            )
        for i in range(stress_launches):
            sig.copy_(staged_sig)
            l2.fill_(POISON_BYTE)
            l2sf.fill_(-1)  # int32 view of 0xFF bytes
            cb_mine.fill_(POISON_BYTE)
            if S > 0:
                sh_tok.fill_(POISON_BYTE)
                sh_sf.fill_(-1)
            launch()
            if (i + 1) % stress_check_every == 0 or i == stress_launches - 1:
                torch.cuda.synchronize()
                ok_l2 = (not rows) or (
                    torch.equal(l2[rows_t], ref_rows)
                    and torch.equal(l2sf[:, pos_t], ref_pos)
                )
                ok_cb = M == 0 or bool(
                    ((cb_mine[:, :M] == ref_cb).all(dim=-1) | ~valid_all).all().item()
                )
                ok_sh = (
                    S == 0
                    or M == 0
                    or (
                        torch.equal(sh_tok[:M], ref_sh_tok)
                        and torch.equal(sh_sf[:, spos], ref_sh_sf)
                    )
                )
                if not (ok_l2 and ok_cb and ok_sh):
                    res.passed = False
                    res.message = f"stress mismatch at launch {i + 1}: l2/sf equal={ok_l2}, combine equal={ok_cb}, shared equal={ok_sh}"
                    break
    if res.passed and perf_iters > 0:
        # early perf signal: per-launch GPU time between two CUDA events (a GPU sleep keeps the CPU ahead); DeepGEMM from its clean state, the replayed kernel with
        # the staged signals restored before every launch (outside the timed window)
        def timed(pre, run):
            for _ in range(3):
                pre()
                run()
            torch.cuda.synchronize()
            evs = [
                (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                for _ in range(perf_iters)
            ]
            for e0, e1 in evs:
                torch.cuda._sleep(5_000_000)
                pre()
                e0.record()
                run()
                e1.record()
            torch.cuda.synchronize()
            ts = sorted(a.elapsed_time(b) * 1e3 for a, b in evs)
            return ts[len(ts) // 2]

        restore_clean_state(env, snap)
        res.dg_us = timed(
            lambda: None,
            lambda: deep_gemm.fp8_fp4_mega_moe(
                y,
                env.tl1,
                env.tl2,
                buffer,
                activation="swiglu",
                **_dg_numerics(num),
                **shared_kwargs,
            ),
        )
        restore_clean_state(env, snap)
        sig.copy_(staged_sig)
        torch.cuda.synchronize()
        res.g_us = timed(lambda: sig.copy_(staged_sig), launch)
    if restore:
        restore_clean_state(env, snap)
    return res


def full_kernel_case(
    env: ReplayEnv,
    num_tokens: int,
    block_m: int = 0,
    masked: float = 0.0,
    seed: int = 0,
    launches: int = 3,
    use_disk_cache: bool = True,
    mode_mask: Optional[int] = None,
    cfg_replace: Optional[dict] = None,
    numerics: Optional[dict] = None,
    x_scale: float = 1.0,
    keep_ref: bool = False,
    poison="auto",
) -> ReplayResult:
    """FULL-kernel EP1 check (dispatch + GEMM + combine of the port, ``mode_mask`` default MODE_ALL) against DeepGEMM's ``y`` on the SAME staged inputs
    (routed + optional shared expert): ``launches`` back-to-back launches of the port's kernel (each into its own ``y``) must all be bit-identical to DeepGEMM's.
    Ring wrap is allowed (no snapshot replay involved).  The buffer must be in DeepGEMM's clean state before the call (it is after any DeepGEMM call).
    ``numerics`` / ``x_scale``: see :func:`replay_case`.  ``poison``: fill the kernel-written regions of the buffer (:data:`POISON_REGIONS`) with 0xFF before EVERY launch of the port
    (DeepGEMM ran first on the same buffer, so without it a write to a wrong address hides behind DeepGEMM's identical stale results); ``"auto"`` = on for the full kernel
    (``mode_mask`` None / MODE_ALL), off for partial kernels (their inputs are the stale regions); ``None`` / ``False`` = off; an int = that byte."""
    import deep_gemm
    from deep_gemm.utils import per_token_cast_to_fp8

    from flashinfer.experimental.sm107_deepgemm_megamoe import jit as sjit
    from flashinfer.experimental.sm107_deepgemm_megamoe.heuristics import (
        get_sm107_kernel_config,
    )
    from flashinfer.experimental.sm107_deepgemm_megamoe.kernel import MODE_ALL

    M, E, topk, Hh, I = (
        num_tokens,
        env.num_experts,
        env.topk,
        env.hidden,
        env.intermediate,
    )
    S = env.num_shared_experts
    buffer = env.buffer
    dev = "cuda"
    num = resolve_numerics(env, numerics)
    torch.manual_seed(seed + M)
    x = _scaled_randn_x(M, Hh, x_scale, dev)
    scores = torch.randn((M, E), dtype=torch.float, device=dev)
    topk_weights, topk_idx = torch.topk(
        scores, topk, dim=-1, largest=True, sorted=False
    )
    if masked > 0:
        topk_idx.masked_fill_(torch.rand_like(topk_idx, dtype=torch.float) < masked, -1)
        topk_weights.masked_fill_(topk_idx < 0, 0)
    shared_kwargs = {}
    if M > 0:
        x_fp8, x_sf = per_token_cast_to_fp8(
            x, use_ue8m0=True, gran_k=32, use_packed_ue8m0=True
        )
        buffer.x[:M].copy_(x_fp8)
        buffer.x_sf[:M].copy_(x_sf)
        buffer.topk_idx[:M].copy_(topk_idx)
        buffer.topk_weights[:M].copy_(topk_weights)
    if S > 0:
        sf_block_m = block_m or deep_gemm.get_block_m_for_mega_moe(
            1, E, env.num_max_tokens, M, topk, "fp8xfp4"
        )
        if M > 0:
            _copy_fp8_sf(
                buffer.shared_l1_acts_sf,
                _to_shared_sf_layout(
                    x_sf, sf_block_m, buffer.shared_l1_acts_sf.shape[0]
                ),
                M,
            )
        shared_kwargs = dict(
            shared_l1_weights=env.shared[0], shared_l2_weights=env.shared[1]
        )
    old_bm = os.environ.get("DG_MEGA_MOE_SM107_BLOCK_M")
    if block_m:
        os.environ["DG_MEGA_MOE_SM107_BLOCK_M"] = str(block_m)
    try:
        y_ref = torch.empty((M, Hh), dtype=torch.bfloat16, device=dev)
        deep_gemm.fp8_fp4_mega_moe(
            y_ref,
            env.tl1,
            env.tl2,
            buffer,
            activation="swiglu",
            **_dg_numerics(num),
            **shared_kwargs,
        )
        torch.cuda.synchronize()
    finally:
        if block_m:
            if old_bm is None:
                os.environ.pop("DG_MEGA_MOE_SM107_BLOCK_M", None)
            else:
                os.environ["DG_MEGA_MOE_SM107_BLOCK_M"] = old_bm
    cfg = get_sm107_kernel_config(
        1,
        E,
        env.num_max_tokens,
        M,
        topk,
        Hh,
        I,
        S,
        num_sms=212,
        block_m=block_m,
        **_cfg_numerics(num),
    )
    if cfg_replace:
        cfg, use_disk_cache = apply_cfg_replace(cfg, cfg_replace, use_disk_cache)
    runner = sjit.get_or_compile_kernel(
        cfg,
        mode_mask=MODE_ALL if mode_mask is None else mode_mask,
        use_disk_cache=use_disk_cache,
    )
    symb = types.SimpleNamespace(
        base_ptr=buffer.buffer.data_ptr(), peer_offsets=[0], group=env.group
    )
    wargs = (env.tl1[0], env.tl1[1], env.tl2[0], env.tl2[1]) + (
        (env.shared[0][0], env.shared[0][1], env.shared[1][0], env.shared[1][1])
        if S > 0
        else ()
    )
    ys = [
        torch.full((M, Hh), -1, dtype=torch.bfloat16, device=dev)
        for _ in range(launches)
    ]
    if poison == "auto":
        poison = POISON_BYTE if (mode_mask is None or mode_mask == MODE_ALL) else None
    elif poison is True:
        poison = POISON_BYTE
    lay = _layout_of(env, cfg)[1] if poison not in (None, False) else None
    for yi in (
        ys
    ):  # back-to-back launches, no host sync in between (each cleans up after itself)
        if lay is not None:
            poison_regions(
                env, lay, value=int(poison)
            )  # stream-ordered, before every launch
        runner(yi, None, M, symb, *wargs)
    torch.cuda.synchronize()
    res = ReplayResult(
        num_tokens=M, block_m=cfg.block_m, y_ref=y_ref if keep_ref else None
    )
    eq = [bool(torch.equal(yi.view(torch.int16), y_ref.view(torch.int16))) for yi in ys]
    res.y_equal = all(eq)
    res.passed = res.y_equal
    res.message = (
        ""
        if res.passed
        else f"y differs from DeepGEMM in launches {[i for i, e in enumerate(eq) if not e]} (ring blocks {cfg.num_ring_blocks})"
    )
    return res
