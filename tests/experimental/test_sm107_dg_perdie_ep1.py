# SPDX-License-Identifier: Apache-2.0
# Port of DeepGEMM (MIT License, Copyright (c) 2025 DeepSeek; see licenses/LICENSE.deepgemm.txt) sm107_fp8_fp4_mega_moe.cuh / scheduler/sm107_mega_moe.cuh.
"""EP1 per-die (locality-domain) execution of the SM107 port against DeepGEMM's per-die kernel (GPU test, Rubin SM107 = compute capability 10.7 with 2 locality domains).

Oracle: ``deep_gemm.fp8_fp4_mega_moe`` with ``DG_MEGA_MOE_SM107_PERDIE=1`` on die-local routed weights (DeepGEMM's own localizer and balanced SM map).  The port runs the SAME localized
weights (the package's ``locality.localize_mega_moe_weights`` -> identical allocation layout) and the SAME SM -> domain map through its two kernel paths:

* the raw runner (``jit.get_or_compile_kernel(cfg_per_die)`` with ``sm_locality_domains=``), ``y`` bit-exact against DeepGEMM per-die AND against DeepGEMM plain (per-tile math does not depend on scheduling);
* the DeepGEMM-compatible frontend (``compat.fp8_fp4_mega_moe`` selects the per-die variant by itself when the routed weights are die-local);
* a traced build (``cfg.trace``): the domain every scheduler warp read equals ``map[%smid]`` for ALL leader CTAs, the total number of successful task claims equals
  ``pool blocks x (L1 + L2 clusters)`` (+ the shared tasks for S = 1), i.e. every (m_block, n_cluster) task is claimed exactly once from the two queues, and ``y`` is bit-exact.

POISON (review TS-01 / TS-05): DeepGEMM and the port share ONE symmetric buffer, so the second kernel of a test would otherwise start on the first kernel's identical results (a premature L2 start of
``scheduler_perdie.wait_l1_claimed`` or a wrong 3D-map address would read the stale, correct ``l2_token``).  Before EVERY launch of EITHER kernel all kernel-written regions of the buffer
(``src_token_topk_idx``, ``token_src_metadata``, ``l1_token``, ``l1_sf``, ``l1_topk_weights``, ``l2_token``, ``l2_sf``, ``combine_token``, and for S = 1 ``shared_l2_token`` / ``shared_l2_sf``) are filled with 0xFF
(the signals = counters / barriers and the staged inputs are left alone) and the outputs are NaN-filled.  Canary tests prove that the poison is complete, is overwritten by a correct kernel, and that a launch
that produces nothing fails the bit-exact comparison; ``test_port_per_die_changing_inputs_between_launches`` changes tokens, routing and M (hence BLOCK_M and the pool geometry) between launches.

S = 0 and S = 1 (one shared expert; PERDIE=routed recipe: shared weights plain, shared tasks scheduled per domain).  E = 32, top-6, T_max 1920 (EP1: fast).
Per-die at R > 4 needs NVL72 (multi-node): see ``_sm107_dg_perdie_full_ep.py`` (documented manual driver).
"""

import os
import sys
import types

import pytest
import torch

pytest.importorskip("cutlass.cute")
deep_gemm = pytest.importorskip("deep_gemm")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import replay_utils as R  # noqa: E402
from deep_gemm.utils import per_token_cast_to_fp8  # noqa: E402
from flashinfer.experimental.sm107_deepgemm_megamoe import jit, locality  # noqa: E402
from flashinfer.experimental.sm107_deepgemm_megamoe.heuristics import (
    get_sm107_kernel_config,
)  # noqa: E402
from flashinfer.experimental.sm107_deepgemm_megamoe.kernel import layout as L  # noqa: E402
from flashinfer.experimental.sm107_deepgemm_megamoe.kernel.trace import (
    trace_buffer_u64s,
)  # noqa: E402

pytestmark = [
    pytest.mark.skipif(
        not torch.cuda.is_available()
        or torch.cuda.get_device_capability() != (10, 7)
        or not locality.is_localization_available(),
        reason="needs a Rubin SM107 GPU (compute capability 10.7) with 2 locality domains",
    ),
    pytest.mark.arch_rubin,
]

E, TOPK, H, I, TMAX, NUM_SMS = 32, 6, 5120, 2304, 1920, 212
POISON = 0xFF


@pytest.fixture(scope="module", autouse=True)
def _oracle_env():
    """Environment defaults of the DeepGEMM oracle, applied for THIS module only and restored afterwards (nothing is mutated at import time)."""
    with pytest.MonkeyPatch.context() as mp:
        if "DG_MEGA_MOE_SM107" not in os.environ:
            mp.setenv(
                "DG_MEGA_MOE_SM107", "1"
            )  # select DeepGEMM's native SM107 MegaMoE kernel
        yield


def _envs(S):
    from deep_gemm.mega import sm107_locality as DL

    env = R.make_env(E, TMAX, TOPK, H, I, 10.0, seed=3 + S, num_shared_experts=S)
    loc1, loc2, _, _ = locality.localize_mega_moe_weights(
        env.tl1, env.tl2
    )  # the package's pure-python allocator: DeepGEMM accepts these tensors too
    assert locality.weights_localized(loc1) and locality.weights_localized(loc2)
    dmap = DL.get_balanced_sm_locality_domains()  # DeepGEMM's probe + registration of the map for ITS per-die launches; the port gets the very same tensor
    return env, loc1, loc2, dmap


def test_host_probe_map_equals_deepgemms_balanced_map():
    from deep_gemm.mega import sm107_locality as DL

    assert torch.equal(
        locality.get_balanced_sm_locality_domains().cpu(),
        DL.get_balanced_sm_locality_domains().cpu(),
    )


@pytest.fixture(scope="module", params=[0, 1], ids=["S0", "S1"])
def setup(request):
    S = request.param
    env, loc1, loc2, dmap = _envs(S)
    return S, env, loc1, loc2, dmap


def _stage(env, S, M, seed, masked=0.0):
    g = torch.Generator(device="cuda")
    g.manual_seed(seed)
    buf = env.buffer
    x = torch.randn((M, H), dtype=torch.bfloat16, device="cuda", generator=g)
    scores = torch.randn((M, E), dtype=torch.float, device="cuda", generator=g)
    tw, ti = torch.topk(scores, TOPK, dim=-1, largest=True, sorted=False)
    if masked > 0:
        ti.masked_fill_(torch.rand(ti.shape, device="cuda", generator=g) < masked, -1)
        tw.masked_fill_(ti < 0, 0)
    xq, xs = per_token_cast_to_fp8(x, use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)
    cfg = get_sm107_kernel_config(
        1,
        E,
        TMAX,
        M,
        TOPK,
        H,
        I,
        S,
        num_sms=NUM_SMS,
        activation_clamp=10.0,
        per_die=True,
        localized_weights=True,
    )
    if M > 0:
        buf.x[:M].copy_(xq)
        buf.x_sf[:M].copy_(xs)
        buf.topk_idx[:M].copy_(ti)
        buf.topk_weights[:M].copy_(tw)
        if S > 0:
            R._copy_fp8_sf(
                buf.shared_l1_acts_sf,
                R._to_shared_sf_layout(xs, cfg.block_m, buf.shared_l1_acts_sf.shape[0]),
                M,
            )
    flat = ti.reshape(-1)
    counts = torch.bincount(flat[flat >= 0].long(), minlength=E)
    return cfg, counts


# ----------------------------------------------------------------------------------------------------------------------
# poison of the kernel-written regions of the symmetric buffer
# ----------------------------------------------------------------------------------------------------------------------
def _poison_spans(cfg, S):
    """Byte ranges of the symmetric buffer that a launch WRITES (everything except the signals = counters / barriers, which must stay clean, and the staged inputs)."""
    lay = L.buffer_layout(hidden=H, intermediate_hidden=I, num_ranks=1, num_experts=E, num_max_tokens_per_rank=cfg.num_max_tokens_per_rank, num_topk=TOPK,
                          num_ring_tokens=cfg.num_ring_tokens, num_sf_ring_tokens=cfg.num_sf_ring_tokens, num_shared_experts=S)  # fmt: skip
    ws = lay.workspace
    spans = {
        "src_token_topk_idx": (
            ws.src_token_topk_idx_offset,
            ws.token_src_metadata_offset,
        ),
        "token_src_metadata": (
            ws.token_src_metadata_offset,
            ws.token_src_metadata_offset
            + ws.num_max_pool_tokens * L.TOKEN_SRC_METADATA_BYTES,
        ),
    }
    for name in (
        "shared_l2_token",
        "shared_l2_sf",
        "l1_token",
        "l1_sf",
        "l1_topk_weights",
        "l2_token",
        "l2_sf",
        "combine_token",
    ):
        r = lay.regions[name]
        if r.nbytes:
            spans[name] = (r.offset, r.end)
    return lay, spans


def _poison(buffer, cfg, S):
    """0xFF into every kernel-written region of ``buffer`` (stream ordered: call it right before the launch)."""
    raw = buffer.buffer.view(torch.uint8)
    lay, spans = _poison_spans(cfg, S)
    assert lay.total_bytes == raw.numel(), (lay.total_bytes, raw.numel())
    for a, b in spans.values():
        raw[a:b].fill_(POISON)
    return spans


def _dg(env, S, loc, perdie, y, cfg):
    """One DeepGEMM launch (per-die on die-local weights or plain) on the poisoned shared buffer."""
    _poison(env.buffer, cfg, S)
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("DG_MEGA_MOE_SM107_PERDIE", "1" if perdie else "0")
        kw = (
            dict(shared_l1_weights=env.shared[0], shared_l2_weights=env.shared[1])
            if S > 0
            else {}
        )
        a, b = loc if perdie else (env.tl1, env.tl2)
        deep_gemm.fp8_fp4_mega_moe(
            y,
            a,
            b,
            env.buffer,
            activation="swiglu",
            activation_clamp=10.0,
            fast_math=True,
            **kw,
        )
        torch.cuda.synchronize()


def _symb(env):
    return types.SimpleNamespace(
        base_ptr=env.buffer.buffer.data_ptr(), peer_offsets=[0], group=env.group
    )


def _port(
    runner, env, S, cfg, loc1, loc2, dmap, y, M, stats, trace=None, num_tokens=None
):
    """One launch of the port's per-die runner on the poisoned shared buffer (no synchronization: back-to-back launches stay back-to-back; the poison is stream ordered)."""
    _poison(env.buffer, cfg, S)
    sh = (
        ()
        if S == 0
        else (env.shared[0][0], env.shared[0][1], env.shared[1][0], env.shared[1][1])
    )
    kw = {} if trace is None else dict(trace=trace)
    runner(
        y,
        stats,
        M if num_tokens is None else num_tokens,
        _symb(env),
        loc1[0],
        loc1[1],
        loc2[0],
        loc2[1],
        *sh,
        sm_locality_domains=dmap,
        **kw,
    )


def _nan(M):
    return torch.full((M, H), float("nan"), dtype=torch.bfloat16, device="cuda")


def _bits(y):
    return y.view(torch.int16)


def _assert_workspace_clean(env):
    sig = env.buffer.buffer.view(torch.uint8)[: L.SIGNALS_SIZEOF]
    for name, off, nbytes in (
        ("expert_recv_count_sum", L.SIGNALS_EXPERT_RECV_COUNT_SUM, 8 * 512),
        ("l1_full_count", L.SIGNALS_L1_FULL_COUNT, 4 << 20),
        ("task_counts", L.SIGNALS_L1_TASK_COUNT, 32),
        ("shared_l2_full_count", L.SIGNALS_SHARED_L2_FULL_COUNT, 4 << 15),
    ):
        assert int((sig[off : off + nbytes] != 0).sum().item()) == 0, (
            f"workspace field {name} not clean"
        )


@pytest.mark.parametrize(
    "M,masked",
    [(0, 0.0), (1, 0.0), (7, 0.0), (64, 0.3), (300, 0.0), (1000, 0.0), (1900, 0.0)],
)
def test_port_per_die_bit_exact_vs_deepgemm_per_die(setup, M, masked):
    S, env, loc1, loc2, dmap = setup
    cfg, counts = _stage(env, S, M, seed=100 + M, masked=masked)
    y_pd = _nan(M)
    y_plain = _nan(M)
    _dg(env, S, (loc1, loc2), True, y_pd, cfg)
    _dg(env, S, (loc1, loc2), False, y_plain, cfg)
    if M:
        assert (
            not torch.isnan(y_pd.float()).any()
            and not torch.isnan(y_plain.float()).any()
        ), "DeepGEMM produced NaN on the poisoned buffer"
        assert torch.equal(_bits(y_pd), _bits(y_plain)), (
            "DeepGEMM per-die != DeepGEMM plain (scheduling must not change the math)"
        )
    runner = jit.get_or_compile_kernel(
        cfg, mode_mask=7, roles="auto", use_disk_cache=False, timeout_sec=60.0
    )
    assert cfg.num_locality_domains == 2 and cfg.localized_weights
    stats = torch.zeros(E, dtype=torch.int32, device="cuda")
    n = 0
    for _ in range(
        3
    ):  # synchronous launches, then back-to-back (the poison of the next launch is stream ordered, no host synchronization)
        y = _nan(M)
        _port(runner, env, S, cfg, loc1, loc2, dmap, y, M, stats)
        torch.cuda.synchronize()
        n += 1
        if M:
            assert torch.equal(_bits(y), _bits(y_pd))
    ys = [_nan(M) for _ in range(4)]
    for yy in ys:
        _port(runner, env, S, cfg, loc1, loc2, dmap, yy, M, stats)
    torch.cuda.synchronize()
    n += 4
    if M:
        assert all(torch.equal(_bits(yy), _bits(y_pd)) for yy in ys)
    assert torch.equal(stats.cpu(), (counts * n).to(torch.int32).cpu()), (
        "cumulative recv stats"
    )
    _assert_workspace_clean(env)


# (M, masked, seed): M and the tokens / routing change from launch to launch; 300 appears twice with different seeds (the two results must differ: a stale read cannot hide)
CHANGING = [
    (300, 0.0, 41),
    (7, 0.0, 42),
    (1000, 0.2, 43),
    (64, 0.0, 44),
    (1900, 0.0, 45),
    (1, 0.0, 46),
    (300, 0.0, 47),
]


def test_port_per_die_changing_inputs_between_launches(setup):
    """Tokens, routing, M (hence BLOCK_M and the pool geometry) change between launches and the port never runs on inputs that DeepGEMM just consumed: the references of all input sets are computed first,
    then the port runs the sequence forwards and backwards on the poisoned buffer, re-staging the inputs of the launch right before it."""
    S, env, loc1, loc2, dmap = setup
    refs = []
    for M, masked, seed in CHANGING:
        cfg, counts = _stage(env, S, M, seed=seed, masked=masked)
        y = _nan(M)
        _dg(env, S, (loc1, loc2), True, y, cfg)
        refs.append((cfg, counts, y))
    assert not torch.equal(_bits(refs[0][2]), _bits(refs[-1][2])), (
        "two input sets of the same M must give different results (otherwise a stale read could hide)"
    )
    stats = torch.zeros(E, dtype=torch.int32, device="cuda")
    expected = torch.zeros(E, dtype=torch.int64)
    order = list(range(len(CHANGING))) + list(reversed(range(len(CHANGING))))
    for k in order:
        M, masked, seed = CHANGING[k]
        cfg, counts = _stage(
            env, S, M, seed=seed, masked=masked
        )  # re-stage: the buffer holds the previous launch's (different) inputs
        runner = jit.get_or_compile_kernel(
            cfg, mode_mask=7, roles="auto", use_disk_cache=False, timeout_sec=60.0
        )
        y = _nan(M)
        _port(runner, env, S, cfg, loc1, loc2, dmap, y, M, stats)
        torch.cuda.synchronize()
        assert torch.equal(_bits(y), _bits(refs[k][2])), (
            f"launch {k} (M={M}, BLOCK_M {cfg.block_m}): y != DeepGEMM per-die reference"
        )
        expected += counts.cpu()
    assert torch.equal(stats.cpu(), expected.to(torch.int32)), "cumulative recv stats"
    _assert_workspace_clean(env)


@pytest.mark.parametrize("who", ["deepgemm", "port"])
def test_canary_poison_is_complete_effective_and_overwritten_by_a_correct_kernel(
    setup, who
):
    """The poison covers exactly the declared kernel-written regions (inside: all 0xFF; outside: the clean signals and the staged inputs unchanged), a CORRECT kernel overwrites every one of them
    (so the list is neither wrong nor incomplete, the poison is not vacuous) and still produces the reference."""
    S, env, loc1, loc2, dmap = setup
    M = 300
    cfg, counts = _stage(env, S, M, seed=777)
    raw = env.buffer.buffer.view(torch.uint8)
    y_ref = _nan(M)
    _dg(
        env, S, (loc1, loc2), True, y_ref, cfg
    )  # leaves the identical, correct results in the buffer: the stale state the poison has to remove
    lay, spans = _poison_spans(cfg, S)
    before = raw.clone()
    _poison(env.buffer, cfg, S)
    mask = torch.zeros(raw.numel(), dtype=torch.bool, device="cuda")
    for a, b in spans.values():
        mask[a:b] = True
    assert bool((raw[mask] == POISON).all()), (
        "a kernel-written region is not fully poisoned"
    )
    assert torch.equal(raw[~mask], before[~mask]), (
        "the poison touched the signals or the staged inputs"
    )
    assert {
        "src_token_topk_idx",
        "token_src_metadata",
        "l1_token",
        "l1_sf",
        "l1_topk_weights",
        "l2_token",
        "l2_sf",
        "combine_token",
    } <= set(spans)
    assert (S == 0) == ("shared_l2_token" not in spans)
    y = _nan(M)
    if who == "deepgemm":
        _dg(env, S, (loc1, loc2), True, y, cfg)
    else:
        runner = jit.get_or_compile_kernel(
            cfg, mode_mask=7, roles="auto", use_disk_cache=False, timeout_sec=60.0
        )
        _port(
            runner,
            env,
            S,
            cfg,
            loc1,
            loc2,
            dmap,
            y,
            M,
            torch.zeros(E, dtype=torch.int32, device="cuda"),
        )
        torch.cuda.synchronize()
    assert torch.equal(_bits(y), _bits(y_ref))
    for name, (a, b) in spans.items():
        assert bool((raw[a:b] != POISON).any()), (
            f"{who} did not overwrite the poisoned region {name}: the list of kernel-written regions is wrong or the kernel does not use it"
        )
    _assert_workspace_clean(env)


def test_canary_a_launch_that_produces_nothing_fails_the_bit_exact_comparison(setup):
    """The oracle is not vacuous: a launch told to process 0 tokens although M = 300 tokens are staged leaves the NaN-filled output untouched, which the comparison of the real tests rejects."""
    S, env, loc1, loc2, dmap = setup
    M = 300
    cfg, counts = _stage(env, S, M, seed=778)
    y_ref = _nan(M)
    _dg(env, S, (loc1, loc2), True, y_ref, cfg)
    runner = jit.get_or_compile_kernel(
        cfg, mode_mask=7, roles="auto", use_disk_cache=False, timeout_sec=60.0
    )
    stats = torch.zeros(E, dtype=torch.int32, device="cuda")
    y = _nan(M)
    _port(runner, env, S, cfg, loc1, loc2, dmap, y, M, stats, num_tokens=0)
    torch.cuda.synchronize()
    assert bool(torch.isnan(y.float()).all()), (
        "a launch without tokens wrote the output"
    )
    assert not torch.equal(_bits(y), _bits(y_ref)), (
        "the bit-exact comparison accepted an output that was never produced"
    )
    assert int(stats.sum().item()) == 0
    _assert_workspace_clean(env)


@pytest.mark.parametrize("M", [8, 256, 1000])
def test_trace_domain_lookup_and_claim_counts(setup, M):
    S, env, loc1, loc2, dmap = setup
    cfg, counts = _stage(env, S, M, seed=7 + M)
    cfg_t = get_sm107_kernel_config(
        1,
        E,
        TMAX,
        M,
        TOPK,
        H,
        I,
        S,
        num_sms=NUM_SMS,
        activation_clamp=10.0,
        per_die=True,
        localized_weights=True,
        trace=True,
    )
    y_ref = _nan(M)
    _dg(env, S, (loc1, loc2), True, y_ref, cfg_t)
    runner = jit.get_or_compile_kernel(
        cfg_t, mode_mask=7, roles="auto", use_disk_cache=False, timeout_sec=60.0
    )
    tb = torch.zeros(trace_buffer_u64s(cfg_t.num_sms), dtype=torch.int64, device="cuda")
    y = _nan(M)
    stats = torch.zeros(E, dtype=torch.int32, device="cuda")
    _port(runner, env, S, cfg_t, loc1, loc2, dmap, y, M, stats, trace=tb)
    torch.cuda.synchronize()
    assert torch.equal(_bits(y), _bits(y_ref)), (
        "traced per-die build: y != DeepGEMM per-die"
    )
    t = tb[: cfg_t.num_sms * 64].view(cfg_t.num_sms, 64).cpu()
    leaders = torch.arange(
        0, cfg_t.num_sms, 2, device="cpu"
    )  # (a default CUDA device may be active: keep the trace analysis on the CPU)
    smid = t[leaders, 46] - 1  # trace.TRACE_SLOT_SMID (scheduler_perdie)
    dom = t[leaders, 47] - 1
    pm = dmap.cpu().to(torch.int64)
    assert bool((smid >= 0).all() and (smid < cfg_t.num_sms).all())
    assert len(set(smid.tolist())) == len(leaders), (
        "two leader CTAs reported the same %smid"
    )
    assert torch.equal(dom, pm[smid]), (
        "the domain read by the scheduler warp != sm_locality_domains[%smid]"
    )
    own, stolen = int(t[leaders, 48].sum()), int(t[leaders, 49].sum())
    nb = sum(-(-int(c) // cfg_t.block_m) for c in counts.tolist())
    expected = nb * (cfg_t.num_l1_clusters + cfg_t.num_l2_clusters)
    if S > 0:
        expected += -(-M // cfg_t.block_m) * (
            cfg_t.l1_shape_n * S // 256 + cfg_t.l2_shape_n // 256
        )
    assert own + stolen == expected, (
        f"claims: own {own} + stolen {stolen} != {expected}"
    )
    assert own > 0 and stolen < own // 4, (
        "work stealing should be the exception with a balanced map"
    )


@pytest.mark.parametrize("M", [64, 1000])
def test_frontend_selects_the_per_die_variant_for_die_local_weights(setup, M):
    """The DeepGEMM-compatible entry (the package's frontend + the runner): die-local routed weights => per-die variant, SM map passed by the frontend."""
    S, env, loc1, loc2, dmap = setup
    from flashinfer.experimental.sm107_deepgemm_megamoe import compat

    cfg, counts = _stage(env, S, M, seed=900 + M)
    y_pd = _nan(M)
    _dg(env, S, (loc1, loc2), True, y_pd, cfg)
    pbuf = compat.get_symm_buffer_for_mega_moe(
        env.group, E, TMAX, TOPK, H, I, num_shared_experts=S
    )  # the port's own symmetric buffer (same inputs staged)
    try:
        for name in ("x", "x_sf", "topk_idx", "topk_weights"):
            getattr(pbuf, name)[:M].copy_(getattr(env.buffer, name)[:M])
        if S > 0:
            pbuf.shared_l1_acts_sf.copy_(env.buffer.shared_l1_acts_sf)
        _poison(pbuf, cfg, S)  # (the frontend's buffer holds no stale results either)
        y = _nan(M)
        kw = (
            dict(shared_l1_weights=env.shared[0], shared_l2_weights=env.shared[1])
            if S > 0
            else {}
        )
        compat.fp8_fp4_mega_moe(
            y,
            loc1,
            loc2,
            pbuf,
            activation="swiglu",
            activation_clamp=10.0,
            fast_math=True,
            **kw,
        )
        torch.cuda.synchronize()
        assert torch.equal(_bits(y), _bits(y_pd))
    finally:
        pbuf.destroy()
