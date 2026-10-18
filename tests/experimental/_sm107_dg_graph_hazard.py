# SPDX-License-Identifier: Apache-2.0
# Adapted from the DeepGEMM Rubin SM107 MegaMoE kernel (deepseek-ai/DeepGEMM, MIT License, Copyright (c) 2025 DeepSeek; see licenses/LICENSE.deepgemm.txt; branch rubin-megamoe-a7a9277-1016 @3ec0eb4) for the FlashInfer CuteDSL port.
# Test helper of test_sm107_dg_dispatch_combine_graph_replay.py (dispatch / combine roles): INDEPENDENT adversarial test of the `%gridid` CUDA-graph replay hazard of the MegaMoE combine-ready flag (DeepGEMM stock vs the port's tag modes).
#
# Mechanism under test (K:1690-1713, K:445-470): rank j pushes `%gridid_j + 1` to every peer (peer_grid_idx), every peer returns it as the combine-ready flag after its L2 write-backs,
# rank j's combine waits `ld.acquire.sys(combine_ready[peer]) == %gridid_j + 1`.  The flags are never reset, `%gridid` is identical for every replay of one CUDA-graph kernel node
# => if rank j's PREVIOUS launch on this workspace was the same graph node, the flag already holds the expected value: the wait passes before the peer finished writing.
#
# One process per rank (torch.multiprocessing.spawn like DeepGEMM's own tests).  Per experiment, N iterations; the STRICT protocol of every iteration is
#   synchronize(); barrier(); synchronize()      <- every rank is idle: no peer can still pull from my inputs (rules out an input-overwrite artifact)
#   copy the iteration's input set into the symmetric buffer, y.fill_(NaN)
#   synchronize(); barrier(); synchronize()      <- aligned start
#   [GPU-side rank-dependent skew sleep]; launch | graph.replay(); synchronize(); compare y bit-exactly with the EAGER DeepGEMM reference of the same set
# Experiments (token `name[:skew_cycles]`):
#   eager     eager launches, inputs alternate A/B (control: unique %gridid per launch)
#   graph     ONE graph, replayed with inputs alternating A/B (the hazard)
#   same      ONE graph, replayed with IDENTICAL inputs (stale == fresh: must pass)
#   two       TWO graphs (gA, gB: two kernel nodes), replayed alternately with inputs A/B (previous launch always the other node)
#   mix       graph replay, then an eager launch of the other set, alternating (previous launch of the replay is an eager launch)
#   async     ONE graph, inputs alternate, NO host sync / barrier in the loop (production-like enqueue), per-launch device-side compare
#   buckets   3 graphs of different M per rank (vLLM-like batch buckets), per-rank random bucket sequence, inputs alternate per bucket use; wrong counted by "same bucket as the previous launch"
#   twonode   ONE graph containing TWO MegaMoE launches on the same workspace (layer 1, layer 2), replayed with changing inputs
# Kernels (`G_KERNELS`): dg (DeepGEMM stock), port-gridid (faithful), port-epoch, port-epoch-pdl, port-gridid-pdl.
# After each strict experiment the workspace tag words are sampled (peer_grid_idx[], combine_ready_grid_idx[], nvl_barrier_counter) to show the tag sequence directly.
# env: R_RANKS(=2) G_E(=32) G_MS (per rank M, e.g. 300/300 or 1500/200) G_N(=300) G_SKEW(=2000000) G_KERNELS G_EXPS G_SEED G_MASKED G_TMAX G_QUIET
#      G_POISON=1|<hex byte>: under the strict protocol (every rank idle between the two synchronize+barrier pairs of an iteration) fill EVERY kernel-written region of the symmetric buffer(s) with this byte before EVERY launch / replay
#      of both kernels (dispatch rings, L1/L2 activations + SF, combine slots, token_src_metadata AND the remotely written src_token_topk_idx; inputs and the workspace signals untouched); the `async` experiment (no host sync) is not poisoned.
import os
import sys
import time
import types

os.environ.setdefault("CUTE_DSL_ARCH", "sm_107a")
os.environ.setdefault("CUDA_MODULE_LOADING", "EAGER")


def env_int(name, default):
    return int(os.environ.get(name, default))


def worker(local_rank, nranks):
    import traceback

    try:
        main(local_rank, nranks)
    except Exception:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(2)
    sys.stdout.flush()
    os._exit(0)


def hash32(*vals):
    h = 0x9E3779B1
    for v in vals:
        h = ((h ^ (int(v) & 0xFFFFFFFF)) * 0x85EBCA6B) & 0xFFFFFFFF
        h ^= h >> 13
        h = (h * 0xC2B2AE35) & 0xFFFFFFFF
        h ^= h >> 16
    return h


def main(local_rank, nranks):
    import torch
    import torch.distributed as dist

    import deep_gemm
    from deep_gemm.utils import per_token_cast_to_fp4, per_token_cast_to_fp8
    from deep_gemm.utils.dist import init_dist

    from flashinfer.experimental.sm107_deepgemm_megamoe import jit
    from flashinfer.experimental.sm107_deepgemm_megamoe.heuristics import (
        get_sm107_kernel_config,
    )
    from flashinfer.experimental.sm107_deepgemm_megamoe.kernel import layout as L

    E = env_int("G_E", 32)
    topk = env_int("G_TOPK", 6)
    Hh, I = env_int("G_H", 5120), env_int("G_I", 2304)
    Tmax = env_int("G_TMAX", 1920)
    seed = env_int("G_SEED", 0)
    masked = float(os.environ.get("G_MASKED", "0"))
    clamp = 10.0
    N = env_int("G_N", 300)
    skew_default = env_int("G_SKEW", 2000000)
    kernels = os.environ.get("G_KERNELS", "dg,port-gridid,port-epoch").split(",")
    exps = os.environ.get(
        "G_EXPS", "eager,graph:0,graph,same,two,mix,async,buckets,twonode"
    ).split(",")
    Ms_all = [int(v) for v in os.environ.get("G_MS", "300").split("/")]
    Ms_all = Ms_all * nranks if len(Ms_all) == 1 else Ms_all
    assert len(Ms_all) == nranks

    rank, nranks, group = init_dist(local_rank, nranks)
    dev = "cuda"
    El = E // nranks
    assert E % nranks == 0
    M = Ms_all[rank]
    pre = f"[r{rank}]"
    quiet = bool(env_int("G_QUIET", 0))
    buffer = deep_gemm.get_symm_buffer_for_mega_moe(
        group,
        E,
        Tmax,
        topk,
        Hh,
        I,
        num_shared_experts=0,
        mma_type="fp8xfp4",
        activation="swiglu",
    )
    base = buffer.buffer.data_ptr()
    peer_offsets = [int(p) - base for p in buffer.handle.buffer_ptrs]
    symb = types.SimpleNamespace(base_ptr=base, peer_offsets=peer_offsets, group=group)
    sig = buffer.buffer.view(torch.uint8)[: L.SIGNALS_SIZEOF]

    def cast_w(w):
        g, n, k = w.shape
        q = torch.empty((g, n, k // 2), device=dev, dtype=torch.int8)
        sf = torch.empty((g, n, k // 32), device=dev, dtype=torch.float)
        for i in range(g):
            q[i], sf[i] = per_token_cast_to_fp4(w[i], use_ue8m0=True, gran_k=32)
        sf = deep_gemm.transform_sf_into_required_layout(sf, n, k, (1, 32), g)
        return q, sf

    torch.manual_seed(seed + 7919 * rank)
    l1w = torch.randn((El, 2 * I, Hh), dtype=torch.bfloat16, device=dev)
    l2w = torch.randn((El, Hh, I), dtype=torch.bfloat16, device=dev)
    l1, l2 = cast_w(l1w), cast_w(l2w)
    del l1w, l2w
    tl1, tl2 = deep_gemm.transform_weights_for_mega_moe(l1, l2, activation="swiglu")
    stats_t = torch.zeros(El, dtype=torch.int32, device=dev)

    def log(*a):
        print(pre, *a, flush=True)

    def make_set(m, tag):
        g = torch.Generator(device=dev)
        g.manual_seed(seed * 1000003 + 7919 * rank + 104729 * tag + 17)
        x = torch.randn((m, Hh), dtype=torch.bfloat16, device=dev, generator=g)
        scores = torch.randn((m, E), dtype=torch.float, device=dev, generator=g)
        tw, ti = torch.topk(scores, topk, dim=-1, largest=True, sorted=False)
        if masked > 0:
            rm = torch.rand(ti.shape, dtype=torch.float, device=dev, generator=g)
            ti.masked_fill_(rm < masked, -1)
            tw.masked_fill_(ti < 0, 0)
        xq, xs = per_token_cast_to_fp8(
            x, use_ue8m0=True, gran_k=32, use_packed_ue8m0=True
        )
        return dict(m=m, x=xq, x_sf=xs, ti=ti, tw=tw)

    b2 = {}  # lazily created SECOND workspace (the `twobuf` experiment: one graph, two MegaMoE launches, one workspace per launch = per-layer workspaces)

    def get_b2():
        if not b2:
            buf2 = deep_gemm.get_symm_buffer_for_mega_moe(
                group,
                E,
                Tmax,
                topk,
                Hh,
                I,
                num_shared_experts=0,
                mma_type="fp8xfp4",
                activation="swiglu",
            )
            base2 = buf2.buffer.data_ptr()
            b2.update(
                buf=buf2,
                symb=types.SimpleNamespace(
                    base_ptr=base2,
                    peer_offsets=[int(p) - base2 for p in buf2.handle.buffer_ptrs],
                    group=group,
                ),
            )
        return b2

    def load(st, second=False):
        m = st["m"]
        b = get_b2()["buf"] if second else buffer
        b.x[:m].copy_(st["x"])
        b.x_sf[:m].copy_(st["x_sf"])
        b.topk_idx[:m].copy_(st["ti"])
        b.topk_weights[:m].copy_(st["tw"])

    def dg_launch(y, second=False):
        deep_gemm.fp8_fp4_mega_moe(
            y,
            tl1,
            tl2,
            get_b2()["buf"] if second else buffer,
            activation="swiglu",
            activation_clamp=clamp,
            fast_math=True,
        )

    def sync_barrier():
        torch.cuda.synchronize()
        dist.barrier(group)
        torch.cuda.synchronize()

    pv = os.environ.get("G_POISON", "0").strip().lower()
    poison_val = None if pv in ("", "0") else (0xFF if pv == "1" else int(pv, 16))
    poison_regions = (
        "shared_l2_token",
        "shared_l2_sf",
        "l1_token",
        "l1_sf",
        "l1_topk_weights",
        "l2_token",
        "l2_sf",
        "combine_token",
    )
    lay_p = []

    def poison_all():
        """strict protocol: call only while every rank is idle (between the two sync_barrier()s of an iteration)"""
        if poison_val is None:
            return
        if not lay_p:
            cfgp = get_sm107_kernel_config(
                nranks, E, Tmax, M, topk, Hh, I, 0, num_sms=212, activation_clamp=clamp
            )
            lay_p.append(
                L.buffer_layout(
                    hidden=Hh,
                    intermediate_hidden=I,
                    num_ranks=nranks,
                    num_experts=E,
                    num_max_tokens_per_rank=cfgp.num_max_tokens_per_rank,
                    num_topk=topk,
                    num_ring_tokens=cfgp.num_ring_tokens,
                    num_sf_ring_tokens=cfgp.num_sf_ring_tokens,
                    num_shared_experts=0,
                )
            )
        lay = lay_p[0]
        for b_ in [buffer] + ([b2["buf"]] if b2 else []):
            raw = b_.buffer.view(torch.uint8)
            for name in poison_regions:
                r = lay[name]
                if r.nbytes:
                    raw[r.offset : r.end].fill_(poison_val)
            w = lay.workspace
            raw[
                w.token_src_metadata_offset : w.token_src_metadata_offset
                + w.num_max_pool_tokens * L.TOKEN_SRC_METADATA_BYTES
            ].fill_(poison_val)
            raw[
                w.src_token_topk_idx_offset : w.src_token_topk_idx_offset
                + E * w.num_max_tokens_per_rank * 4
            ].fill_(poison_val)

    def eager_ref(st):
        load(st)
        y = torch.empty((st["m"], Hh), dtype=torch.bfloat16, device=dev)
        sync_barrier()
        poison_all()
        sync_barrier()
        dg_launch(y)
        torch.cuda.synchronize()
        sync_barrier()
        return y

    # ---- input sets: NS sets for the plain experiments (M of this rank), 3 buckets x 2 sets for `buckets`
    NS = 4
    sets = [make_set(M, s) for s in range(NS)]
    for _s, st in enumerate(sets):
        st["ref"] = eager_ref(st)
    log(
        f"EP{nranks} E={E} M per rank {Ms_all}: {NS} input sets + eager DeepGEMM references done (absmax {max(float(st['ref'].float().abs().max()) for st in sets):.3g})"
    )
    BF = (0.25, 0.5, 1.0)
    bm_per_rank = [
        [max(1, int(round(Ms_all[r] * f))) for f in BF] for r in range(nranks)
    ]
    bsets = None  # built lazily when `buckets` is requested

    rsets = None

    def build_remote_sets():
        """2 input sets whose every route (all 6 top-k) targets experts of OTHER ranks: every row of y is a sum of PEER-written combine rows only"""
        nonlocal rsets
        if rsets is None:
            rsets = []
            for s in range(2):
                g = torch.Generator(device=dev)
                g.manual_seed(seed * 1000003 + 7919 * rank + 104729 * (50 + s) + 17)
                x = torch.randn((M, Hh), dtype=torch.bfloat16, device=dev, generator=g)
                scores = torch.randn((M, E), dtype=torch.float, device=dev, generator=g)
                scores[:, rank * El : (rank + 1) * El] -= (
                    100.0  # never pick an expert of this rank
                )
                tw, ti = torch.topk(scores, topk, dim=-1, largest=True, sorted=False)
                xq, xs = per_token_cast_to_fp8(
                    x, use_ue8m0=True, gran_k=32, use_packed_ue8m0=True
                )
                st = dict(m=M, x=xq, x_sf=xs, ti=ti, tw=tw)
                st["ref"] = eager_ref(st)
                rsets.append(st)
        return rsets

    def tag_words():
        peer = (
            sig[L.SIGNALS_PEER_GRID_IDX : L.SIGNALS_PEER_GRID_IDX + 8 * nranks]
            .view(torch.int64)
            .tolist()
        )
        ready = (
            sig[
                L.SIGNALS_COMBINE_READY_GRID_IDX : L.SIGNALS_COMBINE_READY_GRID_IDX
                + 8 * nranks
            ]
            .view(torch.int64)
            .tolist()
        )
        nvl = int(sig[16:20].view(torch.int32)[0])
        return peer, ready, nvl

    def gather(obj):
        out = [None] * nranks
        dist.all_gather_object(out, obj, group=group)
        return out

    results = []

    def report(kname, ename, skew, n, wrong, extra=""):
        allw = gather(wrong)
        if rank == 0:
            cnt = [len(w) for w in allw]
            firsts = [w[:6] for w in allw]
            results.append(dict(kernel=kname, exp=ename, skew=skew, n=n, wrong=cnt))
            print(
                f"HAZ {kname:16s} {ename:8s} EP{nranks} Ms={Ms_all} skew={skew} N={n}: wrong per rank {cnt} first {firsts} {extra}",
                flush=True,
            )

    def skew_sleep(it, skew):
        if skew:
            torch.cuda._sleep(
                ((it * 7 + rank * 5) % 4) * skew
            )  # rank-dependent, varying per iteration (same pattern as the foundation tests)

    def strict_iter(it, st, y, go, skew, pre=None):
        """one strict iteration; returns True if y == the eager DeepGEMM reference of the set"""
        sync_barrier()
        load(st)
        y.fill_(float("nan"))
        if pre is not None:
            pre()
        poison_all()
        sync_barrier()
        skew_sleep(it, skew)
        go()
        torch.cuda.synchronize()
        return torch.equal(y.view(torch.int16), st["ref"].view(torch.int16))

    def capture(launch):
        g = torch.cuda.CUDAGraph()
        sync_barrier()
        with torch.cuda.graph(g):
            launch()
        sync_barrier()
        return g

    def run_kernel(kname):
        if kname == "dg":

            def mk(m, second=False):
                return lambda y: dg_launch(y, second)
        else:
            _, tm, *rest = kname.split("-")
            pdl = "pdl" in rest
            cfg_cache = {}

            def mk(m, second=False):
                if m not in cfg_cache:
                    cfg = get_sm107_kernel_config(
                        nranks,
                        E,
                        Tmax,
                        m,
                        topk,
                        Hh,
                        I,
                        0,
                        num_sms=212,
                        activation_clamp=clamp,
                    )
                    r_ = jit.get_or_compile_kernel(
                        cfg,
                        mode_mask=7,
                        roles="auto",
                        use_disk_cache=False,
                        tag_mode=tm,
                        pdl=pdl,
                        timeout_sec=60,
                    )
                    cfg_cache[m] = r_
                r_ = cfg_cache[m]
                return lambda y: r_(
                    y,
                    stats_t,
                    m,
                    (get_b2()["symb"] if second else symb),
                    tl1[0],
                    tl1[1],
                    tl2[0],
                    tl2[1],
                )

        launch = mk(M)
        for ex in exps:
            name, _, sk = ex.partition(":")
            skew = int(sk) if sk else skew_default
            y = torch.full((M, Hh), float("nan"), dtype=torch.bfloat16, device=dev)
            wrong, tags = [], []
            extra_pre = ""
            t0 = time.time()
            if name == "eager":
                for it in range(N):
                    ok = strict_iter(it, sets[it % 2], y, lambda: launch(y), skew)
                    tags.append(tag_words())
                    if not ok:
                        wrong.append(it)
            elif name in ("graph", "same"):
                g = capture(lambda: launch(y))
                for it in range(N):
                    ok = strict_iter(
                        it, sets[(it % 2) if name == "graph" else 0], y, g.replay, skew
                    )
                    tags.append(tag_words())
                    if not ok:
                        wrong.append(it)
            elif name == "two":
                ga, gb = capture(lambda: launch(y)), capture(lambda: launch(y))
                gs = (ga, gb)
                for it in range(N):
                    ok = strict_iter(it, sets[it % 2], y, gs[it % 2].replay, skew)
                    tags.append(tag_words())
                    if not ok:
                        wrong.append(it)
            elif name == "mix":
                g = capture(lambda: launch(y))
                y2 = torch.full((M, Hh), float("nan"), dtype=torch.bfloat16, device=dev)
                wrong_e = []
                for it in range(N):
                    ok = strict_iter(
                        it, sets[it % 2], y, g.replay, skew
                    )  # replay (previous launch on the workspace: the eager launch below)
                    tags.append(tag_words())
                    if not ok:
                        wrong.append(it)
                    ok2 = strict_iter(
                        it, sets[(it + 1) % 2], y2, lambda: launch(y2), skew
                    )
                    if not ok2:
                        wrong_e.append(it)
                report(kname, "mix/eager", skew, N, wrong_e)
            elif name == "twobuf":
                get_b2()
                launch2 = mk(M, second=True)
                src1 = dict(
                    x=sets[0]["x"].clone(),
                    x_sf=sets[0]["x_sf"].clone(),
                    ti=sets[0]["ti"].clone(),
                    tw=sets[0]["tw"].clone(),
                    m=M,
                )
                src2 = dict(
                    x=sets[1]["x"].clone(),
                    x_sf=sets[1]["x_sf"].clone(),
                    ti=sets[1]["ti"].clone(),
                    tw=sets[1]["tw"].clone(),
                    m=M,
                )
                y1 = torch.full((M, Hh), float("nan"), dtype=torch.bfloat16, device=dev)
                y2 = torch.full((M, Hh), float("nan"), dtype=torch.bfloat16, device=dev)

                def two_bufs():
                    load(src1)
                    launch(y1)
                    load(src2, second=True)
                    launch2(y2)

                g = capture(two_bufs)
                wrong2 = []
                for it in range(N):
                    a, b = (0, 1) if it % 2 == 0 else (2, 3)
                    sync_barrier()
                    for dst, srcset in ((src1, sets[a]), (src2, sets[b])):
                        for k in ("x", "x_sf", "ti", "tw"):
                            dst[k].copy_(srcset[k])
                    y1.fill_(float("nan"))
                    y2.fill_(float("nan"))
                    poison_all()
                    sync_barrier()
                    skew_sleep(it, skew)
                    g.replay()
                    torch.cuda.synchronize()
                    if not torch.equal(
                        y1.view(torch.int16), sets[a]["ref"].view(torch.int16)
                    ):
                        wrong.append(it)
                    if not torch.equal(
                        y2.view(torch.int16), sets[b]["ref"].view(torch.int16)
                    ):
                        wrong2.append(it)
                report(kname, "twobuf/2", skew, N, wrong2)
            elif name == "stale":
                rs = build_remote_sets()
                g = capture(lambda: launch(y))
                tot = [
                    0,
                    0,
                    0,
                    0,
                    0,
                ]  # rows == reference of THIS replay's inputs, rows == the PREVIOUS replay's output (fully stale), other (mixed fresh/stale slots), mixed rows explained by a cur/prev slot subset, unexplained
                cfg0 = get_sm107_kernel_config(
                    nranks,
                    E,
                    Tmax,
                    M,
                    topk,
                    Hh,
                    I,
                    0,
                    num_sms=212,
                    activation_clamp=clamp,
                )
                lay0 = L.buffer_layout(
                    hidden=Hh,
                    intermediate_hidden=I,
                    num_ranks=nranks,
                    num_experts=E,
                    num_max_tokens_per_rank=cfg0.num_max_tokens_per_rank,
                    num_topk=topk,
                    num_ring_tokens=cfg0.num_ring_tokens,
                    num_sf_ring_tokens=cfg0.num_sf_ring_tokens,
                    num_shared_experts=0,
                )
                cr = lay0["combine_token"]
                bufu8 = buffer.buffer.view(torch.uint8)

                def rows_snapshot():  # the BF16 (slot, token) rows the peers wrote into THIS rank's combine region (valid after the launch completed)
                    out = torch.empty((topk, M, Hh), dtype=torch.bfloat16, device=dev)
                    for k in range(topk):
                        o = cr.offset + k * cr.token_bytes * cr.num_tokens
                        out[k] = (
                            bufu8[o : o + M * cr.token_bytes]
                            .view(torch.bfloat16)
                            .view(M, Hh)
                        )
                    return out

                prev_rows = None
                for it in range(N):
                    st, prv = rs[it % 2], rs[(it + 1) % 2]
                    ok = strict_iter(it, st, y, g.replay, skew)
                    cur_rows = rows_snapshot()
                    if it >= 1:
                        yv = y.view(torch.int16)
                        eq_cur = (yv == st["ref"].view(torch.int16)).all(dim=1)
                        eq_prev = (yv == prv["ref"].view(torch.int16)).all(dim=1)
                        tot[0] += int(eq_cur.sum())
                        tot[1] += int((eq_prev & ~eq_cur).sum())
                        mixed = ~eq_cur & ~eq_prev
                        nm = int(mixed.sum())
                        tot[2] += nm
                        if nm and prev_rows is not None:
                            idx = torch.nonzero(mixed).flatten()
                            C = cur_rows[:, idx, :].float()
                            Pv = prev_rows[:, idx, :].float()
                            Yb = yv[idx]
                            SEG = 16  # columns (32 B): the peers' L2 tiles / NVLink sectors tear a (token, slot) row at sub-row granularity
                            matched = torch.zeros(
                                nm, Hh // SEG, dtype=torch.bool, device=dev
                            )
                            for msk in range(
                                1 << topk
                            ):  # bit k set: slot k taken from the CURRENT replay's rows, clear: from the PREVIOUS replay's rows (stale); chosen independently per 32-B segment
                                acc = torch.zeros(
                                    nm, Hh, dtype=torch.float32, device=dev
                                )
                                for k in range(topk):
                                    acc = acc + (C[k] if (msk >> k) & 1 else Pv[k])
                                matched |= (
                                    (acc.to(torch.bfloat16).view(torch.int16) == Yb)
                                    .view(nm, Hh // SEG, SEG)
                                    .all(dim=2)
                                )
                            row_ok = matched.all(dim=1)
                            tot[3] += int(row_ok.sum())
                            tot[4] += int((~row_ok).sum())
                        if not ok:
                            wrong.append(it)
                    prev_rows = cur_rows
                allt = gather(tot)
                extra_pre = (
                    "| rows (all routes target the other ranks), summed over replays 1..N-1 per rank [fresh, fully-stale(==previous replay's y), mixed(y != fresh), mixed rows reproduced EXACTLY as the ascending-slot "
                    "FP32/BF16 combine of {current rows for some slots, previous-replay rows for the others, chosen per 32-byte segment}, unexplained]: "
                    + str(allt)
                )
            elif name == "async":
                g = capture(lambda: launch(y))
                flags = torch.zeros(N, dtype=torch.int32, device=dev)
                sync_barrier()
                for it in range(N):
                    st = sets[it % 2]
                    load(st)
                    y.fill_(float("nan"))
                    skew_sleep(it, skew)
                    g.replay()
                    flags[it] = (
                        y.view(torch.int16) != st["ref"].view(torch.int16)
                    ).any()
                torch.cuda.synchronize()
                wrong = torch.nonzero(flags).flatten().tolist()
                sync_barrier()
            elif name == "buckets":
                nonlocal_bsets = build_buckets()
                graphs, ybs = [], []
                for b in range(3):
                    yb = torch.full(
                        (bm_per_rank[rank][b], Hh),
                        float("nan"),
                        dtype=torch.bfloat16,
                        device=dev,
                    )
                    lb = mk(bm_per_rank[rank][b])
                    graphs.append(capture(lambda lb=lb, yb=yb: lb(yb)))
                    ybs.append(yb)
                use = [0, 0, 0]
                prev = -1
                same_total = same_wrong = diff_total = diff_wrong = 0
                for it in range(N):
                    b = hash32(rank, it, seed, 3) % 3
                    s = use[b] % 2
                    use[b] += 1
                    ok = strict_iter(
                        it, nonlocal_bsets[b][s], ybs[b], graphs[b].replay, skew
                    )
                    if b == prev:
                        same_total += 1
                        same_wrong += 0 if ok else 1
                    else:
                        diff_total += 1
                        diff_wrong += 0 if ok else 1
                    if not ok:
                        wrong.append(it)
                    prev = b
                allx = gather((same_total, same_wrong, diff_total, diff_wrong))
                if rank == 0:
                    print(
                        f"HAZ {kname:16s} buckets  EP{nranks} bucket M per rank {bm_per_rank} skew={skew}: (same-bucket-as-previous: wrong/total, different bucket: wrong/total) per rank "
                        f"{[(f'{a[1]}/{a[0]}', f'{a[3]}/{a[2]}') for a in allx]}",
                        flush=True,
                    )
            elif name == "twonode":
                # ONE graph with two MegaMoE launches on the same workspace; the inputs of node 1 / node 2 come from static source tensors rewritten between replays
                src1 = dict(
                    x=sets[0]["x"].clone(),
                    x_sf=sets[0]["x_sf"].clone(),
                    ti=sets[0]["ti"].clone(),
                    tw=sets[0]["tw"].clone(),
                    m=M,
                )
                src2 = dict(
                    x=sets[1]["x"].clone(),
                    x_sf=sets[1]["x_sf"].clone(),
                    ti=sets[1]["ti"].clone(),
                    tw=sets[1]["tw"].clone(),
                    m=M,
                )
                y1 = torch.full((M, Hh), float("nan"), dtype=torch.bfloat16, device=dev)
                y2 = torch.full((M, Hh), float("nan"), dtype=torch.bfloat16, device=dev)

                def two_nodes():
                    load(src1)
                    launch(y1)
                    load(src2)
                    launch(y2)

                g = capture(two_nodes)
                wrong2 = []
                for it in range(N):
                    a, b = (0, 1) if it % 2 == 0 else (2, 3)
                    sync_barrier()
                    for dst, srcset in ((src1, sets[a]), (src2, sets[b])):
                        for k in ("x", "x_sf", "ti", "tw"):
                            dst[k].copy_(srcset[k])
                    y1.fill_(float("nan"))
                    y2.fill_(float("nan"))
                    poison_all()
                    sync_barrier()
                    skew_sleep(it, skew)
                    g.replay()
                    torch.cuda.synchronize()
                    tags.append(tag_words())
                    ok1 = torch.equal(
                        y1.view(torch.int16), sets[a]["ref"].view(torch.int16)
                    )
                    ok2 = torch.equal(
                        y2.view(torch.int16), sets[b]["ref"].view(torch.int16)
                    )
                    if not ok1:
                        wrong.append(it)
                    if not ok2:
                        wrong2.append(it)
                report(kname, "twonode/2", skew, N, wrong2)
            elif name == "trace":
                if kname == "dg":
                    continue
                import numpy as np
                from flashinfer.experimental.sm107_deepgemm_megamoe.kernel.trace import (
                    NUM_SLOTS,
                    trace_buffer_u64s,
                )

                _, tm, *rest = kname.split("-")
                cfgt = get_sm107_kernel_config(
                    nranks,
                    E,
                    Tmax,
                    M,
                    topk,
                    Hh,
                    I,
                    0,
                    num_sms=212,
                    activation_clamp=clamp,
                    trace=True,
                )
                rt = jit.get_or_compile_kernel(
                    cfgt,
                    mode_mask=7,
                    roles="auto",
                    use_disk_cache=False,
                    tag_mode=tm,
                    pdl=("pdl" in rest),
                    timeout_sec=60,
                )
                trb = torch.zeros(trace_buffer_u64s(212), dtype=torch.int64, device=dev)
                yt = torch.full((M, Hh), float("nan"), dtype=torch.bfloat16, device=dev)

                def launch_t():
                    rt(yt, stats_t, M, symb, tl1[0], tl1[1], tl2[0], tl2[1], trace=trb)

                for mode in ("eager", "graph"):
                    g = capture(launch_t) if mode == "graph" else None
                    go = launch_t if g is None else g.replay
                    snaps, wrong_t = [], []
                    for it in range(N):
                        ok = strict_iter(
                            it, sets[it % 2], yt, go, skew, pre=lambda: trb.zero_()
                        )
                        snaps.append(
                            trb[: 212 * NUM_SLOTS]
                            .view(212, NUM_SLOTS)
                            .cpu()
                            .numpy()
                            .copy()
                        )
                        if not ok:
                            wrong_t.append(it)
                    allsn = gather(snaps)
                    allw = gather(wrong_t)
                    if rank == 0:
                        R_ = nranks
                        margins, gids, pw, nvalid = [], [], [], min(212, min(Ms_all))
                        for it in range(N):
                            S = [allsn[r][it] for r in range(R_)]
                            t7 = [
                                float(np.median(S[r][:, 7][S[r][:, 7] > 0]))
                                for r in range(R_)
                            ]
                            off = [
                                t7[r] - t7[0] for r in range(R_)
                            ]  # per-rank globaltimer offset, calibrated at NVLink barrier #1 (all ranks leave it within ~1-3 us)
                            t5 = [
                                float(S[r][0, 5]) - off[r] for r in range(R_)
                            ]  # rank r SM0: combine start = ready flags published (trace slot 5 precedes the flag store)
                            t6med = [
                                float(np.median(S[j][:nvalid, 6])) - off[j]
                                for j in range(R_)
                            ]  # median CTA end of the combine of rank j (CTAs that own a chunk)
                            margins.append(
                                [
                                    (t6med[j] - t5[r]) / 1e3
                                    for j in range(R_)
                                    for r in range(R_)
                                    if j != r
                                ]
                            )
                            gids.append([int(S[r][0, 62]) for r in range(R_)])
                            pw.append(
                                [float(np.median(S[r][:nvalid, 40])) for r in range(R_)]
                            )
                        mg = np.array(
                            margins
                        )  # (N, pairs) us: >0 = rank j finished (the median CTA of) its combine AFTER rank r published readiness
                        skip = 1  # the first iteration of a fresh graph/eager sequence follows a different launch (always fresh)
                        print(
                            f"TRACE {kname:16s} {mode:5s} EP{nranks} Ms={Ms_all} skew={skew} N={N}: y wrong {[len(w) for w in allw]}; "
                            f"margin(median CTA combine end of j - readiness publication of r) us: median {np.median(mg[skip:]):.1f}, min {mg[skip:].min():.1f}, "
                            f"fraction < 0: {float((mg[skip:] < 0).mean()):.3f}, < -5us: {float((mg[skip:] < -5).mean()):.3f}; "
                            f"peer-wait cycles (slot 40, median CTA) median over iterations per rank: {[int(v) for v in np.median(np.array(pw)[skip:], axis=0)]}; "
                            f"%gridid stamped at kernel entry (slot 62, per rank) iterations 0-5: {gids[:6]} distinct per rank: {[len({g[r] for g in gids}) for r in range(R_)]}",
                            flush=True,
                        )
                continue
            else:
                raise ValueError(name)
            extra = f"({time.time() - t0:.1f}s) " + extra_pre
            if (
                tags
                and name in ("eager", "graph", "two", "mix", "twonode")
                and rank == 0
                and not quiet
            ):
                # tag sampling: peer_grid_idx[] / combine_ready[] / nvl counter after each of the first iterations + number of distinct tag values over all iterations
                head = [(p, rd, c) for p, rd, c in tags[:5]]
                dist_p = sorted({tuple(p) for p, _, _ in tags})
                deltas = sorted(
                    {tags[i + 1][2] - tags[i][2] for i in range(len(tags) - 1)}
                )
                extra += (
                    f" | tag words after iterations 0-4 (peer_grid_idx[], combine_ready[], nvl_counter): {head} | distinct peer_grid_idx vectors over {len(tags)} iterations: {len(dist_p)}"
                    f" | nvl_barrier_counter deltas between samples: {deltas}"
                )
            report(kname, name, skew if name != "same" else skew, N, wrong, extra)

    def build_buckets():
        nonlocal bsets
        if bsets is None:
            bsets = []
            for b in range(3):
                row = []
                for s in range(2):
                    st = make_set(bm_per_rank[rank][b], 10 + 2 * b + s)
                    st["ref"] = eager_ref(st)
                    row.append(st)
                bsets.append(row)
        return bsets

    for kname in kernels:
        # EAGER_ONLY guards: experiments that need DeepGEMM-only machinery are skipped for kernels that cannot do them
        run_kernel(kname)
    dist.barrier(group)
    if rank == 0:
        if os.environ.get("G_JSON"):
            import json

            with open(os.environ["G_JSON"], "w") as f:
                json.dump(results, f)
        print("HAZARD RUN DONE", flush=True)


if __name__ == "__main__":
    import torch.multiprocessing as mp

    nranks = env_int("R_RANKS", 2)
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = os.environ.get("R_PORT", str(29900 + os.getpid() % 400))
    mp.spawn(worker, args=(nranks,), nprocs=nranks, join=True)
