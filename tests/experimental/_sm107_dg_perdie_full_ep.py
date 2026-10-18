# SPDX-License-Identifier: Apache-2.0
# Adapted from the DeepGEMM Rubin SM107 MegaMoE kernel (deepseek-ai/DeepGEMM, MIT License, Copyright (c) 2025 DeepSeek; see licenses/LICENSE.deepgemm.txt; branch rubin-megamoe-a7a9277-1016 @3ec0eb4) for the FlashInfer CuteDSL port.
"""Per-die (locality-domain) FULL-kernel driver, EP1 ... multi-node.  A documented MANUAL script, NOT a pytest file (leading underscore): per-die at R > 4 needs an NVL72 block.

WHAT IT CHECKS (bitwise, on EVERY launch; the output is NaN-filled and the kernel-written regions of the symmetric buffer are POISONED with 0xFF before every launch, so stale results cannot hide; multi-rank: full poison + synchronize + barrier before synchronous launches and graph replays, in the back-to-back / chaos loops (launch skew, no host synchronization) every region except ``src_token_topk_idx``, which the PEERS write at the start of their kernel)
    y_plain = DeepGEMM ``fp8_fp4_mega_moe`` with ``DG_MEGA_MOE_SM107_PERDIE=0`` on the plain weights
    y_pd    = DeepGEMM with ``DG_MEGA_MOE_SM107_PERDIE=1`` on die-local routed weights (``deep_gemm.mega.sm107_locality``: localizer + balanced SM map) = the ORACLE; ``y_pd == y_plain`` is checked too
    port    = the CuteDSL kernel with ``cfg.num_locality_domains = 2``, ``cfg.localized_weights`` (3D weight tensor maps) and the SAME SM map: ``y == y_pd`` (and, with PERDIE_PORT_PLAIN=1, the port on plain weights: ``y == y_plain``)
    plus cumulative expert stats, a clean workspace afterwards, ``PERDIE_GRAPH`` CUDA-graph replays (``PERDIE_GRAPH_VAR`` input variants loaded between replays), ``PERDIE_STRESS`` chaos launches (every rank its own random set per launch +
    GPU-side launch skew, device-side compare), ``PERDIE_TRACE=1`` (traced build: the domain every scheduler warp read == map[%smid], claims exact, and NO steal below the threshold ``num_m_blocks * C < 2 * (SMs / 2)``),
    ``PERDIE_CANARY=1`` (after every uniform ``rand`` set the DeepGEMM launch must have overwritten the poison of every kernel-written region: the poison list is complete and not vacuous).

HOW TO RUN
    single node, 1-4 GPUs (one process per GPU, torch.multiprocessing.spawn):
        R_RANKS=4 R_E=384 R_TMAX=8192 PERDIE_SETS=256,1024,64/1000/1/300 PERDIE_PORT=1 PERDIE_PORT_PLAIN=1 timeout -s KILL 1500 python3 tests/experimental/_sm107_dg_perdie_full_ep.py
    multi-node (NVL72, one task per node, 4 GPUs per node; DeepGEMM's ``init_dist`` convention: WORLD_SIZE = #nodes, RANK = node index, R_RANKS = LOCAL ranks = 4; global EP size = 4 x #nodes), e.g. DEP8 on 2 nodes:
        sbatch --nodes=2 --segment=2 --exclusive --ntasks-per-node=1 launch.sbatch     with launch.sbatch running, inside the container of the image that has deep_gemm + this package, once per node:
            export WORLD_SIZE=$SLURM_NNODES RANK=$SLURM_NODEID MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1) MASTER_PORT=8361
            R_RANKS=4 R_E=384 R_TMAX=8192 R_S=1 PERDIE_SETS=40,694,7059 PERDIE_PORT=1 PERDIE_PORT_PLAIN=1 timeout -s KILL 1500 python3 tests/experimental/_sm107_dg_perdie_full_ep.py
      (use a different MASTER_PORT for every launch; start all nodes at the same time).  ALWAYS under an EXTERNAL ``timeout -s KILL`` in its own process group (``setsid``): the per-die scheduler wait has no trap, so a hang
      (imbalanced SM map, a bug) can only be ended from outside; kill the whole process group afterwards (``kill -9 -- -$PGID``), the spawned workers survive a plain ``timeout``.
    knobs (all env): R_RANKS R_E R_TOPK R_H R_I R_TMAX R_S (0/1 shared experts) R_MASKED R_SEED R_BM R_CLAMP R_TAGMODE (epoch|gridid) R_REGS=a,b,c R_ROUTE (default routing of every set);
        PERDIE_SETS = sets separated by ',', per-rank M separated by '/' (a single number = every rank; one entry per GLOBAL rank otherwise), a set may end in ':<route>' with route = rand | hot0 (all tokens of all ranks to the SAME
        topk experts, all on rank 0) | hotspread (the topk hot experts on topk different ranks) | hotlast;  PERDIE_PORT PERDIE_PORT_PLAIN PERDIE_LAUNCHES PERDIE_B2B PERDIE_GRAPH PERDIE_GRAPH_VAR PERDIE_STRESS PERDIE_SKEW PERDIE_TRACE PERDIE_ABL PERDIE_MAP=dg|host|even|halves|random:<seed>|
        imbalanced:<n>|allzero (NEVER imbalanced / allzero at small M: they hang)  PERDIE_TIMEOUT_SEC (kernel-side dispatch / combine timeout)  PERDIE_POISON=1 (0 = off, e.g. for timing runs)  PERDIE_CANARY=1  PERDIE_TIME PERDIE_TIME_MODES PERDIE_TRACE_DIR.
    scenarios that were validated (DEP8: M = 40, 136, 322, 694, 1860, 2421, 7059; DEP16: 20, 68, 161, 347, 930, 1210, 5578 = every BLOCK_M regime), one process group per row:
        regimes + mixed     PERDIE_SETS=<the seven M>,<per-rank lists with 0 / 1-token ranks next to the largest M>   PERDIE_PORT_PLAIN=1 PERDIE_GRAPH=2
        steal threshold     PERDIE_SETS=1,2,3,4,6,8,1:hot0,2:hot0,3:hot0,...  PERDIE_TRACE=1      (the regime flips between M = 2 and 4; the traced build asserts that nothing is stolen below the threshold)
        hot expert / ring   PERDIE_SETS=8192:hot0 (and 8192:hotspread, 8192/0/.../8192:hot0)   (pool of the hot rank = 6 x T_max x R tokens, ring wrap)
        chaos               PERDIE_SETS=<10 light sets incl. :hot0 and 0 / 1-token ranks>  PERDIE_STRESS=300000 PERDIE_PORT_PLAIN=0 PERDIE_LAUNCHES=1 PERDIE_B2B=0
        graph, new inputs   PERDIE_SETS=<5 sets>  PERDIE_GRAPH=12 PERDIE_GRAPH_VAR=4 R_TAGMODE=epoch      (never ``gridid`` under graph replay with changing inputs: stale combine tag)
        masked / zero       R_MASKED=0.1 ;  PERDIE_SETS=0,1,0 (every rank 0 tokens)

EXPECTED OUTPUT (rank 0 prints the global verdicts, every rank its own detail lines ``[rK] ...``; the nodes' outputs are independent logs)
    ``[rK] set s (M=..) port per-die launch 0: x ms  y == DeepGEMM y_pd bit-exact: True (rows differing 0/M)``      ``CASE EP<R> E=384 set s M=[...] port per-die: PASS``   (and ``port plain: PASS``)
    ``[r0] CANARY set s: every poisoned region overwritten by DeepGEMM per-die: True``      ``CASE STRESS-CHAOS EP<R> ... launches=N skew<=..: PASS``      ``[rK] STEAL-REGIME ...: ranks violating 'no steal below the threshold': 0``
    ``[rK] cumulative stats == expected over N launches: True; workspace clean: True``      and, last, ``PER-DIE FULL EP<R> RESULT: PASS`` with exit code 0 (any mismatch, canary failure or traceback gives a non-zero exit code).
"""

import os
import sys
import time
import types


def env_int(name, default):
    return int(os.environ.get(name, default))


def worker(local_rank, nranks):
    import traceback

    try:
        ok = main(local_rank, nranks)
    except Exception:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(2)
    sys.stdout.flush()
    os._exit(0 if ok else 1)


def hash32(*vals):
    h = 0x9E3779B1
    for v in vals:
        h = ((h ^ (int(v) & 0xFFFFFFFF)) * 0x85EBCA6B) & 0xFFFFFFFF
        h ^= h >> 13
        h = (h * 0xC2B2AE35) & 0xFFFFFFFF
        h ^= h >> 16
    return h


def skew_cycles(rank, it, seed, skew_max):
    if skew_max <= 0:
        return 0
    r = hash32(rank, it, seed, 17) % 16
    if r < 8:
        return 0
    if r < 11:
        return min(20000, skew_max)
    if r < 13:
        return min(200000, skew_max)
    if r == 13:
        return min(1000000, skew_max)
    if r == 14:
        return min(4000000, skew_max)
    return skew_max


def cast_fp8_for_mega_moe(x):
    """``tests/test_mega_moe.py::_cast_fp8_for_mega_moe``: (fp8 values, packed UE8M0 SF, SF in the TMA (MN-major, 4-aligned) layout)."""
    import torch
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


def to_shared_sf_layout(sf, block_m, num_max_sf_tokens):
    """``tests/test_mega_moe.py::_to_shared_mega_moe_sf_layout`` (UTCCP-transposed inside every BLOCK_M block)."""
    import torch

    num_tokens, packed_sf_k = sf.shape
    aligned_block_m = -(-block_m // 128) * 128
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
    return result


def copy_fp8_sf(dst, src, num_tokens):
    """``tests/test_mega_moe.py::_copy_fp8_sf``."""
    if num_tokens == 0:
        return
    if dst.shape == src.shape:
        dst.copy_(src)
        return
    dst[:num_tokens].copy_(src)
    if num_tokens < dst.shape[0]:
        dst[num_tokens:].copy_(src[-1:].expand(dst.shape[0] - num_tokens, -1))


def make_map(kind, DL, dev):
    """The SM -> locality-domain map (uint8[num_sms] on the device) used by BOTH DeepGEMM (its registered map) and the port."""
    import torch

    if kind == "dg":
        return DL.get_balanced_sm_locality_domains()
    n = torch.cuda.get_device_properties(
        torch.cuda.current_device()
    ).multi_processor_count
    if kind == "host":
        from flashinfer.experimental.sm107_deepgemm_megamoe import locality as HL

        return HL.get_balanced_sm_locality_domains()
    if kind == "even":  # TPC-alternating (what DeepGEMM falls back to)
        return (
            (torch.arange(n // 2, device=dev) % 2).to(torch.uint8).repeat_interleave(2)
        )
    if kind == "allzero":
        return torch.zeros(n, dtype=torch.uint8, device=dev)
    if (
        kind == "halves"
    ):  # first half of the TPCs -> domain 0, second half -> domain 1 (balanced; unlike the probe map's per-TPC alternation)
        m = torch.zeros(n, dtype=torch.uint8, device=dev)
        m[n // 2 :] = 1
        return m
    if kind.startswith("random:"):  # random balanced TPC assignment
        g = torch.Generator()
        g.manual_seed(int(kind.split(":")[1]))
        perm = torch.randperm(n // 2, generator=g, device="cpu")
        tpc = torch.zeros(n // 2, dtype=torch.uint8, device="cpu")
        tpc[perm[: n // 4]] = 1
        return tpc.repeat_interleave(2).to(dev)
    if kind.startswith("imbalanced:"):  # first k TPCs -> domain 1, rest domain 0
        k = int(kind.split(":")[1])
        m = torch.zeros(n, dtype=torch.uint8, device=dev)
        m[: 2 * k] = 1
        return m
    raise ValueError(kind)


def main(local_rank, nranks):
    import numpy as np
    import torch
    import torch.distributed as dist

    import deep_gemm
    from deep_gemm.mega import sm107_locality as DL
    from deep_gemm.utils import per_token_cast_to_fp4, per_token_cast_to_fp8
    from deep_gemm.utils.dist import init_dist

    from flashinfer.experimental.sm107_deepgemm_megamoe import jit
    from flashinfer.experimental.sm107_deepgemm_megamoe.heuristics import (
        get_sm107_kernel_config,
    )
    from flashinfer.experimental.sm107_deepgemm_megamoe.kernel import layout as L

    E = env_int("R_E", 32)
    topk = env_int("R_TOPK", 6)
    Hh = env_int("R_H", 5120)
    I = env_int("R_I", 2304)
    Tmax = env_int("R_TMAX", 1920)
    masked = float(os.environ.get("R_MASKED", "0"))
    seed = env_int("R_SEED", 0)
    clamp = float(os.environ.get("R_CLAMP", "10.0"))
    tag_mode = os.environ.get("R_TAGMODE", "gridid")
    regs_env = os.environ.get("R_REGS", "")
    regs_override = tuple(int(v) for v in regs_env.split(",")) if regs_env else None
    bm_force = env_int("R_BM", 0)
    NS = env_int(
        "R_S", 0
    )  # shared experts (S=1: the production recipe PERDIE=routed: routed weights die-local, shared weights plain, shared tasks scheduled per domain)
    run_port = env_int("PERDIE_PORT", 1)
    run_port_plain = env_int("PERDIE_PORT_PLAIN", 0)
    nsync = env_int("PERDIE_LAUNCHES", 2)
    nb2b = env_int("PERDIE_B2B", 4)
    nstress = env_int("PERDIE_STRESS", 0)
    skew_max = env_int("PERDIE_SKEW", 4000000)
    timeout_sec = float(os.environ.get("PERDIE_TIMEOUT_SEC", "60"))
    ntime = env_int("PERDIE_TIME", 0)
    map_kind = os.environ.get("PERDIE_MAP", "dg")
    nnodes = env_int(
        "WORLD_SIZE", 1
    )  # multi-node (mn_launch.sbatch / DeepGEMM init_dist convention): WORLD_SIZE = #nodes, RANK = node index, R_RANKS = LOCAL ranks per node
    world = nranks * nnodes
    sets_m = []
    sets_route = []
    for s in os.environ.get("PERDIE_SETS", "256").split(","):
        s, _, route = s.partition(":")  # "M" | "M0/M1/..." [":" route kind]
        v = [int(x) for x in s.split("/")]
        sets_m.append(v * world if len(v) == 1 else v)
        sets_route.append(route or os.environ.get("R_ROUTE", "rand"))
        assert len(sets_m[-1]) == world, (s, world)
    K = len(sets_m)

    rank, nranks, group = init_dist(local_rank, nranks)
    dev = "cuda"
    El = E // nranks
    assert E % nranks == 0
    pre = f"[r{rank}]"
    import socket

    print(f"{pre} host={socket.gethostname()} local_rank={local_rank} nodes={nnodes}")
    print(
        f"{pre} package: {jit.__file__}  EP{nranks} E={E} (per rank {El}) S={NS} topk={topk} Tmax={Tmax} sets(M per rank)={sets_m} routes={sets_route} port={run_port} plain-port={run_port_plain} "
        f"map={map_kind} tag_mode={tag_mode} regs_override={regs_override} masked={masked} timeout={timeout_sec}s",
        flush=True,
    )
    buffer = deep_gemm.get_symm_buffer_for_mega_moe(
        group,
        E,
        Tmax,
        topk,
        Hh,
        I,
        num_shared_experts=NS,
        mma_type="fp8xfp4",
        activation="swiglu",
    )
    base = buffer.buffer.data_ptr()
    peer_offsets = (
        [int(p) - base for p in buffer.handle.buffer_ptrs] if nranks > 1 else [0]
    )
    assert len(peer_offsets) == nranks and peer_offsets[rank] == 0, peer_offsets
    symb = types.SimpleNamespace(base_ptr=base, peer_offsets=peer_offsets, group=group)

    # ---- POISON: 0xFF into every kernel-written region of the symmetric buffer before EVERY launch (the signals = counters / barriers and the staged inputs are left alone) ----
    poison_mode = env_int("PERDIE_POISON", 1)
    cfg_l = get_sm107_kernel_config(
        nranks,
        E,
        Tmax,
        max(1, max(sets_m[0])),
        topk,
        Hh,
        I,
        NS,
        num_sms=212,
        activation_clamp=clamp,
        block_m=bm_force,
    )
    lay_p = L.buffer_layout(hidden=Hh, intermediate_hidden=I, num_ranks=nranks, num_experts=E, num_max_tokens_per_rank=cfg_l.num_max_tokens_per_rank, num_topk=topk,
                            num_ring_tokens=cfg_l.num_ring_tokens, num_sf_ring_tokens=cfg_l.num_sf_ring_tokens, num_shared_experts=NS)  # fmt: skip
    assert lay_p.total_bytes == buffer.buffer.nbytes, (
        lay_p.total_bytes,
        buffer.buffer.nbytes,
    )
    ws_p = lay_p.workspace
    poison_spans = [(ws_p.src_token_topk_idx_offset, ws_p.token_src_metadata_offset),
                    (ws_p.token_src_metadata_offset, ws_p.token_src_metadata_offset + ws_p.num_max_pool_tokens * L.TOKEN_SRC_METADATA_BYTES)]  # fmt: skip
    for nm in (
        "shared_l2_token",
        "shared_l2_sf",
        "l1_token",
        "l1_sf",
        "l1_topk_weights",
        "l2_token",
        "l2_sf",
        "combine_token",
    ):
        r_p = lay_p.regions[nm]
        if r_p.nbytes:
            poison_spans.append((r_p.offset, r_p.end))
    raw_p = buffer.buffer.view(torch.uint8)

    poison_spans_loop = poison_spans[1:]  # span 0 = src_token_topk_idx, see poison()

    def poison(loop=False):
        """0xFF into the kernel-written regions, stream ordered.  Synchronous launches / graph replays (``loop=False``): full poison, and in multi-rank runs ``synchronize + barrier`` afterwards, so that every rank has poisoned
        before ANY rank launches.  Back-to-back / chaos launches (``loop=True``: no host synchronization, launch skew between the ranks) exclude ``src_token_topk_idx``: a peer that starts its next launch early WRITES into this
        rank's ``src_token_topk_idx`` at the start of its kernel, before this rank's own (stream ordered) fill ran, and the fill would destroy it (measured: ~35 % of the chaos launches at EP4); every other region is written
        locally, or remotely only after the kernel's first NVL barrier."""
        if poison_mode:
            for a, b in poison_spans_loop if loop else poison_spans:
                raw_p[a:b].fill_(0xFF)
            if not loop and nranks > 1:
                torch.cuda.synchronize()
                dist.barrier(group)

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
    l1 = cast_w(l1w)
    l2 = cast_w(l2w)
    del l1w, l2w
    tl1, tl2 = deep_gemm.transform_weights_for_mega_moe(l1, l2, activation="swiglu")
    shared = None
    if NS > 0:
        IS = I * NS
        g0 = torch.Generator(device=dev)
        g0.manual_seed(seed + 555)
        sh1 = cast_fp8_for_mega_moe(
            torch.randn((IS * 2, Hh), dtype=torch.bfloat16, device=dev, generator=g0)
        )[0::2]
        sh2 = cast_fp8_for_mega_moe(
            torch.randn((Hh, IS), dtype=torch.bfloat16, device=dev, generator=g0)
        )[0::2]
        shared = deep_gemm.transform_weights_for_mega_moe(sh1, sh2, activation="swiglu")
    # die-local routed weights + SM map (DeepGEMM's helpers: the oracle)
    dmap = (
        DL.get_balanced_sm_locality_domains()
    )  # registers the map for DeepGEMM's per-die launches
    (loc1, loc2, _, _) = DL.localize_mega_moe_weights(tl1, tl2)
    print(
        f"{pre} localized weights: l1 {tuple(loc1[0].shape)} stride0={loc1[0].stride(0)} l2 {tuple(loc2[0].shape)} stride0={loc2[0].stride(0)}  map dom0={int((dmap == 0).sum())} dom1={int((dmap == 1).sum())}",
        flush=True,
    )
    pmap = (
        dmap if map_kind == "dg" else make_map(map_kind, DL, dev)
    )  # the port's map (default: the very same tensor)
    if map_kind != "dg":
        print(
            f"{pre} PORT map '{map_kind}': dom0={int((pmap == 0).sum())} dom1={int((pmap == 1).sum())} (DeepGEMM keeps the balanced probe map)",
            flush=True,
        )

    def dg(y, perdie):
        poison()
        os.environ["DG_MEGA_MOE_SM107_PERDIE"] = "1" if perdie else "0"
        a, b = (loc1, loc2) if perdie else (tl1, tl2)
        kw = (
            dict(shared_l1_weights=shared[0], shared_l2_weights=shared[1])
            if shared is not None
            else {}
        )
        deep_gemm.fp8_fp4_mega_moe(
            y,
            a,
            b,
            buffer,
            activation="swiglu",
            activation_clamp=clamp,
            fast_math=True,
            **kw,
        )

    # ---- input sets ----
    def hot_list(route):
        """Hot experts of the routing kinds: 'hot0' = the first topk experts (all on rank 0), 'hotlast' = the last topk, 'hotspread' = round robin over the ranks."""
        if route == "hot0":
            return list(range(topk))
        if route == "hotlast":
            return list(range(E - topk, E))
        if route == "hotspread":
            return [(i % nranks) * El + i // nranks for i in range(topk)]
        raise ValueError(
            f"unknown route kind {route!r} (rand | hot0 | hotlast | hotspread)"
        )

    def gen_inputs(M, gseed, route, bm):
        """Tokens + routing of THIS rank for one set / variant ('rand' draws exactly the stream of the earlier harness versions)."""
        g = torch.Generator(device=dev)
        g.manual_seed(gseed)
        st = dict(M=M, bm=bm, route=route)
        if M > 0:
            x = torch.randn((M, Hh), dtype=torch.bfloat16, device=dev, generator=g)
            scores = torch.randn((M, E), dtype=torch.float, device=dev, generator=g)
            tw, ti = torch.topk(scores, topk, dim=-1, largest=True, sorted=False)
            if (
                route != "rand"
            ):  # every token goes to the SAME topk experts (random order per token)
                perm = torch.argsort(
                    torch.rand((M, topk), dtype=torch.float, device=dev, generator=g),
                    dim=1,
                )
                ti = torch.tensor(hot_list(route), dtype=ti.dtype, device=dev)[perm]
            if masked > 0:
                rm = torch.rand(ti.shape, dtype=torch.float, device=dev, generator=g)
                ti.masked_fill_(rm < masked, -1)
                tw.masked_fill_(ti < 0, 0)
            xq, xs = per_token_cast_to_fp8(
                x, use_ue8m0=True, gran_k=32, use_packed_ue8m0=True
            )
            st.update(x=xq, x_sf=xs, ti=ti, tw=tw)
            flat = ti.reshape(-1)
            hist = torch.bincount(flat[flat >= 0].long(), minlength=E).to(torch.int64)
        else:
            hist = torch.zeros(E, dtype=torch.int64, device=dev)
        st["hist"] = hist
        return st

    sets = []
    for s, ms in enumerate(sets_m):
        M = ms[rank]
        bm = get_sm107_kernel_config(
            nranks,
            E,
            Tmax,
            M,
            topk,
            Hh,
            I,
            NS,
            num_sms=212,
            activation_clamp=clamp,
            block_m=bm_force,
        ).block_m
        st = gen_inputs(
            M, seed * 1000003 + 7919 * rank + 104729 * s + 13, sets_route[s], bm
        )
        st["ms"] = ms
        sets.append(st)
    hist_local = torch.stack([st["hist"] for st in sets])
    gathered = [torch.zeros_like(hist_local) for _ in range(nranks)]
    if nranks > 1:
        dist.all_gather(gathered, hist_local, group=group)
    else:
        gathered = [hist_local]
    HIST = torch.stack(gathered).cpu().numpy()  # (R, K, E)

    def load_set(st):
        M = st["M"]
        if M > 0:
            buffer.x[:M].copy_(st["x"])
            buffer.x_sf[:M].copy_(st["x_sf"])
            buffer.topk_idx[:M].copy_(st["ti"])
            buffer.topk_weights[:M].copy_(st["tw"])
            if (
                NS > 0
            ):  # host-side staging of the shared L1 token SF (stock test: _to_shared_mega_moe_sf_layout + _copy_fp8_sf)
                copy_fp8_sf(
                    buffer.shared_l1_acts_sf,
                    to_shared_sf_layout(
                        st["x_sf"], st["bm"], buffer.shared_l1_acts_sf.shape[0]
                    ),
                    M,
                )

    all_ok = True

    def agree(ok):
        flag = torch.tensor([1 if ok else 0], dtype=torch.int32, device=dev)
        if nranks > 1:
            dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=group)
        return bool(flag.item())

    # ---- DeepGEMM references ----
    for s, st in enumerate(sets):
        M = st["M"]
        if rank == 0:
            print(
                f"{pre} >>> DeepGEMM references of set {s} (M per rank {st['ms']}, route {st['route']}, bm {st['bm']})",
                flush=True,
            )
        load_set(st)
        yp = torch.empty((M, Hh), dtype=torch.bfloat16, device=dev)
        yd = torch.empty((M, Hh), dtype=torch.bfloat16, device=dev)
        if nranks > 1:
            dist.barrier(group)
        dg(yp, False)
        torch.cuda.synchronize()
        if nranks > 1:
            dist.barrier(group)
        dg(yd, True)
        torch.cuda.synchronize()
        if nranks > 1:
            dist.barrier(group)
        st["y_plain"], st["y_pd"] = yp, yd
        if (
            poison_mode
            and env_int("PERDIE_CANARY", 1)
            and M > 0
            and st["route"] == "rand"
            and all(m > 0 for m in st["ms"])
        ):  # canary: a CORRECT kernel overwrote every poisoned region (list complete, poison not vacuous)
            dead = [
                i
                for i, (a, b) in enumerate(poison_spans)
                if not bool((raw_p[a:b] != 0xFF).any())
            ]
            print(
                f"{pre} CANARY set {s}: every poisoned region overwritten by DeepGEMM per-die: {not dead}"
                + (f" (never written: span indices {dead})" if dead else ""),
                flush=True,
            )
            all_ok &= not dead
        eq = torch.equal(yp.view(torch.int16), yd.view(torch.int16)) if M else True
        nan = int(torch.isnan(yd.float()).sum().item()) if M else 0
        print(
            f"{pre} set {s} (M={M}): DeepGEMM per-die y == DeepGEMM plain y: {eq}  (absmax {yd.float().abs().max().item() if M else 0:.3g}, nan {nan})",
            flush=True,
        )
        all_ok &= eq and nan == 0

    # DeepGEMM timing (sanity only; the A/B harness is under benchmarks/)
    def dg_mode(y, sched, weights):
        """DeepGEMM ablation modes: sched 0/1 = DG_MEGA_MOE_SM107_PERDIE, weights 'plain'/'loc' (die-local)."""
        os.environ["DG_MEGA_MOE_SM107_PERDIE"] = "1" if sched else "0"
        a, b = (loc1, loc2) if weights == "loc" else (tl1, tl2)
        kw = (
            dict(shared_l1_weights=shared[0], shared_l2_weights=shared[1])
            if shared is not None
            else {}
        )
        deep_gemm.fp8_fp4_mega_moe(
            y,
            a,
            b,
            buffer,
            activation="swiglu",
            activation_clamp=clamp,
            fast_math=True,
            **kw,
        )

    def time_launch(fn, n):
        ts = []
        for _ in range(n + 3):
            torch.cuda.synchronize()
            if nranks > 1:
                dist.barrier(group)
            torch.cuda._sleep(2_000_000)
            e0, e1 = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            e0.record()
            fn()
            e1.record()
            torch.cuda.synchronize()
            ts.append(e0.elapsed_time(e1) * 1e3)
        ts = sorted(ts[3:])
        return ts[len(ts) // 2]

    if ntime:
        for s, st in enumerate(sets):
            load_set(st)
            yy = torch.empty((st["M"], Hh), dtype=torch.bfloat16, device=dev)
            tp = time_launch(lambda: dg(yy, False), ntime)
            td = time_launch(lambda: dg(yy, True), ntime)
            print(
                f"{pre} TIME set {s} (M={st['M']}): DeepGEMM plain {tp:.1f} us, per-die {td:.1f} us (event time incl. launch gap, median of {ntime})",
                flush=True,
            )

    # ---- port kernels ----
    runners = []
    runners_plain = []
    if run_port:
        for s, st in enumerate(sets):
            kw = dict(num_sms=212, activation_clamp=clamp, block_m=bm_force)
            cfg = get_sm107_kernel_config(
                nranks,
                E,
                Tmax,
                st["M"],
                topk,
                Hh,
                I,
                NS,
                per_die=True,
                localized_weights=True,
                **kw,
            )
            assert cfg.num_locality_domains == 2 and cfg.localized_weights
            lay = L.buffer_layout(
                hidden=Hh,
                intermediate_hidden=I,
                num_ranks=nranks,
                num_experts=E,
                num_max_tokens_per_rank=cfg.num_max_tokens_per_rank,
                num_topk=topk,
                num_ring_tokens=cfg.num_ring_tokens,
                num_sf_ring_tokens=cfg.num_sf_ring_tokens,
                num_shared_experts=NS,
            )
            assert lay.total_bytes == buffer.buffer.nbytes, (
                lay.total_bytes,
                buffer.buffer.nbytes,
            )
            runner = jit.get_or_compile_kernel(
                cfg,
                mode_mask=7,
                roles="auto",
                use_disk_cache=False,
                tag_mode=tag_mode,
                regs_override=regs_override,
                timeout_sec=timeout_sec,
            )
            runners.append(runner)
            pool_blocks = int(
                sum(
                    -(-int(c) // cfg.block_m)
                    for c in sum(
                        HIST[r, s, rank * El : (rank + 1) * El] for r in range(nranks)
                    )
                )
            )
            print(
                f"{pre} set {s} M={st['M']}: PER-DIE cfg bm={cfg.block_m} stages={cfg.num_stages} regs={cfg.regs_dispatch}/{cfg.regs_non_epilogue}/{cfg.regs_epilogue} ring_blocks={cfg.num_ring_blocks} pool_blocks={pool_blocks}"
                f"{' (RING WRAP)' if pool_blocks > cfg.num_ring_blocks else ''} "
                f"domains={cfg.num_locality_domains} l1/l2 clusters per domain={cfg.num_l1_clusters_per_domain}/{cfg.num_l2_clusters_per_domain} -> {runner.name} (compile {runner.compile_seconds:.1f}s)",
                flush=True,
            )
            if run_port_plain:
                cfg0 = get_sm107_kernel_config(
                    nranks, E, Tmax, st["M"], topk, Hh, I, NS, **kw
                )
                runners_plain.append(
                    jit.get_or_compile_kernel(
                        cfg0,
                        mode_mask=7,
                        roles="auto",
                        use_disk_cache=False,
                        tag_mode=tag_mode,
                        regs_override=regs_override,
                        timeout_sec=timeout_sec,
                    )
                )

    sig = buffer.buffer.view(torch.uint8)[: L.SIGNALS_SIZEOF]
    fields = (("expert_send_count", L.SIGNALS_EXPERT_SEND_COUNT, 8 * 2048), ("expert_recv_count", L.SIGNALS_EXPERT_RECV_COUNT, 8 * 2048),
              ("expert_recv_count_sum", L.SIGNALS_EXPERT_RECV_COUNT_SUM, 8 * 512), ("l1_full_count", L.SIGNALS_L1_FULL_COUNT, 4 * (1 << 20)),
              ("l1_empty_count", L.SIGNALS_L1_EMPTY_COUNT, 4 * (1 << 20)), ("l2_full_mask", L.SIGNALS_L2_FULL_MASK, 8 * (1 << 20)),
              ("l2_empty_count", L.SIGNALS_L2_EMPTY_COUNT, 4 * (1 << 20)), ("task_counts", L.SIGNALS_L1_TASK_COUNT, 32),
              ("shared_l2_full_count", L.SIGNALS_SHARED_L2_FULL_COUNT, 4 * (1 << 15)))  # fmt: skip

    stats = torch.zeros(El, dtype=torch.int32, device=dev)
    exp_stats = np.zeros(El, dtype=np.int64)
    total_launches = 0

    sh_args = (
        ()
        if shared is None
        else (shared[0][0], shared[0][1], shared[1][0], shared[1][1])
    )

    def call_pd(s, y, M, loop=False):
        poison(loop)
        runners[s](
            y,
            stats,
            M,
            symb,
            loc1[0],
            loc1[1],
            loc2[0],
            loc2[1],
            *sh_args,
            sm_locality_domains=pmap,
        )

    def call_plain(s, y, M, loop=False):
        poison(loop)
        runners_plain[s](y, stats, M, symb, tl1[0], tl1[1], tl2[0], tl2[1], *sh_args)

    if run_port:
        maxM = max(max(st["ms"]) for st in sets)
        for s, st in enumerate(sets):
            M = st["M"]
            load_set(st)
            modes = [("per-die", call_pd, "y_pd")] + (
                [("plain", call_plain, "y_plain")] if run_port_plain else []
            )
            for label, call, refname in modes:
                ref = st[refname]
                ok = True
                if rank == 0:
                    print(
                        f"{pre} >>> port {label} set {s} (M per rank {st['ms']}, route {st['route']})",
                        flush=True,
                    )
                for it in range(nsync):
                    y = torch.full(
                        (M, Hh), float("nan"), dtype=torch.bfloat16, device=dev
                    )
                    if nranks > 1:
                        dist.barrier(group)
                    t0 = time.time()
                    call(s, y, M)
                    torch.cuda.synchronize()
                    dt = time.time() - t0
                    eq = (
                        torch.equal(y.view(torch.int16), ref.view(torch.int16))
                        if M
                        else True
                    )
                    nrow = (
                        int(
                            (y.view(torch.int16) != ref.view(torch.int16))
                            .any(dim=1)
                            .sum()
                            .item()
                        )
                        if M
                        else 0
                    )
                    if it == 0 or not eq:
                        print(
                            f"{pre} set {s} (M={M}) port {label} launch {it}: {dt * 1e3:.1f} ms  y == DeepGEMM {refname} bit-exact: {eq} (rows differing {nrow}/{M})",
                            flush=True,
                        )
                    ok &= eq
                    exp_stats += HIST[:, s, rank * El : (rank + 1) * El].sum(axis=0)
                    total_launches += 1
                if nb2b:
                    ys = [
                        torch.full(
                            (M, Hh), float("nan"), dtype=torch.bfloat16, device=dev
                        )
                        for _ in range(nb2b)
                    ]
                    if nranks > 1:
                        dist.barrier(group)
                    torch.cuda.synchronize()
                    for it in range(nb2b):
                        call(s, ys[it], M, True)
                    torch.cuda.synchronize()
                    bad = [
                        it
                        for it in range(nb2b)
                        if M
                        and not torch.equal(
                            ys[it].view(torch.int16), ref.view(torch.int16)
                        )
                    ]
                    print(
                        f"{pre} set {s} (M={M}) port {label} back-to-back x{nb2b}: {'all bit-exact' if not bad else 'MISMATCH in launches ' + str(bad[:8])}",
                        flush=True,
                    )
                    ok &= not bad
                    for _ in range(nb2b):
                        exp_stats += HIST[:, s, rank * El : (rank + 1) * El].sum(axis=0)
                    total_launches += nb2b
                ngraph = env_int("PERDIE_GRAPH", 0)
                if (
                    ngraph and label == "per-die"
                ):  # CUDA-graph replays of the per-die runner; PERDIE_GRAPH_VAR = number of input variants (same M per rank, different tokens / routing) loaded between replays
                    nvar = max(1, env_int("PERDIE_GRAPH_VAR", 1))
                    variants = [dict(st, y_ref=ref, histv=HIST[:, s, :])]
                    for v in range(1, nvar):
                        sv = gen_inputs(
                            M,
                            seed * 1000003
                            + 7919 * rank
                            + 104729 * s
                            + 13
                            + 15485863 * v,
                            st["route"],
                            st["bm"],
                        )
                        h_local = sv["hist"].unsqueeze(0)
                        gat = [torch.zeros_like(h_local) for _ in range(nranks)]
                        if nranks > 1:
                            dist.all_gather(gat, h_local, group=group)
                        else:
                            gat = [h_local]
                        sv["histv"] = torch.stack(gat).cpu().numpy()[:, 0, :]
                        load_set(sv)
                        ypv = torch.empty((M, Hh), dtype=torch.bfloat16, device=dev)
                        if nranks > 1:
                            dist.barrier(group)
                        dg(
                            ypv, True
                        )  # DeepGEMM per-die reference of the variant (collective)
                        torch.cuda.synchronize()
                        if nranks > 1:
                            dist.barrier(group)
                        sv["y_ref"] = ypv
                        variants.append(sv)
                    load_set(variants[0])
                    yg = torch.full(
                        (M, Hh), float("nan"), dtype=torch.bfloat16, device=dev
                    )
                    torch.cuda.synchronize()
                    if nranks > 1:
                        dist.barrier(group)
                    g = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g):
                        call(s, yg, M, True)
                    torch.cuda.synchronize()
                    badg = 0
                    for it in range(ngraph):
                        v = variants[it % nvar]
                        load_set(v)
                        yg.fill_(float("nan"))
                        poison()
                        if nranks > 1:
                            dist.barrier(group)
                        g.replay()
                        torch.cuda.synchronize()
                        if M and not torch.equal(
                            yg.view(torch.int16), v["y_ref"].view(torch.int16)
                        ):
                            badg += 1
                        exp_stats += v["histv"][:, rank * El : (rank + 1) * El].sum(
                            axis=0
                        )
                        total_launches += 1
                    load_set(st)
                    print(
                        f"{pre} set {s} (M={M}) port {label} CUDA-graph replays x{ngraph} over {nvar} input variants (tag_mode={tag_mode}): {'all bit-exact' if not badg else str(badg) + ' MISMATCHING replays'}",
                        flush=True,
                    )
                    ok &= badg == 0
                good = agree(ok)
                if rank == 0:
                    print(
                        f"CASE EP{nranks} E={E} set {s} M={st['ms']} port {label}: {'PASS' if good else 'FAIL'}",
                        flush=True,
                    )
                all_ok &= good
                if nranks > 1:
                    dist.barrier(group)
        # ---- ablations (DeepGEMM's two ablation modes): per-die scheduling on PLAIN weights (D=2, 2D maps) and plain scheduler on die-local weights (D=1, 3D maps): y == DeepGEMM plain y ----
        if env_int("PERDIE_ABL", 0):
            for s, st in enumerate(sets):
                M = st["M"]
                load_set(st)
                kwc = dict(num_sms=212, activation_clamp=clamp, block_m=bm_force)
                cfg_a = get_sm107_kernel_config(
                    nranks,
                    E,
                    Tmax,
                    M,
                    topk,
                    Hh,
                    I,
                    NS,
                    per_die=True,
                    localized_weights=False,
                    **kwc,
                )
                cfg_b = get_sm107_kernel_config(
                    nranks,
                    E,
                    Tmax,
                    M,
                    topk,
                    Hh,
                    I,
                    NS,
                    per_die=False,
                    localized_weights=True,
                    **kwc,
                )
                assert (
                    cfg_a.num_locality_domains == 2
                    and not cfg_a.localized_weights
                    and cfg_b.num_locality_domains == 1
                    and cfg_b.localized_weights
                )
                ra = jit.get_or_compile_kernel(
                    cfg_a,
                    mode_mask=7,
                    roles="auto",
                    use_disk_cache=False,
                    tag_mode=tag_mode,
                    regs_override=regs_override,
                    timeout_sec=timeout_sec,
                )
                rb = jit.get_or_compile_kernel(
                    cfg_b,
                    mode_mask=7,
                    roles="auto",
                    use_disk_cache=False,
                    tag_mode=tag_mode,
                    regs_override=regs_override,
                    timeout_sec=timeout_sec,
                )
                ya = torch.full((M, Hh), float("nan"), dtype=torch.bfloat16, device=dev)
                yb = torch.full((M, Hh), float("nan"), dtype=torch.bfloat16, device=dev)
                if nranks > 1:
                    dist.barrier(group)
                poison()
                ra(
                    ya,
                    stats,
                    M,
                    symb,
                    tl1[0],
                    tl1[1],
                    tl2[0],
                    tl2[1],
                    *sh_args,
                    sm_locality_domains=pmap,
                )
                torch.cuda.synchronize()
                if nranks > 1:
                    dist.barrier(group)
                poison()
                rb(yb, stats, M, symb, loc1[0], loc1[1], loc2[0], loc2[1], *sh_args)
                torch.cuda.synchronize()
                exp_stats += 2 * HIST[:, s, rank * El : (rank + 1) * El].sum(axis=0)
                total_launches += 2
                eqa = (
                    torch.equal(ya.view(torch.int16), st["y_plain"].view(torch.int16))
                    if M
                    else True
                )
                eqb = (
                    torch.equal(yb.view(torch.int16), st["y_plain"].view(torch.int16))
                    if M
                    else True
                )
                print(
                    f"{pre} ABLATION set {s} (M={M}): per-die scheduling on PLAIN weights == plain y: {eqa}; plain scheduler on DIE-LOCAL weights == plain y: {eqb}",
                    flush=True,
                )
                all_ok &= eqa and eqb
        # ---- per-die trace build (kTrace + per-die slots): end-to-end check of the SM -> domain lookup and the own/stolen claim statistics ----
        if env_int("PERDIE_TRACE", 0):
            from flashinfer.experimental.sm107_deepgemm_megamoe.kernel.trace import (
                trace_buffer_u64s,
            )

            for s, st in enumerate(sets):
                M = st["M"]
                cfg_t = get_sm107_kernel_config(
                    nranks,
                    E,
                    Tmax,
                    M,
                    topk,
                    Hh,
                    I,
                    NS,
                    num_sms=212,
                    activation_clamp=clamp,
                    block_m=bm_force,
                    per_die=True,
                    localized_weights=True,
                    trace=True,
                )
                runner_t = jit.get_or_compile_kernel(
                    cfg_t,
                    mode_mask=7,
                    roles="auto",
                    use_disk_cache=False,
                    tag_mode=tag_mode,
                    regs_override=regs_override,
                    timeout_sec=timeout_sec,
                )
                tb = torch.zeros(
                    trace_buffer_u64s(cfg_t.num_sms), dtype=torch.int64, device=dev
                )
                load_set(st)
                y = torch.full((M, Hh), float("nan"), dtype=torch.bfloat16, device=dev)
                if nranks > 1:
                    dist.barrier(group)
                runners_t = runner_t
                poison()
                runners_t(
                    y,
                    stats,
                    M,
                    symb,
                    loc1[0],
                    loc1[1],
                    loc2[0],
                    loc2[1],
                    *sh_args,
                    trace=tb,
                    sm_locality_domains=pmap,
                )
                torch.cuda.synchronize()
                exp_stats += HIST[:, s, rank * El : (rank + 1) * El].sum(axis=0)
                total_launches += 1
                eq = (
                    torch.equal(y.view(torch.int16), st["y_pd"].view(torch.int16))
                    if M
                    else True
                )
                t = (
                    tb[: cfg_t.num_sms * 64]
                    .view(cfg_t.num_sms, 64)
                    .cpu()
                    .numpy()
                    .astype(np.int64)
                )
                pm = pmap.cpu().numpy().astype(np.int64)
                leaders = np.arange(0, cfg_t.num_sms, 2)
                smid = t[leaders, 46] - 1
                dom = t[leaders, 47] - 1
                lookup_ok = bool(
                    (smid >= 0).all()
                    and (smid < cfg_t.num_sms).all()
                    and (dom == pm[np.clip(smid, 0, cfg_t.num_sms - 1)]).all()
                )
                distinct = len(set(smid.tolist()))
                own = {d: int(t[leaders][dom == d, 48].sum()) for d in (0, 1)}
                stolen = {d: int(t[leaders][dom == d, 49].sum()) for d in (0, 1)}
                ncl = {d: int((dom == d).sum()) for d in (0, 1)}
                # expected successful claims: (L1 + L2 clusters) x pool blocks of this rank
                nb = int(
                    sum(
                        -(-int(c) // cfg_t.block_m)
                        for c in sum(
                            HIST[r, s, rank * El : (rank + 1) * El]
                            for r in range(nranks)
                        )
                    )
                )
                exp_claims = nb * (cfg_t.num_l1_clusters + cfg_t.num_l2_clusters)
                if (
                    NS > 0
                ):  # + shared L1 (2I*S/256 clusters) and shared L2 (H/256 clusters) per token block of this rank
                    exp_claims += -(-M // cfg_t.block_m) * (
                        cfg_t.l1_shape_n * NS // 256 + cfg_t.l2_shape_n // 256
                    )
                got_claims = sum(own.values()) + sum(stolen.values())
                thr = (
                    cfg_t.num_min_steal_waves * (cfg_t.num_sms // 2)
                )  # SCH:220: stealing only with num_m_blocks * C >= kNumMinStealWaves * (kNumSMs / 2)
                ntb = -(-M // cfg_t.block_m)
                steal_possible = (
                    nb * cfg_t.num_l1_clusters >= thr
                    or nb * cfg_t.num_l2_clusters >= thr
                )
                if NS > 0:
                    steal_possible = (
                        steal_possible
                        or ntb * (cfg_t.l1_shape_n * NS // 256) >= thr
                        or ntb * (cfg_t.l2_shape_n // 256) >= thr
                    )
                steals = sum(stolen.values())
                regime_ok = (
                    steal_possible or steals == 0
                )  # below the threshold NOTHING may be stolen
                rg = torch.tensor(
                    [
                        0 if steal_possible else 1,
                        1 if steal_possible else 0,
                        steals,
                        0 if regime_ok else 1,
                    ],
                    dtype=torch.int64,
                    device=dev,
                )
                if nranks > 1:
                    dist.all_reduce(rg, op=dist.ReduceOp.SUM, group=group)
                if rank == 0:
                    print(
                        f"{pre} STEAL-REGIME set {s} (M per rank {st['ms']}, route {st['route']}): {int(rg[0])} ranks below the steal threshold (OFF), {int(rg[1])} ranks above it (ON), "
                        f"total stolen claims {int(rg[2])}, ranks violating 'no steal below the threshold': {int(rg[3])}",
                        flush=True,
                    )
                print(
                    f"{pre} TRACE set {s} (M={M}, bm={cfg_t.block_m}): y bit-exact {eq}; SM->domain lookup == map[%smid] for all {len(leaders)} leader CTAs: {lookup_ok} ({distinct} distinct %smid, "
                    f"{ncl[0]} CTA-pairs on domain 0 / {ncl[1]} on domain 1); claims own {own} stolen {stolen} total {got_claims} (expected {exp_claims} = {nb} pool blocks x "
                    f"{cfg_t.num_l1_clusters}+{cfg_t.num_l2_clusters} clusters [+ shared]): {got_claims == exp_claims}",
                    flush=True,
                )
                print(
                    f"{pre} steal regime of this rank: {'ON' if steal_possible else 'OFF'} (pool blocks {nb}: L1 {nb * cfg_t.num_l1_clusters} / L2 {nb * cfg_t.num_l2_clusters} cluster tasks vs threshold {thr}; stolen {steals})",
                    flush=True,
                )
                all_ok &= eq and lookup_ok and got_claims == exp_claims and regime_ok
        # stats / clean
        stats_ok = bool(np.array_equal(stats.cpu().numpy().astype(np.int64), exp_stats))
        dirty = []
        for name, off, nbytes in fields:
            nz = int((sig[off : off + nbytes] != 0).sum().item())
            if nz:
                dirty.append((name, nz))
        print(
            f"{pre} cumulative stats == expected over {total_launches} launches: {stats_ok}; workspace clean: {not dirty} {dirty if dirty else ''}",
            flush=True,
        )
        all_ok &= stats_ok and not dirty

        # ---- chaos stress: every rank its own random set per launch, GPU-side skew, device-side compare ----
        if nstress:
            N = nstress
            choice = np.array(
                [
                    [hash32(r, it, seed, 5) % K if K > 1 else 0 for r in range(nranks)]
                    for it in range(N)
                ],
                dtype=np.int64,
            )
            per_launch_stats = np.zeros(El, dtype=np.int64)
            for r in range(nranks):
                per_launch_stats += HIST[
                    r, choice[:, r], rank * El : (rank + 1) * El
                ].sum(axis=0)
            ny = 4
            Ys = [
                torch.full(
                    (max(maxM, 1), Hh), float("nan"), dtype=torch.bfloat16, device=dev
                )
                for _ in range(ny)
            ]
            bad_flags = torch.zeros(N, dtype=torch.int32, device=dev)
            if nranks > 1:
                dist.barrier(group)
            cur = None
            t0 = time.time()
            for it in range(N):
                s = int(choice[it, rank])
                st = sets[s]
                M = st["M"]
                if s != cur:
                    load_set(st)
                    cur = s
                y = Ys[it % ny][:M]
                if M:
                    y.fill_(float("nan"))
                cyc = skew_cycles(rank, it, seed, skew_max)
                if cyc:
                    torch.cuda._sleep(cyc)
                call_pd(s, y, M, True)
                if M:
                    bad_flags[it] = (
                        y.view(torch.int16) != st["y_pd"].view(torch.int16)
                    ).any()
                if (it + 1) % max(500, N // 20) == 0:
                    torch.cuda.synchronize()
                    print(
                        f"{pre}   [chaos] {it + 1}/{N} launches, {time.time() - t0:.1f}s, mismatching so far: {int(bad_flags[: it + 1].sum().item())}",
                        flush=True,
                    )
            torch.cuda.synchronize()
            dt = time.time() - t0
            bad_its = torch.nonzero(bad_flags).flatten().tolist()
            exp_stats += per_launch_stats
            total_launches += N
            print(
                f"{pre}   [chaos] {N} launches in {dt:.1f}s ({dt / N * 1e3:.3f} ms/launch incl. skew); mismatching launches: {len(bad_its)} {bad_its[:12]}",
                flush=True,
            )
            ok = not bad_its
            if nranks > 1:
                dist.barrier(group)
            stats_ok = bool(
                np.array_equal(stats.cpu().numpy().astype(np.int64), exp_stats)
            )
            dirty = []
            for name, off, nbytes in fields:
                nz = int((sig[off : off + nbytes] != 0).sum().item())
                if nz:
                    dirty.append((name, nz))
            print(
                f"{pre}   [chaos] cumulative stats == expected over {total_launches} launches: {stats_ok}; workspace clean: {not dirty} {dirty if dirty else ''}",
                flush=True,
            )
            ok &= stats_ok and not dirty
            good = agree(ok)
            if rank == 0:
                print(
                    f"CASE STRESS-CHAOS EP{nranks} E={E} sets {[st['ms'] for st in sets]} launches={N} skew<={skew_max}: {'PASS' if good else 'FAIL'}",
                    flush=True,
                )
            all_ok &= good

        # ---- DeepGEMM per-die timing next to the port (sanity) ----
        if ntime:
            for s, st in enumerate(sets):
                load_set(st)
                yy = torch.empty((st["M"], Hh), dtype=torch.bfloat16, device=dev)
                t_dgp = time_launch(lambda: dg(yy, True), ntime)
                t_pd = time_launch(lambda: call_pd(s, yy, st["M"]), ntime)
                msg = f"{pre} TIME set {s} (M={st['M']}): DeepGEMM per-die {t_dgp:.1f} us, port per-die {t_pd:.1f} us  ratio {t_pd / t_dgp:.3f}"
                if run_port_plain:
                    t_pl = time_launch(lambda: call_plain(s, yy, st["M"]), ntime)
                    t_dgl = time_launch(lambda: dg(yy, False), ntime)
                    msg += f" | DeepGEMM plain {t_dgl:.1f}, port plain {t_pl:.1f} ratio {t_pl / t_dgl:.3f}"
                print(msg, flush=True)

    # ---- PERDIE_TIME_MODES: the four (scheduler, weights) combinations for DeepGEMM and the port in ONE process (event time, median of N, max over ranks) ----
    ntm = env_int("PERDIE_TIME_MODES", 0)
    if ntm and run_port:
        import torch.distributed as _d

        def rank_max(v):
            t = torch.tensor([v], dtype=torch.float64, device=dev)
            if nranks > 1:
                _d.all_reduce(t, op=_d.ReduceOp.MAX, group=group)
            return float(t.item())

        for s, st in enumerate(sets):
            M = st["M"]
            load_set(st)
            kwc = dict(num_sms=212, activation_clamp=clamp, block_m=bm_force)
            combos = [
                ("plain", False, False),
                ("sched-only", True, False),
                ("weights-only", False, True),
                ("per-die", True, True),
            ]  # (label, per-die scheduler, die-local weights)
            yy = torch.empty((M, Hh), dtype=torch.bfloat16, device=dev)
            row = {}
            for label, sched, loc_w in combos:
                cfg_m = get_sm107_kernel_config(
                    nranks,
                    E,
                    Tmax,
                    M,
                    topk,
                    Hh,
                    I,
                    NS,
                    per_die=sched,
                    localized_weights=loc_w,
                    **kwc,
                )
                rm = jit.get_or_compile_kernel(
                    cfg_m,
                    mode_mask=7,
                    roles="auto",
                    use_disk_cache=False,
                    tag_mode=tag_mode,
                    regs_override=regs_override,
                    timeout_sec=timeout_sec,
                )
                wl = (loc1, loc2) if loc_w else (tl1, tl2)
                kw_run = dict(sm_locality_domains=pmap) if sched else {}

                def run_port_mode(rm=rm, wl=wl, kw_run=kw_run):
                    rm(
                        yy,
                        stats,
                        M,
                        symb,
                        wl[0][0],
                        wl[0][1],
                        wl[1][0],
                        wl[1][1],
                        *sh_args,
                        **kw_run,
                    )

                def run_dg_mode(sched=sched, loc_w=loc_w):
                    dg_mode(yy, sched, "loc" if loc_w else "plain")

                # interleave DG and port launches per iteration (same process, same buffers) to cancel drift
                tp, td = [], []
                for it in range(ntm + 3):
                    for fn, acc in ((run_dg_mode, td), (run_port_mode, tp)):
                        torch.cuda.synchronize()
                        if nranks > 1:
                            dist.barrier(group)
                        torch.cuda._sleep(2_000_000)
                        e0, e1 = (
                            torch.cuda.Event(enable_timing=True),
                            torch.cuda.Event(enable_timing=True),
                        )
                        e0.record()
                        fn()
                        e1.record()
                        torch.cuda.synchronize()
                        if it >= 3:
                            acc.append(e0.elapsed_time(e1) * 1e3)
                        exp_stats += (
                            HIST[:, s, rank * El : (rank + 1) * El].sum(axis=0)
                            if fn is run_port_mode
                            else 0
                        )
                        total_launches += 1 if fn is run_port_mode else 0
                tdm, tpm = sorted(td)[len(td) // 2], sorted(tp)[len(tp) // 2]
                row[label] = (rank_max(tdm), rank_max(tpm))
            base_dg, base_pt = row["plain"]
            msg = (
                f"{pre} TIME-MODES set {s} (M={M}, max over ranks, us): "
                + " | ".join(
                    f"{k}: DG {v[0]:.1f} port {v[1]:.1f} ratio {v[1] / v[0]:.3f} (vs own plain: DG {v[0] / base_dg:.3f} port {v[1] / base_pt:.3f})"
                    for k, v in row.items()
                )
            )
            if rank == 0:
                print(msg, flush=True)

    # ---- PERDIE_TRACE_DIR: kTrace dumps (trace_rank<r>_iter<i>.{bin,json}) of DeepGEMM per-die and of the port per-die, same inputs ----
    tdir = os.environ.get("PERDIE_TRACE_DIR", "")
    if tdir and run_port:
        import json as _json

        from flashinfer.experimental.sm107_deepgemm_megamoe.kernel.trace import (
            trace_buffer_u64s,
        )

        n_iter = env_int("PERDIE_TRACE_ITERS", 6)
        for s, st in enumerate(sets):
            M = st["M"]
            load_set(st)
            nsm = 212
            n_u64 = trace_buffer_u64s(nsm)
            tb = torch.zeros(n_u64, dtype=torch.int64, device=dev)
            yy = torch.empty((M, Hh), dtype=torch.bfloat16, device=dev)
            cfg_t = get_sm107_kernel_config(
                nranks,
                E,
                Tmax,
                M,
                topk,
                Hh,
                I,
                NS,
                num_sms=nsm,
                activation_clamp=clamp,
                block_m=bm_force,
                per_die=True,
                localized_weights=True,
                trace=True,
            )
            rt = jit.get_or_compile_kernel(
                cfg_t,
                mode_mask=7,
                roles="auto",
                use_disk_cache=False,
                tag_mode=tag_mode,
                regs_override=regs_override,
                timeout_sec=timeout_sec,
            )
            for label in ("dg", "port"):
                od = os.path.join(tdir, f"{label}_perdie_M{M}")
                os.makedirs(od, exist_ok=True)
                if label == "dg":
                    os.environ["DG_MEGA_MOE_SM107_TRACE_PTR"] = str(tb.data_ptr())
                for it in range(n_iter + 1):
                    if nranks > 1:
                        dist.barrier(group)
                    tb.zero_()
                    torch.cuda.synchronize()
                    e0, e1 = (
                        torch.cuda.Event(enable_timing=True),
                        torch.cuda.Event(enable_timing=True),
                    )
                    e0.record()
                    if label == "dg":
                        dg(yy, True)
                    else:
                        poison()
                        rt(
                            yy,
                            stats,
                            M,
                            symb,
                            loc1[0],
                            loc1[1],
                            loc2[0],
                            loc2[1],
                            *sh_args,
                            trace=tb,
                            sm_locality_domains=pmap,
                        )
                        exp_stats += HIST[:, s, rank * El : (rank + 1) * El].sum(axis=0)
                        total_launches += 1
                    e1.record()
                    torch.cuda.synchronize()
                    if it == 0:
                        continue  # first launch of each kind may include JIT / first-touch effects
                    tb.cpu().numpy().tofile(
                        os.path.join(od, f"trace_rank{rank}_iter{it - 1}.bin")
                    )
                    with open(
                        os.path.join(od, f"trace_rank{rank}_iter{it - 1}.json"), "w"
                    ) as f:
                        _json.dump(
                            {
                                "rank": rank,
                                "iter": it - 1,
                                "num_sms": nsm,
                                "event_ms": e0.elapsed_time(e1),
                                "slots_per_cta": 64,
                                "event_slots": 4 * 64,
                                "n_u64": n_u64,
                                "world": nranks,
                                "M": M,
                                "block_m": cfg_t.block_m,
                                "kernel": label,
                            },
                            f,
                        )
                if label == "dg":
                    os.environ.pop("DG_MEGA_MOE_SM107_TRACE_PTR", None)
            if rank == 0:
                print(
                    f"{pre} TRACE-DUMP set {s} (M={M}): {n_iter} iterations each of DeepGEMM per-die and port per-die written under {tdir}/{{dg,port}}_perdie_M{M}/ (all ranks)",
                    flush=True,
                )

    # DeepGEMM per-die once more on the buffer the port left behind (clean-state compatibility)
    st = sets[0]
    load_set(st)
    y2 = torch.empty((st["M"], Hh), dtype=torch.bfloat16, device=dev)
    if nranks > 1:
        dist.barrier(group)
    dg(y2, True)
    torch.cuda.synchronize()
    eq2 = (
        torch.equal(y2.view(torch.int16), st["y_pd"].view(torch.int16))
        if st["M"]
        else True
    )
    print(
        f"{pre} DeepGEMM per-die re-run on the buffer left by the port: y == first DeepGEMM per-die y: {eq2}",
        flush=True,
    )
    all_ok &= eq2
    good = agree(all_ok)
    if rank == 0:
        print(
            f"PER-DIE FULL EP{nranks} RESULT: {'PASS' if good else 'FAIL'}", flush=True
        )
    if nranks > 1:
        dist.barrier(group)
    return good


if __name__ == "__main__":
    import torch.multiprocessing as mp

    os.environ.setdefault(
        "CUTE_DSL_ARCH", "sm_107a"
    )  # script defaults (the workers inherit them), NOT applied at import time
    os.environ.setdefault("CUDA_MODULE_LOADING", "EAGER")
    os.environ.setdefault("DG_MEGA_MOE_SM107", "1")
    nranks = env_int(
        "R_RANKS", 1
    )  # LOCAL ranks (processes) of this node; multi-node: WORLD_SIZE = #nodes, RANK = node index, MASTER_ADDR / MASTER_PORT from the launcher (R_PORT overrides the port)
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    if "R_PORT" in os.environ or "WORLD_SIZE" not in os.environ:
        os.environ["MASTER_PORT"] = os.environ.get(
            "R_PORT", str(29900 + os.getpid() % 400)
        )
    mp.spawn(worker, args=(nranks,), nprocs=nranks, join=True)
