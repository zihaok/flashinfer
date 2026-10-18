# SPDX-License-Identifier: Apache-2.0
# Port of DeepGEMM (MIT License, Copyright (c) 2025 DeepSeek; see licenses/LICENSE.deepgemm.txt) sm107_fp8_fp4_mega_moe.cuh / test harness.
# GEMM-core test harness: a mini persistent kernel (grid 2 = one CTA pair) with DeepGEMM's prologue (barrier init counts, TMEM alloc),
# the real ctx classes (SmemMap/Barriers/TmemMap/Workspace), the REAL GEMM roles (tma_a_role, tma_b_role, mma_role, scheduler consumer)
# and test stand-ins for the scheduler warp (publishes a host-given task table) and the epilogue (raw TMEM dump to global memory).
import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda
from cutlass import Int32, Float32

from flashinfer.experimental.sm107_deepgemm_megamoe.kernel import ctx as fctx
from flashinfer.experimental.sm107_deepgemm_megamoe.kernel import comm as fcomm
from flashinfer.experimental.sm107_deepgemm_megamoe.kernel import (
    epilogue as epi_mod,
    mma_warp,
    ptx,
    scheduler as sched,
    tma_warps,
)
from flashinfer.experimental.sm107_deepgemm_megamoe.kernel.dslutil import (
    DslStatic,
    init_fields,
    static_dataclass,
)
from flashinfer.experimental.sm107_deepgemm_megamoe.kernel.trace import Trace


@static_dataclass
class GCtx(DslStatic):
    """read-only role context (the `static_dataclass` + `DslStatic`: see kernel/dslutil.py): no method call inside a dynamic region
    can make the DSL try to carry it."""

    def __init__(self, **kw):
        init_fields(self, **kw)


class Barriers(fctx.Barriers):
    """The Barriers with a dynamic-index-safe pointer: add the index in units of Int64 elements AFTER the recast so that the
    8-byte alignment (required by the TMA `tma_bar` operand) survives a runtime stage index."""

    def _mb(self, name, i):
        base = cute.recast_ptr(self._s.base + self._s.off[name], dtype=cutlass.Int64)
        return ptx.Mbar(base + i)


class SymHost:
    """stand-in for comm.SymBufferHost (only `.base` is used by make_tma_params)"""

    def __init__(self, base):
        self.base = base


class Harness:
    def __init__(
        self, cfg, ntasks_max=8, real_sched=False, real_epilogue=False, grid=2
    ):
        self.cfg = cfg
        self.ntasks_max = ntasks_max
        self.real_sched = real_sched
        self.real_epilogue = real_epilogue
        self.grid = grid

    @cute.jit
    def __call__(
        self,
        symm: cute.Pointer,
        w1: cute.Pointer,
        w1_sf: cute.Pointer,
        w2: cute.Pointer,
        w2_sf: cute.Pointer,
        tasks: cute.Pointer,
        ntasks: Int32,
        out: cute.Pointer,
        taskdump: cute.Pointer,
        stream: cuda.CUstream,
    ):
        cfg = self.cfg
        tp = tma_warps.make_tma_params(
            cfg,
            sym=SymHost(symm.toint()),
            l1_w=w1,
            l1_w_sf=w1_sf,
            l2_w=w2,
            l2_w_sf=w2_sf,
        )
        self.kernel(tp, symm, tasks, ntasks, out, taskdump, Int32(0)).launch(
            grid=(self.grid, 1, 1),
            block=(512, 1, 1),
            cluster=(2, 1, 1),
            smem=cfg.smem_bytes,
            min_blocks_per_mp=1,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        tp: tma_warps.TmaParams,
        symm: cute.Pointer,
        tasks: cute.Pointer,
        ntasks: Int32,
        out: cute.Pointer,
        taskdump: cute.Pointer,
        num_tokens: Int32,
    ):
        cfg = self.cfg
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(tidx // 32)
        lane = tidx % 32
        cta_rank = cute.arch.block_idx_in_cluster()
        smem_base = cute.arch.get_dyn_smem(cutlass.Uint8, alignment=1024)
        smem = fctx.SmemMap(smem_base, cfg)
        ctx = GCtx(
            cfg=cfg,
            tma=tp,
            num_tokens=num_tokens,
            warp_idx=warp_idx,
            lane_idx=lane,
            cta_rank=cta_rank,
            is_leader_cta=cta_rank == 0,
            smem=smem,
            mb=Barriers(smem),
            tmem=fctx.TmemMap(cfg),
            ws=fctx.Workspace(symm.toint(), cfg),
            buf=fctx.BufferMap(symm.toint(), cfg),
            bar=fctx.NamedBarriers(),
            sym=fcomm.SymBuffer(symm.toint(), Int32(0), None, 1),
            tr=Trace(cfg, None, 0),
        )  # tr: disabled trace (cfg.trace False -> no-ops)

        if warp_idx == 0:
            tma_warps.prefetch_tma_params(tp)  # K:142-161
        ptx.cluster_sync_relaxed()  # K:293
        # initialization (K:296-340)
        if warp_idx == 2:
            with cute.arch.elect_one():
                for i in cutlass.range_constexpr(cfg.num_stages):
                    ctx.mb.full(i).init(2 * 2)
                    ctx.mb.empty(i).init(1)
                for i in cutlass.range_constexpr(2):
                    ctx.mb.tmem_full(i).init(1)
                    ctx.mb.tmem_empty(i).init(2 * cfg.num_epilogue_threads)
                for i in cutlass.range_constexpr(2):
                    ctx.mb.task_info_full(i).init(1)
                    ctx.mb.task_info_empty(i).init(2 * cfg.num_epilogue_threads)
            ptx.fence_barrier_init()
        if warp_idx == 3:
            cute.arch.alloc_tmem(
                cfg.tmem_cols_alloc, ctx.smem.tmem_ptr, is_two_cta=True, arch="sm_107"
            )
        ptx.cluster_arrive()
        ptx.cluster_wait()

        if warp_idx == 4:
            tma_warps.tma_a_role(ctx)
        if warp_idx == 5:
            tma_warps.tma_b_role(ctx)
        if warp_idx == 6:
            if ctx.is_leader_cta:
                mma_warp.mma_role(ctx)
        if warp_idx == 7:
            if ctx.is_leader_cta:
                if cutlass.const_expr(self.real_sched):
                    sched.scheduler_role(ctx)
                else:
                    self._mini_scheduler(ctx, tasks, ntasks)
        if warp_idx >= 8:
            if cutlass.const_expr(self.real_epilogue):
                epi_mod.epilogue_role(ctx)
            else:
                self._dump_epilogue(ctx, out, taskdump)
            if warp_idx == 8:
                cute.arch.dealloc_tmem(
                    cute.make_ptr(Float32, 0, cute.AddressSpace.tmem, assumed_align=16),
                    cfg.tmem_cols_alloc,
                    is_two_cta=True,
                    arch="sm_107",
                )
        # harness-only: keep both CTAs alive until everything is done
        ptx.cluster_arrive()
        ptx.cluster_wait()

    @cute.jit
    def _mini_scheduler(self, ctx, tasks: cute.Pointer, ntasks: Int32):
        """Test stand-in for the scheduler warp: publishes the host task table (8 u32 per task) then the sentinel."""
        sched_stage = Int32(0)
        sched_phase = Int32(0)
        t = cute.make_tensor(
            tasks, cute.make_layout((self.ntasks_max, 8), stride=(8, 1))
        )
        for i in cutlass.range(ntasks, unroll=1):
            ctx.mb.task_info_empty(sched_stage).wait(sched_phase ^ 1)
            sched_stage, sched_phase = sched.publish_task(
                ctx,
                Int32(t[i, 0]),
                Int32(t[i, 1]),
                Int32(t[i, 2]),
                Int32(t[i, 3]),
                Int32(t[i, 4]),
                Int32(t[i, 5]),
                Int32(t[i, 6]),
                Int32(t[i, 7]),
                sched_stage,
                sched_phase,
            )
        ctx.mb.task_info_empty(sched_stage).wait(sched_phase ^ 1)
        z = Int32(0)
        sched_stage, sched_phase = sched.publish_task(
            ctx, z, z, z, z, z, z, z, z, sched_stage, sched_phase
        )

    @cute.jit
    def _dump_epilogue(self, ctx, out: cute.Pointer, taskdump: cute.Pointer):
        """Test epilogue: raw FP32 dump of the accumulator of every task: out[task][cta][128 rows][BLOCK_M] (int32 bits)."""
        cfg = self.cfg
        BM = cfg.block_m
        ew = ctx.warp_idx - 8
        wg = ew // 4
        wq = ew % 4
        lane = ctx.lane_idx
        o = cute.make_tensor(
            cute.recast_ptr(out, dtype=Int32),
            cute.make_layout(self.ntasks_max * 2 * 128 * BM),
        )
        td = cute.make_tensor(
            taskdump, cute.make_layout((self.ntasks_max, 8), stride=(8, 1))
        )
        sched_stage = Int32(0)
        sched_phase = Int32(0)
        it = Int32(0)
        task_idx = Int32(0)
        t0, t1, t2, t3, t4, t5, t6, t7, sched_stage, sched_phase = sched.get_next_task(
            ctx, sched_stage, sched_phase
        )
        while t0 != sched.PHASE_NONE:
            accum_stage = it % 2
            accum_phase = (it // 2) & 1
            it = it + 1
            ctx.mb.tmem_full(accum_stage).wait(accum_phase)
            ptx.tcgen05_fence_after_thread_sync()
            sched.release_task_info(ctx, sched_stage)
            if ctx.warp_idx == 8:
                if ctx.lane_idx == 0:
                    if ctx.cta_rank == 0:
                        td[task_idx, 0] = t0
                        td[task_idx, 1] = t1
                        td[task_idx, 2] = t2
                        td[task_idx, 3] = t3
                        td[task_idx, 4] = t4
                        td[task_idx, 5] = t5
                        td[task_idx, 6] = t6
                        td[task_idx, 7] = t7
            base = ((task_idx * 2 + ctx.cta_rank) * 128) * BM
            row = wq * 32 + lane // 4
            for atom in cutlass.range_constexpr(cfg.wg_block_m // 8):
                col = accum_stage * BM + wg * cfg.wg_block_m + atom * 8
                a0, a1, a2, a3 = ptx.tmem_ld_16x256b_x1(col)
                b0, b1, b2, b3 = ptx.tmem_ld_16x256b_x1(col | 0x00100000)
                ptx.tmem_ld_wait()
                tok = wg * cfg.wg_block_m + atom * 8 + 2 * (lane % 4)
                o[base + row * BM + tok] = a0
                o[base + row * BM + tok + 1] = a1
                o[base + (row + 8) * BM + tok] = a2
                o[base + (row + 8) * BM + tok + 1] = a3
                o[base + (row + 16) * BM + tok] = b0
                o[base + (row + 16) * BM + tok + 1] = b1
                o[base + (row + 24) * BM + tok] = b2
                o[base + (row + 24) * BM + tok + 1] = b3
            ptx.tcgen05_fence_before_thread_sync()
            ctx.mb.tmem_empty(accum_stage).arrive_remote(0)
            task_idx = task_idx + 1
            t0, t1, t2, t3, t4, t5, t6, t7, sched_stage, sched_phase = (
                sched.get_next_task(ctx, sched_stage, sched_phase)
            )


class HarnessShared(Harness):
    """Harness for S > 0 builds: the four shared-expert weight pointers are passed through (``make_tma_params`` builds the nine extra maps)."""

    @cute.jit
    def __call__(
        self,
        symm: cute.Pointer,
        w1: cute.Pointer,
        w1_sf: cute.Pointer,
        w2: cute.Pointer,
        w2_sf: cute.Pointer,
        sw1: cute.Pointer,
        sw1_sf: cute.Pointer,
        sw2: cute.Pointer,
        sw2_sf: cute.Pointer,
        tasks: cute.Pointer,
        ntasks: Int32,
        out: cute.Pointer,
        taskdump: cute.Pointer,
        num_tokens: Int32,
        stream: cuda.CUstream,
    ):
        cfg = self.cfg
        tp = tma_warps.make_tma_params(
            cfg,
            sym=SymHost(symm.toint()),
            l1_w=w1,
            l1_w_sf=w1_sf,
            l2_w=w2,
            l2_w_sf=w2_sf,
            shared_l1_w=sw1,
            shared_l1_w_sf=sw1_sf,
            shared_l2_w=sw2,
            shared_l2_w_sf=sw2_sf,
        )
        self.kernel(tp, symm, tasks, ntasks, out, taskdump, num_tokens).launch(
            grid=(self.grid, 1, 1),
            block=(512, 1, 1),
            cluster=(2, 1, 1),
            smem=cfg.smem_bytes,
            min_blocks_per_mp=1,
            stream=stream,
        )
