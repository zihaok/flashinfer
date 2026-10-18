# SPDX-License-Identifier: Apache-2.0
# Port of DeepGEMM (MIT License, Copyright (c) 2025 DeepSeek; see licenses/LICENSE.deepgemm.txt) sm107_fp8_fp4_mega_moe.cuh / ptx helpers.
"""GPU unit tests of ``kernel/ptx.py``.  Needs an SM107 GPU (cluster launch) + CuteDSL 4.8.

Run:  ``python3 tests/experimental/test_sm107_dg_foundation_ptx.py [group ...]``  (groups: a b c)  or via pytest.
Each group compiles ONE test kernel that exercises many helpers; results are checked against torch / python references.
Set ``SM107_DG_DUMP_DIR=<dir>`` to keep PTX/CUBIN/SASS of every compiled test kernel (for SASS inspection).
"""

import importlib.util
import os
import sys
import types

import pytest

torch = pytest.importorskip("torch")
if not (
    torch.cuda.is_available() and torch.cuda.get_device_capability() == (10, 7)
):  # pragma: no cover
    pytest.skip(
        "needs an SM107 (Rubin, compute capability 10.7) device",
        allow_module_level=True,
    )
cutlass = pytest.importorskip("cutlass")

import cuda.bindings.driver as cuda  # noqa: E402
import cutlass.cute as cute  # noqa: E402
from cutlass.cute.runtime import make_ptr  # noqa: E402
from cutlass.cute.typing import AddressSpace  # noqa: E402
from cutlass.cutlass_dsl import Boolean, Float32, Int32, Int64  # noqa: E402
from dataclasses import dataclass  # noqa: E402


def _load_kernel_modules():
    """Import ``kernel/{ptx,comm}.py`` WITHOUT executing ``kernel/__init__.py`` (insulated from the skeleton under construction)."""
    here = os.path.dirname(os.path.abspath(__file__))
    cand = [
        os.path.join(
            here,
            "..",
            "..",
            "flashinfer",
            "experimental",
            "sm107_deepgemm_megamoe",
            "kernel",
        ),
    ]
    kdir = next(
        os.path.abspath(c) for c in cand if os.path.exists(os.path.join(c, "ptx.py"))
    )
    if "sm107_kernel" not in sys.modules:
        pkg = types.ModuleType("sm107_kernel")
        pkg.__path__ = [kdir]
        sys.modules["sm107_kernel"] = pkg
    mods = {}
    for name in ("ptx", "comm"):
        full = f"sm107_kernel.{name}"
        if full in sys.modules:
            mods[name] = sys.modules[full]
            continue
        if not os.path.exists(os.path.join(kdir, f"{name}.py")):
            continue
        spec = importlib.util.spec_from_file_location(
            full, os.path.join(kdir, f"{name}.py")
        )
        m = importlib.util.module_from_spec(spec)
        sys.modules[full] = m
        spec.loader.exec_module(m)
        mods[name] = m
    return mods


_MODS = _load_kernel_modules()
ptx = _MODS["ptx"]


def _opts():
    """Extra compile options: keep PTX/CUBIN/SASS when SM107_DG_DUMP_DIR is set."""
    d = os.environ.get("SM107_DG_DUMP_DIR")
    if not d:
        return None
    os.makedirs(d, exist_ok=True)
    return d


def compile_and_run(kernel, *args, tag="k", stream=None):
    """Compile ``kernel(*args)`` with the DSL (optionally dumping PTX/SASS) and run it once. Returns the compiled object."""
    stream = stream or cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    compiled = cute.compile(kernel, *args, stream)
    compiled(*args, stream)
    torch.cuda.synchronize()
    return compiled


def gptr(t: "torch.Tensor", dtype=cutlass.Int32, align=16):
    return make_ptr(dtype, t.data_ptr(), AddressSpace.gmem, assumed_align=align)


# =====================================================================================================================
# group A: identity, barriers, fences, mbarrier (local + remote), st.async, shared zero, advance_stage
# =====================================================================================================================
OUT_STRIDE = 256  # int32 results per CTA
SMEM_A = 8192


class KernelA:
    """2 CTAs (cluster 2), 128 threads.  smem: [0,4096) data, [4096, 4096+64) mbarriers, [5120,..) st.async landing area."""

    @cute.jit
    def __call__(self, out: cute.Pointer, stream: cuda.CUstream):
        self.kernel(out).launch(
            grid=[2, 1, 1],
            block=[128, 1, 1],
            cluster=[2, 1, 1],
            smem=SMEM_A,
            stream=stream,
        )

    @cute.kernel
    def kernel(self, out: cute.Pointer):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        cta_rank = cute.arch.block_idx_in_cluster()
        warp = tidx // 32
        lane = tidx % 32
        base = (
            out + bidx * OUT_STRIDE
        )  # int32 element offset (each CTA writes its own slice)
        smem = cute.arch.get_dyn_smem(cutlass.Int8, alignment=1024)

        mb_local = ptx.Mbar(smem + 4096)  # count 1
        mb_cnt3 = ptx.Mbar(smem + 4104)  # count 3 (arrive_count)
        mb_try = ptx.Mbar(smem + 4112)  # count 1 (try_wait false -> true)
        mb_remote = ptx.Mbar(
            smem + 4120
        )  # count 1: peer arrives remotely (arrive_remote)
        mb_tx_local = ptx.Mbar(
            smem + 4128
        )  # count 1 + 16 B tx (expect_tx + arrive, completed by a local st.async)
        mb_tx_remote = ptx.Mbar(
            smem + 4136
        )  # count 1 + 16 B tx; peer does arrive_expect_tx_remote + st.async into us
        mb_tx_local2 = ptx.Mbar(
            smem + 4144
        )  # count 1: arrive_expect_tx + local st.async

        # ---- init (thread 0) + fence + cluster sync
        if tidx == 0:
            mb_local.init(1)
            mb_cnt3.init(3)
            mb_try.init(1)
            mb_remote.init(1)
            mb_tx_local.init(1)
            mb_tx_remote.init(1)
            mb_tx_local2.init(1)
        ptx.fence_barrier_init()
        ptx.cluster_sync()

        # ---- identity helpers
        if tidx == 0:
            ptx.st_b32(base + 0, ptx.smid())
            ptx.st_b32(base + 1, Int32(ptx.gridid()))
            ptx.st_b32(base + 2, cta_rank)
        if tidx == 5:
            ptx.st_b32(base + 3, ptx.lane_id())
        if tidx == 0:
            c0 = ptx.clock64()
            g0 = ptx.globaltimer()
            ptx.nanosleep(2000)
            c1 = ptx.clock64()
            g1 = ptx.globaltimer()
            ptx.st_b32(base + 4, Int32(c1 > c0))
            ptx.st_b32(base + 5, Int32(g1 > g0))

        # ---- elect.sync: exactly one lane per warp; its lane id is the lowest
        e = ptx.elect_sync()
        n_elected = ptx.popc(ptx.ballot(e))
        if lane == 0:
            ptx.st_b32(base + 8 + warp, n_elected)
        if e:
            ptx.st_b32(base + 12 + warp, lane)

        # ---- local arrive/wait (count 1)
        if tidx == 0:
            mb_local.arrive()
        mb_local.wait(0)
        # arrive_count: three arrivals in one go
        if tidx == 1:
            mb_cnt3.arrive_count(3)
        mb_cnt3.wait(0)
        if tidx == 0:
            ptx.st_b32(base + 16, 1)

        # ---- try_wait: false before arrive, true after
        t_before = Int32(0)
        if tidx == 0:
            t_before = Int32(mb_try.try_wait(0))
            mb_try.arrive()
            ok = Int32(0)
            ok = Int32(mb_try.try_wait(0))
            ptx.st_b32(base + 17, t_before)
            ptx.st_b32(base + 18, ok)

        # ---- bar.sync (aligned): neighbour exchange through smem
        ptx.st_shared_u32(smem + tidx * 4, tidx * 7 + 1)
        ptx.bar_sync(5, 128)
        nb = ptx.ld_shared_u32(smem + ((tidx + 1) % 128) * 4)
        if tidx < 64:
            ptx.st_b32(base + 32 + tidx, nb)
        ptx.bar_sync(5, 128)
        # ---- barrier.sync (unaligned) over warps 0..2 only (96 threads), runtime-valued operands
        nthr96 = Int32(96)
        if tidx < 96:
            ptx.st_shared_u32(smem + 1024 + tidx * 4, tidx + 100)
            ptx.bar_sync_unaligned(6, nthr96)
            v = ptx.ld_shared_u32(smem + 1024 + ((tidx + 32) % 96) * 4)
            if tidx < 32:
                ptx.st_b32(base + 96 + tidx, v)

        # ---- st.bulk zero of 2048 B (one elected lane), after filling with ones
        ptx.st_shared_u32(smem + 2048 + tidx * 4, Int32(-1))
        ptx.st_shared_u32(smem + 2048 + 512 + tidx * 4, Int32(-1))
        ptx.st_shared_u32(smem + 2048 + 1024 + tidx * 4, Int32(-1))
        ptx.st_shared_u32(smem + 2048 + 1536 + tidx * 4, Int32(-1))
        ptx.bar_sync(5, 128)
        if tidx == 0:
            ptx.st_shared_zero(smem + 2048, 2048)
        ptx.bar_sync(5, 128)
        z = (
            ptx.ld_shared_u32(smem + 2048 + tidx * 4)
            | ptx.ld_shared_u32(smem + 2048 + 512 + tidx * 4)
            | ptx.ld_shared_u32(smem + 2048 + 1024 + tidx * 4)
            | ptx.ld_shared_u32(smem + 2048 + 1536 + tidx * 4)
        )
        if tidx < 32:
            ptx.st_b32(base + 128 + tidx, z)

        # ---- remote arrive: the peer CTA (rank 1) arrives on rank 0's mb_remote
        if cta_rank == 1:
            if tidx == 0:
                mb_remote.arrive_remote(0)
        if cta_rank == 0:
            if tidx == 0:
                mb_remote.wait(0)
                ptx.st_b32(base + 19, 1)

        # ---- arrive_expect_tx + local st.async (own CTA), expect_tx + arrive variant
        if tidx == 0:
            mb_tx_local.expect_tx(16)
            mb_tx_local.arrive()
            ptx.st_async_v4(smem + 5120, 11, 22, 33, 44, mb_tx_local, cta_rank)
            mb_tx_local.wait(0)
            a, b, c, d = ptx.ld_shared_v4_u32(smem + 5120)
            ptx.st_b32(base + 20, a)
            ptx.st_b32(base + 21, b)
            ptx.st_b32(base + 22, c)
            ptx.st_b32(base + 23, d)
            mb_tx_local2.arrive_expect_tx(16)
            ptx.st_async_v4(smem + 5136, 55, 66, 77, 88, mb_tx_local2, cta_rank)
            mb_tx_local2.wait(0)
            a2, b2, c2, d2 = ptx.ld_shared_v4_u32(smem + 5136)
            ptx.st_b32(base + 24, a2 + b2 + c2 + d2)

        # ---- remote arrive_expect_tx + st.async from rank 1 into rank 0
        if cta_rank == 1:
            if tidx == 0:
                mb_tx_remote.arrive_expect_tx_remote(16, 0)
                ptx.st_async_v4(smem + 5152, 101, 102, 103, 104, mb_tx_remote, 0)
        if cta_rank == 0:
            if tidx == 0:
                mb_tx_remote.wait(0)
                a3, b3, c3, d3 = ptx.ld_shared_v4_u32(smem + 5152)
                ptx.st_b32(base + 25, a3)
                ptx.st_b32(base + 26, b3)
                ptx.st_b32(base + 27, c3)
                ptx.st_b32(base + 28, d3)

        # ---- addresses
        if tidx == 0:
            ptx.st_b32(base + 29, mb_local.addr)
            ptx.st_b32(base + 30, mb_local.cluster_addr(1))
            ptx.st_b32(base + 31, mb_local.addr_leader)

        # ---- fences compile + run
        ptx.fence_acq_rel_cta()
        ptx.fence_proxy_async_shared_cta()
        ptx.tcgen05_fence_before_thread_sync()
        ptx.tcgen05_fence_after_thread_sync()

        # ---- advance_stage sequence (7 stages), 20 steps, thread 0
        if tidx == 0:
            stage = Int32(0)
            phase = Int32(0)
            for i in cutlass.range_constexpr(20):
                ptx.st_b32(base + 160 + i, stage | (phase << 8))
                stage, phase = ptx.advance_stage(stage, phase, 7)

        # ---- cluster-scope sync helpers (compile + run)
        ptx.cluster_arrive_relaxed()
        ptx.cluster_wait()
        ptx.cluster_sync_relaxed()
        ptx.cluster_sync()

        # ---- bar.arrive / bar.sync pairing on id 7 (128 threads = 64 arriving + 64 waiting): warps 0-1 produce, warps 2-3 consume
        if warp < 2:
            ptx.st_shared_u32(smem + 3072 + tidx * 4, tidx + 500)
            ptx.bar_arrive(7, 128)
        else:
            ptx.bar_sync(7, 128)
            ptx.st_b32(
                base + (190 + tidx - 64),
                ptx.ld_shared_u32(smem + (3072 + (tidx - 64) * 4)),
            )
        ptx.bar_sync(5, 128)

        # ---- tcgen05.alloc/dealloc .exclusive.cta_group::2 by warp 3 of BOTH CTAs (as DeepGEMM), 64 columns
        ptx.syncwarp()
        if warp == 3:
            ptx.tmem_alloc(smem + 6144, 64, 2, True)
        ptx.cluster_sync()
        taddr_a = ptx.ld_shared_u32(smem + 6144)
        if tidx == 0:
            ptx.st_b32(base + 181, taddr_a)
        ptx.bar_sync(5, 128)
        if warp == 3:
            ptx.tmem_dealloc(taddr_a, 64, 2, True)
        ptx.cluster_sync()


def check_group_a():
    out = torch.zeros(2 * OUT_STRIDE, dtype=torch.int32, device="cuda")
    compile_and_run(KernelA(), gptr(out), tag="a")
    r = out.cpu().view(2, OUT_STRIDE)
    for c in range(2):
        row = r[c]
        assert 0 <= int(row[0]) < 212, f"smid {int(row[0])}"
        assert int(row[1]) != 0, "gridid == 0"
        assert int(row[2]) == c
        assert int(row[3]) == 5, f"lane_id {int(row[3])}"
        assert int(row[4]) == 1 and int(row[5]) == 1, (
            "clock64/globaltimer not monotonic"
        )
        assert row[8:12].tolist() == [1, 1, 1, 1], (
            f"elect count per warp {row[8:12].tolist()}"
        )
        assert row[12:16].tolist() == [0, 0, 0, 0], (
            f"elected lane {row[12:16].tolist()}"
        )
        assert int(row[16]) == 1
        assert int(row[17]) == 0, "try_wait before arrive must be false"
        assert int(row[18]) == 1, "try_wait after arrive must be true"
        exp = [((t + 1) % 128) * 7 + 1 for t in range(64)]
        assert row[32:96].tolist() == exp, "bar.sync neighbour exchange"
        exp96 = [((t + 32) % 96) + 100 for t in range(32)]
        assert row[96:128].tolist() == exp96, "barrier.sync (unaligned) exchange"
        assert row[128:160].tolist() == [0] * 32, "st.bulk zero"
        assert row[20:24].tolist() == [11, 22, 33, 44], (
            f"local st.async {row[20:24].tolist()}"
        )
        assert int(row[24]) == 55 + 66 + 77 + 88
        seq = []
        s = p = 0
        for _ in range(20):
            seq.append(s | (p << 8))
            s = 0 if s == 6 else s + 1
            p ^= int(s == 0)
        assert row[160:180].tolist() == seq, (
            f"advance_stage {row[160:180].tolist()} vs {seq}"
        )
    assert int(r[0][19]) == 1, "remote arrive on rank 0 did not complete"
    assert r[0][25:29].tolist() == [101, 102, 103, 104], (
        f"remote st.async {r[0][25:29].tolist()}"
    )
    a0, ca0, lead0 = (int(r[0][29]), int(r[0][30]), int(r[0][31]))
    print(
        f"  [info] mbar addr (cta0)={a0:#x} cluster_addr(1)={ca0:#x} addr_leader={lead0:#x} ; cta1 addr={int(r[1][29]):#x}"
    )
    assert (ca0 ^ a0) == (1 << 24), (
        "cluster_addr(1) must differ from the local address by the CTA bit (bit 24)"
    )
    assert lead0 == (a0 & 0xFEFFFFFF)
    for c in range(2):
        assert r[c][190:254].tolist() == [t + 500 for t in range(64)], (
            "bar.arrive / bar.sync pairing"
        )
    print(
        f"  [info] TMEM base after tcgen05.alloc.exclusive.cta_group::2 (64 cols): cta0={int(r[0][181]):#x} cta1={int(r[1][181]):#x}"
    )
    assert int(r[0][181]) == 0 and int(r[1][181]) == 0, (
        "TMEM base address must be 0 (DeepGEMM asserts it)"
    )


def test_group_a():
    check_group_a()


# =====================================================================================================================
# group B: global memory ops, shared ops, bulk copies (1-D TMA), atomics
# =====================================================================================================================
SRC32, SRC64, SRCF, DST, CNT, TMAST = 0, 256, 512, 640, 1024, 1152
GEN_DST = 1280  # generic-store destination (224 words)
BYTE_DST = 1536  # st_b8 destination (64 bytes)
OLD32, OLD64, OLDS64, OLDR32, OUTB = 2048, 2304, 2816, 3328, 4096
TOTAL_B = 5120


class KernelB:
    """2 CTAs x 128 threads. CTA 0: loads/stores/shared ops/atomics; CTA 1: 1-D TMA load/store + atomics."""

    @cute.jit
    def __call__(self, buf: cute.Pointer, stream: cuda.CUstream):
        self.kernel(buf).launch(
            grid=[2, 1, 1], block=[128, 1, 1], smem=8192, stream=stream
        )

    @cute.kernel
    def kernel(self, buf: cute.Pointer):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        gt = bidx * 128 + tidx
        smem = cute.arch.get_dyn_smem(cutlass.Int8, alignment=1024)
        o = buf + OUTB

        # ---- loads (CTA 0, 16 threads)
        if bidx == 0:
            if tidx < 16:
                ptx.st_b32(o + (0 * 16 + tidx), ptx.ld_b32(buf + (SRC32 + tidx)))
                ptx.st_b32(
                    o + (1 * 16 + tidx), ptx.ld_volatile_b32(buf + (SRC32 + tidx))
                )
                ptx.st_b32(
                    o + (2 * 16 + tidx), ptx.ld_acquire_gpu_b32(buf + (SRC32 + tidx))
                )
                ptx.st_b32(
                    o + (3 * 16 + tidx), ptx.ld_acquire_sys_b32(buf + (SRC32 + tidx))
                )
                ptx.st_b32(
                    o + (4 * 16 + tidx), ptx.ld_acquire_sys_s32(buf + (SRC32 + tidx))
                )
                ptx.st_b64(o + (96 + 2 * tidx), ptx.ld_b64(buf + (SRC64 + 2 * tidx)))
                ptx.st_b64(
                    o + (128 + 2 * tidx), ptx.ld_volatile_b64(buf + (SRC64 + 2 * tidx))
                )
                ptx.st_b64(
                    o + (160 + 2 * tidx),
                    ptx.ld_acquire_gpu_b64(buf + (SRC64 + 2 * tidx)),
                )
                ptx.st_b64(
                    o + (192 + 2 * tidx),
                    ptx.ld_acquire_sys_b64(buf + (SRC64 + 2 * tidx)),
                )
                ptx.st_f32(o + (224 + tidx), ptx.ld_f32(buf + (SRCF + tidx)))
                a, b, c, d = ptx.ld_v4_b32(buf + (SRC32 + 4 * tidx))
                ptx.st_b32(o + (240 + 4 * tidx + 0), a)
                ptx.st_b32(o + (240 + 4 * tidx + 1), b)
                ptx.st_b32(o + (240 + 4 * tidx + 2), c)
                ptx.st_b32(o + (240 + 4 * tidx + 3), d)

        # ---- generic-space loads (CTA 0, 16 threads)
        if bidx == 0:
            if tidx < 16:
                ptx.st_b32(o + (480 + tidx), ptx.ld_generic_b32(buf + (SRC32 + tidx)))
                ptx.st_b64(
                    o + (496 + 2 * tidx), ptx.ld_generic_b64(buf + (SRC64 + 2 * tidx))
                )
                ga, gb, gc, gd = ptx.ld_generic_v4_b32(buf + (SRC32 + 4 * tidx))
                ptx.st_b32(o + (528 + 4 * tidx + 0), ga)
                ptx.st_b32(o + (528 + 4 * tidx + 1), gb)
                ptx.st_b32(o + (528 + 4 * tidx + 2), gc)
                ptx.st_b32(o + (528 + 4 * tidx + 3), gd)

        # ---- generic-space stores (CTA 0, 32 threads)
        if bidx == 0:
            if tidx < 32:
                g0 = buf + GEN_DST
                ptx.st_generic_b32(g0 + tidx, 3 * tidx + 7)
                ptx.st_generic_b64(
                    g0 + (32 + 2 * tidx), (Int64(tidx) << 33) | Int64(tidx + 3)
                )
                ptx.st_generic_v4_b32(
                    g0 + (96 + 4 * tidx), tidx, tidx * 2, tidx * 3, tidx * 4
                )

        # ---- byte store / 16-bit shared load (CTA 0)
        if bidx == 0:
            if tidx < 64:
                ptx.st_b8(
                    buf.toint() + (BYTE_DST * 4 + tidx), tidx * 7 + 3
                )  # byte address (Int64)
            if tidx == 0:
                ptx.st_shared_u32(smem + 2656, 0xBEEF1234)
                ptx.st_b32(o + 722, ptx.ld_shared_b16(smem + 2656))
                ptx.st_b32(o + 723, ptx.ld_shared_b16(smem + 2658))

        # ---- shared-memory atomic (CTA 0, all 128 threads)
        if bidx == 0:
            if tidx == 0:
                ptx.st_shared_u32(smem + 2640, 0)
            ptx.bar_sync(5, 128)
            old_s = ptx.atom_add_shared_u32(smem + 2640, 1)
            ptx.st_b32(o + (592 + tidx), old_s)
            ptx.bar_sync(5, 128)
            if tidx == 0:
                ptx.st_b32(o + 720, ptx.ld_shared_u32(smem + 2640))

        # ---- stores (CTA 0, 32 threads)
        if bidx == 0:
            if tidx < 32:
                d0 = buf + DST
                ptx.st_b32(d0 + tidx, 7 * tidx + 1)
                ptx.st_b64(
                    d0 + (32 + 2 * tidx), (Int64(tidx) << 32) | Int64(tidx * 2 + 5)
                )
                ptx.st_f32(d0 + (96 + tidx), tidx.to(cutlass.Float32) * 0.25)
                ptx.st_v4_b32(
                    d0 + (128 + 4 * tidx), tidx, tidx + 100, tidx + 200, tidx + 300
                )
                ptx.st_release_sys_b64(
                    d0 + (256 + 2 * tidx), Int64(tidx) * Int64(1000003)
                )
                ptx.st_release_gpu_b64(
                    d0 + (320 + 2 * tidx), Int64(tidx) * Int64(7919) + Int64(11)
                )

        # ---- atomics / reductions (all 256 threads)
        cn = buf + CNT
        o32 = ptx.atom_add_u32(cn + 0, 1)
        ptx.st_b32(buf + (OLD32 + gt), o32)
        o64 = ptx.atom_add_u64(cn + 2, (Int64(1) << 32) + Int64(2))
        ptx.st_b64(buf + (OLD64 + 2 * gt), o64)
        os_ = ptx.atom_add_sys_u64(cn + 4, 3)
        ptx.st_b64(buf + (OLDS64 + 2 * gt), os_)
        orl = ptx.atom_add_release_gpu_u32(cn + 6, 1)
        ptx.st_b32(buf + (OLDR32 + gt), orl)
        ptx.red_add_u32(cn + 7, 1)
        ptx.red_add_s32(cn + 8, -2)
        ptx.red_add_release_gpu_u32(cn + 9, 3)
        ptx.red_add_release_sys_s32(cn + 10, -1)
        if gt < 200:
            ptx.red_xor_release_gpu_b64(cn + 12, Int64(1) << Int64(gt % 64))
        ptx.red_or_release_gpu_b64(cn + 14, Int64(1) << Int64(gt % 64))
        ptx.red_or_release_sys_b64(cn + 16, Int64(1) << Int64(gt % 64))

        # ---- 1-D TMA load + store (CTA 1)
        mb = ptx.Mbar(smem + 4096)
        if bidx == 1:
            if tidx == 0:
                mb.init(1)
            ptx.fence_barrier_init()
            ptx.bar_sync(5, 128)
            if tidx == 0:
                ptx.tma_load_1d(smem + 1024, buf + SRC32, mb, 512)
                mb.arrive_expect_tx(512)
            mb.wait(0)
            ptx.st_b32(o + (320 + tidx), ptx.ld_shared_u32(smem + (1024 + tidx * 4)))
            ptx.st_shared_u32(smem + (1536 + tidx * 4), tidx * 11 + 3)
            ptx.fence_proxy_async_shared_cta()
            ptx.bar_sync(5, 128)
            if tidx == 0:
                ptx.tma_store_1d(buf + TMAST, smem + 1536, 512)
                ptx.tma_store_commit()
                ptx.tma_store_wait(0)

        # ---- shared-memory vector ops (CTA 0, thread 0)
        if bidx == 0:
            if tidx == 0:
                ptx.st_shared_v4_u32(smem + 2560, 1, 2, 3, 4)
                ptx.st_shared_v2_u32(smem + 2576, 5, 6)
                ptx.st_shared_v2_f32(smem + 2584, Float32(1.5), Float32(2.5))
                ptx.st_shared_f32(smem + 2592, Float32(3.5))
                ptx.st_shared_u32(smem + 2596, 9)
                ptx.st_shared_f32(smem + 2608, Float32(1.0))
                ptx.st_shared_f32(smem + 2612, Float32(2.0))
                ptx.st_shared_f32(smem + 2616, Float32(3.0))
                ptx.st_shared_f32(smem + 2620, Float32(4.0))
                a1, b1, c1, d1 = ptx.ld_shared_v4_u32(smem + 2560)
                e1, f1 = ptx.ld_shared_v2_f32(smem + 2584)
                g1 = ptx.ld_shared_f32(smem + 2592)
                h1 = ptx.ld_shared_u32(smem + 2596)
                p1, q1, r1, s1 = ptx.ld_shared_v4_f32(smem + 2608)
                u1 = ptx.ld_shared_u32(smem + 2576)
                v1 = ptx.ld_shared_u32(smem + 2580)
                ptx.st_b32(o + 448, a1)
                ptx.st_b32(o + 449, b1)
                ptx.st_b32(o + 450, c1)
                ptx.st_b32(o + 451, d1)
                ptx.st_f32(o + 452, e1)
                ptx.st_f32(o + 453, f1)
                ptx.st_f32(o + 454, g1)
                ptx.st_b32(o + 455, h1)
                ptx.st_f32(o + 456, p1)
                ptx.st_f32(o + 457, q1)
                ptx.st_f32(o + 458, r1)
                ptx.st_f32(o + 459, s1)
                ptx.st_b32(o + 460, u1)
                ptx.st_b32(o + 461, v1)


def check_group_b():
    import numpy as np

    host = torch.zeros(TOTAL_B, dtype=torch.int32)
    host[SRC32 : SRC32 + 256] = torch.arange(256, dtype=torch.int32) * 5 + 2
    host[SRC64 : SRC64 + 256].view(torch.int64)[:] = (
        torch.arange(128, dtype=torch.int64) * 3 + 1
    )
    host[SRCF : SRCF + 128].view(torch.float32)[:] = (
        torch.arange(128, dtype=torch.float32) * 0.5
    )
    buf = host.cuda()
    compile_and_run(KernelB(), gptr(buf), tag="b")
    r = buf.cpu()
    out = r[OUTB : OUTB + 1024]
    src32 = torch.arange(256, dtype=torch.int32) * 5 + 2
    # loads
    for k in range(5):
        assert out[k * 16 : (k + 1) * 16].tolist() == src32[:16].tolist(), (
            f"load variant {k}"
        )
    exp64 = (torch.arange(16, dtype=torch.int64) * 3 + 1).tolist()
    for k, name in enumerate(
        ("ld_b64", "ld_volatile_b64", "ld_acquire_gpu_b64", "ld_acquire_sys_b64")
    ):
        got = (
            out[96 + 32 * k : 96 + 32 * (k + 1)].contiguous().view(torch.int64).tolist()
        )
        assert got == exp64, f"{name}: {got}"
    assert out[224:240].view(torch.float32).tolist() == [0.5 * i for i in range(16)], (
        "ld_f32"
    )
    assert out[240:304].tolist() == src32[:64].tolist(), "ld_v4_b32"
    # stores
    dst = r[DST : DST + 384]
    assert dst[0:32].tolist() == [7 * t + 1 for t in range(32)], "st_b32"
    assert dst[32:96].contiguous().view(torch.int64).tolist() == [
        (t << 32) | (2 * t + 5) for t in range(32)
    ], "st_b64"
    assert dst[96:128].view(torch.float32).tolist() == [0.25 * t for t in range(32)], (
        "st_f32"
    )
    assert dst[128:256].tolist() == [
        v for t in range(32) for v in (t, t + 100, t + 200, t + 300)
    ], "st_v4_b32"
    assert dst[256:320].contiguous().view(torch.int64).tolist() == [
        t * 1000003 for t in range(32)
    ], "st_release_sys_b64"
    assert dst[320:384].contiguous().view(torch.int64).tolist() == [
        t * 7919 + 11 for t in range(32)
    ], "st_release_gpu_b64"
    # generic-space loads / stores / shared atomic
    assert out[480:496].tolist() == src32[:16].tolist(), "ld_generic_b32"
    assert out[496:528].contiguous().view(torch.int64).tolist() == exp64, (
        "ld_generic_b64"
    )
    assert out[528:592].tolist() == src32[:64].tolist(), "ld_generic_v4_b32"
    gd = r[GEN_DST : GEN_DST + 224]
    assert gd[0:32].tolist() == [3 * t + 7 for t in range(32)], "st_generic_b32"
    assert gd[32:96].contiguous().view(torch.int64).tolist() == [
        (t << 33) | (t + 3) for t in range(32)
    ], "st_generic_b64"
    assert gd[96:224].tolist() == [
        v for t in range(32) for v in (t, 2 * t, 3 * t, 4 * t)
    ], "st_generic_v4_b32"
    assert sorted(out[592:720].tolist()) == list(range(128)) and int(out[720]) == 128, (
        "atom_add_shared_u32"
    )
    assert r[BYTE_DST : BYTE_DST + 16].view(torch.uint8).tolist() == [
        (t * 7 + 3) & 0xFF for t in range(64)
    ], "st_b8"
    assert int(out[722]) == 0x1234 and int(out[723]) == 0xBEEF, "ld_shared_b16"
    # atomics
    cn = r[CNT : CNT + 32]
    assert int(cn[0]) == 256 and sorted(r[OLD32 : OLD32 + 256].tolist()) == list(
        range(256)
    ), "atom_add_u32"
    u = (1 << 32) + 2
    assert int(cn[2:4].contiguous().view(torch.int64)[0]) == 256 * u
    assert sorted(r[OLD64 : OLD64 + 512].contiguous().view(torch.int64).tolist()) == [
        k * u for k in range(256)
    ], "atom_add_u64"
    assert int(cn[4:6].contiguous().view(torch.int64)[0]) == 768
    assert sorted(r[OLDS64 : OLDS64 + 512].contiguous().view(torch.int64).tolist()) == [
        3 * k for k in range(256)
    ], "atom_add_sys_u64"
    assert int(cn[6]) == 256 and sorted(r[OLDR32 : OLDR32 + 256].tolist()) == list(
        range(256)
    ), "atom_add_release_gpu_u32"
    assert int(cn[7]) == 256, "red_add_u32"
    assert int(cn[8]) == -512, "red_add_s32"
    assert int(cn[9]) == 768, "red_add_release_gpu_u32"
    assert int(cn[10]) == -256, "red_add_release_sys_s32"
    xor = int(cn[12:14].contiguous().view(torch.int64)[0]) & ((1 << 64) - 1)
    assert xor == ((1 << 64) - 1) & ~0xFF, f"red_xor_release_gpu_b64 {xor:#x}"
    assert int(cn[14:16].contiguous().view(torch.int64)[0]) == -1, (
        "red_or_release_gpu_b64"
    )
    assert int(cn[16:18].contiguous().view(torch.int64)[0]) == -1, (
        "red_or_release_sys_b64"
    )
    # 1-D TMA
    assert out[320:448].tolist() == src32[:128].tolist(), "tma_load_1d"
    assert r[TMAST : TMAST + 128].tolist() == [t * 11 + 3 for t in range(128)], (
        "tma_store_1d"
    )
    # shared ops
    assert out[448:452].tolist() == [1, 2, 3, 4], "st/ld_shared_v4_u32"
    assert out[452:455].view(torch.float32).tolist() == [1.5, 2.5, 3.5], "shared v2/f32"
    assert int(out[455]) == 9
    assert out[456:460].view(torch.float32).tolist() == [1.0, 2.0, 3.0, 4.0], (
        "ld_shared_v4_f32"
    )
    assert out[460:462].tolist() == [5, 6], "st_shared_v2_u32"
    del np


def test_group_b():
    check_group_b()


# =====================================================================================================================
# group C: warp ops, numerics, stmatrix, tcgen05.ld/st
# =====================================================================================================================
IN_C = 0  # x,y,z,w floats: 4 x 128 words ; second packed bf16 pairs at 512
OUT_C = 1024  # 40 arrays x 128 words
SM_OUT_C = 8192  # smem dump (1536 words)
TMEM_OUT_C = 10240  # 32 lanes x 8 words
TOTAL_C = 12288
FNS_MASK = 0b101100


class KernelC:
    @cute.jit
    def __call__(self, buf: cute.Pointer, stream: cuda.CUstream):
        self.kernel(buf).launch(
            grid=[1, 1, 1], block=[128, 1, 1], smem=8192, stream=stream
        )

    @cute.kernel
    def kernel(self, buf: cute.Pointer):
        tidx, _, _ = cute.arch.thread_idx()
        lane = tidx % 32
        warp = tidx // 32
        smem = cute.arch.get_dyn_smem(cutlass.Int8, alignment=1024)
        x = ptx.ld_f32(buf + (IN_C + tidx))
        y = ptx.ld_f32(buf + (IN_C + 128 + tidx))
        z = ptx.ld_f32(buf + (IN_C + 256 + tidx))
        w = ptx.ld_f32(buf + (IN_C + 384 + tidx))
        pk2 = ptx.ld_b32(buf + (512 + tidx))
        o = buf + OUT_C

        # ---- numerics
        pk = ptx.cvt_rn_bf16x2_f32(
            y, x
        )  # (hi, lo): packed {x low, y high} == __float22bfloat162_rn({x, y})
        ptx.st_b32(o + (0 * 128 + tidx), pk)
        lo, hi = ptx.bf16x2_to_f32x2(pk)
        ptx.st_f32(o + (1 * 128 + tidx), lo)
        ptx.st_f32(o + (2 * 128 + tidx), hi)
        ax, ay = ptx.accumulate_bf16x2(z, w, pk)
        ptx.st_f32(o + (3 * 128 + tidx), ax)
        ptx.st_f32(o + (4 * 128 + tidx), ay)
        ptx.st_f32(o + (5 * 128 + tidx), ptx.add_rn_f32_bf16(z, pk))
        s0, s1 = ptx.fadd2_rn((x, y), (z, w))
        ptx.st_f32(o + (6 * 128 + tidx), s0)
        ptx.st_f32(o + (7 * 128 + tidx), s1)
        m0, m1 = ptx.fmul2_rn((x, y), (z, w))
        ptx.st_f32(o + (8 * 128 + tidx), m0)
        ptx.st_f32(o + (9 * 128 + tidx), m1)
        f0, f1 = ptx.ffma2_rn((x, y), (z, w), (w, x))
        ptx.st_f32(o + (10 * 128 + tidx), f0)
        ptx.st_f32(o + (11 * 128 + tidx), f1)
        ptx.st_f32(o + (12 * 128 + tidx), ptx.ex2_approx(x))
        xs = ptx.ld_f32(buf + (640 + tidx))
        ptx.st_f32(o + (35 * 128 + tidx), ptx.ex2_approx(xs))
        ptx.st_f32(o + (36 * 128 + tidx), ptx.ex2_approx_ftz(xs))
        ptx.st_f32(o + (37 * 128 + tidx), ptx.ex2_approx_ftz(x))
        ptx.st_f32(o + (13 * 128 + tidx), ptx.rcp_approx_ftz(z))
        ptx.st_b32(o + (14 * 128 + tidx), ptx.hmin2_bf16x2(pk, pk2))
        ptx.st_b32(o + (15 * 128 + tidx), ptx.hmax2_bf16x2(pk, pk2))
        ptx.st_b32(
            o + (16 * 128 + tidx),
            ptx.cvt_e4m3x4_f32(x * 20.0, y * 20.0, z * 20.0, w * 20.0),
        )

        # ---- warp-level ops
        ptx.st_b32(o + (17 * 128 + tidx), ptx.ballot((lane % 3) == 0))
        ptx.st_b32(o + (18 * 128 + tidx), Int32(ptx.all_sync(lane < 40)))
        ptx.st_b32(o + (19 * 128 + tidx), Int32(ptx.any_sync(lane == 99)))
        ptx.st_b32(o + (20 * 128 + tidx), ptx.shfl_idx(lane * 3 + 1, 5))
        ptx.st_b32(o + (21 * 128 + tidx), ptx.shfl_xor(lane, 1))
        ptx.st_b32(o + (22 * 128 + tidx), ptx.shfl_up(lane, 2))
        ptx.st_f32(
            o + (23 * 128 + tidx), ptx.shfl_idx(lane.to(cutlass.Float32) * 0.5, 7)
        )
        ptx.st_b32(o + (24 * 128 + tidx), ptx.redux_add_u32(lane + 1))
        ptx.st_b32(o + (25 * 128 + tidx), ptx.redux_add_s32(lane - 16))
        ptx.st_b32(o + (26 * 128 + tidx), ptx.redux_min_u32(lane + 7))
        ptx.st_f32(
            o + (27 * 128 + tidx), ptx.redux_max_f32(lane.to(cutlass.Float32) * 0.5)
        )
        ptx.st_b32(o + (28 * 128 + tidx), ptx.fns_b32(FNS_MASK, 0, lane % 6 + 1))
        ptx.st_b32(o + (29 * 128 + tidx), ptx.popc(tidx))
        ptx.st_b32(o + (30 * 128 + tidx), ptx.clz(tidx))
        ptx.st_b32(o + (31 * 128 + tidx), ptx.ffs(tidx << (tidx % 5)))
        ptx.st_b32(o + (32 * 128 + tidx), ptx.fns_b32(FNS_MASK, 3, 1))
        ptx.st_b32(o + (33 * 128 + tidx), ptx.fns_b32(FNS_MASK, 0, 0))
        ptx.st_b32(o + (34 * 128 + tidx), ptx.fns_b32(FNS_MASK, 2, 0))

        # ---- stmatrix: model matrices v(i, row, col) = i*64 + row*8 + col ; thread holds (row = lane/4, cols 2*(lane%4)+{0,1})
        if warp == 0:
            row = lane // 4
            col = (lane % 4) * 2
            r0 = (0 * 64 + row * 8 + col) | ((0 * 64 + row * 8 + col + 1) << 16)
            r1 = (1 * 64 + row * 8 + col) | ((1 * 64 + row * 8 + col + 1) << 16)
            r2 = (2 * 64 + row * 8 + col) | ((2 * 64 + row * 8 + col + 1) << 16)
            r3 = (3 * 64 + row * 8 + col) | ((3 * 64 + row * 8 + col + 1) << 16)
            a4 = (
                smem + ((lane // 8) * 8 + (lane % 8)) * 16
            )  # x4: 4 matrices x 8 rows x 16 B  -> [0, 512)
            ptx.stmatrix_m8n8_x4_trans_b16(a4, r0, r1, r2, r3)
            a2 = smem + (
                512 + ((lane % 16) // 8) * 8 * 16 + (lane % 8) * 16
            )  # x2 non-trans -> [512, 768)
            ptx.stmatrix_m8n8_x2_b16(a2, r0, r1)
            a3 = smem + (
                768 + ((lane % 16) // 8) * 8 * 16 + (lane % 8) * 16
            )  # x2 trans -> [768, 1024)
            ptx.stmatrix_m8n8_x2_trans_b16(a3, r0, r1)
            # b8 variants: unique bytes per (lane, byte): value = lane*4 + j
            rb0 = (
                (lane * 4)
                | ((lane * 4 + 1) << 8)
                | ((lane * 4 + 2) << 16)
                | ((lane * 4 + 3) << 24)
            )
            rb1 = (
                ((lane * 4 + 128) & 255)
                | (((lane * 4 + 129) & 255) << 8)
                | (((lane * 4 + 130) & 255) << 16)
                | (((lane * 4 + 131) & 255) << 24)
            )
            ab1 = smem + (1024 + (lane % 8) * 16)  # x1 -> [1024, 1152)
            ptx.stmatrix_m16n8_x1_trans_b8(ab1, rb0)
            ab2 = smem + (
                1152 + ((lane % 16) // 8) * 128 + (lane % 8) * 16
            )  # x2 -> [1152, 1408)
            ptx.stmatrix_m16n8_x2_trans_b8(ab2, rb0, rb1)
        ptx.bar_sync(5, 128)
        for k in cutlass.range_constexpr(3):
            ptx.st_b32(
                buf + (SM_OUT_C + k * 128 + tidx),
                ptx.ld_shared_u32(smem + (k * 512 + tidx * 4)),
            )

        # ---- tcgen05.st / tcgen05.ld round trip on a 32-column allocation (warp 0)
        if warp == 0:
            slot = smem + 4096
            cute.arch.alloc_tmem(32, slot, arch="sm_107")
            taddr = ptx.ld_shared_u32(slot)
            ptx.tmem_st_16x256b_x1(
                taddr, lane * 4 + 0, lane * 4 + 1, lane * 4 + 2, lane * 4 + 3
            )
            ptx.tmem_st_16x256b_x1(
                taddr | 0x00100000,
                lane * 4 + 1000,
                lane * 4 + 1001,
                lane * 4 + 1002,
                lane * 4 + 1003,
            )
            ptx.tmem_st_wait()
            ptx.tcgen05_fence_before_thread_sync()
            ptx.tcgen05_fence_after_thread_sync()
            l0, l1, l2, l3 = ptx.tmem_ld_16x256b_x1(taddr)
            h0, h1, h2, h3 = ptx.tmem_ld_16x256b_x1(taddr | 0x00100000)
            ptx.tmem_ld_wait()
            t = buf + (TMEM_OUT_C + lane * 8)
            ptx.st_b32(t + 0, l0)
            ptx.st_b32(t + 1, l1)
            ptx.st_b32(t + 2, l2)
            ptx.st_b32(t + 3, l3)
            ptx.st_b32(t + 4, h0)
            ptx.st_b32(t + 5, h1)
            ptx.st_b32(t + 6, h2)
            ptx.st_b32(t + 7, h3)
            ptx.st_b32(buf + (TMEM_OUT_C + 256), taddr)
            tptr = cute.arch.retrieve_tmem_ptr(cutlass.Float32, 16, slot)
            cute.arch.dealloc_tmem(tptr, 32, arch="sm_107")


def _fma_f32(a, b, c):
    """Exact single-rounded fma for float32 inputs with benign exponent ranges (double arithmetic is exact here)."""
    import numpy as np

    return (a.astype(np.float64) * b.astype(np.float64) + c.astype(np.float64)).astype(
        np.float32
    )


def check_group_c():
    import numpy as np

    rng = np.random.default_rng(1234)

    def rnd(lo, hi, n=128):
        return rng.uniform(lo, hi, n).astype(np.float32)

    def sign():
        return rng.choice(np.array([-1.0, 1.0], dtype=np.float32), 128)

    x = rnd(1, 2) * sign()
    y = rnd(1, 2) * sign()
    z = rnd(1, 2) * sign()
    w = rnd(1, 2) * sign()
    # second packed-bf16 operand of min/max: bf16 roundings of two random floats (no NaN/Inf patterns)
    pa, pb = rnd(-8, 8), rnd(-8, 8)
    pk2 = (
        torch.stack(
            [
                torch.from_numpy(pa).to(torch.bfloat16),
                torch.from_numpy(pb).to(torch.bfloat16),
            ],
            dim=1,
        )
        .contiguous()
        .view(torch.int32)
        .squeeze(1)
        .numpy()
    )
    host = torch.zeros(TOTAL_C, dtype=torch.int32)
    host[0:128] = torch.from_numpy(x.view(np.int32))
    host[128:256] = torch.from_numpy(y.view(np.int32))
    host[256:384] = torch.from_numpy(z.view(np.int32))
    host[384:512] = torch.from_numpy(w.view(np.int32))
    host[512:640] = torch.from_numpy(pk2)
    tt = np.arange(128)
    xs_in = np.where(tt < 8, -126.5 - 3.0 * tt, -20.0 + 0.3 * (tt - 8)).astype(
        np.float32
    )
    host[640:768] = torch.from_numpy(xs_in.view(np.int32))
    buf = host.cuda()
    compile_and_run(KernelC(), gptr(buf), tag="c")
    r = buf.cpu()
    out = r[OUT_C : OUT_C + 40 * 128].view(40, 128)

    def f32(k):
        return out[k].contiguous().view(torch.float32).numpy()

    def i32(k):
        return out[k].numpy()

    # --- cvt.rn.bf16x2.f32: (hi=y -> bits 31:16, lo=x -> bits 15:0)
    xt, yt = torch.from_numpy(x), torch.from_numpy(y)
    exp_pk = (
        torch.stack([xt.to(torch.bfloat16), yt.to(torch.bfloat16)], dim=1)
        .contiguous()
        .view(torch.int32)
        .squeeze(1)
        .numpy()
    )
    assert np.array_equal(i32(0), exp_pk), "cvt_rn_bf16x2_f32"
    elo = xt.to(torch.bfloat16).to(torch.float32).numpy()
    ehi = yt.to(torch.bfloat16).to(torch.float32).numpy()
    assert np.array_equal(f32(1), elo) and np.array_equal(f32(2), ehi), (
        "bf16x2_to_f32x2"
    )
    assert np.array_equal(f32(3), (z + elo).astype(np.float32)), "accumulate_bf16x2.x"
    assert np.array_equal(f32(4), (w + ehi).astype(np.float32)), "accumulate_bf16x2.y"
    assert np.array_equal(f32(5), (z + elo).astype(np.float32)), "add_rn_f32_bf16"
    assert np.array_equal(f32(6), x + z) and np.array_equal(f32(7), y + w), "fadd2_rn"
    assert np.array_equal(f32(8), x * z) and np.array_equal(f32(9), y * w), "fmul2_rn"
    assert np.array_equal(f32(10), _fma_f32(x, z, w)) and np.array_equal(
        f32(11), _fma_f32(y, w, x)
    ), "ffma2_rn"
    ex2 = f32(12)
    ref = np.exp2(x.astype(np.float64))
    assert np.max(np.abs(ex2 - ref) / ref) < 2e-6, (
        f"ex2.approx.ftz rel err {np.max(np.abs(ex2 - ref) / ref)}"
    )
    # DeepGEMM's __expf compiles to the NON-ftz ex2.approx.f32: subnormal results must survive, the ftz variant flushes them
    e_nf, e_ftz = f32(35), f32(36)
    ref_s = np.exp2(xs_in.astype(np.float64))
    normal = ref_s >= 2.0**-126
    assert np.max(np.abs(e_nf[normal] - ref_s[normal]) / ref_s[normal]) < 2e-6, (
        "ex2_approx (normal range)"
    )
    sub = ~normal
    assert np.all(e_nf[sub][ref_s[sub] > 2.0**-147] > 0), (
        "ex2_approx must produce subnormal results"
    )
    assert np.all(e_ftz[sub] == 0.0), "ex2_approx_ftz must flush subnormal results"
    assert np.max(np.abs(f32(37) - ref) / ref) < 2e-6, "ex2_approx_ftz (x)"
    print(
        f"  [info] ex2 of {xs_in[0]} : non-ftz -> {e_nf[0]:.3e} (exact {ref_s[0]:.3e}); ftz -> {e_ftz[0]:.3e}"
    )
    rcp = f32(13)
    refr = 1.0 / z.astype(np.float64)
    assert np.max(np.abs(rcp - refr) / np.abs(refr)) < 2e-6, "rcp.approx.ftz"
    pk_b = torch.from_numpy(exp_pk).view(torch.bfloat16)
    pk2_b = torch.from_numpy(pk2).view(torch.bfloat16)
    assert np.array_equal(
        i32(14), torch.minimum(pk_b, pk2_b).contiguous().view(torch.int32).numpy()
    ), "hmin2_bf16x2"
    assert np.array_equal(
        i32(15), torch.maximum(pk_b, pk2_b).contiguous().view(torch.int32).numpy()
    ), "hmax2_bf16x2"
    q = (
        torch.stack(
            [torch.from_numpy(v * np.float32(20.0)) for v in (x, y, z, w)], dim=1
        )
        .to(torch.float8_e4m3fn)
        .view(torch.uint8)
        .to(torch.int64)
    )
    exp_e4 = q[:, 0] | (q[:, 1] << 8) | (q[:, 2] << 16) | (q[:, 3] << 24)
    got_e4 = torch.from_numpy(i32(16).astype(np.int64) & 0xFFFFFFFF)
    assert torch.equal(got_e4, exp_e4), "cvt_e4m3x4_f32"
    # --- warp ops (per warp the same pattern)
    lanes = np.arange(128) % 32
    mask3 = sum(1 << i for i in range(32) if i % 3 == 0)
    assert np.all((i32(17).astype(np.int64) & 0xFFFFFFFF) == mask3), "ballot"
    assert np.all(i32(18) == 1) and np.all(i32(19) == 0), "all_sync/any_sync"
    assert np.all(i32(20) == 16), "shfl_idx"
    assert np.array_equal(i32(21), lanes ^ 1), "shfl_xor"
    assert np.array_equal(i32(22), np.where(lanes >= 2, lanes - 2, lanes)), "shfl_up"
    assert np.all(f32(23) == 3.5), "shfl_idx (float)"
    assert np.all(i32(24) == 528) and np.all(i32(25) == -16) and np.all(i32(26) == 7), (
        "redux add/min"
    )
    assert np.all(f32(27) == 15.5), "redux_max_f32"
    kth = {1: 2, 2: 3, 3: 5}
    exp_fns = np.array(
        [kth.get(int(l) % 6 + 1, 0xFFFFFFFF) for l in lanes], dtype=np.int64
    )
    assert np.array_equal(i32(28).astype(np.int64) & 0xFFFFFFFF, exp_fns), (
        f"fns_b32: {i32(28)[:8]}"
    )
    t = np.arange(128)
    assert np.array_equal(i32(29), np.array([bin(int(v)).count("1") for v in t])), (
        "popc"
    )
    assert np.array_equal(i32(30), np.array([32 - int(v).bit_length() for v in t])), (
        "clz"
    )

    def ffs_ref(v):
        v &= 0xFFFFFFFF
        return 0 if v == 0 else (v & -v).bit_length()

    assert np.array_equal(
        i32(31), np.array([ffs_ref(int(v) << (int(v) % 5)) for v in t])
    ), "ffs"
    print(
        f"  [info] fns(mask=0b101100, base=3, offset=1) = {int(i32(32)[0]) & 0xFFFFFFFF:#x} ; "
        f"fns(base=0, offset=0) = {int(i32(33)[0]) & 0xFFFFFFFF:#x} ; fns(base=2, offset=0) = {int(i32(34)[0]) & 0xFFFFFFFF:#x}"
    )
    # --- stmatrix (b16): model matrices v(i, row, col) = i*64 + row*8 + col
    smd = (
        r[SM_OUT_C : SM_OUT_C + 3 * 128]
        .contiguous()
        .view(torch.int16)
        .numpy()
        .reshape(-1)
    )  # 1536 B = 768 b16

    def v(i, row, col):
        return i * 64 + row * 8 + col

    exp_x4 = np.zeros((4, 8, 8), dtype=np.int16)  # [matrix][mem row t][mem col s]
    for i in range(4):
        for tt in range(8):
            for ss in range(8):
                exp_x4[i, tt, ss] = v(
                    i, ss, tt
                )  # .trans: memory row t holds register-matrix column t
    assert np.array_equal(smd[0:256].reshape(4, 8, 8), exp_x4), (
        f"stmatrix x4 trans b16:\n{smd[0:64].reshape(8, 8)}"
    )
    exp_n = np.zeros((2, 8, 8), dtype=np.int16)
    exp_t = np.zeros((2, 8, 8), dtype=np.int16)
    for i in range(2):
        for tt in range(8):
            for ss in range(8):
                exp_n[i, tt, ss] = v(i, tt, ss)
                exp_t[i, tt, ss] = v(i, ss, tt)
    assert np.array_equal(smd[256:384].reshape(2, 8, 8), exp_n), (
        "stmatrix x2 non-trans b16"
    )
    assert np.array_equal(smd[384:512].reshape(2, 8, 8), exp_t), "stmatrix x2 trans b16"
    print(
        "  [ok] stmatrix b16 layouts match the documented fragment model (trans: mem[t][s] = reg(row=s, col=t))"
    )
    # --- stmatrix (b8, m16n8): a permutation of the 128 (x1) / 2 x 128 (x2) register bytes; record the byte mapping
    sm_b = r[SM_OUT_C : SM_OUT_C + 3 * 128].contiguous().view(torch.uint8).numpy()
    x1 = sm_b[1024:1152]
    assert sorted(x1.tolist()) == list(range(128)), (
        "stmatrix m16n8 x1 trans b8: not a permutation"
    )
    x2a, x2b = sm_b[1152:1280], sm_b[1280:1408]
    assert sorted(x2a.tolist()) == list(range(128)), (
        "stmatrix m16n8 x2 trans b8 matrix 0"
    )
    assert sorted(x2b.tolist()) == [(i + 128) & 255 for i in range(128)] or sorted(
        x2b.tolist()
    ) == sorted(((np.arange(128) * 1 + 128) & 255).tolist()), (
        "stmatrix m16n8 x2 trans b8 matrix 1"
    )
    pos = {
        int(val): idx for idx, val in enumerate(x1.tolist())
    }  # value = lane*4 + j  ->  byte offset in the 8 x 16 B matrix
    mapping = [[pos[4 * ln + j] for j in range(4)] for ln in range(8)]
    print(
        f"  [info] stmatrix m16n8.x1.trans.b8: byte offset (row*16+col) of (lane, byte j) for lanes 0..7: {mapping}"
    )
    # --- tcgen05.st / ld
    tm = r[TMEM_OUT_C : TMEM_OUT_C + 256].view(32, 8).numpy()
    lanes32 = np.arange(32)
    exp_lo = np.stack([lanes32 * 4 + j for j in range(4)], axis=1)
    exp_hi = np.stack([lanes32 * 4 + 1000 + j for j in range(4)], axis=1)
    assert np.array_equal(tm[:, 0:4], exp_lo), "tcgen05.st/ld lane block 0"
    assert np.array_equal(tm[:, 4:8], exp_hi), "tcgen05.st/ld lane block 1 (+16 lanes)"
    print(
        f"  [info] TMEM base address of a 32-column allocation = {int(r[TMEM_OUT_C + 256]):#x}"
    )


def test_group_c():
    check_group_c()


# =====================================================================================================================
# group D: usage patterns of the role modules inside @cute.jit roles: Mbar views from a frozen dataclass, producer/consumer pipeline with
# loop-carried stage/phase (advance_stage), Mbar passed to a jit helper and waited inside a staged `if`
# =====================================================================================================================
NSTAGES = 4
OFF_FULL, OFF_EMPTY, OFF_DATA = 4096, 4160, 2048


@dataclass(frozen=True, eq=False, init=False)
class PipeBars:
    """Frozen-dataclass view (the DSL treats frozen dataclasses as READ-ONLY inside staged regions): barrier handles at runtime stage index."""

    def __init__(self, base):
        object.__setattr__(self, "base", base)

    def full(self, stage):
        return ptx.Mbar(self.base + (OFF_FULL + stage * 8))

    def empty(self, stage):
        return ptx.Mbar(self.base + (OFF_EMPTY + stage * 8))


@cute.jit
def _pipe_producer(bars: PipeBars, data: cute.Pointer, n_iters: Int32):
    stage = Int32(0)
    phase = Int32(0)
    i = Int32(0)
    while i < n_iters:
        bars.empty(stage).wait(phase ^ 1)
        ptx.st_shared_u32(data + stage * 4, i * 3 + 1)
        bars.full(stage).arrive()
        stage, phase = ptx.advance_stage(stage, phase, NSTAGES)
        i = i + 1


@cute.jit
def _pipe_consumer(
    bars: PipeBars, data: cute.Pointer, out: cute.Pointer, n_iters: Int32
):
    stage = Int32(0)
    phase = Int32(0)
    acc = Int32(0)
    i = Int32(0)
    while i < n_iters:
        bars.full(stage).wait(phase)
        acc = acc + ptx.ld_shared_u32(data + stage * 4) * (i + 1)
        bars.empty(stage).arrive()
        stage, phase = ptx.advance_stage(stage, phase, NSTAGES)
        i = i + 1
    ptx.st_b32(out, acc)


@cute.jit
def _wait_when(mb: ptx.Mbar, phase: Int32, cond: Boolean):
    if cond:
        mb.wait(phase)


class KernelD:
    @cute.jit
    def __call__(self, out: cute.Pointer, n_iters: Int32, stream: cuda.CUstream):
        self.kernel(out, n_iters).launch(
            grid=[1, 1, 1], block=[128, 1, 1], smem=8192, stream=stream
        )

    @cute.kernel
    def kernel(self, out: cute.Pointer, n_iters: Int32):
        tidx, _, _ = cute.arch.thread_idx()
        warp = tidx // 32
        lane = tidx % 32
        smem = cute.arch.get_dyn_smem(cutlass.Int8, alignment=1024)
        bars = PipeBars(smem)
        if tidx == 0:
            for st in cutlass.range_constexpr(NSTAGES):
                bars.full(st).init(1)
                bars.empty(st).init(1)
        ptx.fence_barrier_init()
        ptx.bar_sync(5, 128)
        if warp == 0:
            if lane == 0:
                _pipe_producer(bars, smem + OFF_DATA, n_iters)
        if warp == 1:
            if lane == 0:
                _pipe_consumer(bars, smem + OFF_DATA, out, n_iters)
        # an Mbar handed to a jit helper and waited inside a staged `if`: complete a scratch barrier first, then wait in warps 2-3 only
        scratch = ptx.Mbar(smem + 4224)
        if tidx == 64:
            scratch.init(1)
            scratch.arrive()
        ptx.fence_barrier_init()
        ptx.bar_sync(5, 128)
        _wait_when(scratch, 0, warp >= 2)
        if tidx == 64:
            ptx.st_b32(out + 1, 77)


def check_group_d():
    out = torch.zeros(8, dtype=torch.int32, device="cuda")
    n = 100
    compile_and_run_args = (KernelD(), gptr(out), Int32(n))
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    compiled = cute.compile(compile_and_run_args[0], gptr(out), Int32(n), stream)
    compiled(gptr(out), Int32(n), stream)
    torch.cuda.synchronize()
    exp = sum((3 * i + 1) * (i + 1) for i in range(n))
    got = int(out[0])
    assert got == exp, f"pipeline accumulator {got} != {exp}"
    assert int(out[1]) == 77, (
        "Mbar waited inside a staged if (jit helper) did not complete"
    )
    # a different trip count with the SAME compiled kernel (n is a runtime value) must also work and not hang
    out.zero_()
    compiled(gptr(out), Int32(37), stream)
    torch.cuda.synchronize()
    assert int(out[0]) == sum((3 * i + 1) * (i + 1) for i in range(37))
    print(
        f"  [ok] producer/consumer pipeline over {NSTAGES} stages x {n} iterations with loop-carried (stage, phase): acc={got}"
    )


def test_group_d():
    check_group_d()


CONTRACT_PTX_NAMES = [
    "smid",
    "gridid",
    "lane_id",
    "clock64",
    "globaltimer",
    "elect_sync",
    "ballot",
    "all_sync",
    "any_sync",
    "shfl_idx",
    "shfl_xor",
    "shfl_up",
    "redux_add_s32",
    "redux_add_u32",
    "redux_min_u32",
    "redux_max_f32",
    "fns_b32",
    "popc",
    "clz",
    "ffs",
    "nanosleep",
    "trap",
    "advance_stage",
    "bar_sync",
    "bar_sync_unaligned",
    "bar_arrive",
    "fence_acq_rel_cta",
    "fence_proxy_async_shared_cta",
    "fence_barrier_init",
    "tcgen05_fence_before_thread_sync",
    "tcgen05_fence_after_thread_sync",
    "cluster_arrive_relaxed",
    "cluster_arrive",
    "cluster_wait",
    "cluster_sync_relaxed",
    "cluster_sync",
    "Mbar",
    "mapa_shared",
    "st_shared_zero",
    "ld_shared_u32",
    "ld_shared_f32",
    "ld_shared_v2_f32",
    "ld_shared_v4_f32",
    "ld_shared_v4_u32",
    "ld_shared_b16",
    "st_shared_u32",
    "st_shared_f32",
    "st_shared_v2_f32",
    "st_shared_v2_u32",
    "st_shared_v4_u32",
    "st_async_v4",
    "ld_b32",
    "ld_b64",
    "ld_f32",
    "ld_v4_b32",
    "ld_volatile_b32",
    "ld_volatile_b64",
    "ld_acquire_gpu_b32",
    "ld_acquire_gpu_b64",
    "ld_acquire_sys_b32",
    "ld_acquire_sys_s32",
    "ld_acquire_sys_b64",
    "st_b32",
    "st_b64",
    "st_f32",
    "st_v4_b32",
    "st_b8",
    "st_release_sys_b64",
    "st_release_gpu_b64",
    "atom_add_u32",
    "atom_add_u64",
    "atom_add_sys_u64",
    "atom_add_release_gpu_u32",
    "red_add_u32",
    "red_add_s32",
    "red_add_release_gpu_u32",
    "red_add_release_sys_s32",
    "red_xor_release_gpu_b64",
    "red_or_release_gpu_b64",
    "red_or_release_sys_b64",
    "ld_generic_b32",
    "ld_generic_b64",
    "ld_generic_v4_b32",
    "st_generic_b32",
    "st_generic_b64",
    "st_generic_v4_b32",
    "atom_add_shared_u32",
    "syncwarp",
    "tmem_alloc",
    "tmem_dealloc",
    "tma_load_1d",
    "tma_store_1d",
    "tma_store_commit",
    "tma_store_wait",
    "EVICT_NORMAL",
    "EVICT_FIRST",
    "cvt_rn_bf16x2_f32",
    "add_rn_f32_bf16",
    "accumulate_bf16x2",
    "fadd2_rn",
    "fmul2_rn",
    "ffma2_rn",
    "ex2_approx",
    "ex2_approx_ftz",
    "rcp_approx_ftz",
    "hmin2_bf16x2",
    "hmax2_bf16x2",
    "bf16x2_to_f32x2",
    "cvt_e4m3x4_f32",
    "tmem_ld_16x256b_x1",
    "tmem_ld_wait",
    "tmem_st_16x256b_x1",
    "tmem_st_wait",
    "stmatrix_m16n8_x1_trans_b8",
    "stmatrix_m16n8_x2_trans_b8",
    "stmatrix_m8n8_x4_trans_b16",
    "stmatrix_m8n8_x2_trans_b16",
    "stmatrix_m8n8_x2_b16",
]
CONTRACT_COMM_NAMES = [
    "SymBuffer",
    "SymBufferHost",
    "grid_sync",
    "nvlink_barrier",
    "make_sym_buffer_host",
]
MBAR_METHODS = [
    "init",
    "wait",
    "try_wait",
    "arrive",
    "arrive_count",
    "arrive_expect_tx",
    "expect_tx",
    "arrive_remote",
    "arrive_expect_tx_remote",
    "cluster_addr",
    "addr",
    "addr_leader",
]


def check_names():
    """Every name promised by DESIGN.md section 13 exists (plus the extras documented in the ptx.py header)."""
    missing = [n for n in CONTRACT_PTX_NAMES if not hasattr(ptx, n)]
    assert not missing, f"missing in ptx.py: {missing}"
    missing = [n for n in MBAR_METHODS if not hasattr(ptx.Mbar, n)]
    assert not missing, f"missing Mbar members: {missing}"
    assert (
        ptx.EVICT_NORMAL == 0x1000000000000000 and ptx.EVICT_FIRST == 0x12F0000000000000
    )
    comm = _MODS["comm"]
    missing = [n for n in CONTRACT_COMM_NAMES if not hasattr(comm, n)]
    assert not missing, f"missing in comm.py: {missing}"
    print(
        f"  [ok] {len(CONTRACT_PTX_NAMES)} ptx.py names + Mbar members + comm.py names present"
    )


def test_contract_names():
    check_names()


if __name__ == "__main__":
    groups = sys.argv[1:] or ["a"]
    for g in groups:
        print(f"== group {g}")
        globals().get(f"check_group_{g}", globals().get(f"check_{g}"))()
        print(f"== group {g} PASS")
