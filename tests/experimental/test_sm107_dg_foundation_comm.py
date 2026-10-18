# SPDX-License-Identifier: Apache-2.0
# Port of DeepGEMM (MIT License, Copyright (c) 2025 DeepSeek; see licenses/LICENSE.deepgemm.txt) comm/barrier.cuh + layout/sym_buffer.cuh.
"""GPU tests of ``kernel/comm.py``: SymBuffer by-value peer table, software grid sync, NVLink barrier.

Run:  ``python3 tests/experimental/test_sm107_dg_foundation_comm.py <group>``  with groups
  ``sym``        single GPU: SymBuffer.map with a 32-entry by-value table (+ R=1 identity), base/rank fields
  ``gridsync``   single GPU: grid_sync with 212 CTAs, N iterations (default 100000)
  ``nvl``        2 GPUs (2 processes): remote st/ld/atom.sys/red.sys round trips + the NVLink barrier algorithm, N iterations
Env: SM107_DG_ITERS (default 100000), SM107_DG_DUMP_DIR.
"""

import importlib.util
import os
import sys
import types
from dataclasses import dataclass

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
from cutlass.cutlass_dsl import Int32, Int64  # noqa: E402


def _load_kernel_modules():
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
        if full not in sys.modules:
            spec = importlib.util.spec_from_file_location(
                full, os.path.join(kdir, f"{name}.py")
            )
            m = importlib.util.module_from_spec(spec)
            sys.modules[full] = m
            spec.loader.exec_module(m)
        mods[name] = sys.modules[full]
    return mods


_MODS = _load_kernel_modules()
ptx = _MODS["ptx"]
comm = _MODS["comm"]
ITERS = int(os.environ.get("SM107_DG_ITERS", "100000"))


def compile_and_run(kernel, *args, stream=None):
    stream = stream or cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    compiled = cute.compile(kernel, *args, stream)
    compiled(*args, stream)
    torch.cuda.synchronize()
    return compiled, stream


def gptr(t, dtype=cutlass.Int32, align=16):
    return make_ptr(dtype, t.data_ptr(), AddressSpace.gmem, assumed_align=align)


# =====================================================================================================================
# sym: SymBuffer by-value table
# =====================================================================================================================
@dataclass(frozen=True, eq=False)
class FakeWs:
    """Minimal workspace view over an Int64 base address (same offsets as MegaMoESignals).

    DSL rule (verified): plain Python objects read inside a staged ``if``/``while`` region must be FROZEN dataclasses (or implement the
    DSL value protocol); a plain class raises TYPE_DYNAMIC_EXPR_UNSUPPORTED ("a value carried through this `if` is a plain Python value").
    """

    base: object

    def grid_sync_count(self, idx):
        return self.base + (0 + 4 * idx)

    def nvl_barrier_counter(self):
        return self.base + 16

    def nvl_barrier_signal(self, phase):
        return self.base + (20 + 4 * phase)


class KernelSym:
    def __init__(self, num_ranks):
        self.num_ranks = num_ranks

    @cute.jit
    def __call__(
        self, sym_host: comm.SymBufferHost, out: cute.Pointer, stream: cuda.CUstream
    ):
        sym = sym_host.make_device_object()
        self.kernel(sym, out).launch(grid=[1, 1, 1], block=[128, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, sym: comm.SymBuffer, out: cute.Pointer):
        tidx, _, _ = cute.arch.thread_idx()
        R = self.num_ranks
        if tidx == 0:
            ptx.st_b64(out + 0, sym.base)  # out viewed as int32 pairs: slot 0..1 = base
            ptx.st_b32(out + 2, sym.rank_idx)
            ptx.st_b32(out + 3, Int32(R))
        # Pointer map: write 1000+t into rank t's buffer at byte offset 64
        if tidx < R:
            p = cute.make_ptr(
                cutlass.Int32, sym.base + 64, AddressSpace.gmem, assumed_align=16
            )
            q = sym.map(p, tidx)
            ptx.st_b32(q, 1000 + tidx)
            # Int64 address variant: byte offset 128
            a = sym.map(sym.base + 128, tidx)
            ptx.st_b32(a, 2000 + tidx)


def check_sym():
    for R in (32, 72, 1):
        bufs = [torch.zeros(64, dtype=torch.int32, device="cuda") for _ in range(R)]
        rank = 3 % R
        base = bufs[rank].data_ptr()
        host = comm.make_sym_buffer_host(base, [b.data_ptr() for b in bufs], rank)
        out = torch.zeros(16, dtype=torch.int32, device="cuda")
        compile_and_run(KernelSym(R), host, gptr(out))
        o = out.cpu()
        got_base = int(o[0]) & 0xFFFFFFFF | ((int(o[1]) & 0xFFFFFFFF) << 32)
        assert got_base == base, f"R={R}: base {got_base:#x} != {base:#x}"
        assert int(o[2]) == rank and int(o[3]) == R
        if R == 1:
            # identity: tidx < 1 -> only rank 0, which is the local buffer
            assert int(bufs[0][16]) == 1000 and int(bufs[0][32]) == 2000, (
                "identity map failed"
            )
        else:
            for t in range(R):
                assert int(bufs[t][16]) == 1000 + t, (
                    f"R={R}: peer {t}: [16]={int(bufs[t][16])}"
                )
                assert int(bufs[t][32]) == 2000 + t, (
                    f"R={R}: peer {t}: [32]={int(bufs[t][32])}"
                )
        print(f"  [ok] SymBuffer R={R}")


def check_sym_dynamic():
    """A compiled kernel must follow the RUNTIME values of the SymBufferHost (base/offsets/rank are not baked in at compile time)."""
    R = 32
    bufs1 = [torch.zeros(64, dtype=torch.int32, device="cuda") for _ in range(R)]
    host1 = comm.make_sym_buffer_host(
        bufs1[3].data_ptr(), [b.data_ptr() for b in bufs1], 3
    )
    out = torch.zeros(16, dtype=torch.int32, device="cuda")
    compiled, stream = compile_and_run(KernelSym(R), host1, gptr(out))
    assert all(int(bufs1[t][16]) == 1000 + t for t in range(R))
    # new buffers (different addresses), different own rank, permuted peer order
    bufs2 = [torch.zeros(64, dtype=torch.int32, device="cuda") for _ in range(R)]
    perm = list(range(R))[::-1]
    peers2 = [bufs2[perm[i]].data_ptr() for i in range(R)]
    rank2 = 7
    host2 = comm.make_sym_buffer_host(peers2[rank2], peers2, rank2)
    for b in bufs1:
        b.zero_()
    out.zero_()
    compiled(host2, gptr(out), stream)
    torch.cuda.synchronize()
    o = out.cpu()
    assert int(o[2]) == rank2, f"rank_idx not dynamic: {int(o[2])}"
    assert all(int(b[16]) == 0 for b in bufs1), (
        "old buffers were written: peer table baked into the compiled kernel"
    )
    for i in range(R):
        assert (
            int(bufs2[perm[i]][16]) == 1000 + i and int(bufs2[perm[i]][32]) == 2000 + i
        ), f"peer {i} -> buffer {perm[i]}"
    print(
        "  [ok] compiled kernel follows runtime SymBufferHost values (base/offsets/rank)"
    )


def test_sym():
    check_sym()
    check_sym_dynamic()


def check_tvmffi():
    """TVM-FFI path (DESIGN.md section 2): dataclass SymBufferHost + pointer + torch stream through an `EnableTVMFFI` compile."""
    import tvm_ffi  # noqa: F401 - availability guard of the TVM-FFI path
    from cutlass.cute.runtime import make_fake_stream

    R = 32
    k = KernelSym(R)
    fake_host = comm.SymBufferHost(
        Int64(0), tuple(Int64(0) for _ in range(R)), Int32(0), R
    )
    fake_out = make_ptr(cutlass.Int32, 0, AddressSpace.gmem, assumed_align=16)
    compiled = cute.compile[cute.EnableTVMFFI(True)](
        k, fake_host, fake_out, make_fake_stream()
    )
    bufs = [torch.zeros(64, dtype=torch.int32, device="cuda") for _ in range(R)]
    rank = 5
    host = comm.make_sym_buffer_host(
        bufs[rank].data_ptr(), [b.data_ptr() for b in bufs], rank
    )
    out = torch.zeros(16, dtype=torch.int32, device="cuda")
    errors = {}
    ok = False
    sint = torch.cuda.current_stream().cuda_stream
    for name, call in (
        (
            "host dataclass (python ints) + int pointer + int stream",
            lambda: compiled(host, out.data_ptr(), sint),
        ),
        (
            "host dataclass (python ints) + torch tensor + int stream",
            lambda: compiled(host, out, sint),
        ),
    ):
        try:
            call()
            torch.cuda.synchronize()
            o = out.cpu()
            good = int(o[2]) == rank and all(
                int(bufs[t][16]) == 1000 + t for t in range(R)
            )
            print(
                f"  [tvm-ffi] {name}: {'OK' if good else 'WRONG result'} (rank_idx={int(o[2])})"
            )
            ok = ok or good
            if good:
                break
        except Exception as e:  # noqa: BLE001
            errors[name] = str(e).strip().splitlines()[0][:300]
            print(f"  [tvm-ffi] {name}: FAILED: {errors[name]}")
    assert ok, f"no TVM-FFI calling convention worked: {errors}"
    # dynamic values through TVM-FFI: call again with other buffers / rank
    bufs2 = [torch.zeros(64, dtype=torch.int32, device="cuda") for _ in range(R)]
    host2 = comm.make_sym_buffer_host(
        bufs2[9].data_ptr(), [b.data_ptr() for b in bufs2], 9
    )
    out.zero_()
    compiled(host2, out.data_ptr(), sint)
    torch.cuda.synchronize()
    assert int(out[2].item()) == 9 and all(
        int(bufs2[t][16]) == 1000 + t for t in range(R)
    ), "TVM-FFI call with new SymBufferHost values"
    print(
        "  [tvm-ffi] second call with different base/offsets/rank follows the new values"
    )


def test_tvmffi():
    check_tvmffi()


# =====================================================================================================================
# gridsync: software grid sync, 212 CTAs
# =====================================================================================================================
class KernelGridSync:
    def __init__(self, num_sms, iters, num_threads=128):
        self.num_sms, self.iters, self.num_threads = num_sms, iters, num_threads

    @cute.jit
    def __call__(self, ws_base: Int64, out: cute.Pointer, stream: cuda.CUstream):
        self.kernel(ws_base, out).launch(
            grid=[self.num_sms, 1, 1],
            block=[self.num_threads, 1, 1],
            cluster=[2, 1, 1],
            smem=216 * 1024,
            stream=stream,
        )

    @cute.kernel
    def kernel(self, ws_base: Int64, out: cute.Pointer):
        tidx, _, _ = cute.arch.thread_idx()
        sm_idx, _, _ = cute.arch.block_idx()
        ws = FakeWs(ws_base)
        # per-iteration check of the sync property: before the sync every CTA bumps a shared counter (atomic), after the sync
        # every CTA must observe counter >= (i+1)*num_sms
        counter = cute.make_ptr(
            cutlass.Int32, ws_base + 64, AddressSpace.gmem, assumed_align=16
        )
        errors = Int32(0)
        for i in cutlass.range(self.iters, unroll=1):
            if tidx == 0:
                ptx.red_add_u32(counter, Int32(1))
            comm.grid_sync(ws, 1, self.num_sms, sm_idx, tidx, (0, self.num_threads))
            if tidx == 0:
                seen = ptx.ld_acquire_gpu_b32(counter)
                if seen < Int32((i + 1) * self.num_sms):
                    errors = errors + Int32(1)
            comm.grid_sync(ws, 1, self.num_sms, sm_idx, tidx, (0, self.num_threads))
        if tidx == 0:
            ptx.red_add_u32(out, errors)


def check_gridsync():
    props = torch.cuda.get_device_properties(0)
    n = props.multi_processor_count
    ws = torch.zeros(1024, dtype=torch.int32, device="cuda")
    out = torch.zeros(4, dtype=torch.int32, device="cuda")
    import time

    k = KernelGridSync(n, ITERS)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    t0 = time.time()
    compiled = cute.compile(k, int(ws.data_ptr()), gptr(out), stream)
    t1 = time.time()
    compiled(int(ws.data_ptr()), gptr(out), stream)
    torch.cuda.synchronize()
    t2 = time.time()
    errs = int(out[0])
    print(
        f"  [grid_sync] sms={n} iters={ITERS} compile={t1 - t0:.1f}s run={t2 - t1:.1f}s errors={errs} counter={int(ws[16])}"
    )
    assert errs == 0, f"grid_sync property violated {errs} times"
    assert int(ws[16]) == ITERS * n


def test_gridsync():
    check_gridsync()


# =====================================================================================================================
# nvl: 2 ranks (2 processes, GPUs 0 and 1): remote st/ld/atom.sys/red.sys round trips + NVLink barrier algorithm
# =====================================================================================================================
NVL_BYTES = 1 << 16
OFF_CNT = 128  # u64 cnt[src_rank]: remote atomic counters written by the peer
OFF_W32, OFF_WV4, OFF_WREL, OFF_WATOM, OFF_WRED = 1024, 1040, 1056, 1064, 1072
OFF_PAT = 2048  # 64 words: rank * 1000 + i (host initialised)
OFF_PATQ = 2304  # 8 bytes: rank-specific u64
OFF_RES = 4096  # pulled results (words)
OFF_G32, OFF_G64, OFF_GV4 = (
    1088,
    1096,
    1104,
)  # remote GENERIC-space store targets (written by the peer)
OFF_RES2 = 4608  # results of GENERIC-space pulls (words)
OFF_ERR = 8192  # error counter word


class KernelNvl:
    def __init__(self, num_ranks, num_sms):
        self.R, self.num_sms = num_ranks, num_sms

    @cute.jit
    def __call__(
        self, sym_host: comm.SymBufferHost, iters: Int32, stream: cuda.CUstream
    ):
        sym = sym_host.make_device_object()
        self.kernel(sym, iters).launch(
            grid=[self.num_sms, 1, 1],
            block=[128, 1, 1],
            cluster=[2, 1, 1],
            smem=216 * 1024,
            stream=stream,
        )

    @cute.kernel
    def kernel(self, sym: comm.SymBuffer, iters: Int32):
        tidx, _, _ = cute.arch.thread_idx()
        sm_idx, _, _ = cute.arch.block_idx()
        ws = FakeWs(sym.base)
        me = sym.rank_idx
        peer = (me + 1) % self.R
        N = self.num_sms
        # ---------------- phase 1: remote store / atomic / reduction round trips (pushes), then barrier, then pulls
        if tidx == 0:
            if sm_idx == 0:
                ptx.st_b32(sym.map(sym.base + OFF_W32, peer), 0x1000 + me)
                ptx.st_v4_b32(
                    sym.map(sym.base + OFF_WV4, peer),
                    me * 100 + 1,
                    me * 100 + 2,
                    me * 100 + 3,
                    me * 100 + 4,
                )
                ptx.st_release_sys_b64(
                    sym.map(sym.base + OFF_WREL, peer), (Int64(1) << 40) + Int64(me)
                )
            ptx.atom_add_sys_u64(sym.map(sym.base + OFF_WATOM, peer), 5)
            ptx.red_add_release_sys_s32(sym.map(sym.base + OFF_WRED, peer), -1)
            if sm_idx == 0:
                # DeepGEMM compiles the remote stores through SymBuffer::map pointers to GENERIC st.b32/st.b64/st.v4 (SASS ST.E*)
                ptx.st_generic_b32(sym.map(sym.base + OFF_G32, peer), 0x2000 + me)
                ptx.st_generic_b64(
                    sym.map(sym.base + OFF_G64, peer), (Int64(7) << 40) + Int64(me)
                )
                ptx.st_generic_v4_b32(
                    sym.map(sym.base + OFF_GV4, peer),
                    me * 10 + 1,
                    me * 10 + 2,
                    me * 10 + 3,
                    me * 10 + 4,
                )
        comm.grid_sync(ws, 0, N, sm_idx, tidx, (0, 128))
        comm.nvlink_barrier(
            ws,
            sym,
            self.R,
            N,
            128,
            0,
            1,
            sm_idx,
            tidx,
            (0, 128),
            prologue=False,
            epilogue=True,
        )
        if sm_idx == 0:
            if tidx < 16:
                a, b, c, d = ptx.ld_v4_b32(
                    sym.map(sym.base + (OFF_PAT + 16 * tidx), peer)
                )
                r = sym.base + (OFF_RES * 4 + 16 * tidx)
                ptx.st_v4_b32(r, a, b, c, d)
            if tidx == 16:
                q = sym.map(sym.base + OFF_PATQ, peer)
                ptx.st_b64(sym.base + (OFF_RES * 4 + 256), ptx.ld_acquire_sys_b64(q))
                ptx.st_b64(sym.base + (OFF_RES * 4 + 264), ptx.ld_volatile_b64(q))
                ptx.st_b32(
                    sym.base + (OFF_RES * 4 + 272),
                    ptx.ld_acquire_sys_s32(sym.map(sym.base + OFF_PAT, peer)),
                )
                ptx.st_b32(
                    sym.base + (OFF_RES * 4 + 276),
                    ptx.ld_acquire_sys_b32(sym.map(sym.base + (OFF_PAT + 4), peer)),
                )
            # generic-space remote pulls (DeepGEMM: 10 x ld.v4.b32 per lane from the peer, ST.E.128 into the local ring)
            if tidx < 16:
                ga, gb, gc, gd = ptx.ld_generic_v4_b32(
                    sym.map(sym.base + (OFF_PAT + 16 * tidx), peer)
                )
                ptx.st_v4_b32(
                    sym.base + (OFF_RES2 * 4 + 16 * tidx), ga, gb, gc, gd
                )  # K:707: the LOCAL ring store is st.global.v4
            if tidx == 17:
                ptx.st_generic_b64(
                    sym.base + (OFF_RES2 * 4 + 256),
                    ptx.ld_generic_b64(sym.map(sym.base + OFF_PATQ, peer)),
                )
                ptx.st_generic_b32(
                    sym.base + (OFF_RES2 * 4 + 264),
                    ptx.ld_generic_b32(sym.map(sym.base + (OFF_PAT + 8), peer)),
                )
        # ---------------- phase 2: barrier iterations with a cross-rank visibility check
        errors = Int32(0)
        for i in cutlass.range(iters, unroll=1):
            if tidx == 0:
                ptx.atom_add_sys_u64(sym.map(sym.base + (OFF_CNT + 8 * me), peer), 1)
            comm.grid_sync(ws, 0, N, sm_idx, tidx, (0, 128))
            comm.nvlink_barrier(
                ws,
                sym,
                self.R,
                N,
                128,
                0,
                1,
                sm_idx,
                tidx,
                (0, 128),
                prologue=False,
                epilogue=True,
            )
            if tidx == 0:
                seen = ptx.ld_acquire_sys_b64(sym.base + (OFF_CNT + 8 * peer))
                if seen != Int64(N) * Int64(i + 1):
                    errors = errors + Int32(1)
            comm.nvlink_barrier(
                ws,
                sym,
                self.R,
                N,
                128,
                0,
                2,
                sm_idx,
                tidx,
                (0, 128),
                prologue=True,
                epilogue=True,
            )
        # final barrier of the DeepGEMM kind (prologue grid sync only)
        comm.nvlink_barrier(
            ws,
            sym,
            self.R,
            N,
            128,
            0,
            2,
            sm_idx,
            tidx,
            (0, 128),
            prologue=True,
            epilogue=False,
        )
        if tidx == 0:
            ptx.red_add_u32(sym.base + OFF_ERR * 4, errors)


def _nvl_init_buffer(buf, rank):
    buf.zero_()
    words = buf.view(torch.int32)
    words[OFF_PAT // 4 : OFF_PAT // 4 + 64] = (
        torch.arange(64, dtype=torch.int32, device="cuda") + rank * 1000
    )
    words[OFF_PATQ // 4 : OFF_PATQ // 4 + 2].view(torch.int64)[:] = (
        0x5A5A << 32
    ) + rank


def _nvl_verify(rank, world, buf, iters, num_sms):
    """Host-side checks of everything KernelNvl left in the local buffer; returns a list of error strings."""
    peer = (rank + 1) % world
    w = buf.view(torch.int32).cpu()
    q = buf.view(torch.int64).cpu()
    errs = []

    def chk(cond, what):
        if not cond:
            errs.append(what)

    chk(
        int(w[OFF_W32 // 4]) == 0x1000 + peer,
        f"remote st_b32 {int(w[OFF_W32 // 4]):#x}",
    )
    chk(
        w[OFF_WV4 // 4 : OFF_WV4 // 4 + 4].tolist()
        == [peer * 100 + j for j in (1, 2, 3, 4)],
        "remote st_v4_b32",
    )
    chk(int(q[OFF_WREL // 8]) == (1 << 40) + peer, "remote st_release_sys_b64")
    chk(
        int(q[OFF_WATOM // 8]) == 5 * num_sms,
        f"remote atom_add_sys_u64 {int(q[OFF_WATOM // 8])}",
    )
    chk(int(w[OFF_WRED // 4]) == -num_sms, "remote red_add_release_sys_s32")
    chk(
        w[OFF_RES : OFF_RES + 64].tolist() == [peer * 1000 + j for j in range(64)],
        "remote ld_v4_b32 pull",
    )
    chk(
        int(q[(OFF_RES * 4 + 256) // 8]) == (0x5A5A << 32) + peer,
        "remote ld_acquire_sys_b64",
    )
    chk(
        int(q[(OFF_RES * 4 + 264) // 8]) == (0x5A5A << 32) + peer,
        "remote ld_volatile_b64",
    )
    chk(int(w[OFF_RES + 68]) == peer * 1000, "remote ld_acquire_sys_s32")
    chk(int(w[OFF_RES + 69]) == peer * 1000 + 1, "remote ld_acquire_sys_b32")
    chk(int(w[OFF_G32 // 4]) == 0x2000 + peer, "remote st_generic_b32")
    chk(int(q[OFF_G64 // 8]) == (7 << 40) + peer, "remote st_generic_b64")
    chk(
        w[OFF_GV4 // 4 : OFF_GV4 // 4 + 4].tolist()
        == [peer * 10 + j for j in (1, 2, 3, 4)],
        "remote st_generic_v4_b32",
    )
    chk(
        w[OFF_RES2 : OFF_RES2 + 64].tolist() == [peer * 1000 + j for j in range(64)],
        "remote ld_generic_v4_b32 pull + local st_v4_b32",
    )
    chk(
        int(q[(OFF_RES2 * 4 + 256) // 8]) == (0x5A5A << 32) + peer,
        "remote ld_generic_b64",
    )
    chk(int(w[OFF_RES2 + 66]) == peer * 1000 + 2, "remote ld_generic_b32")
    chk(int(w[OFF_ERR]) == 0, f"barrier visibility errors = {int(w[OFF_ERR])}")
    chk(
        int(q[(OFF_CNT + 8 * peer) // 8]) == num_sms * iters,
        f"cnt[peer] = {int(q[(OFF_CNT + 8 * peer) // 8])}",
    )
    nb = 2 * iters + 2  # barrier #1/#2 per iteration + phase-1 barrier + final barrier
    chk(int(w[16 // 4]) == nb, f"nvl_barrier_counter {int(w[4])} != {nb}")
    sig = [0, 0]
    for kk in range(nb):
        sig[kk & 1] += world * (1 if ((kk >> 1) & 1) == 0 else -1)
    chk([int(w[5]), int(w[6])] == sig, f"nvl signals {[int(w[5]), int(w[6])]} != {sig}")
    return errs


def _nvl_worker(rank, world, port, iters, num_sms):
    import time

    import torch.distributed as dist
    import torch.distributed._symmetric_memory as symm_mem

    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        world_size=world,
        rank=rank,
        device_id=torch.device(f"cuda:{rank}"),
    )
    group = dist.new_group(list(range(world)))
    buf = symm_mem.empty(NVL_BYTES, dtype=torch.int8, device="cuda")
    handle = symm_mem.rendezvous(buf, group=group)
    _nvl_init_buffer(buf, rank)
    group.barrier()
    torch.cuda.synchronize()
    host = comm.make_sym_buffer_host(buf.data_ptr(), list(handle.buffer_ptrs), rank)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    compiled = cute.compile(KernelNvl(world, num_sms), host, Int32(iters), stream)
    group.barrier()
    t0 = time.time()
    compiled(host, Int32(iters), stream)
    torch.cuda.synchronize()
    dt = time.time() - t0
    group.barrier()
    errs = _nvl_verify(rank, world, buf, iters, num_sms)
    print(
        f"  [rank {rank}] iters={iters} sms={num_sms} run={dt:.2f}s ({dt / max(iters, 1) * 1e6:.1f} us/iter incl. 3 grid syncs + 2 NVLink barriers) errors={errs}"
    )
    dist.destroy_process_group()
    if errs:
        raise AssertionError(f"rank {rank}: {errs}")


def check_nvl1():
    """Single rank (EP1): SymBuffer.map is the identity, the NVLink barrier waits for its own signal (target 1)."""
    n = torch.cuda.get_device_properties(0).multi_processor_count
    buf = torch.zeros(NVL_BYTES, dtype=torch.int8, device="cuda")
    _nvl_init_buffer(buf, 0)
    host = comm.make_sym_buffer_host(buf.data_ptr(), [buf.data_ptr()], 0)
    iters = min(ITERS, 20000)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    compiled = cute.compile(KernelNvl(1, n), host, Int32(iters), stream)
    compiled(host, Int32(iters), stream)
    torch.cuda.synchronize()
    errs = _nvl_verify(0, 1, buf, iters, n)
    print(f"  [R=1] iters={iters} sms={n} errors={errs}")
    assert not errs, errs


def test_nvl1():
    check_nvl1()


def check_nvl():
    import socket

    assert torch.cuda.device_count() >= 2, "needs 2 GPUs"
    n = torch.cuda.get_device_properties(0).multi_processor_count
    with socket.socket() as sck:
        sck.bind(("127.0.0.1", 0))
        port = sck.getsockname()[1]
    torch.multiprocessing.spawn(
        _nvl_worker, args=(2, port, ITERS, n), nprocs=2, join=True
    )


def test_nvl():
    check_nvl()


# =====================================================================================================================
# timeout: the device-side watchdog (trap instead of an infinite spin) with a shortened timeout, run in a child process
# =====================================================================================================================
class KernelGridSyncHang:
    """All CTAs but the last call grid_sync -> the missing arrival can never come -> every spinning CTA must trap after TIMEOUT_CYCLES."""

    def __init__(self, num_sms):
        self.num_sms = num_sms

    @cute.jit
    def __call__(self, ws_base: Int64, stream: cuda.CUstream):
        self.kernel(ws_base).launch(
            grid=[self.num_sms, 1, 1],
            block=[128, 1, 1],
            cluster=[2, 1, 1],
            smem=216 * 1024,
            stream=stream,
        )

    @cute.kernel
    def kernel(self, ws_base: Int64):
        tidx, _, _ = cute.arch.thread_idx()
        sm_idx, _, _ = cute.arch.block_idx()
        ws = FakeWs(ws_base)
        if sm_idx != Int32(self.num_sms - 1):
            comm.grid_sync(ws, 0, self.num_sms, sm_idx, tidx, (0, 128))


def _timeout_child():
    import time

    comm.TIMEOUT_CYCLES = (
        4_000_000_000  # ~2 s at 2 GHz; read at trace time by comm._spin_*
    )
    n = torch.cuda.get_device_properties(0).multi_processor_count
    ws = torch.zeros(1024, dtype=torch.int32, device="cuda")
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    compiled = cute.compile(KernelGridSyncHang(n), int(ws.data_ptr()), stream)
    t0 = time.time()
    compiled(int(ws.data_ptr()), stream)
    try:
        torch.cuda.synchronize()
    except Exception as e:  # noqa: BLE001
        print(
            f"CHILD: kernel aborted after {time.time() - t0:.1f}s: {str(e).splitlines()[0][:120]}"
        )
        sys.exit(3)
    print(
        f"CHILD: kernel completed after {time.time() - t0:.1f}s (watchdog did NOT trap)"
    )
    sys.exit(0)


def check_timeout():
    import subprocess
    import time

    t0 = time.time()
    r = subprocess.run(
        [sys.executable, os.path.abspath(__file__), "_timeout_child"],
        capture_output=True,
        text=True,
        timeout=240,
    )
    dt = time.time() - t0
    out = (r.stdout + r.stderr).strip().splitlines()
    print("  " + " | ".join(l for l in out if "CHILD" in l)[:300])
    assert r.returncode == 3, (
        f"child rc={r.returncode}; expected the watchdog to abort the kernel (rc 3): {out[-3:]}"
    )
    assert dt < 120, f"watchdog took {dt:.0f}s"
    print(
        f"  [ok] watchdog trapped the deadlocked grid_sync (child exited rc=3 after {dt:.1f}s total incl. compile)"
    )


def test_timeout():
    check_timeout()


if __name__ == "__main__":
    groups = sys.argv[1:] or ["sym"]
    if groups == ["_timeout_child"]:
        _timeout_child()
    for g in groups:
        print(f"== group {g}")
        globals()[f"check_{g}"]()
        print(f"== group {g} PASS")
