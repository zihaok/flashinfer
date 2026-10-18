# SPDX-License-Identifier: Apache-2.0
# Port of DeepGEMM (MIT License, Copyright (c) 2025 DeepSeek; see licenses/LICENSE.deepgemm.txt) sm107_fp8_fp4_mega_moe.cuh / kernel skeleton EP2 test.
"""EP2 (2 GPUs, 2 processes) test of the kernel skeleton with STUB roles: the real launch geometry (212 CTAs x 512 threads, 2-CTA clusters, shipped smem) running
N rounds of the DeepGEMM sequence (grid sync A, NVLink barrier #1, NVLink barrier #2) between the two ranks through torch symmetric memory peer pointers.

Run: ``pytest tests/experimental/test_sm107_dg_foundation_skeleton_ep2.py`` (2 GPUs); ``python test_sm107_dg_foundation_skeleton_ep2.py [iters] [bm]``.
"""

from __future__ import annotations

import socket
import sys
import time

import pytest
import torch


def _is_sm107() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (10, 7)


pytestmark = pytest.mark.skipif(
    torch.cuda.device_count() < 2 or not _is_sm107(),
    reason="needs 2 SM107 (Rubin, compute capability 10.7) GPUs",
)

NBYTES = 1 << 20


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _worker(rank: int, world: int, port: int, iters: int, block_m: int, mode_mask: int):
    import torch.distributed as dist
    import torch.distributed._symmetric_memory as symm_mem

    from flashinfer.experimental.sm107_deepgemm_megamoe import jit
    from flashinfer.experimental.sm107_deepgemm_megamoe.heuristics import (
        get_sm107_kernel_config,
    )
    from flashinfer.experimental.sm107_deepgemm_megamoe.kernel import comm

    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", init_method=f"tcp://127.0.0.1:{port}", world_size=world, rank=rank,
                            device_id=torch.device(f"cuda:{rank}"))  # fmt: skip
    group = dist.new_group(list(range(world)))
    buf = symm_mem.empty(NBYTES, dtype=torch.int8, device="cuda")
    handle = symm_mem.rendezvous(buf, group=group)
    buf.zero_()
    counters = torch.zeros(64, dtype=torch.int64, device="cuda")
    group.barrier()
    torch.cuda.synchronize()

    cfg = get_sm107_kernel_config(
        world, 384, 3840, 1024, 6, 5120, 2304, 0, activation_clamp=10.0, block_m=block_m
    )
    build = jit.make_build(cfg, roles="stubs", mode_mask=mode_mask, debug_marks=True)

    build = type(build)(**{**build.__dict__, "debug_nvlink_iters": iters})
    runner = jit.get_or_build = jit.get_or_compile_build(build)
    host = comm.make_sym_buffer_host(buf.data_ptr(), list(handle.buffer_ptrs), rank)
    sym = (
        host.base_address,
        tuple(int(o) for o in host.offsets),
        int(host.rank),
        world,
    )
    group.barrier()
    t0 = time.time()
    runner.call_raw(0, 0, 0, sym, 0, 0, 0, 0, trace_ptr=counters.data_ptr())
    torch.cuda.synchronize()
    dt = time.time() - t0
    group.barrier()
    c = counters.tolist()
    words = buf.view(torch.int32).cpu()
    nvl_counter = int(words[16 // 4])
    sig = [int(words[20 // 4]), int(words[24 // 4])]
    ok_marks = c[0:16] == [cfg.num_sms] * 16 and c[16:32] == [cfg.num_sms] * 16
    # each iteration = barrier #1 + barrier #2 => 2 increments; signals return to 0 after an even number of uses per phase
    expect_counter = 2 * iters
    print(
        f"  [rank {rank}] BM={block_m} mode={mode_mask} iters={iters}: {dt:.3f}s ({dt / max(iters, 1) * 1e6:.1f} us/iter incl. grid sync + 2 NVLink barriers) "
        f"marks_ok={ok_marks} nvl_counter={nvl_counter} (expect {expect_counter}) signals={sig}",
        flush=True,
    )
    dist.destroy_process_group()
    assert ok_marks, c
    assert nvl_counter == expect_counter, (nvl_counter, expect_counter)


def run(iters: int = 2000, block_m: int = 128, mode_mask: int = 7):
    port = _free_port()
    torch.multiprocessing.spawn(
        _worker, args=(2, port, iters, block_m, mode_mask), nprocs=2, join=True
    )


@pytest.mark.parametrize("block_m", (16, 128, 240))
def test_skeleton_ep2_nvlink_rounds(block_m):
    run(1000, block_m)


def test_skeleton_ep2_mode_masks():
    for mode in (1, 3, 5, 7):
        run(200, 128, mode)


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    bm = int(sys.argv[2]) if len(sys.argv) > 2 else 128
    run(n, bm)
    print("EP2 OK")
