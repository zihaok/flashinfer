# SPDX-License-Identifier: Apache-2.0
# Port of DeepGEMM (MIT License, Copyright (c) 2025 DeepSeek; see licenses/LICENSE.deepgemm.txt) sm107_fp8_fp4_mega_moe.cuh / multi-rank symmetric buffer tests.
"""Multi-rank (EP2/EP4/... on one NVLink domain) parity of the SM107 MegaMoE ``SymmBuffer`` with DeepGEMM's.

Needs one Rubin GPU per rank (torch refuses two ranks on one device) and ``deep_gemm`` importable; skipped unless launched with
``torchrun``::

    torchrun --nproc-per-node 2 -m pytest tests/experimental/test_sm107_deepgemm_megamoe_symm_multirank.py -x -s

Checks, per rank: every view of ``SymmBuffer`` (shape / strides / dtype / byte offset) equals DeepGEMM's; the peer offset table is
``ptrs[i] - ptrs[rank]`` of the rendezvous handle; the layout is identical on all ranks (a pattern written into the ``x_sf`` region
of every rank is read back through the symmetric-memory peer buffers at the layout's offset).
"""

import os

import pytest

torch = pytest.importorskip("torch")

WORLD = int(os.environ.get("WORLD_SIZE", "1"))
_RUBIN = torch.cuda.is_available() and torch.cuda.get_device_capability() == (10, 7)
pytestmark = [
    pytest.mark.arch_rubin,
    pytest.mark.skipif(
        WORLD < 2 or not _RUBIN,
        reason="needs torchrun with >= 2 ranks, one Rubin (cc 10.7) GPU each",
    ),
]


@pytest.fixture(scope="module")
def group():
    import torch.distributed as dist

    local = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local)
    if not dist.is_initialized():
        dist.init_process_group("nccl", device_id=torch.device("cuda", local))
    yield dist.new_group(list(range(dist.get_world_size())))
    dist.barrier()
    dist.destroy_process_group()


def _describe(buf, names):
    """Layout of a symmetric buffer: byte size and (shape, stride, dtype, byte offset) of every view (``names``: our SymmBuffer.view_names; DeepGEMM's object has the same attributes)."""
    base = buf.buffer.data_ptr()
    d = {"nbytes": buf.buffer.nbytes, "T": buf.num_max_tokens_per_rank}
    for n in names:
        t = getattr(buf, n)
        d[n] = (
            None
            if t is None
            else (tuple(t.shape), tuple(t.stride()), t.dtype, t.data_ptr() - base)
        )
    return d


@pytest.mark.parametrize(
    "E,H,I", [(384, 5120, 2304), (256, 4096, 2048)], ids=["v41", "v4"]
)
@pytest.mark.parametrize("T,S", [(1920, 1), (8192, 0)])
def test_symm_buffer_matches_deepgemm_multirank(group, E, H, I, T, S):
    import torch.distributed as dist

    deep_gemm = pytest.importorskip("deep_gemm")
    from flashinfer.experimental.sm107_deepgemm_megamoe import compat

    rank, world = dist.get_rank(), dist.get_world_size()
    if E % world:
        pytest.skip("experts not divisible by world size")
    ref = deep_gemm.get_symm_buffer_for_mega_moe(
        group, E, T, 6, H, I, num_shared_experts=S
    )
    mine = compat.get_symm_buffer_for_mega_moe(
        group, E, T, 6, H, I, num_shared_experts=S
    )
    assert _describe(ref, mine.view_names) == _describe(mine, mine.view_names)

    ptrs = list(mine.handle.buffer_ptrs)
    assert (
        len(ptrs) == world
        and mine.peer_offsets == [int(p) - int(ptrs[rank]) for p in ptrs]
        and mine.peer_offsets[rank] == 0
    )

    # layout identical on every rank: write a rank-specific pattern into the x_sf region, read the peers' through the handle
    off, nbytes = mine.layout.regions["input_sf"]
    mine.buffer[off : off + 16].fill_(rank + 1)
    torch.cuda.synchronize()
    dist.barrier(group)
    for peer in range(world):
        peer_buf = mine.handle.get_buffer(peer, (nbytes,), torch.int8, off)
        assert int(peer_buf[:16].min()) == int(peer_buf[:16].max()) == peer + 1, (
            peer,
            peer_buf[:16],
        )
    dist.barrier(group)
    mine.destroy()
    del ref
