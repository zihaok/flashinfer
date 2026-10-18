# SPDX-License-Identifier: Apache-2.0
# Port of DeepGEMM (MIT License, Copyright (c) 2025 DeepSeek; see licenses/LICENSE.deepgemm.txt) sm107_fp8_fp4_mega_moe.cuh / tests/test_mega_moe.py usage.
"""EXPERIMENTAL: run the SM107 (Rubin) DeepGEMM-style MegaMoE through FlashInfer's DeepGEMM-compatible API.

One process per GPU, expert parallelism = world size; needs a Rubin GPU (compute capability 10.7), CUDA 13.4 and the NVLink fabric for
symmetric memory.  Weights are random but valid (packed MXFP4 + UE8M0 scales, FP8 shared expert); the example prints the output
statistics and, with ``--check deepgemm`` (needs ``deep_gemm`` with the Rubin kernel), compares the result bit-for-bit with DeepGEMM's own
``fp8_fp4_mega_moe`` run on the same operands.

    # one GPU (EP1), DeepSeek-V4.1-Flash shape (H=5120, I=2304, 384 experts, top-6), one fused shared expert
    python examples/experimental/sm107_deepgemm_megamoe.py --shared 1 --tokens 1024 --check deepgemm
    # expert parallelism over 4 GPUs, die-local routed weights (the production recipe: only the routed pair is die-local)
    torchrun --nproc-per-node 4 examples/experimental/sm107_deepgemm_megamoe.py --tokens 1024 --shared 1 --per-die

The first call of each BLOCK_M variant compiles a kernel (seconds, cached on disk afterwards): ``--precompile`` builds all variants a rank
can select up front, as a serving engine does at warmup (compiling inside a CUDA graph capture is an error).  The package reads no
environment variable; the variables set below only configure ``torch.distributed`` for the single-process case.
"""

import argparse
import os
import socket

import torch
import torch.distributed as dist

import flashinfer.mega_moe_deepgemm as mega
from flashinfer.experimental.sm107_deepgemm_megamoe.staging import (
    prepare_megamoe_inputs,
)  # only to stage DeepGEMM's own buffer in --check deepgemm
from flashinfer.experimental.sm107_deepgemm_megamoe.weights import sf_to_required_layout


def _random_mxfp4_weights(num_experts, n, k, device, generator):
    """Packed FP4 weights ``(E, n, k/2)`` int8 and UE8M0 scales in the kernel's MN-major int32 layout (random but valid values)."""
    w = torch.randint(
        -128,
        128,
        (num_experts, n, k // 2),
        dtype=torch.int8,
        device=device,
        generator=generator,
    )
    exp = torch.randint(
        116,
        123,
        (num_experts, n, k // 32),
        dtype=torch.int32,
        device=device,
        generator=generator,
    )  # scales 2^-11 .. 2^-5 (weights ~0.01-0.1)
    return w, sf_to_required_layout((exp << 23).view(torch.float32), n, k, num_experts)


def _random_fp8_weights(n, k, device, generator):
    """FP8 E4M3 shared-expert weights ``(n, k)`` and UE8M0 scales (2^-8) in the kernel's layout."""
    w = (torch.randn((n, k), device=device, generator=generator) * 0.05).to(
        torch.float8_e4m3fn
    )
    return w, sf_to_required_layout(
        torch.full((n, k // 32), 2.0**-8, device=device), n, k
    )


def _init_distributed() -> int:
    """``torchrun`` sets the rendezvous variables; a plain ``python`` launch becomes a one-process (EP1) job."""
    if "RANK" not in os.environ:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        os.environ.update(
            RANK="0",
            WORLD_SIZE="1",
            LOCAL_RANK="0",
            MASTER_ADDR="127.0.0.1",
            MASTER_PORT=str(port),
        )
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
    return local_rank


def _deepgemm_reference(
    group, args, x, topk_ids, topk_weights, l1_t, l2_t, s1, s2, block_m, stats_shape
):
    """The same operands through DeepGEMM's own kernel (its own symmetric buffer; the port's transformed weights are bit-identical to DeepGEMM's)."""
    import deep_gemm

    E, H, I, S = args.num_experts, args.hidden, args.intermediate_hidden, args.shared
    dg_buf = deep_gemm.get_symm_buffer_for_mega_moe(
        group, E, args.max_tokens, args.topk, H, I, num_shared_experts=S
    )
    num_tokens = x.size(0)
    prepare_megamoe_inputs(
        x, topk_weights, topk_ids, dg_buf.x[:num_tokens], dg_buf.x_sf[:num_tokens], dg_buf.topk_idx[:num_tokens],
        dg_buf.topk_weights[:num_tokens], shared_x_sf=dg_buf.shared_l1_acts_sf if S else None, shared_block_m=block_m if S else None,
    )  # fmt: skip
    y = torch.empty_like(x)
    deep_gemm.fp8_fp4_mega_moe(
        y,
        l1_t,
        l2_t,
        dg_buf,
        shared_l1_weights=s1,
        shared_l2_weights=s2,
        activation_clamp=args.clamp,
    )
    torch.cuda.synchronize()
    del dg_buf
    return y


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--num-experts", type=int, default=384)
    parser.add_argument("--hidden", type=int, default=5120)
    parser.add_argument("--intermediate-hidden", type=int, default=2304)
    parser.add_argument("--topk", type=int, default=6)
    parser.add_argument(
        "--shared",
        type=int,
        default=1,
        choices=(0, 1),
        help="fused FP8 shared expert (intermediate size = --intermediate-hidden)",
    )
    parser.add_argument("--tokens", type=int, default=1024, help="tokens on this rank")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=8192,
        help="num_max_tokens_per_rank (aligned up to 1920)",
    )
    parser.add_argument("--clamp", type=float, default=10.0, help="SwiGLU clamp")
    parser.add_argument(
        "--per-die",
        action="store_true",
        help="die-local routed weights + per-die scheduling (needs a GPU with 2 locality domains)",
    )
    parser.add_argument(
        "--precompile",
        action="store_true",
        help="build every BLOCK_M variant first (as at engine warmup)",
    )
    parser.add_argument(
        "--check",
        choices=("none", "deepgemm"),
        default="none",
        help="compare bit-for-bit with DeepGEMM's kernel on the same operands",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=0,
        help="also time this many back-to-back calls (CUDA events)",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.check == "deepgemm":
        os.environ.setdefault(
            "DG_MEGA_MOE_SM107", "1"
        )  # DeepGEMM launches its SM107 (Rubin) kernel only with this set; without it the reference would be its SM100 kernel

    local_rank = _init_distributed()
    rank, world = dist.get_rank(), dist.get_world_size()
    group = dist.new_group(list(range(world)))
    E, H, I, topk, S = (
        args.num_experts,
        args.hidden,
        args.intermediate_hidden,
        args.topk,
        args.shared,
    )
    E_l = E // world
    dev = torch.device("cuda", local_rank)
    gen = torch.Generator(device=dev).manual_seed(args.seed * 1000 + rank)

    # 1. symmetric buffer (collective), weights in the DeepGEMM layout (pure torch transform, bit-exact with deep_gemm's)
    buf = mega.get_symm_buffer_for_mega_moe(
        group, E, args.max_tokens, topk, H, I, num_shared_experts=S
    )
    l1_t, l2_t = mega.transform_weights_for_mega_moe(
        _random_mxfp4_weights(E_l, 2 * I, H, dev, gen),
        _random_mxfp4_weights(E_l, H, I, dev, gen),
    )
    s1 = s2 = None
    if S:
        s1, s2 = mega.transform_weights_for_mega_moe(
            _random_fp8_weights(2 * I * S, H, dev, gen),
            _random_fp8_weights(H, I * S, dev, gen),
        )
    plain = (l1_t, l2_t)
    if args.per_die:
        assert mega.is_localization_available(), (
            "--per-die needs a Rubin GPU that reports 2 locality domains"
        )
        mega.get_balanced_sm_locality_domains()  # probes the SM -> domain map once; must run outside CUDA graph capture
        l1_t, l2_t, _, _ = mega.localize_mega_moe_weights(
            l1_t, l2_t
        )  # (E_l, N, K/2) -> (2, E_l, N/2, K/2); the launch picks the per-die kernel

    if args.precompile:
        built = mega.precompile_mega_moe_variants(
            world, E, args.max_tokens, topk, H, I, S, activation_clamp=args.clamp, per_die=args.per_die, localized_weights=args.per_die
        )  # fmt: skip
        if rank == 0:
            print(f"precompiled BLOCK_M variants: {built}")

    # 2. stage this call's inputs into the symmetric buffer (BF16 -> MXFP8 + packed UE8M0 scales + routing + shared-expert scale layout)
    x = torch.randn((args.tokens, H), dtype=torch.bfloat16, device=dev, generator=gen)
    scores = torch.randn((args.tokens, E), device=dev, generator=gen)
    topk_weights, topk_ids = torch.topk(scores, topk, dim=-1, sorted=False)
    topk_weights = topk_weights.softmax(-1)
    block_m = mega.stage_inputs_for_mega_moe(buf, x, topk_ids, topk_weights)

    # 3. run: the BLOCK_M kernel variant is selected from y.size(0), as in DeepGEMM
    y = torch.empty((args.tokens, H), dtype=torch.bfloat16, device=dev)
    stats = torch.zeros(E_l, dtype=torch.int32, device=dev)
    mega.fp8_fp4_mega_moe(y, l1_t, l2_t, buf, shared_l1_weights=s1, shared_l2_weights=s2, cumulative_local_expert_recv_stats=stats, activation_clamp=args.clamp)  # fmt: skip
    torch.cuda.synchronize()
    finite = bool(torch.isfinite(y.float()).all())
    print(
        f"rank {rank}/{world}: BLOCK_M={block_m} shared={S} per_die={args.per_die} finite={finite} y.abs().mean()={y.float().abs().mean().item():.4f} "
        f"tokens received by local experts={int(stats.sum())}"
    )
    assert finite and y.float().abs().sum() > 0

    if args.per_die:  # the per-die kernel must reproduce the plain kernel bit-for-bit
        y_plain = torch.empty_like(y)
        mega.fp8_fp4_mega_moe(
            y_plain,
            *plain,
            buf,
            shared_l1_weights=s1,
            shared_l2_weights=s2,
            activation_clamp=args.clamp,
        )
        torch.cuda.synchronize()
        same = torch.equal(y.view(torch.int16), y_plain.view(torch.int16))
        print(f"rank {rank}: per-die output bit-identical to the plain kernel: {same}")
        assert same
    if args.check == "deepgemm":
        y_ref = _deepgemm_reference(
            group, args, x, topk_ids, topk_weights, *plain, s1, s2, block_m, (E_l,)
        )
        same = torch.equal(y.view(torch.int16), y_ref.view(torch.int16))
        print(
            f"rank {rank}: bit-identical to DeepGEMM's fp8_fp4_mega_moe: {same} (max |diff| = {(y.float() - y_ref.float()).abs().max().item():.3g})"
        )
        assert same

    if args.iters:  # back-to-back launches of the same call (restage not needed: the symmetric buffer keeps the inputs)
        start, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        dist.barrier()
        start.record()
        for _ in range(args.iters):
            mega.fp8_fp4_mega_moe(
                y,
                l1_t,
                l2_t,
                buf,
                shared_l1_weights=s1,
                shared_l2_weights=s2,
                activation_clamp=args.clamp,
            )
        end.record()
        torch.cuda.synchronize()
        print(
            f"rank {rank}: {start.elapsed_time(end) * 1000 / args.iters:.1f} us per call (eager, {args.iters} back-to-back launches)"
        )

    buf.destroy()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
