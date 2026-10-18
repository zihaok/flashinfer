# SPDX-License-Identifier: Apache-2.0
# Port of DeepGEMM (MIT License, Copyright (c) 2025 DeepSeek; see licenses/LICENSE.deepgemm.txt) sm107_fp8_fp4_mega_moe.cuh / test helper.
"""Helper script (not a test): launch DeepGEMM's SM107 MegaMoE once per BLOCK_M at EP1 with ``DG_PRINT_CONFIGS=1``.

DeepGEMM prints (HEUR:255-265, IMPL:142-143) the SM100-sized ``MegaMoEConfig(...)`` line once per key and the shipped
``SM107 MegaMoE: stages=..., smem=...`` line on every launch; ``test_sm107_deepgemm_megamoe_heuristics.py`` parses both and
compares them with ``heuristics.py``.  Numerics are irrelevant here (random weight bytes, zero activations): only the host
configuration is exercised, so the weights are random bytes in the DeepGEMM layout (no quantisation pass).

usage (Rubin GPU, DeepGEMM 3ec0eb4 importable as ``deep_gemm``; one launch compiles one kernel, minutes each on a cold JIT cache):
    DG_MEGA_MOE_SM107=1 DG_PRINT_CONFIGS=1 python _sm107_dg_print_probe.py [--block-ms 16,32,...]
"""

import argparse
import os
import sys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden", type=int, default=5120)
    parser.add_argument("--intermediate-hidden", type=int, default=2304)
    parser.add_argument("--num-experts", type=int, default=384)
    parser.add_argument("--num-topk", type=int, default=6)
    parser.add_argument("--num-max-tokens-per-rank", type=int, default=1920)
    parser.add_argument("--num-tokens", type=int, default=64)
    parser.add_argument("--block-ms", type=str, default="16,32,64,128,192,240")
    args = parser.parse_args()

    os.environ.setdefault("DG_MEGA_MOE_SM107", "1")
    os.environ.setdefault("DG_PRINT_CONFIGS", "1")
    os.environ.setdefault(
        "MASTER_PORT", str(29000 + os.getpid() % 2000)
    )  # the node is shared: avoid DeepGEMM's default port
    import torch
    import deep_gemm
    from deep_gemm.utils.dist import init_dist

    H, I, E, topk, T = (
        args.hidden,
        args.intermediate_hidden,
        args.num_experts,
        args.num_topk,
        args.num_max_tokens_per_rank,
    )
    _, num_ranks, group = init_dist(0, 1)
    assert num_ranks == 1
    buf = deep_gemm.get_symm_buffer_for_mega_moe(
        group, E, T, topk, H, I, num_shared_experts=0
    )

    def align4(n: int) -> int:
        return (n + 3) // 4 * 4

    def sf(
        n: int, k: int
    ) -> (
        torch.Tensor
    ):  # int32 (E, N, K/128), MN-major (what transform_sf_into_required_layout returns)
        t = torch.empty_strided(
            (E, n, k // 128),
            ((k // 128) * align4(n), 1, align4(n)),
            dtype=torch.int32,
            device="cuda",
        )
        return t.fill_(0x7F7F7F7F)  # UE8M0 exponent 127 = scale 1.0

    l1_w = torch.randint(
        -128, 127, (E, 2 * I, H // 2), dtype=torch.int8, device="cuda"
    )  # packed FP4 (two values per byte)
    l2_w = torch.randint(-128, 127, (E, H, I // 2), dtype=torch.int8, device="cuda")
    l1_sf, l2_sf = sf(2 * I, H), sf(H, I)

    for i, bm in enumerate(int(b) for b in args.block_ms.split(",")):
        m = (
            args.num_tokens + i
        )  # distinct M -> distinct DG_PRINT_CONFIGS key per BLOCK_M
        os.environ["DG_MEGA_MOE_SM107_BLOCK_M"] = str(
            bm
        )  # read through getenv at every launch
        scores = torch.randn((m, E), device="cuda")
        buf.topk_idx[:m].copy_(torch.topk(scores, topk, dim=-1, sorted=False).indices)
        buf.topk_weights[:m].fill_(1.0 / topk)
        y = torch.empty((m, H), dtype=torch.bfloat16, device="cuda")
        print(f"PROBE bm={bm} M={m}", flush=True)
        deep_gemm.fp8_fp4_mega_moe(
            y, (l1_w, l1_sf), (l2_w, l2_sf), buf, activation_clamp=10.0
        )
        torch.cuda.synchronize()
        print(f"PROBE done bm={bm}", flush=True)
    buf.destroy()
    sys.stdout.flush()


if __name__ == "__main__":
    main()
