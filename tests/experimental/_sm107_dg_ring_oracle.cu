// SPDX-License-Identifier: Apache-2.0
// Contains code of DeepGEMM (MIT License, Copyright (c) 2025 DeepSeek; see
// licenses/LICENSE.deepgemm.txt). Test helper of test_sm107_deepgemm_megamoe_heuristics.py
// (compiled with nvcc against DeepGEMM's headers): exact C++ oracle of the SM-count dependent host
// math of DeepGEMM's MegaMoE (ring sizing), evaluated at ARBITRARY SM counts. The functions come
// from DeepGEMM's own headers (scheduler/mega_moe.cuh, layout/mega_moe.cuh); the ring loop is the
// verbatim body of csrc/apis/mega_moe.hpp build_symm_buffer_layout (lines 123-155).  Prints: R E T
// topk H I sms ring_tokens sf_ring_tokens
#include <cstdio>
#include <cstdlib>
// the asserts only guard shapes this grid never violates; the device `asm("trap;")` of
// DG_UNIFIED_ASSERT is not assemblable by the aarch64 host pass
#define DG_UNIFIED_ASSERT(cond) \
  do {                          \
    (void)sizeof(cond);         \
  } while (0)
#include <algorithm>
#include <deep_gemm/layout/mega_moe.cuh>
#include <deep_gemm/scheduler/mega_moe.cuh>

using namespace deep_gemm;

static int ceil_div_i(int a, int b) { return (a + b - 1) / b; }

int main() {
  const int shapes[2][4] = {{384, 5120, 2304, 6}, {256, 4096, 2048, 6}};  // E, H, I, topk
  const int sms_list[] = {148, 152, 160, 196, 208, 212, 132, 120};
  const int T_list[] = {1920, 3840, 5760, 9600, 15360};
  const int R_list[] = {1, 2, 3, 4, 6, 8, 12, 16, 24, 32};
  for (const auto& sh : shapes) {
    const int E = sh[0], H = sh[1], I = sh[2], topk = sh[3];
    for (const int sms : sms_list)
      for (const int T : T_list)
        for (const int R : R_list) {
          if (E % R != 0) continue;
          const int epr = E / R;
          const int num_active_topk = std::min(topk, epr);
          const int num_max_routed_tokens = T * R * num_active_topk;
          int num_ring_tokens = 0;
          for (const auto& block_m : layout::kCandidateBlockM) {
            const auto num_pool_blocks = ceil_div_i(num_max_routed_tokens, block_m) + epr;
            const auto num_live_pool_blocks =
                sched::get_num_max_live_pool_blocks(num_pool_blocks, sms, H, I);
            num_ring_tokens = std::max(num_ring_tokens, num_live_pool_blocks * block_m);
          }
          num_ring_tokens = math::align(num_ring_tokens, layout::kLCMCandidateBlockM);
          int num_sf_ring_tokens = 0;
          for (auto block_m : layout::kCandidateBlockM)
            num_sf_ring_tokens = std::max(num_sf_ring_tokens,
                                          layout::get_num_sf_ring_tokens(num_ring_tokens, block_m));
          printf("%d %d %d %d %d %d %d %d %d\n", R, E, T, topk, H, I, sms, num_ring_tokens,
                 num_sf_ring_tokens);
        }
  }
  // direct function grid
  for (const int sms : sms_list)
    for (int nb = 1; nb <= 20000; nb += 37)
      printf("LIVE %d %d %d\n", nb, sms, sched::get_num_max_live_pool_blocks(nb, sms, 5120, 2304));
  return 0;
}
