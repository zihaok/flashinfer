# sm107_deepgemm_megamoe (EXPERIMENTAL)

CuTe DSL port of DeepGEMM's Rubin (SM107) **MegaMoE** kernel: one persistent, communication-fused kernel that does token dispatch over
symmetric memory, the two grouped GEMMs with a fused SwiGLU / re-quantisation epilogue, and the combine, for
**MXFP8 activations x MXFP4 weights -> BF16** (DeepSeek-V4.1-Flash: H=5120, I=2304, 384 experts, top-6, optional fused shared expert).
It reproduces DeepGEMM's `sm107_fp8_fp4_mega_moe_impl` mapping (CTA / cluster / warp roles, tiles, pipeline depths, scheduling order, communication
protocol, epilogue numerics) without a C++ compile in the development loop, behind the **same API names, symmetric-buffer byte layout and weight layout**
as `deep_gemm.fp8_fp4_mega_moe`, so a DeepGEMM-based caller (e.g. vLLM's native MegaMoE path) can switch the library.

> **Status: experimental, opt-in.** Both the API (`flashinfer.mega_moe_deepgemm`, every function decorated with `@flashinfer_experimental_api`) and the
> backend are experimental: no compatibility guarantees, may change or be removed without deprecation. Calling an API function **is** the opt-in (it warns once
> with `ExperimentalWarning`); the backend is never selected by `backend="auto"`, is **JIT-only** (never registered in `flashinfer/aot.py`) and needs **no environment
> variable**: the package neither reads nor writes `os.environ` (host files and import behavior are enforced by `tests/experimental/test_sm107_deepgemm_megamoe_policy.py`).

## Requirements and limits

* Hardware / toolchain: compute capability exactly **10.7 (Rubin, `sm_107a`)**, CUDA 13.4, CuTe DSL >= 4.8 (the target chip is passed to `cute.compile`; compiling needs a current CUDA device, `jit.check_residency` loads the
  module on it: there is no GPU-less compile-only mode), torch with `torch.distributed._symmetric_memory`, cuda-python 13.4 (per-die helpers). The kernel is persistent with software grid
  syncs: it uses the whole device (`kNumSMs` = `multi_processor_count`, 212 on VR200) and all CTAs must be co-resident (never share a GPU with another persistent kernel).
* Expert parallelism DEP1/2/4/8/16/32 (`num_experts % num_ranks == 0`, at most 32 ranks: the peer table is passed by value; DEP8+ spans several nodes of one NVL72 block with NVLink
  fabric handles); the symmetric buffer is torch `symm_mem` (no NVSHMEM). `get_unsupported_reason_for_mega_moe(...)` tells whether a call can be served (see "Supported geometries and options"); it also requires at least 12 experts per rank (`heuristics.MIN_VALIDATED_EXPERTS_PER_RANK`:
  E=384 at DEP32 is the smallest value that has run) and is the only place that limits the rank count to 32 for the direct API's callers (`fp8_fp4_mega_moe` refuses more than 32 ranks as well).
* Numerics / format contract is DeepGEMM's: `activation="swiglu"` only (optional clamp, alpha, beta), `recipe=(1, 1, 32)`, UE8M0 scales, packed FP4 routed weights (`int8`, two E2M1 values
  per byte), FP8 E4M3 shared-expert weights, `fp32` top-k weights, `int64` ids, `hidden` and `intermediate_hidden` multiples of 128 (plus the combine-chunking constraint of the kernel,
  e.g. `hidden=4864` is rejected). Not ported: BF16xBF16 / SiTU / SwiGLU-step / NVFP4 activations (callers keep DeepGEMM for them), dispatch modes 1/2, `UMMA_K=32`, `BLOCK_K=256` and
  the other DeepGEMM tuning environment knobs.
* Per-call kernel variant: the BLOCK_M (16/32/64/128/192/240) is chosen from `y.size(0)` of the calling rank, exactly like DeepGEMM; up to six kernels are compiled per
  (shape, EP, T_max, shared experts) key (cold compile about 1 s each on VR200, about 10 ms from the disk cache afterwards). Precompile them at engine start (`precompile_mega_moe_variants`);
  compiling inside a CUDA-graph capture raises an error. CUDA-graph capture and replay of the launch is supported once the variants exist.
* PDL is not exercised (`pdl=False`); multi-node DEP8/16/32 needs an NVL72 allocation.

### Supported geometries and options

DeepGEMM accepts more than the kernel has been run with; the port refuses what it does not implement (`NotImplementedError` from `fp8_fp4_mega_moe` / `precompile_mega_moe_variants`, a reason string from the support predicate) instead of computing wrong rows:

* **Validated geometries** (`heuristics.VALIDATED_GEOMETRIES`, queried by `get_unsupported_reason_for_mega_moe`, which a DeepGEMM-API caller such as vLLM uses to fall through to DeepGEMM per call; the moe_ep backend applies it in `validate_init`): only shapes that
  ran bit-exact against DeepGEMM on Rubin GPUs are admitted.

  | hidden | intermediate_hidden | top-k | num_experts | evidence |
  |---|---|---|---|---|
  | 5120 | 2304 | 6 | 384 (DEP1-DEP32), 192, 96, 32 | DeepSeek-V4.1-Flash: E=384 in the bit-exactness / A/B campaigns and the vLLM layer validation (EP1, EP4); E=192 / 96 / 32 are test sizes of the same shape (EP1 full-kernel and per-die tests, EP4 harness) |

  | 5120 | 2304 | 3 | 128, at least 12 experts per rank (DEP1, 2, 4, 8; DEP16 / DEP32 would be 8 / 4 per rank and are refused) | MTP / DSpark draft-model layers of vLLM's DeepSeek-V4.1 deployment: EP4 (32 experts per rank), plain / masked / hot-expert / zipf / mixed per-rank M, M = 1 .. 4096: 120 / 120 (case, M, rank) cells bit-equal to DeepGEMM through vLLM's `FlashInferMoeEp` adapter; EP1 full-kernel shape plan 15 / 15 |

  The EP size is a second dimension of the gate: a call is admitted only with at least 12 experts per rank (the smallest value that has run is E=384 at DEP32); smaller values (1-expert ranks, fewer experts per rank than top-k) take code paths that never ran.
  Other shapes (DeepSeek-V4-Flash 4096 / 2048 / 256, ...) are structurally derivable but were not run: they are refused until a GPU result extends the table. `allow_unvalidated=True` (predicate) / `allow_unvalidated_geometry=True` (moe_ep config) skip the table for kernel development;
  the direct API (`get_symm_buffer_for_mega_moe`, `fp8_fp4_mega_moe`) accepts any structurally valid geometry at the caller's risk.
* **Shared experts**: none or one merged FP8 shared expert (`num_shared_experts` in {0, 1}); more than one is refused.
* **`fast_math`**: both `True` (DeepGEMM's default: `__expf` + `rcp.approx`) and `False` (IEEE `expf` as nvcc expands it + true division) are implemented and bit-exact vs DeepGEMM at EP1 (`tests/experimental/test_sm107_dg_gemm_core_numerics.py`).
* **Die-local weights: routed only** (the production recipe: the routed pair is localized, DeepGEMM runs it with `DG_MEGA_MOE_SM107_PERDIE=1`, vLLM selects it with `VLLM_DSV41_MEGAMOE_PERDIE=routed`). Die-local *shared* weights (the caller localizes the shared pair as well; vLLM `VLLM_DSV41_MEGAMOE_PERDIE=1`) are refused: the kernel's weight tensor maps only implement the routed 3D layout.
* **Scale-factor layout**: the weight scale tensors must have the dense TMA layout `stride(-1) == mn` (what `deep_gemm.transform_weights_for_mega_moe` / `transform_sf_into_required_layout` produce for multiple-of-4 `mn`); a padded outer stride, which DeepGEMM accepts, is refused.

## API (DeepGEMM-compatible)

```python
import flashinfer.mega_moe_deepgemm as mm   # core, decorated, thin

buf = mm.get_symm_buffer_for_mega_moe(group, num_experts, num_max_tokens_per_rank, num_topk, hidden, intermediate_hidden,
                                      num_shared_experts=1)               # collective; buf.x, buf.x_sf, buf.topk_idx, buf.topk_weights, ...
w1, w2 = mm.transform_weights_for_mega_moe((l1_w, l1_sf), (l2_w, l2_sf))   # pure torch, bit-exact with deep_gemm.transform_weights_for_mega_moe
mm.precompile_mega_moe_variants(world, num_experts, num_max_tokens_per_rank, topk, hidden, intermediate_hidden, num_shared_experts)  # before any graph capture
block_m = mm.stage_inputs_for_mega_moe(buf, x_bf16, topk_ids, topk_weights)  # BF16 -> MXFP8 + UE8M0 + routing (+ shared-expert scale layout), one Triton launch
mm.fp8_fp4_mega_moe(y, w1, w2, buf, shared_l1_weights=..., shared_l2_weights=..., activation_clamp=10.0)
bm = mm.get_block_m_for_mega_moe(num_ranks, num_experts, num_max_tokens_per_rank, num_tokens, num_topk, "fp8xfp4")
```

`stage_inputs_for_mega_moe` is bit-exact with DeepGEMM's `per_token_cast_to_fp8` and with vLLM's Triton `prepare_megamoe_inputs` (callers that stage with their own kernel write the same
buffer views, which are bit-identical to DeepGEMM's). `examples/experimental/sm107_deepgemm_megamoe.py` runs the whole sequence (EP1 or `torchrun`), optionally compares the result bit-for-bit
with DeepGEMM's kernel and times it.

A DeepGEMM-based caller switches the library by pointing its `deep_gemm` handle at these names (same signatures, buffer and weight layouts). vLLM's native MegaMoE layer (`DeepseekV4MegaMoEExperts`:
backend-owned weights, its own Triton staging, fused shared expert, `block_m` warmup, per-die weights) was validated this way against DeepGEMM's Rubin kernel at EP1 and EP4 (bit-exact outputs, CUDA-graph
capture and replay, layer timing within a few percent); that adapter is a vLLM-side change and not part of this package.

### Kernel build options (`set_kernel_options`)

Explicit arguments replace DeepGEMM's `DG_MEGA_MOE_SM107_*` environment variables. `mm.set_kernel_options(tag_mode="epoch", regs_override=None, faithful_regs=False, timeout_sec=300.0, block_m=0,
smem_capacity=232448, max_stages=0)` (keywords; no argument restores the defaults; `mm.get_kernel_options()`); variants are keyed by their options, so call it before the first launch / precompile.

* `tag_mode="epoch"` (default) tags the combine-readiness handshake with a per-launch counter and is exact under CUDA-graph replay. `"gridid"` is DeepGEMM's `%gridid + 1`: identical for every replay
  of one CUDA-graph node, it returns **stale** results when one captured launch is replayed back to back with changing inputs (reproduced with stock DeepGEMM; keep `"gridid"` for A/B parity runs only).
* Register split (`setmaxnreg`): the default is the spill-free **96/88/160 at every EP size**. DeepGEMM uses 96/88/160 only when experts-per-rank > 64 (DEP1-4 for 384 experts) and 48/40/208 otherwise (DEP>=8),
  where its 48-register dispatch cap spills. `faithful_regs=True` restores DeepGEMM's own split (parity / mapping runs); `regs_override=(dispatch, non_epilogue, epilogue)` sets an explicit split
  (mutually exclusive with `faithful_regs`; e.g. `(56, 40, 200)` is spill-free as well).
* `timeout_sec`: spin-wait trap timeout baked into the kernel (DeepGEMM: 300 s); `block_m`: force one variant (`DG_MEGA_MOE_SM107_BLOCK_M`); `smem_capacity` / `max_stages`: `DG_MEGA_MOE_SM107_SMEM` / `_STAGES`.
* The moe_ep backend takes the same fields in its config (`tag_mode`, `regs_override`, `faithful_regs`, `timeout_sec`, `block_m_override`, `launch_smem_capacity`, `stages_cap`) and scopes them to its own launches.
* Developer knob: `Sm107KernelConfig.tma_l2_promotion` / `Sm107MegaMoeConfig.tma_l2_promotion` (`None` = the kernel's own policy, `0 / 64 / 128 / 256` force the L2 promotion of the weight TMA maps; part of the compile key) is
  for kernel experiments; the launch API above always uses the kernel's policy.

## Per-die execution (Rubin locality domains)

A Rubin GPU has two locality domains ("dies"). DeepGEMM's per-die variant (`DG_MEGA_MOE_SM107_PERDIE=1` with die-local routed weights, the production recipe) keeps the routed expert weights die-local, runs two task queues (one per
domain, with stealing) and reads the weights through 3D tensor maps. The host side lives in `locality.py`, **pure Python on cuda-python 13.4 (no `deep_gemm`, no C++ compile)**:

```python
w1, w2 = mm.transform_weights_for_mega_moe(l1, l2)
if mm.is_localization_available():                       # cc 10.7 and exactly 2 locality domains
    mm.get_balanced_sm_locality_domains()                # probes the SM -> domain map once (outside CUDA graph capture), cached
    w1, w2, _, _ = mm.localize_mega_moe_weights(w1, w2)  # (E_l, N, K/2) -> (2, E_l, N/2, K/2); scales stay put
mm.fp8_fp4_mega_moe(y, w1, w2, buf, ...)                 # die-local weights select the per-die kernel variant by themselves
```

* `localize(t, dim=-2)`: `(*B, N, K) -> (2, *B, N/2, K)`, slice `d` allocated in locality domain `d` with the CUDA 13.4 native driver API (`cuMemCreate` on `CU_MEM_LOCATION_TYPE_DEVICE_LOCALITY_DOMAIN`, slices padded to the
  2 MiB allocation granularity, exported to torch through `__cuda_array_interface__`). `localize_mega_moe_weights(l1, l2, shared_l1=None, shared_l2=None)` localizes the routed pair (the production recipe); the shared pair can be localized
  too (DeepGEMM accepts that) but the port's launch refuses die-local shared weights, keep it plain. `weights_localized(...)` / `is_localized(...)` introspect.
* `get_balanced_sm_locality_domains()`: `uint8[num_sms]` indexed by the physical `%smid`, from DeepGEMM's latency probe (embedded as PTX, JIT-loaded by the driver) and balanced to equal TPC counts (an imbalanced map
  hangs per-die scheduling). It allocates and synchronizes: call it before CUDA graph capture. The launch passes its pointer as the kernel argument `sm_locality_domains`.
* Selection is automatic (per-die iff the routed weights are die-local); `frontend.set_per_die_mode(True / False)` forces per-die scheduling on plain weights / the plain scheduler on die-local weights
  (DeepGEMM's two ablation modes). Die-local weights must be the full tensors returned by `localize` (the kernel bakes the padded slice stride into its tensor map).
* The port has no limit on the slice stride; DeepGEMM's per-die launcher asserts `w.stride(0) <= INT_MAX` elements, so e.g. 384 local experts at EP1 (slice 2.27 GB) cannot run per-die in DeepGEMM (DEP>=2 production shapes can).
* Limits: one CUDA device per process, exactly 2 domains, granularity 2 MiB (checked), no EPLB (die-local weights cannot be moved expert by expert).

## moe_ep (Tier 2) backend

`sm107_mxfp8_mxfp4_bf16_deepgemm_cutedsl` (`@register_mega_kernel`, config `Sm107_Mxfp8_Mxfp4_Bf16_DeepgemmCutedsl_MegaMoeConfig`) drives the kernel from `MoEEpLayer` / `MoEEpMegaLayer` with a
`MegaConfig` like FlashInfer's other mega kernels: canonical pre-quantized `MoEWeightPack` (packed MXFP4 + E8M0 scales as `uint8` / `float8_e8m0fnu`, or `float32` UE8M0 values as produced for the
`sm100_fp8_fp4_bf16_deepgemm` backend; the same pack feeds both), BF16 hidden states quantized while staging, `layer.warmup(t)` before capture, `per_die=True` localizes the routed weights at preprocessing.
The backend module is imported on demand: `flashinfer.moe_ep` resolves the kernel name through a lazy registry (`moe_ep/core/kernel/registry.py`) that imports `flashinfer.experimental.sm107_deepgemm_megamoe.backend` at the first `create_mega_kernel` of that name, and
`Sm107_Mxfp8_Mxfp4_Bf16_DeepgemmCutedsl_MegaMoeConfig` is a lazy attribute of `flashinfer.moe_ep` (or import it from `flashinfer.experimental.sm107_deepgemm_megamoe.config`); `import flashinfer`, `flashinfer.aot` and `flashinfer.moe_ep` do not import the experimental package.

Behavior notes of the backend: `apply_topk_in_fc1=False` is refused (the kernel always applies the top-k weight in the FC1 epilogue); a weight pack that carries extra fields (fused shared experts: `shared_*` tensors) and the NVFP4-only `MoEEpTensors` scalars (`fc1_alpha`, `fc2_alpha`, `fc1_norm_const`) are refused instead of being dropped;
the first launch of a workspace (what `layer.warmup(t)` does, including a one-token warm-up) builds EVERY BLOCK_M variant of the geometry (`precompile_on_warmup=True`, about 1-1.5 s per variant cold, about 0.1 s from the disk cache; `False`: variants compile at their first use), so a later CUDA graph capture at another token count finds its variant and `stage_inputs` refuses a capture whose variant is not built;
`tag_mode="gridid"` warns at construction and CUDA graph capture is refused under it (DeepGEMM's `%gridid` tag is stale under graph replay).

**What is missing for parity with the native (DeepGEMM-API) path ("Route A")**: (1) *fused shared experts*: the kernel supports them, but the canonical `MoEWeightPack` / `MoEEpTensors` have no shared-expert fields
(FP8 `shared_w13 [2I, H]`, `shared_w2 [H, I]` and their UE8M0 scales; the SM120 Cake backend's `CakePrequantizedMoEWeightsWithSharedExpert` is the pattern), so the config rejects `num_shared_experts > 0`; (2) *per-die in an adapter
that aliases the transformed weights back into framework parameters*: die-local weights `(2, E_l, N/2, K/2)` are not expert-major, which such adapters (vLLM's `flashinfer_moe_ep`) require, so an API for per-expert views of
the transformed weights would be needed; `per_die=True` itself works through `MoEEpLayer`; (3) pre-quantized activations (`quantize_input=False`) are not supported; (4) the padding mask (`is_padding`) of the native path is not carried by `MoEEpTensors` (`stage_inputs_for_mega_moe` accepts it).

## Layout of the package

| file | content |
|---|---|
| `config.py` | `Sm107KernelConfig` (every compile-time constant of one kernel variant), `Sm107MegaMoeConfig`, the moe_ep backend config |
| `heuristics.py` | pure-Python bit-exact port of DeepGEMM's host logic: BLOCK_M selection, ring sizing, stages / smem, `SharedStorage` offsets, TMEM, registers, symmetric-buffer byte layout, `MegaMoESignals` offsets, per-die tile coordinates and TMA geometry, the support predicate |
| `workspace.py` | `SymmBuffer` (DeepGEMM-compatible), `get_symm_buffer_for_mega_moe` |
| `weights.py` | `transform_weights_for_mega_moe` (gate/up 8-row interleave, UTCCP SF transpose), `sf_to_required_layout` |
| `staging.py` / `staging_triton.py` | input staging: torch reference + single-launch Triton kernel (default on CUDA, bit-identical; adapted from vLLM's `prepare_megamoe_inputs`, Apache-2.0) |
| `locality.py` | per-die host helpers: die-local weight allocation, SM -> locality-domain map (latency probe + balancer), introspection |
| `frontend.py` / `compat.py` | `fp8_fp4_mega_moe` with DeepGEMM's host asserts, per-call variant selection, kernel options, drop-in names |
| `backend.py` | moe_ep (Tier 2) backend; imported on demand, never by the package `__init__` |
| `jit.py` | compile + disk cache of the kernel variants (`get_or_compile_kernel`, `is_compiled`, `precompile_variants`) |
| `kernel/` | the CuTe DSL kernel: `kernel.py` (host launcher, prologue, role dispatch), `scheduler.py` / `scheduler_perdie.py`, `tma_warps.py`, `mma_warp.py`, `epilogue.py` (GEMM roles), `dispatch.py`, `combine.py` (communication roles), `ptx*.py`, `comm.py`, `layout.py`, `trace.py`; `docs/sm107_deepgemm_megamoe/DESIGN.md` is the design contract |

## Tests, examples, benchmark

All tests live under `tests/experimental/` (excluded from `pytest tests/`; run explicitly). Host side (CPU / one Rubin GPU; `deep_gemm` is the oracle where available):

```bash
pytest tests/experimental/test_sm107_deepgemm_megamoe_policy.py      # policy: lazy import, no aot, warn-once, no environment access, key coverage
pytest tests/experimental/test_sm107_deepgemm_megamoe_heuristics.py  # CPU tables; GPU + deep_gemm: zero mismatches vs DeepGEMM's C++ host functions
pytest tests/experimental/test_sm107_deepgemm_megamoe_weights.py     # bit-exact vs deep_gemm.transform_weights_for_mega_moe
pytest tests/experimental/test_sm107_deepgemm_megamoe_staging.py     # bit-exact vs per_token_cast_to_fp8 (+ vLLM's Triton kernel)
pytest tests/experimental/test_sm107_deepgemm_megamoe_api.py         # signatures, SymmBuffer layout vs DeepGEMM, host asserts, kernel options
pytest tests/experimental/test_sm107_deepgemm_megamoe_backend.py     # moe_ep backend (stub kernel): weights, lifecycle, capture readiness
pytest tests/experimental/test_sm107_deepgemm_megamoe_locality.py    # per-die helpers vs DeepGEMM's ops (needs a 2-domain GPU)
FLASHINFER_SM107_DG_PRINT_CONFIGS=1 pytest ...heuristics.py -k print_configs  # compares stages/smem with DG_PRINT_CONFIGS output (opt-in: compiles DeepGEMM kernels)
```

Kernel (one Rubin GPU unless noted; reference = DeepGEMM with `DG_MEGA_MOE_SM107=1`): `test_sm107_dg_full_kernel_vs_deepgemm.py` (full kernel, S=0 and S=1, all BLOCK_M, ring wrap, masked routes, graph replay),
`test_sm107_dg_perdie_ep1.py` / `test_sm107_dg_perdie_model.py` (per-die), `test_sm107_dg_foundation_*.py` (layout, JIT, PTX helpers, comm, stub-role skeleton), `test_sm107_dg_gemm_core_*.py`, `test_sm107_dg_dispatch_combine*.py`;
multi-rank: `torchrun --nproc-per-node 2 -m pytest tests/experimental/test_sm107_deepgemm_megamoe_symm_multirank.py` (symmetric-buffer layout on all ranks) and the EP2/EP4 harnesses `tests/experimental/_sm107_dg_full_ep.py` / `_sm107_dg_perdie_full_ep.py`.

Runnable example: `examples/experimental/sm107_deepgemm_megamoe.py` (`python ... --shared 1 --check deepgemm`, `torchrun --nproc-per-node 4 ... --per-die`; with `--check deepgemm` the script sets `DG_MEGA_MOE_SM107=1` for the DeepGEMM reference, without it DeepGEMM would run its SM100 kernel). A/B benchmark against DeepGEMM (kernel durations from the profiler,
interleaved, per-M ratios; multi-node via `WORLD_SIZE` = nodes): `benchmarks/moe_ep/backends/mega/kernel/sm107/bench_moe_ep_sm107_deepgemm_cutedsl_mega.py`, e.g.
`python <bench> --mode bench --impl both --num-processes 4 --num-shared-experts 1,0 --m-list core --ab-swap --out ab.jsonl` (see its docstring for `check`, `gate0`, `graphrace`, `stress` modes and
`--port-tag-mode` / `--port-regs` / `--perdie`).

## Attribution

Port of DeepGEMM (MIT License, DeepSeek) `sm107_fp8_fp4_mega_moe.cuh` and its host code; PTX helper patterns derived from FlashInfer's vendored NVIDIA CuTe-DSL MegaMoE (BSD-3-Clause); the Triton staging kernel is adapted from vLLM's
`prepare_megamoe_inputs` (Apache-2.0).

## License and attribution

The package is Apache-2.0 (FlashInfer). It is a port of DeepGEMM's Rubin MegaMoE kernel `sm107_fp8_fp4_mega_moe`, of its host heuristics and of its Python API: DeepGEMM is MIT-licensed, Copyright (c) 2025 DeepSeek (`licenses/LICENSE.deepgemm.txt`, `NOTICE`; the headers of the derived files say so). The inline-PTX helpers and NVLink communication primitives (`kernel/ptx.py`, `ptx_d.py`, `ptx_g.py`, `comm.py`, `mma_warp.py`) derive from FlashInfer's vendored NVIDIA CuTe-DSL MegaMoE (BSD-3-Clause, NVIDIA CORPORATION & AFFILIATES); `staging_triton.py` is adapted from vLLM's `prepare_megamoe_inputs` (Apache-2.0, Copyright contributors to the vLLM project).
