# SM107 DeepGEMM-compatible MegaMoE: kernel design

Design document of the CuTe DSL kernel `DeepGemmSm107MegaMoeKernel` (`flashinfer/experimental/sm107_deepgemm_megamoe/kernel/`): launch geometry and argument ABI, compile-time configuration,
the device-side object model and DSL idioms, the prologue / epilogue sequence, the shared-memory, mbarrier and TMEM maps, named barriers, the symmetric-buffer workspace, the register policy,
the PTX helper API, the trace slots, and the measured deviations from DeepGEMM. The kernel is a port of DeepGEMM's `sm107_fp8_fp4_mega_moe_impl`: the kernel mapping (warp roles, tiles, pipeline,
scheduling order, communication protocol, epilogue numerics) is DeepGEMM's; the host side (heuristics, symmetric-buffer layout, weight layout) is documented in the package README.

**Source references.** Line references of the form `K:<lines>`, `IMPL:<lines>`, `HEUR:<lines>`, `LAY:<lines>`, `SCH:<lines>` point into DeepGEMM (branch `rubin-megamoe-a7a9277-1016` @ `3ec0eb4`):
`K` = `impls/sm107_fp8_fp4_mega_moe.cuh` (device kernel), `IMPL` = `csrc/jit_kernels/impls/sm107_fp8_fp4_mega_moe.hpp` (host launcher), `HEUR` = `csrc/jit_kernels/heuristics/mega_moe.hpp`,
`LAY` = `layout/mega_moe.cuh`, `SCH` / `SCH0` = `scheduler/sm107_mega_moe.cuh` / `scheduler/mega_moe.cuh`.

## 1. File split (all under `flashinfer/experimental/sm107_deepgemm_megamoe/`)

| file | content |
|---|---|
| `config.py` (package level) | `Sm107KernelConfig` (frozen dataclass: EVERY compile-time constant) + `Sm107MegaMoeConfig` |
| `kernel/config.py` | `KernelBuild` (= `Sm107KernelConfig` + `mode_mask`, `pdl`, `arch`), constants (warp ids, barrier ids, mode bits, register split, TMEM cols) |
| `kernel/layout.py` | pure-Python (no cutlass import) byte layouts: `SharedStorage` offsets, `MegaMoESignals` offsets, symmetric-buffer regions; cross-checks `cfg.smem_offsets` |
| `kernel/ctx.py` | `KernelCtx` + device-side views (`SmemMap`, `Barriers`, `Workspace`, `BufferMap`, `TmemMap`, `NamedBarriers`, `Trace`) |
| `kernel/ptx.py` | every inline-PTX helper (section 13) |
| `kernel/comm.py` | `SymBuffer` (+ host payload), `grid_sync`, `nvlink_barrier` |
| `kernel/trace.py` | DeepGEMM-compatible phase trace (section 14) |
| `kernel/kernel.py` | `DeepGemmSm107MegaMoeKernel`: host `@cute.jit __call__` + `@cute.kernel` body, prologue, role dispatch, epilogue-end TMEM free |
| `kernel/stubs.py` | stub role functions used while a role module does not exist yet (section 5.3) |
| `kernel/__init__.py` | exports `DeepGemmSm107MegaMoeKernel`, `KernelBuild`, `MODE_*` |
| `jit.py` (package level) | compile + disk cache: `get_or_compile_kernel(cfg, ...)` |
| `kernel/scheduler.py`, `tma_warps.py`, `mma_warp.py`, `epilogue.py`, `ptx_g.py` | `scheduler_role`, `tma_a_role`, `tma_b_role`, `mma_role`, `epilogue_role` (+ optional `make_tma_params`) |
| `kernel/dispatch.py`, `combine.py`, `ptx_d.py` | `dispatch_role`, `combine_role` |
| `heuristics.py`, `workspace.py`, `weights.py`, `staging.py`, `frontend.py` ... | host side; `Sm107KernelConfig` factory `heuristics.get_sm107_kernel_config(...)` |

Import rules: role modules import `from . import ptx`, `from .comm import ...`, `from .config import ...`, `from .layout import ...`
(never `from .kernel import ...`: `kernel.py` imports the role modules, not the other way round). No cutlass import at module level in
`layout.py`/`config.py` (they stay CPU-importable). Every new file carries the header
`# SPDX-License-Identifier: Apache-2.0` + `# Port of DeepGEMM (MIT License, DeepSeek) sm107_fp8_fp4_mega_moe.cuh / <file>; PTX helper patterns derived from FlashInfer's vendored NVIDIA CuTe-DSL MegaMoE (BSD-3-Clause).`

---------------------------------------------------------------------------------------------------------------------------
## 2. Launch geometry and kernel argument ABI

Launch (exactly DeepGEMM's, `IMPL:357-362`, K:70): `grid=(cfg.num_sms,1,1)` (212 on Rubin VR200; `num_sms` = `multi_processor_count`, compile-time constant of the
kernel), `block=(512,1,1)`, `cluster=(2,1,1)`, dynamic smem = `cfg.smem_bytes` (231156/229348/232404/217508/224148/216708 for BLOCK_M 16/32/64/128/192/240),
`min_blocks_per_mp=1` (== `__launch_bounds__(512,1)`: `nvvm.maxntid=512`, `nvvm.minctasm=1`), `use_pdl=build.pdl` (default **False**, like DeepGEMM `enable_pdl=false`).
All 212 CTAs must be co-resident (software grid syncs): `get_max_active_clusters(2) >= num_sms/2` is asserted by `jit.py` at plan time.
The kernel always executes `griddepcontrol.wait` once in the prologue (no-op without the PDL attribute); it never executes `griddepcontrol.launch_dependents`.

`DeepGemmSm107MegaMoeKernel(build: KernelBuild)`; `__call__` positional ABI (TVM-FFI friendly: pointers + scalars + the by-value peer table; no torch objects):

| # | name | type | meaning |
|---|---|---|---|
| 1 | `y` | `cute.Pointer` gmem BF16 | output `(num_tokens, H)` row-major contiguous, 16 B aligned |
| 2 | `stats` | `cute.Pointer` gmem Int32 (address 0 = none) | `cumulative_local_expert_recv_stats` (`E/R` int32) |
| 3 | `num_tokens` | `Int32` | live tokens of THIS rank (== `y.size(0)`), runtime |
| 4 | `sym` | `comm.SymBufferHost(base, offsets[R], rank)` | base address of the local symmetric buffer, peer offsets `peer_base[i]-local_base` (by value, R<=32 entries here; DSL>=4.7 allows any width), own rank |
| 5-8 | `l1_w`, `l1_w_sf`, `l2_w`, `l2_w_sf` | gmem pointers (Uint8 / Int32) | packed-FP4 routed weights `(E_l, 2I, H/2)` bytes, `(E_l, H, I/2)`; int32 UE8M0 SFs in the DeepGEMM MN-major / UTCCP layout |
| 9-12 | `shared_l1_w`, `shared_l1_w_sf`, `shared_l2_w`, `shared_l2_w_sf` | gmem pointers, address 0 when S=0 | shared-expert FP8 weights (later slice; the args exist from v1) |
| 13 | `trace` | `cute.Pointer` gmem Uint64 (0 = none) | `num_sms*(64+4*64)` u64 trace buffer, only read when `cfg.trace` |
| 14 | `sm_locality_domains` | gmem Uint8 pointer (0 = none) | per-die builds (`cfg.num_locality_domains == 2`): `uint8[num_sms]` indexed by the PHYSICAL SM id (`%smid`), balanced map required; read by the leader CTA's scheduler warp (`perdie.load_domain`); `jit.MegaMoeRunner` refuses to launch/bind such a build without it |
| 15 | `stream` | `cuda.CUstream` (fake stream for TVM-FFI compile) | |

**TVM-FFI calling convention (measured; implemented in `jit.py`)**: the compiled function has the 15 parameters of the table in this order; `cute.Pointer` parameters are `DataPointer`s and accept Python **ints**
(`tensor.data_ptr()`, 0 for "none"; torch tensors are rejected); `num_tokens` is an int; `sym` is passed as the nested tuple `(base, (off_0, ..., off_{R-1}), rank, R)` (the dataclass fields in order incl. the constexpr
`max_ranks`, which must be passed back as the same int); `stream` is an explicit int handle (`torch.cuda.current_stream().cuda_stream`): the `EnvStream` mode cannot be used because it needs a GPU-tensor parameter.
`jit.MegaMoeRunner` hides all of this (host-level call with torch tensors + `SymmBuffer`; `call_raw(...)` takes the ints).

Everything else the kernel needs is DERIVED: all regions of the symmetric buffer (`x`, `x_sf`, `topk_idx`, `topk_weights`, ring buffers, combine buffers, workspace) from
`sym.base` + `layout.buffer_layout(cfg)` (byte-identical to DeepGEMM's `MegaMoEBuffer`), the 18 tensor maps / TMA atoms from those regions + the weight pointers + `cfg` (built in `__call__`, section 4.4).
Host preconditions (asserted by `frontend.py`, repeated in `jit.py`): the symmetric buffer is zeroed at the first launch, all ranks use identical `cfg`-independent layout
(layout is BLOCK_M independent), `num_tokens <= T_max`, `E_l = E/R`, `R <= 32` for the by-value table of this port.

---------------------------------------------------------------------------------------------------------------------------
## 3. Compile-time configuration

`cfg` = the package's `Sm107KernelConfig` (field names are final; see `config.py` there). The kernel reads ONLY these fields (never environment variables):
`num_ranks, num_experts, num_experts_per_rank, hidden, intermediate_hidden, num_shared_experts, num_topk, num_sms, num_max_tokens_per_rank, block_m, block_n(=128), block_k(=128), umma_k(=64),
load_block_m(=BM/2), store_block_m, sf_block_m, sf_block_n(=128), wg_block_m(=BM/2), num_stages, smem_bytes, accum_stages(=2), tma_store_stages(=2), schedule_stages(=2),
smem_offsets{...}, smem_reusable_bytes, regs_dispatch/regs_non_epilogue/regs_epilogue, tmem_accum_cols, tmem_sfa_cols, tmem_sfb_cols, tmem_cols_alloc, tmem_sfa_col, tmem_sfb_col,
num_ring_tokens, num_sf_ring_tokens, num_ring_blocks, num_shared_sf_tokens, num_bytes_per_pull, num_tokens_per_warp(=32//topk), num_experts_per_lane, num_ranks_per_lane,
num_combine_chunks(=2), combine_chunk_bytes(=5120), combine_chunk_uint4_per_lane(=10), clamp, alpha, beta, fast_math, weight_packed_fp4, num_locality_domains, trace, debug_flags`.
`KernelBuild(cfg, mode_mask=7, pdl=False, arch="sm_107a", ptxas_options, opt_level, tag_mode="epoch", regs_override=None, ...)` adds the fields that are not part of DeepGEMM's template list; the compile-cache name digest covers cfg digest + `mode_mask` + `pdl` + `arch` + `trace` + DSL options + `tag_mode` + `regs_override` (jit.py).
`timeout_sec` (default 300 = DeepGEMM's `kNumTimeoutCycles` at 2 GHz) is the spin-wait trap timeout baked into the PTX of every software barrier / signal wait: `jit.compile_kernel_object` sets `comm.TIMEOUT_CYCLES` and `ptx_d.NUM_TIMEOUT_CYCLES` for the duration of a compile, and it is part of the digest (PTX-verified: `timeout_sec=10` bakes `20000000000`, default `600000000000`); hang tests pass a short `timeout_sec` (e.g. 10).
`jit.py` passes the target chip explicitly (`cute.GPUArch(build.arch)`) and does NOT touch the environment (the DSL auto-detects `sm_107a` on SM107 devices; compile-only hosts without that GPU export `CUTE_DSL_ARCH=sm_107a`).
Roles read the config as `ctx.cfg.<field>`; every such value is a Python int/float/bool (constexpr). Use `cutlass.const_expr(...)` / `range_constexpr` for anything derived from it.
**Numerics options and refused knobs**: `fast_math` True (DeepGEMM `kFastMath`: `__expf` + `rcp.approx`) and False (IEEE `expf` exactly as nvcc 13.4 expands it + `div.rn.f32`: `ptx_g.expf_ieee` / `div_rn_f32`, `epilogue._epilogue_l1` `const_expr(cfg.fast_math)` branch) are BOTH implemented (`expf_ieee` == nvcc `expf` over all 2^32 float bit patterns, `div_rn_f32` == nvcc `a/b` on 16.8 M pairs, 216 units x 9 inputs vs `deep_gemm(fast_math=False)` bit-exact at EP1); `clamp` None / inf (compiled out) / any value, `alpha`, `beta` are implemented. The roles REFUSE at trace time (Python assert naming `cfg.<field>`, "not implemented by the CuteDSL port") non-default `epi_prefetch_stages` (!= 0), `tma_store_stages` (!= 2), `debug_flags` (!= 0), `weight_prefetch` (!= 0): they are part of the compile key but the roles ignore them (the package's heuristics set exactly the defaults).

### 3.1 mode_mask (staged testing, compile-time constexpr)

| bit | name | when 1 | when 0 |
|---|---|---|---|
| 0 (`MODE_DISPATCH`=1) | dispatch roles | `dispatch_role(ctx)` runs (counting, pushes, grid syncs A, NVLink barrier #1, **R1**, pull loop, **R2**, cleanup, NVLink barrier #2) | warps 0-3 only execute **R1 then R2** (`barrier.sync 1,384` twice) and exit |
| 1 (`MODE_GEMM`=2) | GEMM roles | `scheduler_role`, `tma_a_role`, `tma_b_role`, `mma_role`, `epilogue_role` run | those warps skip their bodies (the epilogue warps still execute R1 and the TMEM free) |
| 2 (`MODE_COMBINE`=4) | combine / epilogue tail | `combine_role(ctx)` runs (peer_grid_idx load, epilogue grid sync idx 1, SM0 ready flags, **R2**, combine loop) | epilogue warps execute **R2 only** |

Test-only build flags (all in the compile-cache digest, never set by the production path): `KernelBuild.debug_marks` (visit counters in `trace_ptr`), `debug_gridsync_iters` (stub dispatch role runs N software grid syncs), `dispatch_only_sync` (-> `ctx.d_test_pull_done_sync`: `dispatch.py` adds one grid sync idx 1 before R2 in dispatch-only launches), `roles="stubs"`.

`mode_mask=7` is the production kernel. The skeleton rule: a disabled role still executes every CTA-level rendezvous it would otherwise take part in, in the same order
(cluster syncs of the prologue, the 384-thread barrier id 1 twice per dispatch/epilogue warp, TMEM alloc/free). Cross-CTA/cross-rank protocols (grid syncs, NVLink barriers, counters) are skipped
consistently on ALL CTAs by the disabled bit, so staged tests must stage the corresponding global state on the host (tests: `tests/experimental/replay_utils.py`).

---------------------------------------------------------------------------------------------------------------------------
## 4. Device-side object model (KernelCtx) and DSL idioms

### 4.1 Idiom rules (the DSL traces Python; **[V]** = verified by experiment with CuTe DSL 4.8.0)
1. **Every role function / helper with dynamic control flow is `@cute.jit`** (plain Python functions cannot contain dynamic `if/while/for`); pure arithmetic helpers may be plain. **[V]**
2. **`break` / `continue` / `return` are NOT allowed inside staged `if`/`while`/`for` regions** — kernel or `@cute.jit` — (`error[UNSUP_EARLY_EXIT]: Early exit (break|continue|return) is not allowed in ... A staged for/while/if runs as a whole`).
   Use a predicate-carried loop: `active = Boolean(True)`; `while active: ...; if done: active = Boolean(False)` (nested carried loops work). `while True:` without a carried predicate is an infinite loop. **[V]**
3. **A variable assigned inside an `if` (or loop) and used after it must be initialised BEFORE the `if`** (`v = Int32(0); if c: v = ...`), otherwise `NameError`. **[V]**
4. **No closures that capture runtime (dynamic) values** inside staged regions (`error[SCOPE_CLOSURE_CAPTURE]: Function f captures variable n`): use module-level helpers with explicit arguments. `lambda: ...` sync scopes are therefore
   replaced by static descriptors (`comm.grid_sync(..., sync_scope=(bar_id, nthreads))`). **[V]**
5. **Objects whose METHODS you call inside a staged region** (`ctx.mb.full(i).init(1)`, `sched.get_next_task()` ...) are treated as possibly mutated and carried through the region (the root name of the attribute chain; `self` and
   module names are exempt). A plain Python class fails with `TYPE_DYNAMIC_EXPR_UNSUPPORTED ... plain Python value (a KernelCtx)`. Such objects (the context, views, role-local helpers like a scheduler) MUST be declared
   `@static_dataclass class X(DslStatic)` from `kernel/dslutil.py` (frozen dataclass = read-only + empty MLIR value list; both are needed because the two DSL preprocessor paths — native, and the python reference path selected e.g. by
   `CUTE_DSL_KEEP=...` — treat them differently); set fields in `__init__` with `init_fields(self, a=..., b=...)`. Objects implementing the DSL value protocol (`ptx.Mbar`) are fine as they are. **[V]**
6. **`ctx` (and every static object) is read-only inside roles**: attribute assignment inside a dynamic loop is a compile error (`TYPE_DYNAMIC_EXPR_UNSUPPORTED ... a value carried through this while is a plain Python value`).
   Loop-carried state (stage/phase counters, task info, k-block counters, ...) lives in LOCAL Int32/Boolean variables (or tuples returned by nested `@cute.jit` helpers). **[V]**
   Do not alias a non-static member into a local and call methods on it inside a region (`mb = ctx.mb; mb.full(i)` — fine only because `Barriers` is static; a hand-written class must follow rule 5).
7. Compile-time knobs (`cfg.*`, `mode_mask`) are Python values: branch with `if cutlass.const_expr(...)`, unroll with `cutlass.range_constexpr(...)` (a plain `range(n)` over an `Int32` is a dynamic scf loop). Plain Python `NamedTuple`s of
   Int32 work as kernel arguments and pass through to roles. **[V]**
8. Never call `cute.arch.sync_threads()` / `cute.arch.barrier()` without an explicit id: it is `bar.sync 0` and id 0 is the dispatch barrier. All CTA barriers go through `ctx.bar.*` (section 9) or `ptx.bar_sync*`.
9. `with cute.arch.elect_one():` is a region (no early exits inside); `ptx.elect_sync()` returns a predicate for `if` use. **[V]** (`elect.sync` PTX confirmed in the skeleton)
10. **Dynamic barrier/stage index**: build mbarrier pointers as `cute.recast_ptr(base + static_off, dtype=Int64) + i` (index in Int64 units AFTER the recast); `recast_ptr(base + (off + i*8))` loses the alignment fact and the TMA `tma_bar` operand is rejected (`ctx.mb.*` already does this).
11. All PTX goes through `ptx.py` (or the role's own `ptx_g.py`/`ptx_d.py`); every asm is `has_side_effects=True` and memory-clobbering where ordering matters. Do not add fences/proxy fences that DeepGEMM does not have.
13. **Address arithmetic is Int64; products of Python ints are folded on the host.** A DSL `Int32 * python_int` STAYS Int32 and wraps modulo 2**32 (measured, DSL 4.8.0: `Int32(5) * 335544320` -> Int32 `1677721600`; `700_000_000 + Int32(5) * 335544320` -> Int32 `-1917245696`; `3_000_000_000 + Int32(5)` -> Int64 (a literal that does not fit Int32 promotes); `Int64(Int32(5)) * 335544320` -> Int64). The symmetric buffer is > 2 GiB for production shapes (DEP32 E=384 T_max >= 7680, DEP4 E=384 T_max >= 24960, EP1 E=96 T_max >= 32640), so every byte offset that can reach 2 GiB must be `Int64(runtime_value) * Int64(host_folded_constant)`; `BufferMap.region_addr`, `Workspace.addr` and `DispatchAddrs.*` are the sanctioned forms. Guarded by `tests/experimental/test_sm107_dg_foundation_addr.py` (CPU window scan of the real `region_addr` with the DSL's own scalar semantics + PTX grep: the combine slot stride only in `mul.wide`/`mul.lo.s64`, no 32-bit `mul.lo`/`mad.lo` with an immediate >= 4 MiB). **[V]**
12. DSL tooling: `CUTE_DSL_KEEP=ptx,cubin` (+ `CUTE_DSL_DUMP_DIR=<dir>`) dumps PTX/cubin of every compile (the DSL's own SASS dump needs `nvdisasm` on its lookup path; otherwise run `nvdisasm` / `cuobjdump -sass` on the kept cubin); `cute.compile[cute.PtxasOptions("--register-usage-level=10"), cute.OptLevel(3), cute.EnableTVMFFI(True)]`.

### 4.2 `KernelCtx` (constructed by `kernel.py` right after the prologue, identical object for every role; `kernel/ctx.py`)

```
ctx.cfg                      Sm107KernelConfig (constexpr)
ctx.build                    KernelBuild (mode_mask, pdl, arch)
-- ids (runtime Int32; warp_idx is warp-uniform: it comes from threadIdx.x // 32 via make_warp_uniform)
ctx.sm_idx                   blockIdx.x                               (K: sm_idx)
ctx.thread_idx               threadIdx.x                              (0..511)
ctx.warp_idx                 canonical warp index (0..15)
ctx.lane_idx                 %laneid
ctx.cta_rank                 block rank in cluster (0 leader, 1 peer)
ctx.is_leader_cta            Boolean(cta_rank == 0)
-- arguments
ctx.y, ctx.stats, ctx.num_tokens, ctx.trace_ptr, ctx.sm_locality_domains
ctx.sym                      comm.SymBuffer: .base (Int64), .rank_idx (Int32), .map(ptr_or_addr, rank) -> same kind (identity for R=1, compile-time)
ctx.l1_w, ctx.l1_w_sf, ctx.l2_w, ctx.l2_w_sf (+ shared_*)             gmem Pointers
ctx.tma                      TmaParams (opaque to the kernel body), or None (section 4.4)
-- views
ctx.smem                     SmemMap   typed smem pointers (section 6)
ctx.mb                       Barriers  mbarrier handles (section 7)
ctx.tmem                     TmemMap   TMEM columns (section 8); base column is 0 on both CTAs (asserted)
ctx.ws                       Workspace gmem field accessors of MegaMoESignals + src_token_topk_idx + token_src_metadata (section 10)
ctx.buf                      BufferMap addresses of the symmetric-buffer regions (section 10)
ctx.bar                      NamedBarriers (section 9)
ctx.tr                       Trace     (no-op unless cfg.trace)
```
Fields are Python attributes holding DSL values; all accessors return typed `cute.Pointer`s (gmem/smem) or `Int64` addresses (documented per accessor). Stage/phase helpers:
`ptx.advance_stage(stage, phase, num_stages) -> (stage, phase)` (DeepGEMM `advance_pipeline`: phase flips when `stage` wraps to 0).

### 4.3 Role functions (all `@cute.jit`, signature `def <name>(ctx) -> None`)

| function | module | warps (per CTA) | runs when | what the KERNEL does around it |
|---|---|---|---|---|
| `dispatch_role(ctx)` | `dispatch.py` | 0-3 (128 thr; `setmaxnreg.dec cfg.regs_dispatch` already executed) | bit0 | after return the warps exit (nothing follows) |
| `tma_a_role(ctx)` | `tma_warps.py` | 4 (both CTAs) | bit1 | `setmaxnreg.dec cfg.regs_non_epilogue` before |
| `tma_b_role(ctx)` | `tma_warps.py` | 5 (both CTAs) | bit1 | same |
| `mma_role(ctx)` | `mma_warp.py` | 6 (`is_leader_cta` only: the kernel calls it only on the leader CTA) | bit1 | same |
| `scheduler_role(ctx)` | `scheduler.py` | 7 (leader CTA only: kernel guard) | bit1 | same |
| `epilogue_role(ctx)` | `epilogue.py` | 8-15 (256 thr, 2 warpgroups; `setmaxnreg.inc cfg.regs_epilogue` executed) | bit1 | before: **R1** (`bar.sync_unaligned 1, 384`); after: TMEM free by epilogue warp 0 (always) |
| `combine_role(ctx)` | `combine.py` | 8-15 | bit2 | before: nothing; if bit2=0 the kernel executes **R2** itself |

Role bodies create their own role-local objects (scheduler consumer, pipeline counters...) at their start; nothing GEMM-specific is built by the kernel except barriers/TMEM.
Sequencing inside the epilogue warps (K:1291-1812): `R1` -> `epilogue_role` -> TMEM free (epilogue warp 0, `tcgen05.dealloc.cta_group::2`, all lanes) -> `combine_role`
(`[combine smem carve, peer_grid_idx load, grid_sync idx 1 over the 256 epilogue threads, SM0 st.release.sys ready flags, ctx.bar.r2(), combine loop]`).
**R2 contract:** `combine_role` MUST execute `ctx.bar.dispatch_epilogue_sync()` (barrier id 1, 384 threads, unaligned) exactly once, at the DeepGEMM position (K:1710, after the ready flags,
before the combine loop). `dispatch_role` MUST execute `ctx.bar.dispatch_epilogue_sync()` twice: R1 at K:491 (after NVLink barrier #1, before the pull loop) and R2 at K:752 (after the pull loop, before cleanup).
The epilogue's R1 (K:1292) is executed by the kernel before `epilogue_role` (the GEMM roles do NOT execute it).

### 4.4 TMA atoms / tensor maps (CuTe DSL atoms)
CuteDSL TMA atoms (`cpasync.make_tiled_tma_atom`) are created in `__call__` (host side) from the symmetric-buffer regions + weight pointers + `cfg`, passed to the kernel as one opaque `tma_params` argument and exposed as `ctx.tma`.
The factory is `tma_warps.make_tma_params(cfg, *, sym, l1_w, l1_w_sf, l2_w, l2_w_sf, shared_l1_w, shared_l1_w_sf, shared_l2_w, shared_l2_w_sf) -> TmaParams` (called by `kernel.py` only when the module exists and `build.roles != "stubs"`;
otherwise `ctx.tma is None`). `TmaParams` must implement the DSL dynamic protocol incl. `__extract_mlir_attributes__` (without it the atoms lose `grid_constant`); its fields are the atoms/tensors (`ctx.tma.<name>_atom/_t`).
The tensor-map prefetch of K:142-161 is optional: if `tma_warps.prefetch_tma_params(tma_params)` exists the kernel calls it from warp 0 before the prologue (`cpasync.prefetch_descriptor` does not legalize for 2-SM atoms in DSL 4.8.0, so
it is skipped).
Outcome: DSL atoms `CopyBulkTensorTileG2SOp(CtaGroup.TWO)` emit `cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint.cta_group::2` with the peer bit of the barrier address
cleared and an explicit cache policy; packed FP4 weights are viewed as a UINT8 tensor (box 64 B x 128 rows, SW64) which yields the identical smem image and expect_tx bytes as `16U4_ALIGN8B` (the `Float4E2M1FN`/ALIGN8B dtype itself is untested; documented deviation).
**L2_256B host tensor maps**: the DSL atoms encode L2_128B promotion, DeepGEMM L2_256B (-2.6..-4.1 % GEMM time at BLOCK_M >= 128); `TmaParams` therefore also carries host-created `TensorMap` leaves (routed `l1_w`/`l2_w`, shared `shared_l1_w`/`shared_l2_w`; only when `tma_warps.l2_promotion_bytes(cfg) > 0`, BLOCK_M >= 128 by default) as extra grid-constant kernel parameters (+128 B each: BM128 DEP4 S=0 parameter bank 2240 -> 2496 B, S=1 about 3.1 KB < 4 KB); `kernel.py` / `jit.py` unchanged (the compile cache key hashes `kernel/tma_warps.py`; export + disk-cache reload are covered by the tests).
Fallback if a map cannot be built by the DSL: host-encoded `CUtensorMap`s (`cuda.bindings.driver.cuTensorMapEncodeTiled` and `CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B` exist in cuda-python 13.4.1) in a 2304 B device buffer passed as one more pointer argument.

---------------------------------------------------------------------------------------------------------------------------
## 5. Prologue / epilogue sequence executed by the kernel (K:292-401, 1642-1648)

All 512 threads, in order (`kernel.py::_kernel`):
1. (warp 0) `prefetch.tensormap` of the 18 tensor maps (no-op while `ctx.tma is None`)  — K:142-161.
2. `cluster_arrive_relaxed(); cluster_wait()`  — K:293 (`cluster_sync_with_relaxed_arrive`).
3. Warp-specialised init — K:296-340:
   * warp 0: elected lane zeroes the `expert_token_count` region (`align(4E, 1024)` bytes at offset 0) (`st.bulk.weak.shared::cta` if ptxas accepts it, else vector stores);
   * warp 1: lanes `<4` init `dispatch_barriers[i]` (count 1); `fence_barrier_init`;
   * warp 2: elected lane inits `full[s]=4`, `empty[s]=1` (s < stages), `tmem_full[2]=1`, `tmem_empty[2]=512`, `combine[16]=1`, `task_info_full[2]=1`, `task_info_empty[2]=512`; `fence_barrier_init`;
   * warp 3: `tcgen05.alloc.exclusive.cta_group::2` (`ptx.tmem_alloc(slot, cols, 2, True)`; DeepGEMM's cubin uses the `.exclusive` form for all six BLOCK_M, `cute.arch.alloc_tmem` only emits it above 512 columns) of `cfg.tmem_cols_alloc` columns, address written to `tmem_ptr_in_smem`.
4. `cluster_arrive(); cluster_wait()` (full `cluster_sync`) — K:341.
5. `griddepcontrol.wait` (all threads, BEFORE the role split) — K:344. Trace slot 0/60 stamped by thread 0.
6. Role split with `setmaxnreg` (warp ranges and values: section 11). Roles then run as in section 4.3.
7. Epilogue warps only, after `epilogue_role`: TMEM free `tcgen05.dealloc.exclusive.cta_group::2` (`ptx.tmem_dealloc(0, cols, 2, True)`) by epilogue warp 0 (K:1645-1648; "same logical warp on both CTAs").
8. After the last role returns, the kernel flushes the trace accumulators (`kTrace` only) and exits. There is no CTA/cluster exit barrier (as in DeepGEMM): the MMA warp's final `tmem_empty` drain wait keeps the leader alive for remote arrivals.

---------------------------------------------------------------------------------------------------------------------------
## 6. Shared memory map (`SharedStorage`, K:250-275; authoritative offsets = `cfg.smem_offsets`, recomputed independently by `layout.smem_offsets(cfg)`)

Dynamic smem base is 1024-aligned. Members in order (byte offsets from the base; `align` = round up to 1024 unless noted): `expert_token_count[E] u32` @0 (size `4E`);
`dispatch_send_buffer[1][16]` @`align(4E,1024)`; `smem_d` union @ `+1024` (L1: `fp8 [2 WG][2 stages][SBM*64]`, L2: `bf16 [2 WG][SBM*128]`; size `align(512*SBM,1024)`);
`smem_a[stages][(BM/2)*128]` fp8; `smem_b[stages][8192]` packed fp4 (1024-aligned start); `smem_sfa[stages][SF_BLOCK_M] u32`; `smem_sfb[stages][128] u32`;
`amax_reduction[8][SBM/2] float2`; `task_infos[2]` (32 B each, `alignas(16)`); then the 8-byte mbarriers in this order: `dispatch_barriers[4]`, `full_barriers[stages]`, `empty_barriers[stages]`,
`tmem_full_barriers[2]`, `tmem_empty_barriers[2]`, `combine_barriers[16]`, `task_info_full_barriers[2]`, `task_info_empty_barriers[2]`; `tmem_ptr_in_smem` (u32).
`kNumReusableSmemBytes = offset(dispatch_barriers)`: the combine phase reuses `[0, reusable)` (3 slots x 8 warps x 5120 B = 122880 B, K:1677-1680) — all of items before the barriers are dead by then.

Offsets for E=384, V4.1, shipped stage counts (`layout.py` reproduces and asserts them, CPU test `test_sm107_dg_foundation_layout.py`):

| member | BM=16 (13 stg) | BM=32 (12) | BM=64 (11) | BM=128 (8) | BM=192 (7) | BM=240 (6) |
|---|---|---|---|---|---|---|
| `expert_token_count` | 0 | 0 | 0 | 0 | 0 | 0 |
| `dispatch_send_buffer` | 2048 | 2048 | 2048 | 2048 | 2048 | 2048 |
| `smem_d` (size) | 3072 (4096) | 3072 (8192) | 3072 (8192) | 3072 (16384) | 3072 (16384) | 3072 (20480) |
| `smem_a` (size) | 7168 (13312) | 11264 (24576) | 11264 (45056) | 19456 (65536) | 19456 (86016) | 23552 (92160) |
| `smem_b` (size) | 20480 (106496) | 35840 (98304) | 56320 (90112) | 84992 (65536) | 105472 (57344) | 115712 (49152) |
| `smem_sfa` | 126976 | 134144 | 146432 | 150528 | 162816 | 164864 |
| `smem_sfb` | 133632 | 140288 | 152064 | 154624 | 169984 | 171008 |
| `amax_reduction` | 140288 | 146432 | 157696 | 158720 | 173568 | 174080 |
| `task_infos` | 140544 | 146944 | 158208 | 159744 | 174592 | 175360 |
| `dispatch_barriers` (= reusable) | 140608 | 147008 | 158272 | 159808 | 174656 | 175424 |
| `full_barriers` | 140640 | 147040 | 158304 | 159840 | 174688 | 175456 |
| `empty_barriers` | 140744 | 147136 | 158392 | 159904 | 174744 | 175504 |
| `tmem_full_barriers` | 140848 | 147232 | 158480 | 159968 | 174800 | 175552 |
| `tmem_empty_barriers` | 140864 | 147248 | 158496 | 159984 | 174816 | 175568 |
| `combine_barriers` | 140880 | 147264 | 158512 | 160000 | 174832 | 175584 |
| `task_info_full_barriers` | 141008 | 147392 | 158640 | 160128 | 174960 | 175712 |
| `task_info_empty_barriers` | 141024 | 147408 | 158656 | 160144 | 174976 | 175728 |
| `tmem_ptr_in_smem` | 141040 | 147424 | 158672 | 160160 | 174992 | 175744 |
| sizeof (end rounded to 1024) | 141312 | 147456 | 158720 | 160768 | 175104 | 176128 |
| **dynamic smem requested (`cfg.smem_bytes`)** | 231156 | 229348 | 232404 | 217508 | 224148 | 216708 |

`ctx.smem` (typed views, all `cute.Pointer` in `AddressSpace.smem` with the right dtype/alignment; stage arguments are Int32 runtime values):
`smem.expert_token_count` (u32 ptr), `smem.send_buffer`, `smem.d_l1(wg, s)` (fp8), `smem.d_l2(wg)` (bf16), `smem.a(stage)` (fp8, 1024-aligned), `smem.b(stage)` (packed fp4/u8 ptr),
`smem.sfa(stage)` (u32), `smem.sfb(stage)` (u32), `smem.amax(warp)` (float2 ptr), `smem.task_info(slot)` (u32 ptr, 32 B), `smem.tmem_ptr` (u32 ptr),
`smem.reusable(offset_bytes)` (u8 ptr into the combine overlay), `smem.base` (u8 ptr), `smem.addr(ptr)` (Int32 shared address). Offsets are exactly `cfg.smem_offsets[...]`; stage strides: `a`: `(BM/2)*128` B,
`b`: 8192 B, `sfa`: `SF_BLOCK_M*4` B, `sfb`: 512 B.

---------------------------------------------------------------------------------------------------------------------------
## 7. mbarrier map (K:296-340)

`ctx.mb.<name>(i)` returns an `Mbar` (thin handle over the smem pointer; methods in section 13.4). Initialisation is done by the kernel (section 5); counts:

| barrier | init count | arrivals per phase | waiter(s) |
|---|---|---|---|
| `dispatch_barriers[4]` | 1 | unused in mode 3 (initialised for parity) | - |
| `full[stage]` | **4** (2 CTAs x (A+B)) | leader TMA-A `arrive_and_expect_tx`, peer TMA-A remote `arrive` on CTA0, leader TMA-B `arrive_and_expect_tx`, peer TMA-B remote `arrive` on CTA0; plus `complete_tx` of both CTAs' 2-SM loads (landing on CTA0's barrier, peer bit of the barrier address cleared: mask `0xFEFFFFFF`) | MMA warp (leader), `wait(phase)` |
| `empty[stage]` | 1 | one `tcgen05.commit.cta_group::2...multicast::cluster` (CTA mask 0b11, arrives on both CTAs at the same offset) per k-block | TMA-A and TMA-B of each CTA (own barrier), `wait(phase^1)` |
| `tmem_full[2]` | 1 | commit (mask 0b11) after the last k-block of a task | epilogue threads of each CTA, `wait(accum_phase)` |
| `tmem_empty[2]` | **512** (2 x 256) | every epilogue thread of both CTAs: `tcgen05.fence::before_thread_sync` then remote `arrive(cta 0)` | MMA warp (leader) `wait(accum_phase^1)` + final drain wait |
| `combine[16]` | 1 | elected lane `arrive_and_set_tx(chunk bytes)` after each 1D load | owning epilogue warp (2 per warp) |
| `task_info_full[2]` | 1 | scheduler lane c: remote `arrive.expect_tx(32)` on CTA c + `st.async` completing the 32 B | TMA-A, TMA-B, MMA(leader), epilogue |
| `task_info_empty[2]` | **512** | every epilogue thread of both CTAs after `tmem_full` (`fence.acq_rel.cta` then remote arrive) on slot `sched_stage^1` | scheduler warp, `wait(phase^1)` |

`ctx.mb.<name>(i)` builds `Mbar(recast_ptr(base + static_off, Int64) + i)` (idiom rule 10: runtime `i` keeps its 8-byte alignment fact). Phase conventions: smem stages flip `phase` when `stage` wraps; accumulator stage `i%2`, phase `(i/2)&1`; producers start with `phase^1` so the first wait passes.

---------------------------------------------------------------------------------------------------------------------------
## 8. TMEM map (K:283-290)

`ctx.tmem` — all values are plain Python ints (columns) because the base address is 0 on both CTAs (kernel asserts `tmem_ptr_in_smem == 0` in the epilogue):
`tmem.accum_col(stage) = stage*cfg.block_m`, `tmem.sfa_col = cfg.tmem_sfa_col (2*BM)`, `tmem.sfb_col = cfg.tmem_sfb_col (2*BM + SF_BLOCK_M/32)`, `tmem.alloc_cols = cfg.tmem_cols_alloc`.
A TMEM address as seen by `tcgen05.ld/st/cp/mma` operands is `lane_bits << 16 | column` (no warp bits: hardware ignores them; epilogue warp w can only reach lane quadrant `w % 4`). Allocation by warp 3 (`cta_group::2`, both CTAs), free by epilogue warp 0.

| BM | used cols (2*BM + SFA + SFB) | alloc | SFA col | SFB col |
|---|---|---|---|---|
| 16 | 40 | 64 | 32 | 36 |
| 32 | 72 | 128 | 64 | 68 |
| 64 | 136 | 256 | 128 | 132 |
| 128 | 264 | 512 | 256 | 260 |
| 192 | 396 | 512 | 384 | 392 |
| 240 | 492 | 512 | 480 | 488 |

---------------------------------------------------------------------------------------------------------------------------
## 9. Named barriers, cluster syncs, grid syncs (ids 0..4 are reserved for the kernel)

| id | threads | warps | kind | name in `ctx.bar` | DeepGEMM |
|---|---|---|---|---|---|
| 0 | 128 | 0-3 (dispatch) | aligned `bar.sync 0,128` | `ctx.bar.dispatch()` | `kDispatchBarrierIdx` |
| 1 | 384 | 0-3 + 8-15 | **unaligned** `barrier.sync 1,384` | `ctx.bar.dispatch_epilogue_sync()` (R1/R2) | `kDispatchWithEpilogueBarrierIdx` |
| 2 | 256 | 8-15 | aligned `bar.sync 2,256` | `ctx.bar.epilogue()` | `kEpilogueFullBarrierIdx` |
| 3, 4 | 128 | 8-11 / 12-15 (WG0 / WG1) | aligned `bar.sync 3+wg,128` | `ctx.bar.epilogue_wg(wg)` | `kEpilogueWGBarrierStartIdx + wg` |
| 5..15 | free | | | | (tests may use 8..15; other named-barrier helpers must not use 0..4) |

Grid-sync counters in `MegaMoESignals.grid_sync_count[4]`: idx 0 = dispatch (used 3x per launch: grid sync A, NVLink barrier #1 tail, NVLink barrier #2 head), idx 1 = epilogue (once).
NVLink barrier tags: 1 = before pull (`prologue=False, epilogue=True`), 2 = after workspace clean (`prologue=True, epilogue=False`). `comm.grid_sync(...)`/`comm.nvlink_barrier(...)` signatures in section 13.5.
Cluster scope: `cluster_arrive_relaxed+cluster_wait` (prologue 1), `cluster_arrive+cluster_wait` (prologue 2); no other cluster barrier exists in the kernel.

---------------------------------------------------------------------------------------------------------------------------
## 10. Workspace (`MegaMoESignals`) and symmetric buffer

`ctx.ws` returns typed gmem pointers into the LOCAL buffer (add `sym.map(p, rank)` for remote access). Byte offsets (V4.1; `layout.SIGNALS_*`; kNumDeviceLocalityDomains = 2):

| field | offset | dtype / count | accessor |
|---|---|---|---|
| `grid_sync_count[4]` | 0 | u32 | `ws.grid_sync_count(idx)` |
| `nvl_barrier_counter` | 16 | u32 | `ws.nvl_barrier_counter()` |
| `nvl_barrier_signals[2]` | 20 | s32 | `ws.nvl_barrier_signal(phase)` |
| `l1_task_count[2]` / `l2_task_count[2]` / `shared_l1_task_count[2]` / `shared_l2_task_count[2]` | 28 / 36 / 44 / 52 | u32 per locality domain | `ws.l1_task_count(dom=0)` ... |
| `combine_ready_grid_idx[72]` | 128 | u64 | `ws.combine_ready_grid_idx(rank)` |
| `peer_grid_idx[72]` | 768 | u64 | `ws.peer_grid_idx(rank)` |
| `expert_send_count[2048]` | 1408 | u64 (lo32 count, hi32 #SMs) | `ws.expert_send_count(e)` |
| `expert_recv_count[2048]` | 17792 | u64, index `src_rank*EPR + local_expert` | `ws.expert_recv_count(src_rank, local_expert)` |
| `expert_recv_count_sum[512]` | 34176 | u64 (lo32 total, hi32 = kNumSMs*kNumRanks when complete) | `ws.expert_recv_count_sum(local_expert)` |
| `l1_full_count[2^20]` | 38272 | u32 | `ws.l1_full_count(ring_block)` |
| `l1_empty_count[2^20]` | 4232576 | u32 | `ws.l1_empty_count(ring_block)` |
| `l2_full_mask[2^20]` | 8426880 | u64 | `ws.l2_full_mask(ring_block)` |
| `l2_empty_count[2^20]` | 16815488 | u32 | `ws.l2_empty_count(ring_block)` |
| `shared_l2_full_count[2^15]` | 21009792 | u32 | `ws.shared_l2_full_count(i)` |
| (end of `MegaMoESignals`) | 21140864 | | |
| `src_token_topk_idx[EPR][R][T_max]` | 21140864 | u32 | `ws.src_token_topk_idx(local_expert, src_rank, slot)` |
| `token_src_metadata[max_pool]` | +E*T*4 | 12 B `{rank, token, topk}` | `ws.token_src_metadata(pool_token)` (12 B pointer stride) |

`ctx.buf` (regions after the workspace; offsets from `layout.buffer_layout(cfg)` = DeepGEMM `MegaMoEBuffer`, LAY:345-457; the V4.1 DEP8 T=9600 S=1 example is checked in `tests/experimental/test_sm107_dg_foundation_layout.py`):
`x` (fp8 `(T,H)`), `x_sf` (int32 `(T,H/128)`), `topk_idx` (int64 `(T,6)`), `topk_weights` (f32), `shared_l1_acts_sf`, `shared_l2_token`, `shared_l2_sf`, `l1_token` (fp8 ring `(kNumRingTokens, H)`), `l1_sf` (int32 `(H/128, kNumSFRingTokens)`
MN-major), `l1_topk_weights` (f32 ring), `l2_token` (fp8 ring `(kNumRingTokens, I)`), `l2_sf` (int32 `(I/128, kNumSFRingTokens)`), `combine_token` (bf16 `(top-k slots, T, H)`; `combine_token.rank_buffer(slot).data(token)`).
Each accessor returns `Int64` generic addresses (use `cute.make_ptr` / the typed `ptr_at` helpers for typed access). The rule "never derive a cross-rank offset from a rank's own BLOCK_M" is encoded in `layout.py`
(rings are sized from `cfg.num_ring_tokens`, which is the max over all BLOCK_M candidates).

---------------------------------------------------------------------------------------------------------------------------
## 11. Register split, setmaxnreg, warp roles

| warps | role | `setmaxnreg` (E/R>64, i.e. DEP4 at E=384 / E/R<=64: DEP8/16/32) |
|---|---|---|
| 0-3 | dispatch (128 thr) | `dec` 96 / 48 |
| 4 | TMA-A | `dec` 88 / 40 |
| 5 | TMA-B | `dec` 88 / 40 |
| 6 | MMA (leader only does work; peer warp falls through) | `dec` 88 / 40 |
| 7 | scheduler (leader only does work; peer warp falls through) | `dec` 88 / 40 |
| 8-15 | epilogue (2 x 128 thr) | `inc` 160 / 208 |

`cfg.regs_*` hold the numbers; sum `96*128+88*128+160*256 = 64512 = 48*128+40*128+208*256` (<= 65536). Each warp executes `setmaxnreg` as the FIRST statement of its role branch (K:401,829,... ); all warps of a warpgroup execute the same value.
Warp-role predicates: `warp_idx < 4`; `== 4`; `== 5`; `== 6`; `== 7`; `>= 8` (K:399-1212).

**Product default register policy**: `KernelBuild.regs` = `DEFAULT_REGS` = **96/88/160 at EVERY EP size**. DeepGEMM's own split is 96/88/160 only when experts-per-rank > 64 and 48/40/208 otherwise (DEP>=8), where the 48-register dispatch cap spills the pull loop (measured: the cliff is between 48 and 56 dispatch registers, 56/40/200 behaves like 96/88/160; EP4 E=192 full kernel: 48/40/208 is 10 % (M=256) / 51 % (M=1000) slower than 56/40/200, stock DeepGEMM at its own split 1.85x slower than the tuned port at M=1000). `faithful_regs=True` (`jit.get_or_compile_kernel(..., faithful_regs=True)`) restores DeepGEMM's split (`cfg.regs_*`) for parity / mapping runs (use it for DEP>=8 comparisons against DeepGEMM); `regs_override` (explicit split) is mutually exclusive with it. At E/R > 64 all three coincide.

**`KernelBuild.regs_override=(r_dispatch, r_non_epilogue, r_epilogue)`** (added because DeepGEMM's own DEP>=8 kernel spills in the dispatch warps at 48 registers): replaces `cfg.regs_*` in the role split (`KernelBuild.regs`); each value a multiple of 8 in [24, 256] with `128*r0 + 128*r1 + 256*r2 <= 64512` (validated, `ValueError` otherwise); part of the digest and of `name()` (`..._m7_r96-88-160_<digest>`); `jit.make_build/get_or_compile_kernel(..., regs_override=...)`. FAITHFUL = `None` (DeepGEMM's split: 96/88/160 when E/R > 64 else 48/40/208); TUNED = e.g. `(96, 88, 160)` at every EP size. The PTX of the real kernel carries exactly the requested `setmaxnreg` immediates (checked: `dec 96; 4 x dec 88; inc 160` at E=32 where the default is 48/40/208).

---------------------------------------------------------------------------------------------------------------------------
## 12. Behaviour guaranteed to roles at entry (what staged tests may rely on)

* All barriers initialised and visible cluster-wide; TMEM allocated; `griddepcontrol.wait` done; trace stamp 0 recorded.
* `smem` is garbage except `expert_token_count[0..align(4E,1024))` zeroed (and mbarriers initialised).
* The symmetric buffer is in the post-cleanup state of the previous launch (all counters zero except the self-cycling ones: `grid_sync_count`, `nvl_barrier_counter/signals`, `peer_grid_idx`, `combine_ready_grid_idx`).
* For `mode_mask` without bit0: the host must have staged `expert_recv_count_sum` (lo32 = count, hi32 = num_sms*num_ranks), the ring tokens/SFs/weights/metadata and `l1_full_count[slot] = BLOCK_M` per used ring block (`tests/experimental/replay_utils.py` does exactly this).

---------------------------------------------------------------------------------------------------------------------------
## 13. `ptx.py` / `comm.py` API (semantics in docstrings; all `@dsl_user_op`/`@cute.jit`; `ga` = gmem address operand = `cute.Pointer` (gmem) or `Int64`; `sp` = smem `cute.Pointer`)

### 13.1 identity / warp / math
`smid()`, `gridid() -> Int64 (%gridid)`, `lane_id()`, `clock64()`, `globaltimer()`, `elect_sync() -> Boolean` (elect.sync), `ballot(pred) -> Int32`, `all_sync(pred)`, `any_sync(pred)`,
`shfl_idx(v, lane)`, `shfl_xor(v, mask)`, `redux_add_s32/u32(v)`, `redux_min_u32(v)`, `redux_max_f32(v)` (`redux.sync.max.f32`), `fns_b32(mask, base, offset)`, `popc(x)`, `ffs(x)`,
`nanosleep(ns)`; `advance_stage(stage, phase, n)`.
### 13.2 global memory (scope/semantics explicit in the name; SASS must match DeepGEMM's: `.STRONG.GPU`, `.STRONG.SYS`, `ATOMG.E.ADD.64...`, `REDG`, `LDG.E.128`)
loads: `ld_b32(ga)`, `ld_b64(ga)`, `ld_v4_b32(ga) -> 4xInt32` (plain `ld.global.v4.b32`, remote pulls), `ld_volatile_b32/b64(ga)`, `ld_acquire_gpu_b32/b64(ga)`, `ld_acquire_sys_b32/s32/b64(ga)`, `ld_f32(ga)`;
stores: `st_b32(ga, v)`, `st_b64(ga, v)`, `st_v4_b32(ga, x, y, z, w)`, `st_release_sys_b64(ga, v)`, `st_release_gpu_b64(ga, v)`, `st_f32(ga, v)`;
**generic-space variants** (DeepGEMM dereferences the generic pointer returned by `SymBuffer::map`, so accesses THROUGH `map` are generic `ld`/`st`, SASS `LD.E*`/`ST.E*`; local workspace/ring accesses stay `.global`, SASS `LDG/STG`): `ld_generic_b32/b64/v4_b32(ga)`, `st_generic_b32/b64/v4_b32(ga, ...)`. Use them where DeepGEMM's PTX does (verified from its kernel.ptx line info): K:661 remote token pull (generic `ld.v4.b32`), K:678/697 remote weight/SF loads (generic `ld.b32`), K:707 LOCAL ring store (`st.global.v4.b32`, NOT generic), K:446 remote index push (generic `st.b32`), K:460/468 remote `peer_grid_idx`/`expert_recv_count` pushes (generic `st.b64`), K:1630 L2-epilogue remote BF16 write-back (generic `st.v4.b32`). Remote atomics/reductions/acquire loads/release stores are explicit `.global` in DeepGEMM's asm and use the `ga` helpers below.
atomics: `atom_add_u32(ga, v)->old` (`atom.global.add.u32`), `atom_add_u64(ga, v)->old`, `atom_add_sys_u64(ga, v)->old` (`atom.sys.global.add.u64`), `atom_add_release_gpu_u32(ga, v)->old` (`atom.release.gpu.global.add.u32`),
`red_add_u32/s32(ga, v)` (`red.gpu.global.add.*`, relaxed), `red_add_release_gpu_u32(ga, v)`, `red_add_release_sys_s32(ga, v)`, `red_xor_release_gpu_b64(ga, v)`, `red_or_release_gpu_b64(ga, v)`.
### 13.3 shared memory / fences / tcgen05
`ld_shared_u32/f32/v2_f32/v4_u32(sp)`, `st_shared_u32/f32/v2_f32/v4_u32(sp, ...)`, `st_shared_zero(sp, nbytes)` (the `st.bulk` substitute/instruction), `fence_acq_rel_cta()`, `fence_proxy_async_shared_cta()`,
`fence_barrier_init()` (`fence.mbarrier_init.release.cluster`), `tcgen05_fence_before_thread_sync()`, `tcgen05_fence_after_thread_sync()`, `tmem_ld_16x256b_x1(taddr)->4xInt32`, `tmem_ld_wait()` (`tcgen05.wait::ld`),
`stmatrix_m16n8_x1_trans_b8(sp, r)`, `stmatrix_m16n8_x2_trans_b8(sp, r0, r1)`, `stmatrix_m8n8_x4_trans_b16(sp, r0..r3)`, `stmatrix_m8n8_x2_trans_b16`, `stmatrix_m8n8_x2_b16`.
### 13.4 mbarrier / cluster / bulk copy
`Mbar` handle (smem pointer): `.init(count)`, `.wait(phase)` (try_wait.parity spin with the 0x989680 suspend hint, as `ClusterTransactionBarrier::wait`), `.try_wait(phase)->Boolean`, `.arrive()`, `.arrive_count(n)`,
`.arrive_expect_tx(bytes)`, `.expect_tx(bytes)`, `.arrive_remote(cta_rank)` (`mapa` + `mbarrier.arrive.shared::cluster.b64`), `.arrive_expect_tx_remote(bytes, cta_rank)`, `.cluster_addr(cta_rank)` (Int32, `mapa.shared::cluster`),
`.addr` (Int32 shared::cta address; `addr & 0xFEFFFFFF` = leader-barrier form used by 2-SM TMA loads);
`mapa_shared(addr_u32, cta_rank)->Int32`, `st_async_v4(dst_sp, x, y, z, w, mbar, cta_rank)` (`st.async.shared::cluster.mbarrier::complete_tx::bytes.u32.v4`, dst/mbarrier mapped to `cta_rank`);
`tma_load_1d(dst_sp, src_ga, mbar, nbytes, hint=EVICT_FIRST)`, `tma_store_1d(dst_ga, src_sp, nbytes, hint=EVICT_NORMAL)` (explicit `.L2::cache_hint` operands), `tma_store_commit()` (`cp.async.bulk.commit_group`),
`tma_store_wait(n)` (`cp.async.bulk.wait_group n`, NOT `.read`). Constants `EVICT_NORMAL=0x1000000000000000`, `EVICT_FIRST=0x12F0000000000000`.
Descriptor-based TMA / UTCCP / MMA forms live in `ptx_g.py`.
### 13.5 math/cvt (numerics-critical; DeepGEMM modifiers exactly)
`cvt_rn_bf16x2_f32(hi, lo)->Int32` (`cvt.rn.bf16x2.f32`), `add_rn_f32_bf16(acc, bf16_bits)->Float32` (`add.rn.f32.bf16`), `fmul2_rn/fadd2_rn/ffma2_rn((a0,a1),(b0,b1))` (packed f32x2 with `.rn`, no ftz), **`ex2_approx(x)` (`ex2.approx.f32`, NO `.ftz`: DeepGEMM's `__expf` compiles to `mul.f32 x, log2e` + `ex2.approx.f32`; subnormal results differ from the ftz form, verified)**, `ex2_approx_ftz(x)` (NOT used by DeepGEMM),
`rcp_approx_ftz(x)`, `hmin2_bf16x2(a,b)`, `hmax2_bf16x2(a,b)`, `bf16x2_to_f32x2(r)`.
### 13.6 comm.py
`SymBuffer` (device by-value, R<=72 constexpr entries; `.base`, `.rank_idx`, `.map(x, rank)` identity when `max_ranks==1`) + `SymBufferHost` (kernel argument payload);
`grid_sync(ws, idx, num_sms, sm_idx, thread_idx_in_group, sync_scope)` (`sync_scope` = STATIC tuple `(bar_id, nthreads)` e.g. `(0, 128)` dispatch / `(2, 256)` epilogue — closures over runtime values are not allowed in staged regions; `ws` must be a static (frozen) object like `ctx.ws`) and `nvlink_barrier(ws, sym, num_ranks, num_sms, num_threads, grid_idx, tag, sm_idx, thread_idx, sync_scope, prologue, epilogue)` with DeepGEMM's exact
counter protocol (DeepGEMM `comm/barrier.cuh`): `sync_scope` is the same STATIC `(bar_id, nthreads)` tuple as for `grid_sync` (the CTA-level rendezvous is `bar.sync bar_id, nthreads`); callables are not allowed in staged regions.

---------------------------------------------------------------------------------------------------------------------------
## 14. Trace (`kTrace`, DeepGEMM-compatible slots)

`ctx.tr.stamp(slot)` (globaltimer ns, thread 0 of a role), `ctx.tr.stamp_clock(slot)` (clock64), `ctx.tr.add(slot, cycles)` (register accumulation, flushed at kernel end with `atom.add` per non-zero slot).
Slots (u64 per CTA, 64 slots + 4x64 event log after `num_sms*64`): 0 start (after PDL wait), 7 after NVLink barrier #1, 1 pull done, 2 first MMA task, 3 MMA done, 4 epilogue GEMM done, 5 combine start, 6 combine end,
60/61 clock64 start/end; per-die (cfg.trace only): 46 = `%smid + 1` of the leader CTA's scheduler warp, 47 = `locality_domain + 1` read from the map, accumulators 48 / 49 = successful task claims from the own / the other domain's queue (`scheduler_perdie.TRACE_SLOT_*`); trace builds: 62 = raw `%gridid` of the launch; stall sums: see `trace.py` (kept identical to the DeepGEMM kernel; the events log window starts at k-block 400). All `ctx.tr.*` calls compile to nothing when `cfg.trace` is False.

---------------------------------------------------------------------------------------------------------------------------
## 15. Build, cache and test hooks

* `jit.get_or_compile_kernel(cfg, mode_mask=7, pdl=False, roles="auto", ptxas_options="--register-usage-level=10", ...)` returns a `MegaMoeRunner` (host-level call with torch tensors + `SymmBuffer`; `.call_raw(...)` with ints).
  In-process memo (argument-keyed first level: about 4 us for a repeated call with the same cfg object; name-keyed second level 0.1-0.2 ms) -> disk cache via `flashinfer.jit.cute_dsl_core.build_and_load_cute_dsl_kernel`
  (warm load about 12 ms) -> compile + TVM-FFI export (about 1 s per variant for the full kernel). Compiling inside a CUDA-graph capture raises; `jit.is_compiled(cfg, ...)` tells whether a variant is already in the memo.
  `precompile_variants([...])`, `jit.stats()`. `MegaMoeRunner.sym_payload` caches the by-value peer table per `SymmBuffer` object: the entry pins the buffer and is re-validated against its current base pointer and
  peer offsets on every call.
* Foundation tests: `tests/experimental/test_sm107_dg_foundation_{layout (CPU),ptx,comm,skeleton,skeleton_ep2,jit}.py`; the full kernel against DeepGEMM: `test_sm107_dg_full_kernel_vs_deepgemm.py` (EP1),
  `test_sm107_dg_dispatch_combine_full.py` (EP2 / EP4), `test_sm107_dg_gemm_core_*.py` (GEMM core).
* **Skeleton with stub roles**: `KernelBuild(cfg, roles="stubs", debug_marks=True)`: every warp bumps u64 visit counters in `trace_ptr` ([0..15] entry, [16..31] exit);
  `python tests/experimental/test_sm107_dg_foundation_skeleton.py --bm 128 --dep 4 --modes 7 --graph`. Roles are plugged in by creating the role module: `kernel.py` picks the module's function if `kernel/<module>.py` exists
  (a module that exists but fails to import or lacks the function RAISES: stubs never mask errors), else `stubs.<function>`; `roles="stubs"` forces the stubs.

## 16. Measured facts and deviations from DeepGEMM

| item | result |
|---|---|
| TMA atom for packed FP4 | **PASS**: DSL atoms with `CtaGroup.TWO` give the exact 2-SM TMA PTX (peer bit cleared, cache hint); packed-FP4 weights as a UINT8 byte view (box 64 B x 128, SW64) match the expected smem image; `Float4E2M1FN`/`16U4_ALIGN8B` dtype not tried (identical image/expect_tx). ABI = DSL atoms (4.4). `cpasync.prefetch_descriptor` fails to legalize for 2-SM atoms. |
| `setmaxnreg` vs ptxas | **partial**: ptxas accepts the full split (`setmaxnreg.dec.sync.aligned.u32 96/88` or `48/40`, `inc 160/208`, each warp executes it as first statement of its branch; PTX `.reqntid 512 .minnctapersm 1`; cubin REG:128 (launch cap) STACK:0 SHARED:1024 LOCAL:0 for all 12 skeleton variants, exactly the DeepGEMM cubin attributes REG 128 / SHARED 1024). **DeepGEMM's own DEP8 cubin (48/40/208) SPILLS in the dispatch pull loop** (STACK 224, STL 65 / LDL 81; the DEP4 cubin has only cold-path STLs), so "no spills" is NOT a parity target at DEP>=8 — the parity target is the same memory-op sequence. Resource usage of the real kernel at the default split: DEP4 STACK 0; DEP8 STACK 48 / 9 STL / 10 LDL; DEP16/32 STACK 40; `regs_override=(96,88,160)` removes all spills at DEP8/32. `--register-usage-level` IS honoured (see the `--register-usage-level` row). |
| launch geometry | **PASS**: `launch(grid=[212], block=[512], cluster=[2], smem=cfg.smem_bytes, min_blocks_per_mp=1, use_pdl=...)` works for all six BLOCK_M (incl. 232404 B for BM=64); `get_max_active_clusters(2) == 106` (== num_sms/2: full co-residency possible; cluster 4 would give only 46); CUDA-graph capture + replay x20 OK; `griddepcontrol.wait` emitted before the role split; 212-CTA software grid sync x1e5: PASS. |
| by-value peer table / TVM-FFI | **PASS**: `SymBufferHost` -> one kernel arg (byval struct, DSL>=4.7 any width): R=4/8/16/32/72 compile + run; TVM-FFI export/reload: pointers as ints, `sym` as nested tuple incl. constexpr R, explicit int stream handle (env stream impossible without a GPU-tensor param); cold compile+export 0.37 s (stubs), disk-cache load 17 ms; graph capture of the runner OK. |
| mbarriers | native `cute.arch.mbarrier_*` + `ptx.Mbar` with DeepGEMM's exact counts (full=4, empty=1, tmem_full=1, tmem_empty=512, combine=1, task_info_full=1, task_info_empty=512) verified in the PTX of the skeleton; stock pipeline classes are NOT used (they have different arrive counts). |
| early exit in dynamic regions | **NOT allowed** (`UNSUP_EARLY_EXIT`) in kernel and `@cute.jit` regions; predicate-carried `while` loops (also nested) work; see 4.1 rules 2-6 for the other measured restrictions (closures, init-before-if, frozen/static objects, no ctx mutation). |
| K=64 block-scaled MMA emitter | **PASS**: raw-PTX `tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale` K=64 (idesc bit 31, sf ids 0/2, desc `lo += 4/2`) + UTCCP + commit multicast bit-exact for all six BLOCK_M, N=16..240, L1/L2 shapes; lives in `kernel/ptx_g.py` / `mma_warp.py` (derived from the vendored `dynamic_mainloop.py`). DSL 4.8 also has native `tcgen05.SM107BlockScaledMmaMXF8F6F4Op`; TMEM columns on sm_107: `get_max_tmem_alloc_cols == 576`. |
| `%gridid` / combine tag | **MEASURED, REPRODUCED IN DEEPGEMM ITSELF**: `%gridid` is unique and increasing for direct launches (also with the PDL attribute and across streams) but **identical for every replay of the same CUDA-graph kernel node** (7,8,9,7,8,9,...; replays consume no new ids). DeepGEMM's combine-readiness tag `%gridid+1` (K:460/K:1713, flags never reset) is therefore NOT per-launch unique under graph replay: a stale flag from the previous replay releases the combine before the senders finished their L2 writes. EP2 test with inputs that change between replays of one captured graph + rank launch skew (EP2, one process per rank, DeepGEMM `y` as the oracle): **DeepGEMM stock returns stale `y` in 59 of 60 replays (rank 0, Ms=[300,300] and [1500,200], skew 2e6 cycles), 44/300 and 209/300 (Ms=[64,64], masked 0.3, skew 5e5), 149/300 and 150/300 (Ms=[1900,1900], ring wrap), and WITHOUT any artificial skew 99/100 (rank 0, Ms=[300,300] and [1500,200]) and 84/100 (rank 1, Ms=[64,64]); standalone DeepGEMM-only reproducer: eager 0/60, graph 43/60 and 3/60 wrong**; the port in `tag_mode="gridid"` fails identically (faithful); the port in `tag_mode="epoch"` is bit-exact (60/60 on both ranks). Identical-input replays cannot see it. Port: `ctx.launch_tag` (`KernelBuild.tag_mode`): **`"epoch"` = `(nvl_barrier_counter >> 1) + 1` latched at CTA start after the PDL wait (unique per launch; DEFAULT)**, `"gridid"` = DeepGEMM (kept for A/B parity runs / mapping comparisons only). The dispatch role (P5 push) and the combine role (wait) both use `ctx.launch_tag`. The same fix (a per-launch counter instead of `%gridid`) applies to DeepGEMM. |
| `--register-usage-level` / per-region budgets | **PASS**: the DSL passes `cute.PtxasOptions("--register-usage-level=N")` to ptxas (three different SASS/STACK sizes for none/10/0 on a DEP8 dispatch-only variant); the cubin's `USETMAXREG` immediates are exactly DeepGEMM's (DEP4: `DEALLOC 0x60`, 4x`DEALLOC 0x58`, `TRY_ALLOC 0xa0`; DEP8: `0x30`, 4x`0x28`, `0xd0`), ptxas honours each region's budget (spills when over). Dispatch region (mode_mask=1, BM=128): DEP4 STACK 0 / STL 0 / LDL 0; DEP8 STACK 48 / STL 9 / LDL 10 (DeepGEMM DEP8 cubin: STACK 224, STL 65, LDL 81 in the dispatch region). |
| **Int32 address overflow** | `BufferMap.region_addr("combine_token", slot=<runtime Int32>)` computed `r.offset + slot * token_bytes * num_tokens` in Int32 (section 4.1 rule 13). PTX of the old formulation (EP1 E=96 S=0 T_max=34560, BLOCK_M 128): `mad.lo.s32 %r958, %r957, 353894400, 542973824; cvt.s64.s32 %rd460, %r958` (slot 5: 2312445824 wraps to -1982521472, i.e. the L2-epilogue write-back of top-k slot 5 went 1.98 GB BELOW the buffer); fixed PTX: `mul.wide.s32 %rd460, %r958, 353894400`. Windows (old formulation wrong): EP1 E=96 S=0 T_max >= 32640 (slot 5), EP1 E=32 S=1 T_max >= 26880 (shared slot 6), DEP4 E=384 S=0 T_max >= 24960, DEP8 >= 19200, DEP16 >= 13440, DEP32 >= 7680 (slots 4-6), 19-29 wrapping T_max values per (R, E, S) up to 86400. **Why a comparison against DeepGEMM did not catch it**: a harness that runs DeepGEMM first and then the port on the SAME buffer with identical inputs hides it, because a combine write that lands at a wrong address reads back the stale (identical, correct) DeepGEMM data. With a poison fill (0xFF over every kernel-written region before every launch of either kernel) the old formulation FAILS both windows (all 64 rows NaN, every launch) and the fixed one passes bit-exact. |
| host launch overhead | `MegaMoeRunner.bind(...)` 7.8 us/call, `runner(...)` 8.4 us/call (Python + TVM-FFI dispatch + async launch of the stub kernel, measured over 2000 calls) |
| **full-kernel EP1 integration** (mode_mask=7, all roles) | **PASS, bit-exact vs DeepGEMM `y`** (`torch.equal` on int16 views) for heuristic BLOCK_M and forced BLOCK_M in {16,32,64,128,192,240}, M in {0,1,2,7,33..1920}, masked routes, E=32/96/384, ring wrap (pool > ring), 2-3 synchronous launches + 4-16 back-to-back launches per case, 1000/300/2000 back-to-back stress launches, 64 CUDA-graph replays (`tag_mode` gridid and epoch); stats == counts x launches, workspace clean, DeepGEMM re-run on the port's buffer == first DeepGEMM `y`; `regs_override` 96/88/160 (E=32) and 48/40/208 (E=384) also bit-exact |
| **full-kernel EP2/EP4 integration** (2-4 GPUs, one process per rank, DeepGEMM `y` per rank as the oracle) | **PASS, bit-exact on every rank**: S=0 at EP2: equal M (256, 1000), imbalanced per-rank M => mixed BLOCK_M across ranks ([300,20], [0,64], [1,1000], [1920,7], [7,7], [100,1500], [0,0], [0,300], [257,1]), masks 0.4 + 16 graph replays + 16 back-to-back, forced BLOCK_M 16..240, `regs_override`, E=384, stress (500 / 100 / 1000 back-to-back); S=1 at EP2: the same families (equal, mixed M, masks + graph + back-to-back, six forced BLOCK_M, E=384, 500 stress); EP4: mixed M/BLOCK_M, 2000-launch skew stress (S=0); **graph replay with changing inputs: see the `%gridid` row** |

Other measured facts: `st.bulk.weak.shared::cta` is accepted by ptxas through inline asm (used for the `expert_token_count` zeroing, one elected lane, exactly like DeepGEMM); `barrier.sync 1,384` (unaligned) and `barrier.cluster.arrive.relaxed.aligned` forms are emitted as in DeepGEMM;
the DSL emits `.reqntid 512` (exact block size) where DeepGEMM's `__launch_bounds__(512,1)` gives `.maxntid 512` — same 128-register launch cap; cubin symbol names embed the Python repr of the fake arguments (harmless).

## 17. Known gaps

* Shared experts: one merged FP8 shared expert is implemented and validated (the GEMM role modules declare `SUPPORTS_SHARED_EXPERTS = True`; `jit.MegaMoeRunner` refuses to launch a build with shared experts when a role module does not
  declare it). More than one shared expert (`num_shared_experts > 1`) has not been validated and is refused by the host API.
* Per-die execution (`num_locality_domains == 2`, DeepGEMM `kNumLocalityDomains = 2`, `kLocalizedWeights`): the scheduler warp of the leader CTA reads `sm_locality_domains[%smid]` (argument 14) through `perdie.load_domain`;
  `scheduler_perdie.claim / wait_l1_claimed` implement the per-domain task counters (`ws.*_task_count(d)`, stride 1), the N-cluster split `[d*C/2, (d+1)*C/2)`, stealing (>= 2 waves) and the L2 wait on both domains' L1 counters;
  with `cfg.localized_weights` the routed weight maps are 3D `(rows of a slice, K bytes, 2)` (`perdie.make_weight_pair`, compile-time slice stride `heuristics.get_perdie_domain_stride_bytes`) and TMA-B loads `(k, half_row, half_idx)`
  (`perdie.b_tile_coords`). Die-local SHARED weights (`kLocalizedSharedWeights`, DeepGEMM `PERDIE=all`) are NOT implemented (the shared weight maps are always plain 2D): the host API refuses them and the runner refuses to launch such a build.
  A balanced SM -> domain map is required (an imbalanced map hangs the per-die scheduling, like DeepGEMM). `num_locality_domains == 1` builds are instruction-identical to the pre-per-die kernel.
* `fast_math=False` (IEEE `expf` / division) is not ported (`assert cfg.fast_math` in `epilogue.py`); BF16 / FP8 routed weights (`weight_packed_fp4 = False`) are not ported.
* The DeepGEMM knobs `kNumEpiPF` (TMEM-load prefetch stages), `kWeightPrefetch` (L2 prefetch of weight tiles) and `kDebugFlags` are not ported (the configuration fields exist; the shipped default 0 produces identical code).
* `%gridid` under CUDA-graph replay: DeepGEMM's combine-readiness tag is stale for back-to-back replays of one graph node with changing inputs (section 16); the port's default `tag_mode="epoch"` is exact.
* The DSL cannot dump SASS itself in an environment without `nvdisasm` on its lookup path; use `nvdisasm` / `cuobjdump -sass` on the kept cubin (`CUTE_DSL_KEEP=ptx,cubin`).
