# SPDX-License-Identifier: Apache-2.0
# Port of DeepGEMM (MIT License, Copyright (c) 2025 DeepSeek; see licenses/LICENSE.deepgemm.txt) sm107_fp8_fp4_mega_moe.cuh / policy tests.
"""FlashInfer experimental-package policy checks for the SM107 DeepGEMM-style MegaMoE (flashinfer/experimental/README.md):

* the public functions live in core (``flashinfer.mega_moe_deepgemm``), are marked ``@flashinfer_experimental_api`` (warn once,
  ``is_experimental``) and keep DeepGEMM's signatures;
* importing core / ``flashinfer.aot`` / ``flashinfer.moe_ep`` / the core wrapper module does NOT import the experimental package
  (subprocess ``sys.modules`` checks); importing the package itself is lightweight (no torch.distributed symmetric memory,
  cutlass, nvshmem, cuda bindings at import time);
* nothing is registered in ``flashinfer/aot.py``;
* no environment variables are read by the host-side files (FlashInfer experimental policy);
* every field of ``Sm107KernelConfig`` is either part of the compile key or explicitly classified as derived (so a new
  field cannot silently escape the kernel-name digest).
"""

import dataclasses
import inspect
import os
import subprocess
import sys
import textwrap

import pytest

import flashinfer
from flashinfer.experimental.sm107_deepgemm_megamoe import config as cfgmod

PKG = "flashinfer.experimental.sm107_deepgemm_megamoe"
PKG_DIR = os.path.dirname(cfgmod.__file__)
HOST_FILES = (
    "__init__.py",
    "config.py",
    "heuristics.py",
    "workspace.py",
    "weights.py",
    "staging.py",
    "frontend.py",
    "compat.py",
    "backend.py",
    "locality.py",
)


def _run(code: str) -> str:
    proc = subprocess.run(
        [sys.executable, "-W", "ignore", "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
    return proc.stdout


def test_core_import_does_not_import_the_experimental_package():
    out = _run(
        f"""
        import sys
        import flashinfer, flashinfer.aot, flashinfer.moe_ep
        import flashinfer.mega_moe_deepgemm as core
        bad = sorted(m for m in sys.modules if m == "{PKG}" or m.startswith("{PKG}."))
        print("IMPORTED", bad)
        """
    )
    assert "IMPORTED []" in out, out


def test_package_import_is_lightweight():
    out = _run(
        f"""
        import sys
        import flashinfer
        before = set(sys.modules)
        import {PKG} as pkg
        new = sorted(set(sys.modules) - before)
        heavy = [m for m in new if m.split(".")[0] in ("cutlass", "nvshmem", "cuda", "tvm_ffi", "triton")
                 or m.startswith("torch.distributed._symmetric_memory")]
        lazy = [m for m in new if m in ("{PKG}.workspace", "{PKG}.frontend", "{PKG}.compat", "{PKG}.staging", "{PKG}.weights", "{PKG}.locality")]
        print("HEAVY", heavy, "LAZY", lazy)
        # lazy attributes resolve on demand
        assert pkg.SymmBuffer.__name__ == "SymmBuffer" and callable(pkg.fp8_fp4_mega_moe)
        print("LAZY_OK")
        """
    )
    assert "HEAVY [] LAZY []" in out and "LAZY_OK" in out, out


def test_importing_the_package_does_not_touch_the_environment():
    out = _run(
        f"""
        import os
        import flashinfer
        before = dict(os.environ)
        import flashinfer.mega_moe_deepgemm as core
        import {PKG} as pkg
        from {PKG} import compat, frontend, workspace, locality, backend, jit  # noqa: F401
        after = dict(os.environ)
        print("ENV_CHANGED", sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k)))
        """
    )
    assert "ENV_CHANGED []" in out, out


def test_no_aot_registration():
    with open(os.path.join(os.path.dirname(flashinfer.__file__), "aot.py")) as fh:
        aot_src = fh.read()
    for needle in ("sm107_deepgemm_megamoe", "mega_moe_deepgemm"):
        assert needle not in aot_src, (
            f"{needle} must not be registered in flashinfer/aot.py (experimental features are JIT-only)"
        )


def test_core_wrappers_are_experimental_apis_with_deepgemm_signatures():
    import flashinfer.mega_moe_deepgemm as core
    from flashinfer.experimental.sm107_deepgemm_megamoe import compat

    for name in ("get_symm_buffer_for_mega_moe", "transform_weights_for_mega_moe", "fp8_fp4_mega_moe", "get_block_m_for_mega_moe", "get_token_alignment_for_mega_moe",
                 "is_localization_available", "get_balanced_sm_locality_domains", "localize", "localize_mega_moe_weights",
                 "set_kernel_options", "get_kernel_options"):  # fmt: skip
        fn = getattr(core, name)
        assert getattr(fn, "is_experimental", False), (
            f"{name} must carry @flashinfer_experimental_api"
        )
        params = lambda f: [
            (p.name, p.kind, p.default)
            for p in inspect.signature(f).parameters.values()
        ]  # noqa: E731  (annotations may differ)
        assert params(fn) == params(getattr(compat, name)), name
        assert (
            "experimental" in (fn.__doc__ or "").lower()
        )  # the decorator prepends the status banner
    # the class is re-exported lazily
    assert core.SymmBuffer is compat.SymmBuffer
    with pytest.raises(AttributeError):
        core.does_not_exist  # noqa: B018


def test_every_public_core_function_is_an_experimental_api():
    import flashinfer.mega_moe_deepgemm as core

    names = [n for n in core.__all__ if n != "SymmBuffer"]
    assert len(names) >= 14
    for name in names:
        assert getattr(getattr(core, name), "is_experimental", False), (
            f"{name} must carry @flashinfer_experimental_api"
        )


def test_experimental_warning_emitted_once_per_function():
    out = _run(
        """
        import warnings
        import flashinfer
        from flashinfer.experimental import ExperimentalWarning
        import flashinfer.mega_moe_deepgemm as core
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            a = core.get_block_m_for_mega_moe(4, 384, 8192, 1024, 6, "fp8xfp4")
            b = core.get_block_m_for_mega_moe(4, 384, 8192, 64, 6, "fp8xfp4")
            c = core.get_token_alignment_for_mega_moe()
        n_bm = sum(1 for x in w if issubclass(x.category, ExperimentalWarning) and "SM107" in str(x.message))
        print("RESULT", a, b, c, n_bm)
        """
    )
    # one warning per decorated function (two decorated functions were called), values are the bit-exact heuristics
    assert "RESULT 128 16 1920 2" in out, out


def test_host_files_read_no_environment_variables():
    for name in HOST_FILES:
        with open(os.path.join(PKG_DIR, name)) as fh:
            src = fh.read()
        for needle in ("os.environ", "os.getenv", "getenv("):
            assert needle not in src, (
                f"{name} reads the environment ({needle}); FlashInfer experimental policy: use config fields"
            )


# fields that are DERIVED from the compile-key fields (or pure documentation of them): everything else must key the cache
_DERIVED_NOT_KEYED = {
    "num_experts_per_rank", "load_block_m", "load_block_n", "wg_block_m", "umma_m", "umma_n", "umma_block_k", "atom_m",
    "num_atoms_per_store", "num_store_blocks_per_wg", "shared_block_k", "l1_out_block_n", "l1_shape_n", "l1_shape_k",
    "l2_shape_n", "l2_shape_k", "shared_l2_shape_k", "num_l1_block_ns", "num_l2_block_ns", "num_l1_clusters",
    "num_l2_clusters", "swizzle_a_mode", "swizzle_b_mode", "swizzle_cd_mode", "swizzle_shared_mode", "accum_stages",
    "tma_store_stages", "schedule_stages", "smem_sizeof", "smem_end", "smem_reusable_bytes", "smem_offsets", "smem_sizes",
    "num_threads", "num_dispatch_warps", "num_mma_non_epilogue_warps", "num_epilogue_warps", "num_epilogue_warpgroups",
    "tmem_accum_cols", "tmem_sfa_cols", "tmem_sfb_cols", "tmem_cols_used", "tmem_sfa_col", "tmem_sfb_col", "num_ring_blocks",
    "num_shared_sf_tokens", "num_max_pool_tokens", "num_pull_chunks", "symm_buffer_bytes", "num_tokens_per_warp",
    "num_activate_lanes", "num_experts_per_lane", "num_ranks_per_lane", "num_combine_chunks", "combine_chunk_bytes",
    "combine_chunk_uint4_per_lane", "pull_uint4_per_lane", "per_die",
    "num_l1_clusters_per_domain", "num_l2_clusters_per_domain", "num_min_steal_waves", "domain_counter_stride",
}  # fmt: skip


def test_every_config_field_is_keyed_or_classified_as_derived():
    fields = {f.name for f in dataclasses.fields(cfgmod.Sm107KernelConfig)}
    keyed = set(cfgmod.Sm107KernelConfig._COMPILE_KEY_FIELDS)
    assert keyed <= fields, keyed - fields
    unclassified = fields - keyed - _DERIVED_NOT_KEYED
    assert not unclassified, (
        f"new Sm107KernelConfig fields must be added to _COMPILE_KEY_FIELDS or classified as derived: {sorted(unclassified)}"
    )
    assert not (_DERIVED_NOT_KEYED - fields), _DERIVED_NOT_KEYED - fields


def test_readme_exists_and_states_experimental_status():
    with open(os.path.join(PKG_DIR, "README.md")) as fh:
        text = fh.read().lower()
    for needle in ("experimental", "opt-in", "sm107", "tests/experimental"):
        assert needle in text, needle
