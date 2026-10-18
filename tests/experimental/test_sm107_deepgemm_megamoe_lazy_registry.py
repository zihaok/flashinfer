# SPDX-License-Identifier: Apache-2.0
# Port of DeepGEMM (MIT License, Copyright (c) 2025 DeepSeek; see licenses/LICENSE.deepgemm.txt) sm107_fp8_fp4_mega_moe.cuh / registry tests.
"""Lazy resolution of the experimental SM107 mega kernel in ``flashinfer.moe_ep`` (CPU only, no CUDA device needed).

The two core touch points of the experimental package are the PEP 562 ``__getattr__`` / ``__dir__`` of ``flashinfer/moe_ep/__init__.py`` (the pure-Python
config class is resolved on first access) and the lazy kernel-name map of ``flashinfer/moe_ep/core/kernel/registry.py`` (``create_mega_kernel`` imports the
backend module on first use).  Every other backend test imports the backend explicitly first, so none of them can notice if the lazy path breaks: a change that
makes the config import eager, drops the lazy entry or breaks the name resolution would pass them all.  This module follows the whole path in ONE fresh
interpreter, in the order a user meets it:

``import flashinfer.moe_ep`` (no module of the experimental package) -> ``dir()`` / unknown attribute -> ``moe_ep.Sm107_..._MegaMoeConfig`` (config modules only, no backend /
JIT / workspace) -> ``is_mega_kernel_config`` (knows the lazy name) -> ``create_mega_kernel`` of an unknown name (error lists the lazy kernel, imports nothing) ->
``create_mega_kernel(cfg)`` (imports the backend by name) -> a second call (no re-import).
"""

import json
import pathlib
import subprocess
import sys
import textwrap

import pytest

CONFIG_NAME = "Sm107_Mxfp8_Mxfp4_Bf16_DeepgemmCutedsl_MegaMoeConfig"


def _lazy_registry_present() -> bool:
    """The lazy registry is a core patch (applied by the PR cut); the port tree and the snapshot overlay have no core files, so these tests skip there."""
    import importlib.util

    spec = importlib.util.find_spec("flashinfer.moe_ep")
    return bool(
        spec
        and spec.origin
        and CONFIG_NAME in pathlib.Path(spec.origin).read_text(encoding="utf-8")
    )


pytestmark = pytest.mark.skipif(
    not _lazy_registry_present(),
    reason="needs the lazy-registry core patch of flashinfer.moe_ep (applied by the PR cut; the port tree and the snapshot overlay have no core files)",
)
KERNEL_NAME = "sm107_mxfp8_mxfp4_bf16_deepgemm_cutedsl"
PKG = "flashinfer.experimental.sm107_deepgemm_megamoe"
HEAVY = (
    "backend",
    "frontend",
    "jit",
    "workspace",
    "compat",
    "staging",
    "weights",
    "locality",
    "kernel",
)  # everything that needs torch / CUDA / cutlass

_CHILD = f"""
import json, sys, types

def loaded():
    return sorted(m for m in sys.modules if m == "{PKG}" or m.startswith("{PKG}."))

res = {{}}
import flashinfer, flashinfer.aot
import flashinfer.moe_ep as moe_ep
from flashinfer.moe_ep.core.kernel import registry

res["modules_after_core_import"] = loaded()
res["name_in_dir"] = "{CONFIG_NAME}" in dir(moe_ep)
res["modules_after_dir"] = loaded()
try:
    moe_ep.ThisNameDoesNotExist
    res["bogus_attribute"] = "no error"
except AttributeError as exc:
    res["bogus_attribute"] = str(exc)

# a mega kernel that does not live under flashinfer.experimental is unaffected
sm100_cfg = moe_ep.Sm100_Fp8_Fp4_Bf16_Deepgemm_MegaMoeConfig(intermediate_size=2304, top_k=6, activation_clamp=10.0, fast_math=True)
res["sm100_is_mega_config"] = registry.is_mega_kernel_config(sm100_cfg)
res["modules_after_sm100_check"] = loaded()

cfg_cls = getattr(moe_ep, "{CONFIG_NAME}")
res["modules_after_config"] = loaded()
pkg_config_cls = getattr(sys.modules["{PKG}.config"], "{CONFIG_NAME}")
res["config_is_package_class"] = cfg_cls is pkg_config_cls
cfg = cfg_cls(intermediate_size=2304, top_k=6, gate_up_clamp=10.0)
res["config_kernel_name"] = cfg.kernel_name
res["is_mega_config_before_backend_import"] = registry.is_mega_kernel_config(cfg)
res["backend_loaded_before_create"] = "{PKG}.backend" in sys.modules

try:
    registry.create_mega_kernel(types.SimpleNamespace(kernel_name="no_such_mega_kernel"))
    res["unknown_kernel"] = "no error"
except KeyError as exc:
    res["unknown_kernel"] = str(exc)
res["backend_loaded_after_unknown_kernel"] = "{PKG}.backend" in sys.modules

try:
    kernel = registry.create_mega_kernel(cfg)
    backend_module = sys.modules.get("{PKG}.backend")
    res["backend_loaded_after_create"] = backend_module is not None
    res["kernel_class"] = type(kernel).__module__ + "." + type(kernel).__qualname__
    res["kernel_name"] = kernel.kernel_name()
    res["supports_output_view"] = bool(kernel.supports_output_view)
    again = registry.create_mega_kernel(cfg)
    res["second_create_is_a_new_instance_of_the_same_class"] = again is not kernel and type(again) is type(kernel)
    res["backend_module_not_reimported"] = sys.modules.get("{PKG}.backend") is backend_module
except Exception as exc:
    res["create_error"] = type(exc).__name__ + ": " + str(exc)
print("RESULT " + json.dumps(res))
"""


@pytest.fixture(scope="module")
def probe():
    proc = subprocess.run(
        [sys.executable, "-W", "ignore", "-c", textwrap.dedent(_CHILD)],
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert proc.returncode == 0, proc.stdout[-1500:] + proc.stderr[-3000:]
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")]
    assert lines, proc.stdout[-1500:]
    return json.loads(lines[-1][len("RESULT ") :])


def test_core_import_imports_no_module_of_the_experimental_package(probe):
    assert probe["modules_after_core_import"] == []


def test_dir_lists_the_lazy_config_without_importing_it(probe):
    assert probe["name_in_dir"] is True
    assert probe["modules_after_dir"] == []


def test_unknown_attribute_of_moe_ep_raises_attribute_error(probe):
    assert "has no attribute 'ThisNameDoesNotExist'" in probe["bogus_attribute"]


def test_other_mega_kernel_configs_resolve_without_the_experimental_package(probe):
    assert probe["sm100_is_mega_config"] is True
    assert probe["modules_after_sm100_check"] == []


def test_resolving_the_config_imports_the_config_modules_only(probe):
    loaded = set(probe["modules_after_config"])
    assert f"{PKG}.config" in loaded
    heavy = sorted(m for m in loaded if m.split(".")[3:4] and m.split(".")[3] in HEAVY)
    assert heavy == [], f"resolving the config must not import {heavy}"
    assert probe["backend_loaded_before_create"] is False


def test_lazily_resolved_config_is_the_package_config_class(probe):
    assert probe["config_is_package_class"] is True
    assert probe["config_kernel_name"] == KERNEL_NAME


def test_registry_knows_the_experimental_kernel_before_its_backend_is_imported(probe):
    assert probe["is_mega_config_before_backend_import"] is True


def test_unknown_kernel_error_lists_the_lazy_kernel_and_imports_nothing(probe):
    assert (
        "no_such_mega_kernel" in probe["unknown_kernel"]
        and KERNEL_NAME in probe["unknown_kernel"]
    )
    assert probe["backend_loaded_after_unknown_kernel"] is False


def test_create_mega_kernel_imports_the_backend_on_first_use(probe):
    assert "create_error" not in probe, probe.get("create_error")
    assert probe["backend_loaded_after_create"] is True
    assert (
        probe["kernel_class"] == f"{PKG}.backend.Sm107DeepgemmCutedslMegaKernelBackend"
    )
    assert probe["kernel_name"] == KERNEL_NAME
    assert probe["supports_output_view"] is True


def test_create_mega_kernel_is_repeatable_without_reimporting_the_backend(probe):
    assert "create_error" not in probe, probe.get("create_error")
    assert probe["second_create_is_a_new_instance_of_the_same_class"] is True
    assert probe["backend_module_not_reimported"] is True
