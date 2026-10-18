# SPDX-License-Identifier: Apache-2.0
# Port of DeepGEMM (MIT License, Copyright (c) 2025 DeepSeek; see licenses/LICENSE.deepgemm.txt) sm107_fp8_fp4_mega_moe.cuh / layout/mega_moe.cuh.
"""CPU tests of the pure-Python byte layouts (kernel/layout.py) against DeepGEMM's numbers (shared-memory offsets, symmetric-buffer regions, signal offsets)."""

import importlib
import importlib.util
import os
import sys

import pytest


def _load_layout():
    name = "flashinfer.experimental.sm107_deepgemm_megamoe.kernel.layout"
    try:
        return importlib.import_module(name)
    except (
        Exception
    ):  # CPU-only checkout without flashinfer on the path: load the file directly
        path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "flashinfer",
            "experimental",
            "sm107_deepgemm_megamoe",
            "kernel",
            "layout.py",
        )
        spec = importlib.util.spec_from_file_location(
            "sm107dg_layout_standalone", os.path.abspath(path)
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        return mod


L = _load_layout()

# (V4.1, E=384, shipped stage counts)
EXPECTED_SMEM = {
    16: dict(smem_d=(3072, 4096), smem_a=(7168, 13312), smem_b=(20480, 106496), smem_sfa=126976, smem_sfb=133632, amax_reduction=140288,
             task_infos=140544, dispatch_barriers=140608, full_barriers=140640, empty_barriers=140744, tmem_full_barriers=140848,
             tmem_empty_barriers=140864, combine_barriers=140880, task_info_full_barriers=141008, task_info_empty_barriers=141024,
             tmem_ptr_in_smem=141040, end=141044, sizeof=141312),
    32: dict(smem_d=(3072, 8192), smem_a=(11264, 24576), smem_b=(35840, 98304), smem_sfa=134144, smem_sfb=140288, amax_reduction=146432,
             task_infos=146944, dispatch_barriers=147008, full_barriers=147040, empty_barriers=147136, tmem_full_barriers=147232,
             tmem_empty_barriers=147248, combine_barriers=147264, task_info_full_barriers=147392, task_info_empty_barriers=147408,
             tmem_ptr_in_smem=147424, end=147428, sizeof=147456),
    64: dict(smem_d=(3072, 8192), smem_a=(11264, 45056), smem_b=(56320, 90112), smem_sfa=146432, smem_sfb=152064, amax_reduction=157696,
             task_infos=158208, dispatch_barriers=158272, full_barriers=158304, empty_barriers=158392, tmem_full_barriers=158480,
             tmem_empty_barriers=158496, combine_barriers=158512, task_info_full_barriers=158640, task_info_empty_barriers=158656,
             tmem_ptr_in_smem=158672, end=158676, sizeof=158720),
    128: dict(smem_d=(3072, 16384), smem_a=(19456, 65536), smem_b=(84992, 65536), smem_sfa=150528, smem_sfb=154624, amax_reduction=158720,
              task_infos=159744, dispatch_barriers=159808, full_barriers=159840, empty_barriers=159904, tmem_full_barriers=159968,
              tmem_empty_barriers=159984, combine_barriers=160000, task_info_full_barriers=160128, task_info_empty_barriers=160144,
              tmem_ptr_in_smem=160160, end=160164, sizeof=160768),
    192: dict(smem_d=(3072, 16384), smem_a=(19456, 86016), smem_b=(105472, 57344), smem_sfa=162816, smem_sfb=169984, amax_reduction=173568,
              task_infos=174592, dispatch_barriers=174656, full_barriers=174688, empty_barriers=174744, tmem_full_barriers=174800,
              tmem_empty_barriers=174816, combine_barriers=174832, task_info_full_barriers=174960, task_info_empty_barriers=174976,
              tmem_ptr_in_smem=174992, end=174996, sizeof=175104),
    240: dict(smem_d=(3072, 20480), smem_a=(23552, 92160), smem_b=(115712, 49152), smem_sfa=164864, smem_sfb=171008, amax_reduction=174080,
              task_infos=175360, dispatch_barriers=175424, full_barriers=175456, empty_barriers=175504, tmem_full_barriers=175552,
              tmem_empty_barriers=175568, combine_barriers=175584, task_info_full_barriers=175712, task_info_empty_barriers=175728,
              tmem_ptr_in_smem=175744, end=175748, sizeof=176128),
}  # fmt: skip


@pytest.mark.parametrize("bm", sorted(EXPECTED_SMEM))
def test_smem_offsets_match_reference(bm):
    s = L.SHIPPED_V41[bm]
    o = L.smem_offsets(
        block_m=bm,
        num_stages=s["num_stages"],
        num_experts=384,
        store_block_m=s["store_block_m"],
    )
    spec = EXPECTED_SMEM[bm]
    assert o["expert_token_count"] == 0
    assert o["dispatch_send_buffer"] == 2048
    for name, want in spec.items():
        if isinstance(want, tuple):
            assert (o[name], o[name + ".size"]) == want, name
        elif name in ("end", "sizeof"):
            assert o[name] == want, name
        else:
            assert o[name] == want, name
    assert o["reusable_bytes"] == spec["dispatch_barriers"]
    # combine overlay needs 3 slots x 8 warps x 5120 B
    assert o["reusable_bytes"] >= 3 * 8 * 5120
    assert o["sizeof"] <= s["smem_bytes"]


def test_signals_layout():
    assert L.SIGNALS_SIZEOF == 21_140_864
    assert L.SIGNALS_L2_FULL_MASK % 8 == 0 and L.SIGNALS_EXPERT_SEND_COUNT % 128 == 0


@pytest.mark.parametrize(
    "bm,cols",
    [
        (16, (32, 4, 4, 64, 32)),
        (32, (64, 4, 4, 128, 64)),
        (64, (128, 4, 4, 256, 128)),
        (128, (256, 4, 4, 512, 256)),
        (192, (384, 8, 4, 512, 384)),
        (240, (480, 8, 4, 512, 480)),
    ],
)
def test_tmem_cols(bm, cols):
    assert L.tmem_cols(bm) == cols


def test_buffer_layout_worked_example_dep8():
    b = L.buffer_layout(
        hidden=5120,
        intermediate_hidden=2304,
        num_ranks=8,
        num_experts=384,
        num_max_tokens_per_rank=9600,
        num_topk=6,
        num_ring_tokens=51840,
        num_sf_ring_tokens=829440,
        num_shared_experts=1,
    )
    exp = dict(input_token=41554304, input_sf=90706304, input_topk_idx=92242304, input_topk_weights=92703104, shared_l1_sf=92933504,
               shared_l2_token=117509504, shared_l2_sf=139627904, l1_token=150687104, l1_sf=416107904, l1_topk_weights=548818304,
               l2_token=549025664, l2_sf=668465024, combine_token=728184704)  # fmt: skip
    for k, v in exp.items():
        assert b[k].offset == v, k
    assert b.workspace.nbytes == 41_554_304
    assert b.workspace.num_max_pool_tokens == 472_320
    assert b.total_bytes == 1_416_312_704
    assert b["combine_token"].nbytes == 7 * 9600 * 10240


@pytest.mark.parametrize(
    "ranks,ring,sfring,mib",
    [
        (4, 30720, 491520, 1123.8),
        (8, 51840, 829440, 1350.7),
        (16, 97920, 1566720, 1845.5),
        (32, 190080, 3041280, 2835.1),
    ],
)
def test_symm_buffer_total_mib(ranks, ring, sfring, mib):
    b = L.buffer_layout(
        hidden=5120,
        intermediate_hidden=2304,
        num_ranks=ranks,
        num_experts=384,
        num_max_tokens_per_rank=9600,
        num_topk=6,
        num_ring_tokens=ring,
        num_sf_ring_tokens=sfring,
        num_shared_experts=1,
    )
    assert round(b.total_bytes / 2**20, 1) == mib


def _package_dir():
    import pathlib

    try:
        spec = importlib.util.find_spec(
            "flashinfer.experimental.sm107_deepgemm_megamoe"
        )
        if spec is not None and spec.origin:
            return pathlib.Path(spec.origin).parent
    except Exception:  # noqa: BLE001 - CPU-only checkout
        pass
    return pathlib.Path(
        os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "..",
                "flashinfer",
                "experimental",
                "sm107_deepgemm_megamoe",
            )
        )
    )


def test_package_never_reads_or_writes_the_environment():
    """FlashInfer experimental policy: no module of the port (jit, kernel/*, frontend, ...) touches ``os.environ`` / ``os.getenv`` (AST scan: comments and docstrings are ignored)."""
    import ast

    offenders = []
    for path in sorted(_package_dir().rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Attribute)
                and node.attr in ("environ", "getenv", "putenv", "unsetenv")
                and isinstance(node.value, ast.Name)
                and node.value.id == "os"
            ):
                offenders.append(f"{path.name}:{node.lineno} os.{node.attr}")
            elif (
                isinstance(node, ast.ImportFrom)
                and node.module == "os"
                and any(
                    a.name in ("environ", "getenv", "putenv", "unsetenv")
                    for a in node.names
                )
            ):
                offenders.append(f"{path.name}:{node.lineno} from os import ...")
    assert not offenders, offenders
