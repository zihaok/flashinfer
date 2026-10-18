# SPDX-License-Identifier: Apache-2.0
# Port of DeepGEMM (MIT License, Copyright (c) 2025 DeepSeek; see licenses/LICENSE.deepgemm.txt) sm107_fp8_fp4_mega_moe.cuh / layout/mega_moe.cuh.
"""Int32 address-overflow regression tests of the symmetric-buffer address arithmetic (DESIGN.md section 16, "Int32 address overflow").

Background.  A CuTe-DSL ``Int32 * python_int`` stays Int32 and wraps modulo 2**32.  ``BufferMap.region_addr(..., slot=<runtime Int32>)`` used to evaluate
``r.offset + slot * token_bytes * num_tokens`` in Int32, so the L2-epilogue write-back address of a combine slot was wrong (below the buffer, or aliasing) as soon as that
offset crossed 2 GiB: DEP32 E=384 with T_max >= 7680, DEP4 E=384 with T_max >= 24960, EP1 E=96 with T_max >= 32640, shared slot at EP1 S=1 from T_max = 26880 ...

Layers:

1. CPU window scan: the REAL ``BufferMap.region_addr`` evaluated with the DSL's own scalar semantics (real ``cutlass.Int32`` / ``Int64`` objects outside a jit region wrap and
   promote exactly like the compiled arithmetic) against exact Python integers, over (R, E, S, T_max, BLOCK_M): every region, every combine slot, boundary tokens, runtime and
   Python-int operands.
2. CPU: the scan has teeth: the former formulation is wrong in the known windows.
3. PTX: the REAL-roles kernel compiled in wrap-window configurations keeps the combine slot stride only in 64-bit multiplies and has no 32-bit ``mul.lo`` / ``mad.lo`` with a huge
   immediate anywhere (needs a CUDA device like every other compile test of the package).

Run: ``pytest tests/experimental/test_sm107_dg_foundation_addr.py`` (the PTX layer needs an SM107 device).
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import warnings

import pytest

cutlass = pytest.importorskip("cutlass")

from cutlass import Int32, Int64  # noqa: E402

BASE = (
    0x7F5A_0000_0000  # a plausible device VA; exact integer expectation = BASE + offset
)
INT32_MAX = (1 << 31) - 1


def _kernel_config(
    num_ranks: int, num_experts: int, t_max: int, num_shared: int, block_m: int
):
    from flashinfer.experimental.sm107_deepgemm_megamoe.heuristics import (
        get_sm107_kernel_config,
    )

    return get_sm107_kernel_config(
        num_ranks,
        num_experts,
        t_max,
        1024,
        6,
        5120,
        2304,
        num_shared,
        num_sms=212,
        activation_clamp=10.0,
        block_m=block_m,
    )


def _buffer_map(cfg):
    from flashinfer.experimental.sm107_deepgemm_megamoe.kernel.ctx import BufferMap

    return BufferMap(Int64(BASE), cfg)


def _old_region_addr(bm, name: str, token=0, slot=0):
    """The former formulation, verbatim (kept only to prove that the scan detects it)."""
    r = bm.layout[name]
    off = (
        r.offset + slot * r.token_bytes * r.num_tokens
        if not (isinstance(slot, int) and slot == 0)
        else r.offset
    )
    if isinstance(token, int) and token == 0:
        return bm.base + Int64(off)
    return bm.base + Int64(off) + Int64(token) * r.token_bytes


def _value(x) -> int:
    return int(getattr(x, "value", x))


def _expected(r, slot: int, token: int) -> int:
    return BASE + r.offset + slot * r.token_bytes * r.num_tokens + token * r.token_bytes


def _scan_point(bm, fn, tag, mismatches):
    """All (region, slot, token) forms of one configuration; appends (tag, name, slot, token, got, expected) for every wrong address."""
    for name, r in bm.layout.regions.items():
        if r.num_tokens == 0:
            continue
        slots = range(r.num_slots) if name == "combine_token" else (0,)
        for slot in slots:
            for token in sorted({0, 1, r.num_tokens - 1}):
                want = _expected(r, slot, token)
                forms = [
                    dict(token=Int32(token), slot=Int32(slot)),
                    dict(token=token, slot=slot),
                ]
                if slot == 0:
                    forms.append(dict(token=Int32(token)))
                if token == 0:
                    forms.append(dict(slot=Int32(slot)))
                for kw in forms:
                    got = _value(fn(bm, name, **kw))
                    if got != want:
                        mismatches.append(
                            (
                                tag,
                                name,
                                slot,
                                token,
                                got - BASE,
                                want - BASE,
                                sorted(kw),
                            )
                        )
                        break


# (R, E): EP1 small/medium/production expert counts, and the DEP2..DEP32 production shapes
SCAN_SHAPES = [
    (1, 32),
    (1, 96),
    (1, 384),
    (2, 384),
    (4, 384),
    (8, 384),
    (16, 384),
    (32, 384),
]
SCAN_BLOCK_MS = (16, 128, 240)


def _scan(num_ranks: int, num_experts: int, num_shared: int, fn, t_values):
    mismatches, windows, points = [], 0, 0
    with warnings.catch_warnings():
        warnings.simplefilter(
            "ignore"
        )  # the DSL warns on every Int32 wrap-around of the (deliberately evaluated) old formulation
        for t_max in t_values:
            for block_m in SCAN_BLOCK_MS:
                try:
                    cfg = _kernel_config(
                        num_ranks, num_experts, t_max, num_shared, block_m
                    )
                except (
                    AssertionError,
                    ValueError,
                ):  # configuration refused by the host heuristics (not a kernel address question)
                    continue
                bm = _buffer_map(cfg)
                r = bm.layout["combine_token"]
                last_slot_off = (
                    r.offset + (r.num_slots - 1) * r.token_bytes * r.num_tokens
                )
                windows += int(last_slot_off > INT32_MAX)
                points += 1
                _scan_point(
                    bm,
                    fn,
                    (
                        num_ranks,
                        num_experts,
                        num_shared,
                        cfg.num_max_tokens_per_rank,
                        block_m,
                    ),
                    mismatches,
                )
    return mismatches, windows, points


T_SCAN = [
    1920 * k for k in range(1, 46)
]  # 1920 .. 86400 tokens per rank (the host aligns T_max up to multiples of 1920)


@pytest.mark.parametrize("num_shared", [0, 1])
@pytest.mark.parametrize("num_ranks,num_experts", SCAN_SHAPES)
def test_region_addr_is_exact_over_the_int32_wrap_windows(
    num_ranks, num_experts, num_shared
):
    mismatches, windows, points = _scan(
        num_ranks,
        num_experts,
        num_shared,
        lambda bm, name, **kw: bm.region_addr(name, **kw),
        T_SCAN,
    )
    assert points >= 20, "the scan must cover many configurations"
    assert windows > 0, (
        "the scanned T_max range must reach the >2 GiB windows (otherwise this test proves nothing)"
    )
    assert not mismatches, (
        f"{len(mismatches)} wrong addresses in {points} configurations ({windows} in a wrap window); first: {mismatches[:3]}"
    )


@pytest.mark.parametrize("num_shared", [0, 1])
@pytest.mark.parametrize("num_ranks,num_experts", SCAN_SHAPES)
def test_scan_has_teeth_the_former_formulation_is_wrong_in_the_windows(
    num_ranks, num_experts, num_shared
):
    mismatches, windows, _ = _scan(
        num_ranks, num_experts, num_shared, _old_region_addr, T_SCAN
    )
    assert windows > 0
    assert mismatches, (
        "the old Int32 formulation must be flagged by the scan (otherwise the scan does not test the overflow)"
    )
    assert {m[1] for m in mismatches} == {"combine_token"}, (
        "only the runtime-slot combine address wrapped"
    )


def test_known_windows_of_the_reviewer_finding():
    """Spot checks of the reported windows (DEP32 E=384 T_max=7680 slot 5; EP1 S=1 T_max=26880 shared slot 6), BLOCK_M 128."""
    for num_ranks, num_experts, num_shared, t_max, bad_slot in [
        (32, 384, 0, 7680, 5),
        (1, 32, 1, 26880, 6),
        (1, 96, 0, 32640, 5),
    ]:
        bm = _buffer_map(_kernel_config(num_ranks, num_experts, t_max, num_shared, 128))
        r = bm.layout["combine_token"]
        want = BASE + r.offset + bad_slot * r.token_bytes * r.num_tokens
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            old = _value(
                _old_region_addr(
                    bm, "combine_token", token=Int32(0), slot=Int32(bad_slot)
                )
            )
        new = _value(
            bm.region_addr("combine_token", token=Int32(0), slot=Int32(bad_slot))
        )
        assert old != want, (
            f"the old formulation was expected to wrap at {(num_ranks, num_experts, num_shared, t_max, bad_slot)}"
        )
        assert new == want


def test_region_addr_python_int_operands_are_folded_on_the_host():
    cfg = _kernel_config(4, 384, 24960, 0, 128)
    bm = _buffer_map(cfg)
    r = bm.layout["combine_token"]
    got = bm.region_addr("combine_token", token=7, slot=5)
    assert (
        _value(got)
        == BASE + r.offset + 5 * r.token_bytes * r.num_tokens + 7 * r.token_bytes
    )
    assert _value(bm.region_addr("combine_token")) == BASE + r.offset
    assert _value(bm.region_addr("combine_token", slot=0, token=0)) == BASE + r.offset


# ---------------------------------------------------------------------------------------------------------------------
# PTX layer (compile only; real roles)
# ---------------------------------------------------------------------------------------------------------------------
_COMPILE_SCRIPT = r"""
import sys
from flashinfer.experimental.sm107_deepgemm_megamoe import jit
from flashinfer.experimental.sm107_deepgemm_megamoe.heuristics import get_sm107_kernel_config
R, E, T, S, BM = (int(v) for v in sys.argv[1:6])
cfg = get_sm107_kernel_config(R, E, T, 1024, 6, 5120, 2304, S, activation_clamp=10.0, block_m=BM)
kern = jit.compile_kernel_object(jit.make_build(cfg, roles="auto", mode_mask=7))
open(sys.argv[6], "w").write(kern.__ptx__ or "")  # the compiled object carries its PTX when CUTE_DSL_KEEP=ptx is set
print("T_MAX", cfg.num_max_tokens_per_rank, "STRIDE", cfg.hidden * 2 * cfg.num_max_tokens_per_rank, flush=True)
"""

_MULADD = re.compile(
    r"^\s*(?:@!?%p\d+\s+)?((?:mul|mad)\.(?:lo|wide)\.[su](?:32|64))\s+(.*);"
)
_WIDE_OK = {
    "mul.wide.s32",
    "mul.wide.u32",
    "mad.wide.s32",
    "mad.wide.u32",
    "mul.lo.s64",
    "mul.lo.u64",
    "mad.lo.s64",
    "mad.lo.u64",
}
_HUGE = (
    1 << 22
)  # 4 MiB: no smem / TMEM / small-index immediate gets near this; a 10 KiB row times T tokens always does


def ptx_address_findings(ptx: str, stride: int):
    """(stride_uses, bad_stride_uses, huge_32bit_muls) of a kernel PTX: lines with the combine-slot stride as an operand, those of them that are not 64-bit multiplies,
    and every 32-bit mul.lo / mad.lo with an immediate operand of absolute value >= 4 MiB."""
    stride_uses, bad_stride, huge32 = [], [], []
    token = re.compile(rf"(?<![\w.%-]){stride}(?![\w.%])")
    for line in ptx.splitlines():
        m = _MULADD.match(line)
        if token.search(line):
            stride_uses.append(line.strip())
            if not (m and m.group(1) in _WIDE_OK):
                bad_stride.append(line.strip())
        if m and m.group(1).endswith("32") and ".lo." in m.group(1):
            for operand in m.group(2).split(","):
                operand = operand.strip()
                if re.fullmatch(r"-?\d+", operand) and abs(int(operand)) >= _HUGE:
                    huge32.append(line.strip())
                    break
    return stride_uses, bad_stride, huge32


def test_ptx_scanner_flags_the_32bit_form_and_accepts_the_64bit_form():
    old = "\tmad.lo.s32 \t%r958, %r957, 353894400, 542973824;\n\tcvt.s64.s32 \t%rd1, %r958;\n"
    new = "\tmul.wide.s32 \t%rd460, %r958, 353894400;\n\tadd.s64 \t%rd461, %rd50, %rd460;\n"
    uses, bad, huge = ptx_address_findings(old, 353894400)
    assert uses and bad and huge
    uses, bad, huge = ptx_address_findings(new, 353894400)
    assert uses and not bad and not huge
    uses, bad, huge = ptx_address_findings(
        "\tmul.hi.u32 \t%r1, %r2, 715827883;\n\tmul.lo.s32 \t%r3, %r4, 6;\n", 353894400
    )
    assert (
        not uses and not bad and not huge
    )  # magic-number division and small multiplies are legitimate


@pytest.mark.parametrize(
    "num_ranks,num_experts,t_max,num_shared,block_m",
    [
        (
            1,
            96,
            34560,
            0,
            128,
        ),  # EP1 slot 5 wraps (reproduction: EP1, E=96, T_max=32768)
        (1, 32, 26880, 1, 64),  # EP1 S=1: the shared slot 6 wraps
        (32, 384, 7680, 0, 128),  # DEP32 production shape: slot 5 wraps
        (4, 384, 24960, 1, 192),  # DEP4 production shape with the shared expert
    ],
)
def test_ptx_combine_slot_stride_only_in_64bit_math(
    tmp_path, num_ranks, num_experts, t_max, num_shared, block_m
):
    torch = pytest.importorskip("torch")
    if not (
        torch.cuda.is_available() and torch.cuda.get_device_capability() == (10, 7)
    ):
        pytest.skip(
            "needs an SM107 (Rubin, compute capability 10.7) device (cute.compile of the real-roles kernel for sm_107a)"
        )
    env = dict(os.environ, CUTE_DSL_KEEP="ptx", CUTE_DSL_DUMP_DIR=str(tmp_path))
    out = subprocess.run(
        [sys.executable, "-c", _COMPILE_SCRIPT, str(num_ranks), str(num_experts), str(t_max), str(num_shared), str(block_m), str(tmp_path / "kernel.ptx")],
        env=env, capture_output=True, text=True, timeout=900,
    )  # fmt: skip
    assert out.returncode == 0, out.stderr[-2000:]
    stride = int(re.search(r"STRIDE (\d+)", out.stdout).group(1))
    ptx = (tmp_path / "kernel.ptx").read_text()
    assert len(ptx) > 10000, (
        f"the compiled object carried no PTX ({len(ptx)} chars; CUTE_DSL_KEEP=ptx in the environment?)"
    )
    uses, bad_stride, huge32 = ptx_address_findings(ptx, stride)
    assert uses, (
        f"the combine slot stride {stride} does not appear in the PTX (the epilogue's runtime-slot address was constant-folded or moved)"
    )
    assert not bad_stride, (
        f"combine slot stride {stride} used outside 64-bit multiplies: {bad_stride[:3]}"
    )
    assert not huge32, (
        f"32-bit mul.lo / mad.lo with a huge immediate (Int32 address arithmetic that can wrap): {huge32[:3]}"
    )
