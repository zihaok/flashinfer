# SPDX-License-Identifier: Apache-2.0
# Port of DeepGEMM (MIT License, Copyright (c) 2025 DeepSeek; see licenses/LICENSE.deepgemm.txt) sm107_fp8_fp4_mega_moe.cuh / API tests.
"""API surface of the SM107 DeepGEMM-compatible MegaMoE package (Tier 1).

* signatures equal DeepGEMM's (introspection; vLLM introspects ``num_shared_experts`` / ``shared_l1_weights`` /
  ``shared_l2_weights`` of ``get_symm_buffer_for_mega_moe`` / ``fp8_fp4_mega_moe`` and needs ``get_block_m_for_mega_moe``);
* ``SymmBuffer`` (1 GPU, EP1): every view has the same shape / strides / dtype / byte offset as DeepGEMM's, ``nbytes`` equal;
* ``fp8_fp4_mega_moe`` host checks (``csrc/apis/mega_moe.hpp:293-416``) and per-call BLOCK_M variant selection from ``y.size(0)``,
  exercised against a stub kernel factory (no kernel is compiled or launched).
"""

import inspect
import math

import pytest

from flashinfer.experimental.sm107_deepgemm_megamoe import compat

torch = pytest.importorskip("torch")

# Expected parameter lists (DeepGEMM 3ec0eb4 deep_gemm/mega/__init__.py); `name=default` where a default exists
_EXPECTED = {
    "get_symm_buffer_for_mega_moe": [
        "group", "num_experts", "num_max_tokens_per_rank", "num_topk", "hidden", "intermediate_hidden",
        "num_shared_experts=0", "use_fp8_dispatch=None", "mma_type='fp8xfp4'", "activation='swiglu'",
    ],
    "SymmBuffer": [
        "group", "num_experts", "num_max_tokens_per_rank", "num_topk", "hidden", "intermediate_hidden",
        "num_shared_experts=0", "mma_type='fp8xfp4'", "activation='swiglu'", "base=None",
    ],
    "transform_weights_for_mega_moe": ["l1_weights", "l2_weights", "activation='swiglu'"],
    "fp8_fp4_mega_moe": [
        "y", "l1_weights", "l2_weights", "sym_buffer", "shared_l1_weights=None", "shared_l2_weights=None",
        "cumulative_local_expert_recv_stats=None", "recipe=(1, 1, 32)", "activation='swiglu'", "activation_clamp=None",
        "fast_math=True", "activation_alpha=1.0", "activation_beta=0.0",
    ],
    "get_block_m_for_mega_moe": ["num_ranks", "num_experts", "num_max_tokens_per_rank", "num_tokens", "num_topk", "mma_type"],
}  # fmt: skip


def _sig_str(obj):
    f = obj.__init__ if inspect.isclass(obj) else obj
    out = []
    for name, p in inspect.signature(f).parameters.items():
        if name == "self":
            continue
        out.append(
            name if p.default is inspect.Parameter.empty else f"{name}={p.default!r}"
        )
    return out


@pytest.mark.parametrize("name", sorted(_EXPECTED))
def test_signatures_match_deepgemm(name):
    mine = _sig_str(getattr(compat, name))
    if (
        name == "get_block_m_for_mega_moe"
    ):  # DeepGEMM's Python wrapper has no default for mma_type; the port allows 'fp8xfp4'
        assert [m.split("=")[0] for m in mine] == _EXPECTED[name], mine
    else:
        assert mine == _EXPECTED[name], mine
    # ... and against the real DeepGEMM when it is importable (the vLLM-facing introspection contract)
    try:
        import deep_gemm
    except Exception:
        return
    ref_obj = {"SymmBuffer": deep_gemm.SymmBuffer}.get(
        name, getattr(deep_gemm, name, None)
    )
    if ref_obj is None:
        return
    ref = _sig_str(ref_obj)
    assert [r.split("=")[0] for r in ref] == [m.split("=")[0] for m in mine], (
        name,
        ref,
        mine,
    )
    if name != "get_block_m_for_mega_moe":
        assert ref == mine, (name, ref, mine)


def test_alignment_and_block_m_helpers():
    assert compat.get_token_alignment_for_mega_moe() == 1920
    assert compat.get_block_m_for_mega_moe(4, 384, 8192, 1024, 6, "fp8xfp4") == 128


def test_get_block_m_follows_a_forced_block_m():
    """As in DeepGEMM (``DG_MEGA_MOE_SM107_BLOCK_M`` is seen by the Python helper and by the launch), a forced BLOCK_M (``KernelOptions.block_m``) is what
    ``get_block_m_for_mega_moe`` returns: a caller stages the shared-expert scale rows with it."""
    from flashinfer.experimental.sm107_deepgemm_megamoe import frontend

    heuristic = compat.get_block_m_for_mega_moe(4, 384, 9600, 100, 6, "fp8xfp4")
    assert heuristic == 16
    with frontend.kernel_options_scope(frontend.KernelOptions(block_m=128)):
        assert compat.get_block_m_for_mega_moe(4, 384, 9600, 100, 6, "fp8xfp4") == 128
        assert compat.get_block_m_for_mega_moe(4, 384, 9600, 1900, 6, "fp8xfp4") == 128
        with pytest.raises(
            (AssertionError, ValueError)
        ):  # the arguments are still validated like the heuristic does
            compat.get_block_m_for_mega_moe(4, 384, 9600, 100, 6, "bf16")
    assert (
        compat.get_block_m_for_mega_moe(4, 384, 9600, 100, 6, "fp8xfp4") == heuristic
    )  # the scope restored the default options
    previous = frontend.get_kernel_options()
    try:
        frontend.set_kernel_options(block_m=64)
        assert compat.get_block_m_for_mega_moe(4, 384, 9600, 100, 6, "fp8xfp4") == 64
    finally:
        frontend.set_kernel_options(previous)


# ------------------------------------------------------------------------------------------------------ SymmBuffer vs DeepGEMM
_RUBIN = torch.cuda.is_available() and torch.cuda.get_device_capability() == (10, 7)
pytestmark_gpu = pytest.mark.skipif(
    not _RUBIN, reason="needs an SM107 (Rubin, cc 10.7) device"
)


def _ep1_group():
    import atexit
    import socket

    import torch.distributed as dist

    if not dist.is_initialized():
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        dist.init_process_group(
            "gloo", init_method=f"tcp://127.0.0.1:{port}", rank=0, world_size=1
        )
        atexit.register(lambda: dist.is_initialized() and dist.destroy_process_group())
    return dist.new_group([0])


def _describe(buf):
    base = buf.buffer.data_ptr()
    out = {"nbytes": buf.buffer.nbytes, "T": buf.num_max_tokens_per_rank}
    for name in ("x", "x_sf", "topk_idx", "topk_weights", "shared_l1_acts", "shared_l1_acts_sf", "shared_l2_acts",
                 "shared_l2_acts_sf", "l1_acts", "l1_acts_sf", "l2_acts", "l2_acts_sf"):  # fmt: skip
        t = getattr(buf, name)
        out[name] = (
            None
            if t is None
            else (tuple(t.shape), tuple(t.stride()), t.dtype, t.data_ptr() - base)
        )
    return out


@pytestmark_gpu
@pytest.mark.parametrize(
    "shape", [(384, 5120, 2304), (256, 4096, 2048)], ids=["v41", "v4"]
)
@pytest.mark.parametrize("T,S", [(1, 0), (1920, 1), (3000, 1), (8192, 0)])
def test_symm_buffer_views_equal_deepgemm_ep1(shape, T, S):
    deep_gemm = pytest.importorskip("deep_gemm")
    E, H, I = shape
    group = _ep1_group()
    ref = deep_gemm.get_symm_buffer_for_mega_moe(
        group, E, T, 6, H, I, num_shared_experts=S
    )
    mine = compat.get_symm_buffer_for_mega_moe(
        group, E, T, 6, H, I, num_shared_experts=S
    )
    d_ref, d_mine = _describe(ref), _describe(mine)
    assert d_ref == d_mine, {
        k: (d_ref[k], d_mine[k]) for k in d_ref if d_ref[k] != d_mine[k]
    }
    # vLLM-facing attributes
    assert (
        mine.x.device.type == "cuda"
        and mine.num_max_tokens_per_rank == ref.num_max_tokens_per_rank
    )
    assert (mine.shared_l1_acts_sf is None) == (S == 0)
    # the byte count equals the C++ answer and the buffer starts zeroed
    assert (
        mine.buffer.nbytes
        == deep_gemm._C.get_symm_buffer_size_for_mega_moe(
            1, E, mine.num_max_tokens_per_rank, 6, H, I, "fp8xfp4", "swiglu", S
        )[0]
    )
    assert int(mine.buffer[: 1 << 20].abs().sum()) == 0
    assert mine.peer_offsets == [0] and mine.base_ptr == mine.buffer.data_ptr()
    del ref
    mine.destroy()
    assert mine.buffer is None and mine.x is None and mine.handle is None


@pytestmark_gpu
def test_symm_buffer_base_reuse_and_bad_activation():
    group = _ep1_group()
    big = compat.SymmBuffer(group, 8, 1920, 2, 1024, 512, num_shared_experts=1)
    small = compat.SymmBuffer(
        group, 8, 1920, 2, 1024, 512, num_shared_experts=0, base=big
    )
    assert (
        small.buffer is big.buffer
        and small.handle is big.handle
        and small.shared_l1_acts_sf is None
    )
    with pytest.raises(AssertionError):
        compat.SymmBuffer(
            group, 8, 3840, 2, 1024, 512, base=big
        )  # larger config does not fit
    with pytest.raises(AssertionError):
        compat.SymmBuffer(group, 8, 1920, 2, 1024, 512, activation="relu")
    with pytest.warns(DeprecationWarning):
        compat.get_symm_buffer_for_mega_moe(
            group, 8, 1920, 2, 1024, 512, use_fp8_dispatch=True
        ).destroy()


# ------------------------------------------------------------------------------------------------------ fp8_fp4_mega_moe host path
class _Recorder:
    def __init__(self):
        self.calls = []

    def factory(self, cfg):
        def runner(*args):
            self.calls.append((cfg, args))

        return runner


def _sf(g, n, k, dev="cuda"):
    k128 = -(-k // 128)
    n4 = -(-n // 4) * 4
    shape, strides = (
        ((g, n, k128), (k128 * n4, 1, n4)) if g is not None else ((n, k128), (1, n4))
    )
    return torch.empty_strided(shape, strides, dtype=torch.int32, device=dev).zero_()


def _weights(E=8, H=1024, I=512, S=0):
    l1 = (
        torch.zeros((E, 2 * I, H // 2), dtype=torch.int8, device="cuda"),
        _sf(E, 2 * I, H),
    )
    l2 = (torch.zeros((E, H, I // 2), dtype=torch.int8, device="cuda"), _sf(E, H, I))
    if S:
        s1 = (
            torch.zeros((2 * I * S, H), dtype=torch.float8_e4m3fn, device="cuda"),
            _sf(None, 2 * I * S, H),
        )
        s2 = (
            torch.zeros((H, I * S), dtype=torch.float8_e4m3fn, device="cuda"),
            _sf(None, H, I * S),
        )
    else:
        s1 = s2 = None
    return l1, l2, s1, s2


@pytestmark_gpu
def test_fp8_fp4_mega_moe_selects_variant_from_this_calls_token_count():
    from flashinfer.experimental.sm107_deepgemm_megamoe import frontend

    group = _ep1_group()
    buf = compat.SymmBuffer(group, 8, 1920, 2, 1024, 512, num_shared_experts=1)
    l1, l2, s1, s2 = _weights(S=1)
    stats = torch.zeros(8, dtype=torch.int32, device="cuda")
    rec = _Recorder()
    frontend.set_kernel_factory(rec.factory)
    try:
        seen = {}
        for M in (0, 1, 7, 100, 500, 1500, 1920):
            y = torch.empty((M, 1024), dtype=torch.bfloat16, device="cuda")
            compat.fp8_fp4_mega_moe(
                y, l1, l2, buf, s1, s2, stats, activation_clamp=10.0
            )
            cfg, args = rec.calls[-1]
            assert cfg.block_m == compat.get_block_m_for_mega_moe(
                1, 8, buf.num_max_tokens_per_rank, M, 2, "fp8xfp4"
            )
            assert args[0] is y and args[1] is stats and args[2] == M and args[3] is buf
            assert (
                args[4] is l1[0]
                and args[5] is l1[1]
                and args[6] is l2[0]
                and args[7] is l2[1]
            )
            assert (
                args[8] is s1[0]
                and args[9] is s1[1]
                and args[10] is s2[0]
                and args[11] is s2[1]
            )
            assert (cfg.num_shared_experts, cfg.num_experts, cfg.num_experts_per_rank, cfg.num_ranks, cfg.hidden, cfg.intermediate_hidden,
                    cfg.num_topk, cfg.num_max_tokens_per_rank, cfg.clamp) == (1, 8, 8, 1, 1024, 512, 2, 1920, 10.0)  # fmt: skip
            seen.setdefault(cfg.block_m, cfg)
        assert len(seen) >= 3  # the variant really changes with the live token count
        # no shared experts, no stats; shared-less buffer reuse
        rec.calls.clear()
        l1b, l2b, _, _ = _weights()
        compat.fp8_fp4_mega_moe(
            torch.empty((5, 1024), dtype=torch.bfloat16, device="cuda"), l1b, l2b, buf
        )
        assert (
            rec.calls[-1][0].num_shared_experts == 0
            and rec.calls[-1][1][1] is None
            and rec.calls[-1][1][8] is None
        )
        # numerics knobs reach the config (and the compile key)
        c0 = rec.calls[-1][0]
        compat.fp8_fp4_mega_moe(
            torch.empty((5, 1024), dtype=torch.bfloat16, device="cuda"),
            l1b,
            l2b,
            buf,
            activation_clamp=6.0,
            activation_alpha=1.5,
        )
        c1 = rec.calls[-1][0]
        assert (
            math.isinf(c0.clamp)
            and c1.clamp == 6.0
            and c1.alpha == 1.5
            and c0.cache_key() != c1.cache_key()
        )
        # precompile hook builds exactly the variants a rank can select
        rec.calls.clear()
        built = []
        frontend.set_kernel_factory(
            lambda cfg: built.append(cfg.block_m) or (lambda *a: None)
        )
        bms = frontend.precompile_mega_moe_variants(
            1, 8, 1920, 2, 1024, 512, 1, num_sms=212
        )
        assert built == list(bms) and set(bms) == {
            compat.get_block_m_for_mega_moe(1, 8, 1920, m, 2, "fp8xfp4")
            for m in range(0, 1921)
        }
    finally:
        frontend.set_kernel_factory(None)
        buf.destroy()


@pytestmark_gpu
def test_fp8_fp4_mega_moe_host_asserts():
    from flashinfer.experimental.sm107_deepgemm_megamoe import frontend

    group = _ep1_group()
    buf = compat.SymmBuffer(group, 8, 1920, 2, 1024, 512, num_shared_experts=1)
    l1, l2, s1, s2 = _weights(S=1)
    y = torch.empty((16, 1024), dtype=torch.bfloat16, device="cuda")
    frontend.set_kernel_factory(_Recorder().factory)
    call = lambda **kw: compat.fp8_fp4_mega_moe(
        kw.pop("y", y), kw.pop("l1", l1), kw.pop("l2", l2), kw.pop("buf", buf), **kw
    )  # noqa: E731
    try:
        call(shared_l1_weights=s1, shared_l2_weights=s2)  # baseline passes
        with pytest.raises(AssertionError, match="recipe"):
            call(recipe=(1, 1, 128))
        with pytest.raises(AssertionError):
            call(activation="gelu")
        with pytest.raises(NotImplementedError):
            call(activation="swiglustep")
        with pytest.raises(AssertionError, match="together"):
            call(shared_l1_weights=s1)
        with pytest.raises(AssertionError, match="activation_clamp"):
            call(activation_clamp=-1.0)
        with pytest.raises(AssertionError, match="alpha"):
            call(activation_alpha=float("nan"))
        with pytest.raises(AssertionError, match="beta"):
            call(activation_beta=float("inf"))
        with pytest.raises(AssertionError, match="packed FP4"):
            call(l1=(l1[0].to(torch.float8_e4m3fn), l1[1]))
        with pytest.raises(AssertionError, match="contiguous"):
            call(l2=(l2[0].transpose(1, 2).contiguous().transpose(1, 2), l2[1]))
        with pytest.raises(AssertionError, match="int32"):
            call(l1=(l1[0], l1[1].to(torch.float32)))
        with pytest.raises(AssertionError, match="stride"):
            call(
                l1=(l1[0], torch.zeros((8, 1024, 8), dtype=torch.int32, device="cuda"))
            )  # row-major SF (not MN-major)
        with pytest.raises(AssertionError, match="size\\(-1\\)"):
            call(l1=(l1[0], _sf(8, 1024, 2048)))
        # options DeepGEMM accepts and the kernel does not implement: refused (NotImplementedError), never computed with wrong rows
        s1x2, s2x2 = _weights(S=2)[
            2:
        ]  # two shared experts (shared intermediate = 2 x routed)
        with pytest.raises(NotImplementedError, match="num_shared_experts=2"):
            call(shared_l1_weights=s1x2, shared_l2_weights=s2x2)
        padded = torch.empty_strided(
            (8, 1024, 8), (8 * 1028, 1, 1028), dtype=torch.int32, device="cuda"
        ).zero_()  # outer stride 1028 > 1024: legal for DeepGEMM
        with pytest.raises(NotImplementedError, match="stride\\(-1\\) == mn"):
            call(l1=(l1[0], padded))
        with pytest.raises(NotImplementedError, match="stride\\(-1\\) == mn"):
            call(
                shared_l1_weights=(
                    s1[0],
                    torch.empty_strided(
                        (1024, 8), (1, 1028), dtype=torch.int32, device="cuda"
                    ).zero_(),
                ),
                shared_l2_weights=s2,
            )
        with pytest.raises(AssertionError, match="hidden"):
            call(l2=_weights(H=2048)[1])
        with pytest.raises(AssertionError, match="expert counts"):
            call(l2=_weights(E=4)[1])
        with pytest.raises(AssertionError, match="experts mismatch"):
            call(l1=_weights(E=4)[0], l2=_weights(E=4)[1])
        with pytest.raises(AssertionError, match="stats|cumulative"):
            call(
                cumulative_local_expert_recv_stats=torch.zeros(
                    8, dtype=torch.int64, device="cuda"
                )
            )
        with pytest.raises(AssertionError, match="elements"):
            call(
                cumulative_local_expert_recv_stats=torch.zeros(
                    7, dtype=torch.int32, device="cuda"
                )
            )
        with pytest.raises(AssertionError, match="y must"):
            call(y=torch.empty((16, 1024), dtype=torch.float16, device="cuda"))
        with pytest.raises(AssertionError, match="exceeds"):
            call(y=torch.empty((1921, 1024), dtype=torch.bfloat16, device="cuda"))
        with pytest.raises(AssertionError, match="shared_l1_k|hidden"):
            call(
                shared_l1_weights=(
                    torch.zeros((1024, 2048), dtype=torch.float8_e4m3fn, device="cuda"),
                    _sf(None, 1024, 2048),
                ),
                shared_l2_weights=s2,
            )
        with pytest.raises(AssertionError, match="float8_e4m3fn"):
            call(
                shared_l1_weights=(s1[0].view(torch.int8), s1[1]), shared_l2_weights=s2
            )
        # a buffer that was created without room for the shared experts of this call
        small = compat.SymmBuffer(group, 8, 1920, 2, 1024, 512, num_shared_experts=0)
        with pytest.raises(AssertionError, match="needs"):
            compat.fp8_fp4_mega_moe(y, l1, l2, small, s1, s2)
        small.destroy()
    finally:
        frontend.set_kernel_factory(None)
        buf.destroy()


@pytestmark_gpu
def test_unported_options_are_refused_before_any_kernel_is_built(monkeypatch):
    """The REAL path (no stub factory): the refusals happen on the host, before ``jit.get_or_compile_kernel`` is reached."""
    from flashinfer.experimental.sm107_deepgemm_megamoe import frontend, jit

    group = _ep1_group()
    buf = compat.SymmBuffer(group, 8, 1920, 2, 1024, 512, num_shared_experts=1)
    l1, l2, s1, s2 = _weights(S=1)
    y = torch.empty((16, 1024), dtype=torch.bfloat16, device="cuda")
    frontend.set_kernel_factory(None)
    reached = []
    monkeypatch.setattr(
        jit,
        "get_or_compile_kernel",
        lambda cfg, **kw: reached.append(cfg) or (lambda *a, **k: None),
    )
    try:
        s1x2, s2x2 = _weights(S=2)[2:]
        with pytest.raises(NotImplementedError, match="num_shared_experts=2"):
            compat.fp8_fp4_mega_moe(y, l1, l2, buf, s1x2, s2x2)
        # the precompile hook and the cached-runner choke point refuse the same options
        with pytest.raises(NotImplementedError, match="die-local shared"):
            frontend.precompile_mega_moe_variants(
                1,
                8,
                1920,
                2,
                1024,
                512,
                1,
                num_sms=212,
                per_die=True,
                localized_weights=True,
                localized_shared_weights=True,
            )
        with pytest.raises(NotImplementedError, match="num_shared_experts=2"):
            frontend._cached_runner(
                1, 8, 1920, 16, 2, 1024, 512, 2, 212, math.inf, 1.0, 0.0, True
            )
        assert reached == []
        compat.fp8_fp4_mega_moe(
            y, l1, l2, buf, s1, s2
        )  # the supported call still reaches the (patched) JIT layer
        assert len(reached) == 1
    finally:
        frontend.set_kernel_factory(None)
        buf.destroy()


def test_runner_refuses_die_local_shared_weights_and_pins_the_symmetric_buffer_it_caches():
    """``jit.MegaMoeRunner`` (the real runner class, fake compiled function): a build with die-local shared weights is not launchable, and the peer-table cache cannot
    return the payload of another buffer that happens to reuse an ``id`` / base address."""
    import types

    from flashinfer.experimental.sm107_deepgemm_megamoe import heuristics, jit

    cfg = heuristics.get_sm107_kernel_config(
        2,
        384,
        1920,
        100,
        6,
        5120,
        2304,
        0,
        num_sms=212,
        per_die=True,
        localized_weights=True,
        localized_shared_weights=True,
    )
    runner = jit.MegaMoeRunner(
        types.SimpleNamespace(
            cfg=cfg, roles="auto", mode_mask=7, name=lambda: "variant"
        ),
        lambda *a: None,
        0.0,
        "test",
    )
    assert (
        runner.unsupported is not None
        and "localized_shared_weights" in runner.unsupported
    )
    with pytest.raises(NotImplementedError, match="localized_shared_weights"):
        runner.call_raw(0, 0, 1, (0, (0, 0), 0, 2), 0, 0, 0, 0, domains_ptr=1)
    # plain config: sym_payload
    cfg0 = heuristics.get_sm107_kernel_config(
        2, 384, 1920, 100, 6, 5120, 2304, 0, num_sms=212
    )
    runner = jit.MegaMoeRunner(
        types.SimpleNamespace(
            cfg=cfg0, roles="auto", mode_mask=7, name=lambda: "variant"
        ),
        lambda *a: None,
        0.0,
        "test",
    )

    class Group:
        def __init__(self, rank):
            self._rank = rank

        def rank(self):
            return self._rank

    def buf(base, offsets, rank):
        return types.SimpleNamespace(
            base_ptr=base, peer_offsets=list(offsets), group=Group(rank)
        )

    a = buf(1 << 40, [0, 4096], 0)
    p1 = runner.sym_payload(a)
    assert (
        p1 == (1 << 40, (0, 4096), 0, 2) and runner.sym_payload(a) is p1
    )  # cached per buffer object
    a.peer_offsets[1] = (
        8192  # the buffer's table changed under the same object and base: the payload follows
    )
    assert runner.sym_payload(a) == (1 << 40, (0, 8192), 0, 2)
    a.base_ptr = 2 << 40
    assert runner.sym_payload(a)[0] == 2 << 40
    # many buffers (the cache is bounded and entries pin their buffer, so a recycled id() can never alias a live entry)
    others = [buf(i << 30, [0, 1024 * i], i % 2) for i in range(1, 80)]
    assert [runner.sym_payload(o)[1:3] for o in others] == [
        ((0, 1024 * i), i % 2) for i in range(1, 80)
    ]
    assert len(runner._sym_cache) <= 65 and all(
        entry[0] is not None for entry in runner._sym_cache.values()
    )
    with pytest.raises(ValueError, match="peers"):
        runner.sym_payload(buf(1 << 40, [0], 0))


def test_missing_jit_layer_is_a_clear_error(monkeypatch):
    import sys

    from flashinfer.experimental.sm107_deepgemm_megamoe import frontend, heuristics

    cfg = heuristics.get_sm107_kernel_config(
        4, 384, 1920, 100, 6, 5120, 2304, 1, num_sms=212
    )
    frontend.set_kernel_factory(None)
    monkeypatch.setitem(
        sys.modules, "flashinfer.experimental.sm107_deepgemm_megamoe.jit", None
    )  # `from .jit import ...` -> ImportError
    with pytest.raises(NotImplementedError, match="JIT layer"):
        frontend._get_kernel(cfg)


@pytestmark_gpu
def test_frontend_drives_the_foundation_runner_abi():
    """Cross-role check: the package's `fp8_fp4_mega_moe` -> the `jit.MegaMoeRunner` (torch tensors / SymmBuffer -> the compiled ABI of
    docs/sm107_deepgemm_megamoe/DESIGN.md section 2: ints for pointers, the peer table as (base, offsets, rank, R), the current torch stream), with a fake
    compiled function in place of the DSL kernel."""
    try:
        from flashinfer.experimental.sm107_deepgemm_megamoe import frontend, jit
    except (
        ImportError
    ) as exc:  # pragma: no cover - foundation layer not importable on this host
        pytest.skip(f"jit layer not importable: {exc}")
    group = _ep1_group()
    buf = compat.SymmBuffer(group, 8, 1920, 2, 1024, 512, num_shared_experts=1)
    l1, l2, s1, s2 = _weights(S=1)
    stats = torch.zeros(8, dtype=torch.int32, device="cuda")
    y = torch.empty((100, 1024), dtype=torch.bfloat16, device="cuda")
    captured = []

    def factory(cfg):
        try:
            build = jit.make_build(cfg)
        except Exception as exc:  # pragma: no cover
            pytest.skip(f"KernelBuild not constructible here: {exc}")
        return jit.MegaMoeRunner(
            build, lambda *args: captured.append(args), 0.0, "fake"
        )

    frontend.set_kernel_factory(factory)
    try:
        compat.fp8_fp4_mega_moe(y, l1, l2, buf, s1, s2, stats, activation_clamp=10.0)
        assert len(captured) == 1
        (
            y_ptr,
            stats_ptr,
            num_tokens,
            sym,
            l1w,
            l1s,
            l2w,
            l2s,
            sl1w,
            sl1s,
            sl2w,
            sl2s,
            trace,
            domains,
            stream,
        ) = captured[0]
        assert (y_ptr, stats_ptr, num_tokens) == (y.data_ptr(), stats.data_ptr(), 100)
        assert sym == (
            buf.base_ptr,
            (0,),
            0,
            1,
        )  # (base, peer offsets, rank, num_ranks)
        assert (l1w, l1s, l2w, l2s) == (
            l1[0].data_ptr(),
            l1[1].data_ptr(),
            l2[0].data_ptr(),
            l2[1].data_ptr(),
        )
        assert (sl1w, sl1s, sl2w, sl2s) == (
            s1[0].data_ptr(),
            s1[1].data_ptr(),
            s2[0].data_ptr(),
            s2[1].data_ptr(),
        )
        assert (trace, domains) == (
            0,
            0,
        ) and stream == torch.cuda.current_stream().cuda_stream
        # a second call with the same variant reuses the memoised runner (the factory is not asked again)
        compat.fp8_fp4_mega_moe(y, l1, l2, buf, s1, s2, stats, activation_clamp=10.0)
        assert len(captured) == 2
    finally:
        frontend.set_kernel_factory(None)
        buf.destroy()


# ------------------------------------------------------------------------------------------------------ kernel build options
def test_kernel_options_validation():
    from flashinfer.experimental.sm107_deepgemm_megamoe import frontend

    assert (
        frontend.KernelOptions() == frontend.get_kernel_options()
        and frontend.KernelOptions().tag_mode == "epoch"
    )
    assert (
        frontend.KernelOptions().faithful_regs is False
        and frontend.KernelOptions(faithful_regs=True).faithful_regs is True
    )
    assert frontend.KernelOptions(regs_override=[96, 88, 160]).regs_override == (
        96,
        88,
        160,
    )  # normalised (and hashable: it keys the runner cache)
    # (faithful_regs = DeepGEMM's own split and an explicit regs_override exclude each other)
    for bad in (dict(tag_mode="x"), dict(regs_override=(1, 2)), dict(regs_override=(96, 88, 161)), dict(regs_override=(0, 8, 8)), dict(timeout_sec=0),
                dict(block_m=96), dict(smem_capacity=0), dict(max_stages=-1),
                dict(faithful_regs=True, regs_override=(96, 88, 160))):  # fmt: skip
        with pytest.raises(AssertionError):
            frontend.KernelOptions(**bad)


@pytestmark_gpu
def test_kernel_options_reach_the_jit_layer_and_key_the_variants(monkeypatch):
    """``set_kernel_options`` / ``kernel_options_scope``: tag_mode, regs_override, timeout_sec, faithful_regs go to ``jit.get_or_compile_kernel``; block_m / max_stages
    shape the config; every distinct option set is its own memoised runner."""
    from flashinfer.experimental.sm107_deepgemm_megamoe import frontend, jit

    group = _ep1_group()
    buf = compat.SymmBuffer(group, 8, 1920, 2, 1024, 512)
    l1, l2, _, _ = _weights()
    calls = []
    monkeypatch.setattr(
        jit,
        "get_or_compile_kernel",
        lambda cfg, **kw: calls.append((cfg, kw)) or (lambda *a, **k: None),
    )
    frontend.set_kernel_factory(
        None
    )  # the real path: _get_kernel -> jit.get_or_compile_kernel (patched above); also clears the memo
    y = torch.empty((5, 1024), dtype=torch.bfloat16, device="cuda")
    try:
        compat.fp8_fp4_mega_moe(y, l1, l2, buf)
        assert (
            calls[-1][1]
            == dict(
                tag_mode="epoch",
                regs_override=None,
                timeout_sec=300.0,
                faithful_regs=False,
            )
            and len(calls) == 1
        )
        frontend.set_kernel_options(
            tag_mode="gridid", regs_override=(56, 40, 200), timeout_sec=7.0
        )
        compat.fp8_fp4_mega_moe(y, l1, l2, buf)
        assert (
            calls[-1][1]
            == dict(
                tag_mode="gridid",
                regs_override=(56, 40, 200),
                timeout_sec=7.0,
                faithful_regs=False,
            )
            and len(calls) == 2
        )
        compat.fp8_fp4_mega_moe(y, l1, l2, buf)  # memoised per option set
        assert len(calls) == 2
        with frontend.kernel_options_scope(
            frontend.KernelOptions(block_m=64, max_stages=3)
        ):
            compat.fp8_fp4_mega_moe(y, l1, l2, buf)
        cfg, kw = calls[-1]
        assert (
            cfg.block_m == 64
            and cfg.num_stages <= 3
            and kw["tag_mode"] == "epoch"
            and len(calls) == 3
        )  # forced BLOCK_M, capped stages, scope's own defaults
        assert (
            frontend.get_kernel_options().tag_mode == "gridid"
        )  # the scope restored the process-wide options
        frontend.set_kernel_options(
            faithful_regs=True
        )  # DeepGEMM's own setmaxnreg split: its own variant, handed to the JIT layer
        compat.fp8_fp4_mega_moe(y, l1, l2, buf)
        assert (
            calls[-1][1]
            == dict(
                tag_mode="epoch",
                regs_override=None,
                timeout_sec=300.0,
                faithful_regs=True,
            )
            and len(calls) == 4
        )
        # precompile honours a forced BLOCK_M and the options
        calls.clear()
        built = frontend.precompile_mega_moe_variants(
            1, 8, 1920, 2, 1024, 512, kernel_options=frontend.KernelOptions(block_m=32)
        )
        assert built == (32,) and [c[0].block_m for c in calls] == [32]
        assert frontend.built_block_ms(
            1, 8, 1920, 2, 1024, 512, kernel_options=frontend.KernelOptions(block_m=32)
        ) == frozenset({32})
        assert (
            frontend.built_block_ms(
                1,
                8,
                1920,
                2,
                1024,
                512,
                kernel_options=frontend.KernelOptions(block_m=64, tag_mode="gridid"),
            )
            == frozenset()
        )
    finally:
        frontend.set_kernel_options()  # defaults
        frontend.set_kernel_factory(None)
        buf.destroy()
