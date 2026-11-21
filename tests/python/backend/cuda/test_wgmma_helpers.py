# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# ruff: noqa: E501
"""Hopper WGMMA helpers (backend/cuda/lang/wgmma.py) and named barriers.

GPU tests need an sm_90 device: ``wgmma`` exists only on sm_90a. Every kernel is
compiled with ``perf_remarks="error"``, so a chain that ptxas serializes fails
here, and the chain-structure test also checks the SASS.
"""

import os
import re
import shutil
import subprocess
import tempfile

import numpy as np
import pytest

import tvm_tirx
import tvm_tirx.testing
from tvm_tirx.backend.cuda.lang import NamedBarrier, NamedBarriers
from tvm_tirx.backend.cuda.lang import wgmma as W
from tvm_tirx.script import tirx as T
from tvm_tirx.testing import env

TARGET = {"kind": "cuda", "arch": "sm_90a"}
STRICT = {"cuda": {"perf_remarks": "error"}}
requires_sm90 = pytest.mark.skipif(
    not env.has_cuda_compute(9, exact=True), reason="wgmma needs an sm_90 device"
)


# ---------------------------------------------------------------------------
# Pure-Python checks
# ---------------------------------------------------------------------------


def test_shapes():
    op = W.WGMMA(256, "bfloat16")
    assert (op.k, op.num_acc, op.k_bytes) == (16, 128, 32)
    assert op.mnemonic == "wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16"
    fp8 = W.WGMMA(128, "float8_e4m3fn", b_dtype="float8_e5m2")
    assert (fp8.k, fp8.num_acc) == (32, 64)
    assert fp8.mnemonic == "wgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e5m2"
    assert W.WGMMA(64, "float16", acc_dtype="float16").num_acc == 16
    for bad in (dict(n=12), dict(n=264), dict(n=64, ab_dtype="float32"),
                dict(n=64, ab_dtype="float16", b_dtype="bfloat16")):  # fmt: skip
        with pytest.raises(ValueError):
            W.WGMMA(**bad)


@pytest.mark.parametrize("n", [8, 64, 256])
def test_acc_coord_is_a_bijection(n):
    seen = {W.acc_coord(t, r) for t in range(128) for r in range(n // 2)}
    assert seen == {(i, j) for i in range(64) for j in range(n)}


def test_a_fragment_is_the_first_16_accumulator_columns():
    frag = {
        (t, r, h): W.a_fragment_coord(t, r, h)
        for t in range(128)
        for r in range(4)
        for h in range(2)
    }
    assert set(frag.values()) == {(i, k) for i in range(64) for k in range(16)}
    assert all(frag[t, r, h] == W.acc_coord(t, 2 * r + h) for (t, r, h) in frag)


def test_named_barriers():
    bars = NamedBarriers()
    assert [bars.alloc(256).barrier_id for _ in range(3)] == [1, 2, 3]
    with pytest.raises(ValueError, match="reserved"):
        NamedBarrier(0, 128)
    with pytest.raises(ValueError, match="multiple of 32"):
        NamedBarrier(1, 100)
    full = NamedBarriers()
    for _ in range(15):
        full.alloc(32)
    with pytest.raises(ValueError, match="out of named barriers"):
        full.alloc(32)
    # Ids used directly (e.g. warpgroup_sync(wg + 1) for 2 warpgroups) are reserved.
    bars = NamedBarriers(reserved=(1, 2))
    assert [bars.alloc(256).barrier_id for _ in range(2)] == [3, 4]
    bars.reserve(10)
    assert bars.alloc(128).barrier_id == 5 and bars.free == 16 - len({0, 1, 2, 3, 4, 5, 10})
    with pytest.raises(ValueError, match="already allocated"):
        bars.reserve(3)
    with pytest.raises(ValueError, match="outside"):
        bars.reserve(0)


# ---------------------------------------------------------------------------
# GEMM tiles through the helpers: C[64, N] = A[64, K] @ B^T, one chain over K
# ---------------------------------------------------------------------------

_TORCH = {"bfloat16": "bfloat16", "float16": "float16", "float8_e4m3fn": "float8_e4m3fn"}
_SWIZZLE_ENUM = {32: 1, 64: 2, 128: 3}  # CU_TENSOR_MAP_SWIZZLE_*


def _tile_kernel(op: W.WGMMA, b_major="K", swizzle=128, k_bytes=128):
    """One warpgroup. A [64, K] and B (K-major [N, K] or MN-major [K, N]) arrive by
    swizzled TMA, one box per swizzle atom, stacked as SmemTile expects. One chain
    of k_bytes / 32 steps covers K, crossing atoms when k_bytes > swizzle."""
    dt = op.ab_dtype
    eb = 32 // op.k
    KE = k_bytes // eb  # K extent in elements
    N = op.n
    S = swizzle
    AE = S // eb  # elements per swizzle row
    steps = k_bytes // op.k_bytes
    sw = _SWIZZLE_ENUM[S]
    b_shape = (N, KE) if b_major == "K" else (KE, N)
    b_atoms = k_bytes // S if b_major == "K" else N * eb // S
    b_rows = N if b_major == "K" else KE
    rs = op.a_in_registers
    trans_b = b_major == "MN"
    tx_bytes = (64 + N) * k_bytes

    # fmt: off
    @T.function
    def main(A: T.Tensor((64, KE), dt, align=16), B: T.Tensor(b_shape, dt, align=16),
             C: T.Tensor((64, N), "float32", align=16)):
        A_map: T.let[T.handle("tensormap")] = T.stack_alloca("tensormap", 1)
        T.call_packed("runtime.cuTensorMapEncodeTiled", A_map, dt, 2, A.data,
                      KE, 64, KE * eb, AE, 64, 1, 1, 0, sw, 0, 0)
        B_map: T.let[T.handle("tensormap")] = T.stack_alloca("tensormap", 1)
        if T.constexpr(b_major == "K"):
            T.call_packed("runtime.cuTensorMapEncodeTiled", B_map, dt, 2, B.data,
                          KE, N, KE * eb, AE, N, 1, 1, 0, sw, 0, 0)
        else:
            T.call_packed("runtime.cuTensorMapEncodeTiled", B_map, dt, 2, B.data,
                          N, KE, N * eb, AE, KE, 1, 1, 0, sw, 0, 0)
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=128), backend_config=STRICT)
        tx = T.cuda.thread_idx("x")
        A_s = T.alloc_tensor(ty_args=[T.Tensor((64 * KE,), dt, scope="shared", align=1024)])
        B_s = T.alloc_tensor(ty_args=[T.Tensor((N * KE,), dt, scope="shared", align=1024)])
        bar = T.shared_scalar("uint64")
        acc = T.alloc_tensor(ty_args=[T.Tensor((op.num_acc,), "float32", scope="local")])
        if tx == 0:
            T.ptx.mbarrier.init.shared.b64(T.address_of(bar), T.uint32(1))
        T.ptx.fence.proxy.async_.shared__cta()
        T.cuda.cta_sync()
        if tx == 0:
            for j in T.unroll(k_bytes // S):  # A: one box per K atom
                T.ptx["cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"](
                    A_s.ptr_to([j * 64 * AE]), T.address_of(A_map), j * AE, 0, T.address_of(bar))
            for j in T.unroll(b_atoms):  # B: one box per K atom (K-major) or MN atom (MN-major)
                T.ptx["cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"](
                    B_s.ptr_to([j * b_rows * AE]), T.address_of(B_map), j * AE, 0, T.address_of(bar))
            T.ptx.mbarrier.arrive.expect_tx.shared.b64(T.address_of(bar), T.uint32(tx_bytes))
        T.cuda.mbarrier_wait(T.address_of(bar), 0)

        a_tile = W.SmemTile(A_s.data, rows=64, swizzle=S)
        b_tile = W.SmemTile(B_s.data, rows=b_rows, major=b_major, swizzle=S)
        b_steps = b_tile.k_steps(op, steps)
        if T.constexpr(rs):
            # A fragments straight from global memory, per the rs register map.
            a_half = T.alloc_tensor(ty_args=[T.Tensor((steps * 8,), dt, scope="local")])
            for s in T.unroll(steps):
                for r in T.unroll(4):
                    for h in T.unroll(2):
                        row, k = W.a_fragment_coord(tx, r, h)
                        a_half[s * 8 + r * 2 + h] = A[row, s * op.k + k]
            a_regs = a_half.view("uint32")
            W.chain(op, acc, [_Slice(a_regs, s * 4) for s in range(steps)], b_steps,
                    accumulate=False, transpose_b=trans_b)
        else:
            W.chain(op, acc, a_tile.k_steps(op, steps), b_steps, accumulate=False, transpose_b=trans_b)
        W.wait(0, (acc, op.num_acc))
        for r in T.unroll(op.num_acc):
            row, col = W.acc_coord(tx, r)
            C[row, col] = acc[r]
    # fmt: on
    return main


class _Slice:
    """A 4-register window of a local uint32 tensor, as an rs A operand."""

    def __init__(self, tensor, start):
        self.tensor, self.start = tensor, start

    def __getitem__(self, i):
        return self.tensor[self.start + i]

    def __iter__(self):
        return iter(self.tensor[self.start + i] for i in range(4))

    def __len__(self):
        return 4


def _compile(func):
    target = tvm_tirx.target.Target(TARGET)
    return tvm_tirx.compile(tvm_tirx.IRModule({"main": func}), target=target, pipeline="tirx")


def _run_tile(op, b_major="K", swizzle=128, k_bytes=128):
    import torch

    eb = 32 // op.k
    KE = k_bytes // eb
    torch.manual_seed(0)
    dt = getattr(torch, _TORCH[op.ab_dtype])
    a = (torch.randn(64, KE, device="cuda") / 2).to(dt)
    b_nk = (torch.randn(op.n, KE, device="cuda") / 2).to(dt)
    b = b_nk if b_major == "K" else b_nk.t().contiguous()
    c = torch.zeros(64, op.n, device="cuda")
    mod = _compile(_tile_kernel(op, b_major, swizzle, k_bytes))
    mod(a, b, c)
    torch.cuda.synchronize()
    ref = a.float() @ b_nk.float().t()
    torch.testing.assert_close(c, ref, rtol=1e-3, atol=2e-2)
    return mod


def test_k_offsets_cross_swizzle_atoms():
    op, fp8 = W.WGMMA(64, "bfloat16"), W.WGMMA(64, "float8_e4m3fn")
    tile = W.SmemTile(None, rows=64, swizzle=128)
    assert [tile.k_offset_bytes(op, k) for k in range(6)] == [0, 32, 64, 96, 8192, 8224]
    assert [W.SmemTile(None, rows=128, swizzle=32).k_offset_bytes(fp8, k) for k in range(3)] == [
        0,
        4096,
        8192,
    ]
    mn = W.SmemTile(None, rows=64, major="MN", swizzle=64)
    assert [mn.k_offset_bytes(op, k) for k in range(3)] == [0, 1024, 2048]
    with pytest.raises(ValueError, match="no-swizzle"):
        W.SmemTile(None, rows=64, swizzle=0)
    with pytest.raises(ValueError, match="K-major"):
        mn.k_offset_bytes(fp8, 1)


_K_MAJOR_SWEEP = [
    (dt, sw, kb)
    for dt in ("bfloat16", "float8_e4m3fn")
    for sw in (128, 64, 32)
    for kb in sorted({sw, 256})
]


@requires_sm90
@pytest.mark.parametrize("dt, swizzle, k_bytes", _K_MAJOR_SWEEP,
                         ids=[f"{d[:4]}-sw{s}-k{k}B" for d, s, k in _K_MAJOR_SWEEP])  # fmt: skip
def test_k_major_chain_across_swizzle_atoms(dt, swizzle, k_bytes):
    """K extents of one atom and of 256 B (2, 4 or 8 atoms) at every swizzle."""
    _run_tile(W.WGMMA(64, dt), "K", swizzle, k_bytes)


_MN_SWEEP = [(sw, kb) for sw in (128, 64, 32) for kb in (128, 256)]


@requires_sm90
@pytest.mark.parametrize("swizzle, k_bytes", _MN_SWEEP, ids=[f"sw{s}-k{k}B" for s, k in _MN_SWEEP])
def test_mn_major_b_across_atoms(swizzle, k_bytes):
    """MN-major B (bf16 n128 = 2-8 MN atoms) with K = 64 and 128 rows at every swizzle."""
    _run_tile(W.WGMMA(128, "bfloat16"), "MN", swizzle, k_bytes)


@requires_sm90
@pytest.mark.parametrize(
    "op, b_major, swizzle, k_bytes",
    [
        (W.WGMMA(128, "bfloat16"), "K", 128, 128),
        (W.WGMMA(256, "bfloat16"), "K", 128, 128),
        (W.WGMMA(128, "float16"), "K", 128, 128),
        (W.WGMMA(128, "float8_e4m3fn"), "K", 128, 256),
        (W.WGMMA(128, "bfloat16", a_in_registers=True), "K", 128, 256),
        (W.WGMMA(64, "float16", a_in_registers=True), "K", 64, 128),
        (W.WGMMA(64, "bfloat16", a_in_registers=True), "MN", 64, 256),
    ],
    ids=["ss-bf16-n128", "ss-bf16-n256", "ss-f16-n128", "ss-e4m3-n128-k256B", "rs-bf16-n128-k256B",
         "rs-f16-n64-sw64", "rs-bf16-n64-mnB-sw64-k256B"],
)  # fmt: skip
def test_tile_numerics(op, b_major, swizzle, k_bytes):
    _run_tile(op, b_major, swizzle, k_bytes)


def _sass_chains(cubin: bytes):
    with tempfile.NamedTemporaryFile(suffix=".cubin") as f:
        f.write(cubin)
        f.flush()
        sass = subprocess.run(
            ["cuobjdump", "-sass", f.name], capture_output=True, text=True, check=True
        ).stdout
    chains = []
    for line in sass.splitlines():
        if "WARPGROUP.ARRIVE" in line:
            chains.append([])
        elif re.search(r"\b[HQIB]GMMA\.", line):
            if not chains:
                chains.append([])
            chains[-1].append("gsb0" in line)
    return chains


@pytest.mark.skipif(not env.has_cuda(), reason="needs NVRTC")
@pytest.mark.skipif(shutil.which("cuobjdump") is None, reason="needs cuobjdump")
@pytest.mark.parametrize("op", [W.WGMMA(128, "bfloat16"), W.WGMMA(128, "bfloat16", a_in_registers=True)],
                         ids=["ss", "rs"])  # fmt: skip
def test_chain_is_one_pipelined_group(op):
    """A 4-step chain is ONE warpgroup arrive with gsb0 only on its last GMMA, and
    compiles with no ptxas performance remark (perf_remarks="error"). Compile-only
    for sm_90a, so it runs on any machine with NVRTC and cuobjdump."""
    from tvm_tirx.backend.cuda.compiler import compile_source

    previous = os.environ.get("TVM_TIRX_COMPILE_FORCE_FALLBACK")
    os.environ["TVM_TIRX_COMPILE_FORCE_FALLBACK"] = "1"
    try:
        src = _compile(_tile_kernel(op)).mod.imports[0].inspect_source()
    finally:
        if previous is None:
            del os.environ["TVM_TIRX_COMPILE_FORCE_FALLBACK"]
        else:
            os.environ["TVM_TIRX_COMPILE_FORCE_FALLBACK"] = previous
    result = compile_source(src, {"cuda": {"arch": "sm_90a", "perf_remarks": "error"}})
    assert result.remarks == ()
    assert _sass_chains(result.binary) == [[False, False, False, True]]


# ---------------------------------------------------------------------------
# Named barriers: two warpgroups ping-pong
# ---------------------------------------------------------------------------


@requires_sm90
def test_named_barrier_ping_pong():
    """Warpgroups 0 and 1 take strict turns through two named barriers. Thread 0 of
    the active warpgroup appends a shared counter to the log, so the log must be
    0, 1, 2, ... with even entries from warpgroup 0 and odd from warpgroup 1."""
    import torch

    rounds = 8

    # fmt: off
    @T.function
    def main(log: T.Tensor((2 * rounds,), "int32"), who: T.Tensor((2 * rounds,), "int32")):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=256))
        tx = T.cuda.thread_idx("x")
        wg = tx // 128
        bars = NamedBarriers()
        wg0_done = bars.alloc(256)
        wg1_done = bars.alloc(256)
        counter = T.shared_scalar("int32")
        if tx == 0:
            counter = 0
        T.cuda.cta_sync()
        if wg == 0:
            for i in T.serial(rounds):
                if i > 0:
                    wg1_done.sync()
                if tx == 0:
                    log[2 * i] = counter
                    who[2 * i] = 0
                    counter = counter + 1
                wg0_done.arrive()
            wg1_done.sync()
        else:
            for i in T.serial(rounds):
                wg0_done.sync()
                if tx == 128:
                    log[2 * i + 1] = counter
                    who[2 * i + 1] = 1
                    counter = counter + 1
                wg1_done.arrive()
    # fmt: on

    log = torch.full((2 * rounds,), -1, dtype=torch.int32, device="cuda")
    who = torch.full((2 * rounds,), -1, dtype=torch.int32, device="cuda")
    _compile(main)(log, who)
    torch.cuda.synchronize()
    assert log.tolist() == list(range(2 * rounds))
    assert who.tolist() == [0, 1] * rounds


# ---------------------------------------------------------------------------
# One-warpgroup attention forward: S = Q K^T (ss), softmax on the accumulator
# fragment, O = P V (rs, P from registers, V MN-major)
# ---------------------------------------------------------------------------


@requires_sm90
def test_fa_forward_microkernel():
    """hd128: QK^T runs over 256 B of K (2 swizzle atoms), and V is MN-major with 2
    MN atoms, so both atom paths of SmemTile are exercised."""
    import torch

    M, KV, HD = 64, 64, 128
    E = 64  # bf16 elements per 128-byte swizzle row
    scale = 1.0 / np.sqrt(HD)
    qk = W.WGMMA(KV, "bfloat16")  # S[64, KV] = Q[64, HD] @ K[KV, HD]^T
    pv = W.WGMMA(HD, "bfloat16", a_in_registers=True)  # O[64, HD] = P[64, KV] @ V[KV, HD]
    log2e = 1.4426950408889634
    tma = "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
    qk_steps = HD * 2 // qk.k_bytes  # 8 k steps over 256 B: 2 swizzle atoms
    pv_steps = KV // pv.k

    # fmt: off
    @T.function
    def main(Q: T.Tensor((M, HD), "bfloat16", align=16), K: T.Tensor((KV, HD), "bfloat16", align=16),
             V: T.Tensor((KV, HD), "bfloat16", align=16), Out: T.Tensor((M, HD), "float32", align=16)):
        Q_map: T.let[T.handle("tensormap")] = T.stack_alloca("tensormap", 1)
        T.call_packed("runtime.cuTensorMapEncodeTiled", Q_map, "bfloat16", 2, Q.data, HD, M, HD * 2, E, M, 1, 1, 0, 3, 0, 0)
        K_map: T.let[T.handle("tensormap")] = T.stack_alloca("tensormap", 1)
        T.call_packed("runtime.cuTensorMapEncodeTiled", K_map, "bfloat16", 2, K.data, HD, KV, HD * 2, E, KV, 1, 1, 0, 3, 0, 0)
        V_map: T.let[T.handle("tensormap")] = T.stack_alloca("tensormap", 1)
        T.call_packed("runtime.cuTensorMapEncodeTiled", V_map, "bfloat16", 2, V.data, HD, KV, HD * 2, E, KV, 1, 1, 0, 3, 0, 0)
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=128), backend_config=STRICT)
        tx = T.cuda.thread_idx("x")
        Q_s = T.alloc_tensor(ty_args=[T.Tensor((M * HD,), "bfloat16", scope="shared", align=1024)])
        K_s = T.alloc_tensor(ty_args=[T.Tensor((KV * HD,), "bfloat16", scope="shared", align=1024)])
        V_s = T.alloc_tensor(ty_args=[T.Tensor((KV * HD,), "bfloat16", scope="shared", align=1024)])
        bar = T.shared_scalar("uint64")
        s_acc = T.alloc_tensor(ty_args=[T.Tensor((qk.num_acc,), "float32", scope="local")])
        o_acc = T.alloc_tensor(ty_args=[T.Tensor((pv.num_acc,), "float32", scope="local")])
        p_half = T.alloc_tensor(ty_args=[T.Tensor((qk.num_acc,), "bfloat16", scope="local")])
        row_max = T.alloc_tensor(ty_args=[T.Tensor((2,), "float32", scope="local")])
        row_sum = T.alloc_tensor(ty_args=[T.Tensor((2,), "float32", scope="local")])
        if tx == 0:
            T.ptx.mbarrier.init.shared.b64(T.address_of(bar), T.uint32(1))
        T.ptx.fence.proxy.async_.shared__cta()
        T.cuda.cta_sync()
        if tx == 0:
            # Q, K: one box per K atom (K-major). V: one box per HD column atom (MN-major).
            for j in T.unroll(HD // E):
                T.ptx[tma](Q_s.ptr_to([j * M * E]), T.address_of(Q_map), j * E, 0, T.address_of(bar))
                T.ptx[tma](K_s.ptr_to([j * KV * E]), T.address_of(K_map), j * E, 0, T.address_of(bar))
                T.ptx[tma](V_s.ptr_to([j * KV * E]), T.address_of(V_map), j * E, 0, T.address_of(bar))
            T.ptx.mbarrier.arrive.expect_tx.shared.b64(T.address_of(bar), T.uint32((M + 2 * KV) * HD * 2))
        T.cuda.mbarrier_wait(T.address_of(bar), 0)

        # S = Q K^T over HD = 256 bytes: 8 k steps across 2 swizzle atoms.
        q_tile = W.SmemTile(Q_s.data, rows=M)
        k_tile = W.SmemTile(K_s.data, rows=KV)
        W.chain(qk, s_acc, q_tile.k_steps(qk, qk_steps), k_tile.k_steps(qk, qk_steps), accumulate=False)
        W.wait(0, (s_acc, qk.num_acc))

        # softmax over each row: a thread holds rows r0 (regs with reg % 4 < 2) and r0 + 8;
        # the 4 lanes of a quad hold the rest of each row.
        for h in T.unroll(2):
            row_max[h] = T.float32(-3.0e38)
            row_sum[h] = T.float32(0.0)
        for r in T.unroll(qk.num_acc):
            h = (r % 4) // 2
            row_max[h] = T.max(row_max[h], s_acc[r])
        for h in T.unroll(2):
            for d in T.unroll(1, 3):
                row_max[h] = T.max(row_max[h], T.cuda.__shfl_xor_sync(T.uint32(0xFFFFFFFF), row_max[h], d, 32))
        for r in T.unroll(qk.num_acc):
            h = (r % 4) // 2
            p = T.exp2((s_acc[r] - row_max[h]) * T.float32(scale * log2e))
            row_sum[h] = row_sum[h] + p
            p_half[r] = T.Cast("bfloat16", p)
        for h in T.unroll(2):
            for d in T.unroll(1, 3):
                row_sum[h] = row_sum[h] + T.cuda.__shfl_xor_sync(T.uint32(0xFFFFFFFF), row_sum[h], d, 32)

        # O = P V: the 16-column slices of P, packed in pairs, are the rs A fragments.
        # V is MN-major: KV rows of K, HD = 2 MN atoms of 128 bytes.
        p_regs = p_half.view("uint32")
        v_tile = W.SmemTile(V_s.data, rows=KV, major="MN")
        W.chain(pv, o_acc, [_Slice(p_regs, s * 4) for s in range(pv_steps)],
                v_tile.k_steps(pv, pv_steps), accumulate=False, transpose_b=True)
        W.wait(0, (o_acc, pv.num_acc))
        for r in T.unroll(pv.num_acc):
            row, col = W.acc_coord(tx, r)
            Out[row, col] = o_acc[r] / row_sum[(r % 4) // 2]
    # fmt: on

    torch.manual_seed(0)
    q = torch.randn(M, HD, device="cuda").to(torch.bfloat16)
    k = torch.randn(KV, HD, device="cuda").to(torch.bfloat16)
    v = torch.randn(KV, HD, device="cuda").to(torch.bfloat16)
    o = torch.zeros(M, HD, device="cuda")
    _compile(main)(q, k, v, o)
    torch.cuda.synchronize()
    ref = torch.softmax(q.float() @ k.float().t() * scale, dim=-1) @ v.float()
    torch.testing.assert_close(o, ref, rtol=2e-2, atol=2e-2)


@requires_sm90
def test_producer_and_two_ping_pong_consumers():
    """The FA3 / warp-specialized GEMM shape: 384 threads with setmaxnreg.

    Warpgroup 0 produces into a 2-stage shared ring guarded by full/empty mbarriers.
    Warpgroups 1 and 2 consume alternate items and take strict turns through two
    256-thread named barriers from the kernel's NamedBarriers. The barrier ids are
    allocated after reserving the ids a warpgroup_sync(wg + 1) user would take.
    Every item must be consumed intact and in order.
    """
    import torch

    items, stages = 12, 2

    # fmt: off
    @T.function
    def main(sums: T.Tensor((items,), "float32"), order: T.Tensor((items,), "int32")):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=384),
                       kernel_attrs=T.cuda.KernelAttributes(min_blocks_per_sm=1))
        tx = T.cuda.thread_idx("x")
        wg = tx // 128
        lane = tx % 128
        bars = NamedBarriers(reserved=(1, 2, 3))  # as if warpgroup_sync(wg + 1) were also used
        turn0 = bars.alloc(256)  # consumer 0 finished its item
        turn1 = bars.alloc(256)  # consumer 1 finished its item
        buf = T.alloc_tensor(ty_args=[T.Tensor((stages, 128), "float32", scope="shared")])
        full = T.alloc_tensor(ty_args=[T.Tensor((stages,), "uint64", scope="shared", align=8)])
        empty = T.alloc_tensor(ty_args=[T.Tensor((stages,), "uint64", scope="shared", align=8)])
        seq = T.shared_scalar("int32")
        if tx == 0:
            seq = 0
            for s in T.unroll(stages):
                T.ptx.mbarrier.init.shared.b64(T.address_of(full[s]), T.uint32(128))
                T.ptx.mbarrier.init.shared.b64(T.address_of(empty[s]), T.uint32(128))
        T.cuda.cta_sync()
        if wg == 0:
            T.ptx["setmaxnreg.dec.sync.aligned.u32"](40)
            for i in T.serial(items):
                T.cuda.mbarrier_wait(T.address_of(empty[i % stages]), ((i // stages) & 1) ^ 1)
                buf[i % stages, lane] = T.Cast("float32", i * 1000 + lane)
                T.ptx.mbarrier.arrive.shared.b64(T.address_of(full[i % stages]), T.uint32(1))
        else:
            T.ptx["setmaxnreg.inc.sync.aligned.u32"](232)
            c = wg - 1
            for j in T.serial(items // 2):
                i = 2 * j + c
                if c == 0:
                    if j > 0:
                        turn1.sync()  # consumer 1 finished item i - 1
                else:
                    turn0.sync()  # consumer 0 finished item i - 1
                T.cuda.mbarrier_wait(T.address_of(full[i % stages]), (i // stages) & 1)
                v = buf[i % stages, lane]
                total = T.cuda.warp_sum(v)
                if lane % 32 == 0:
                    T.cuda.atomic_add(T.address_of(sums[i]), total)
                if lane == 0:
                    order[seq] = i
                    seq = seq + 1
                T.ptx.mbarrier.arrive.shared.b64(T.address_of(empty[i % stages]), T.uint32(1))
                if c == 0:
                    turn0.arrive()
                else:
                    if j < items // 2 - 1:
                        turn1.arrive()
    # fmt: on

    sums = torch.zeros(items, device="cuda")
    order = torch.full((items,), -1, dtype=torch.int32, device="cuda")
    _compile(main)(sums, order)
    torch.cuda.synchronize()
    expected = [float(sum(i * 1000 + t for t in range(128))) for i in range(items)]
    assert order.tolist() == list(range(items))
    assert sums.tolist() == expected
