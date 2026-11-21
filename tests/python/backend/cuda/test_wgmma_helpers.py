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


# ---------------------------------------------------------------------------
# GEMM tiles through the helpers: C[64, N] = A[64, K] @ B^T, one 4-step chain
# ---------------------------------------------------------------------------

_TORCH = {"bfloat16": "bfloat16", "float16": "float16", "float8_e4m3fn": "float8_e4m3fn"}


def _tile_kernel(op: W.WGMMA, b_major="K"):
    """One warpgroup; A [64, KB] and B (K-major [N, KB] or MN-major [KB, N]) arrive
    by 128B-swizzled TMA; one chain of 4 k-steps covers the 128-byte K block."""
    dt = op.ab_dtype
    eb = 32 // op.k
    KB = 128 // eb  # one 128-byte swizzle row of K
    N = op.n
    steps = KB // op.k
    assert steps == 4
    n_atoms = N * eb // 128  # MN-major: 128-byte column blocks of N
    b_shape = (N, KB) if b_major == "K" else (KB, N)
    rs = op.a_in_registers
    trans_b = b_major == "MN"

    # fmt: off
    @T.function
    def main(A: T.Tensor((64, KB), dt, align=16), B: T.Tensor(b_shape, dt, align=16),
             C: T.Tensor((64, N), "float32", align=16)):
        A_map: T.let[T.handle("tensormap")] = T.stack_alloca("tensormap", 1)
        T.call_packed("runtime.cuTensorMapEncodeTiled", A_map, dt, 2, A.data,
                      KB, 64, KB * eb, KB, 64, 1, 1, 0, 3, 0, 0)
        B_map: T.let[T.handle("tensormap")] = T.stack_alloca("tensormap", 1)
        if T.constexpr(b_major == "K"):
            T.call_packed("runtime.cuTensorMapEncodeTiled", B_map, dt, 2, B.data,
                          KB, N, KB * eb, KB, N, 1, 1, 0, 3, 0, 0)
        else:
            T.call_packed("runtime.cuTensorMapEncodeTiled", B_map, dt, 2, B.data,
                          N, KB, N * eb, 128 // eb, KB, 1, 1, 0, 3, 0, 0)
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=128), backend_config=STRICT)
        tx = T.cuda.thread_idx("x")
        A_s = T.alloc_tensor(ty_args=[T.Tensor((64, KB), dt, scope="shared", align=1024)])
        B_s = T.alloc_tensor(ty_args=[T.Tensor((N * KB,), dt, scope="shared", align=1024)])
        bar = T.shared_scalar("uint64")
        acc = T.alloc_tensor(ty_args=[T.Tensor((op.num_acc,), "float32", scope="local")])
        if tx == 0:
            T.ptx.mbarrier.init.shared.b64(T.address_of(bar), T.uint32(1))
        T.ptx.fence.proxy.async_.shared__cta()
        T.cuda.cta_sync()
        if tx == 0:
            T.ptx["cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"](
                A_s.data, T.address_of(A_map), 0, 0, T.address_of(bar))
            if T.constexpr(b_major == "K"):
                T.ptx["cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"](
                    B_s.data, T.address_of(B_map), 0, 0, T.address_of(bar))
            else:
                for j in T.unroll(n_atoms):
                    T.ptx["cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"](
                        B_s.ptr_to([j * KB * (128 // eb)]), T.address_of(B_map), j * (128 // eb), 0,
                        T.address_of(bar))
            T.ptx.mbarrier.arrive.expect_tx.shared.b64(T.address_of(bar), T.uint32((64 + N) * KB * eb))
        T.cuda.mbarrier_wait(T.address_of(bar), 0)

        desc_a = W.smem_descriptor(A_s.data)
        if T.constexpr(b_major == "K"):
            desc_b = W.smem_descriptor(B_s.data)
            b_steps = [W.advance_descriptor(desc_b, s * op.k_bytes) for s in range(steps)]
        else:
            desc_b = W.smem_descriptor(B_s.data, major="MN", mn_atom_stride_bytes=KB * 128)
            b_steps = [W.advance_descriptor(desc_b, s * op.k * 128) for s in range(steps)]
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
            a_steps = [W.advance_descriptor(desc_a, s * op.k_bytes) for s in range(steps)]
            W.chain(op, acc, a_steps, b_steps, accumulate=False, transpose_b=trans_b)
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


def _run_tile(op, b_major="K"):
    import torch

    eb = 32 // op.k
    KB = 128 // eb
    torch.manual_seed(0)
    dt = getattr(torch, _TORCH[op.ab_dtype])
    a = (torch.randn(64, KB, device="cuda") / 2).to(dt)
    b_nk = (torch.randn(op.n, KB, device="cuda") / 2).to(dt)
    b = b_nk if b_major == "K" else b_nk.t().contiguous()
    c = torch.zeros(64, op.n, device="cuda")
    mod = _compile(_tile_kernel(op, b_major))
    mod(a, b, c)
    torch.cuda.synchronize()
    ref = a.float() @ b_nk.float().t()
    torch.testing.assert_close(c, ref, rtol=1e-3, atol=1e-2)
    return mod


@requires_sm90
@pytest.mark.parametrize(
    "op, b_major",
    [
        (W.WGMMA(64, "bfloat16"), "K"),
        (W.WGMMA(128, "bfloat16"), "K"),
        (W.WGMMA(256, "bfloat16"), "K"),
        (W.WGMMA(128, "float16"), "K"),
        (W.WGMMA(128, "float8_e4m3fn"), "K"),
        (W.WGMMA(128, "bfloat16", a_in_registers=True), "K"),
        (W.WGMMA(64, "float16", a_in_registers=True), "K"),
        (W.WGMMA(128, "bfloat16"), "MN"),
        (W.WGMMA(64, "bfloat16", a_in_registers=True), "MN"),
    ],
    ids=["ss-bf16-n64", "ss-bf16-n128", "ss-bf16-n256", "ss-f16-n128", "ss-e4m3-n128",
         "rs-bf16-n128", "rs-f16-n64", "ss-bf16-n128-mnB", "rs-bf16-n64-mnB"],
)  # fmt: skip
def test_tile_numerics(op, b_major):
    _run_tile(op, b_major)


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
    import torch

    M = KV = HD = 64  # one 64-query tile, one 64-key block, head dim 64 (= one 128B row)
    scale = 1.0 / np.sqrt(HD)
    qk = W.WGMMA(KV, "bfloat16")  # S[64, KV] = Q[64, HD] @ K[KV, HD]^T
    pv = W.WGMMA(HD, "bfloat16", a_in_registers=True)  # O[64, HD] = P[64, KV] @ V[KV, HD]
    steps = 4
    log2e = 1.4426950408889634

    # fmt: off
    @T.function
    def main(Q: T.Tensor((M, HD), "bfloat16", align=16), K: T.Tensor((KV, HD), "bfloat16", align=16),
             V: T.Tensor((KV, HD), "bfloat16", align=16), Out: T.Tensor((M, HD), "float32", align=16)):
        Q_map: T.let[T.handle("tensormap")] = T.stack_alloca("tensormap", 1)
        T.call_packed("runtime.cuTensorMapEncodeTiled", Q_map, "bfloat16", 2, Q.data, HD, M, HD * 2, HD, M, 1, 1, 0, 3, 0, 0)
        K_map: T.let[T.handle("tensormap")] = T.stack_alloca("tensormap", 1)
        T.call_packed("runtime.cuTensorMapEncodeTiled", K_map, "bfloat16", 2, K.data, HD, KV, HD * 2, HD, KV, 1, 1, 0, 3, 0, 0)
        V_map: T.let[T.handle("tensormap")] = T.stack_alloca("tensormap", 1)
        T.call_packed("runtime.cuTensorMapEncodeTiled", V_map, "bfloat16", 2, V.data, HD, KV, HD * 2, HD, KV, 1, 1, 0, 3, 0, 0)
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=128), backend_config=STRICT)
        tx = T.cuda.thread_idx("x")
        Q_s = T.alloc_tensor(ty_args=[T.Tensor((M, HD), "bfloat16", scope="shared", align=1024)])
        K_s = T.alloc_tensor(ty_args=[T.Tensor((KV, HD), "bfloat16", scope="shared", align=1024)])
        V_s = T.alloc_tensor(ty_args=[T.Tensor((KV, HD), "bfloat16", scope="shared", align=1024)])
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
            T.ptx["cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"](
                Q_s.data, T.address_of(Q_map), 0, 0, T.address_of(bar))
            T.ptx["cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"](
                K_s.data, T.address_of(K_map), 0, 0, T.address_of(bar))
            T.ptx["cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"](
                V_s.data, T.address_of(V_map), 0, 0, T.address_of(bar))
            T.ptx.mbarrier.arrive.expect_tx.shared.b64(T.address_of(bar), T.uint32((M + 2 * KV) * HD * 2))
        T.cuda.mbarrier_wait(T.address_of(bar), 0)

        # S = Q K^T
        dq = W.smem_descriptor(Q_s.data)
        dk = W.smem_descriptor(K_s.data)
        W.chain(qk, s_acc, [W.advance_descriptor(dq, s * 32) for s in range(steps)],
                [W.advance_descriptor(dk, s * 32) for s in range(steps)], accumulate=False)
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
        p_regs = p_half.view("uint32")
        dv = W.smem_descriptor(V_s.data, major="MN", mn_atom_stride_bytes=KV * 128)
        W.chain(pv, o_acc, [_Slice(p_regs, s * 4) for s in range(steps)],
                [W.advance_descriptor(dv, s * 16 * 128) for s in range(steps)],
                accumulate=False, transpose_b=True)
        W.wait(0, (o_acc, pv.num_acc))
        for r in T.unroll(pv.num_acc):
            row, col = W.acc_coord(tx, r)
            Out[row, col] = o_acc[r] / row_sum[(r % 4) // 2]
    # fmt: on

    torch.manual_seed(0)
    q, k, v = (torch.randn(64, 64, device="cuda").to(torch.bfloat16) for _ in range(3))
    o = torch.zeros(64, 64, device="cuda")
    _compile(main)(q, k, v, o)
    torch.cuda.synchronize()
    ref = torch.softmax(q.float() @ k.float().t() * scale, dim=-1) @ v.float()
    torch.testing.assert_close(o, ref, rtol=2e-2, atol=2e-2)
