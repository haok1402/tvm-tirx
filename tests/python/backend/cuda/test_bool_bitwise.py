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
"""And / Or of operands that are safe to evaluate unconditionally print as `&` / `|`.

C `&&` / `||` hand NVRTC a branch to lower. For a per-register mask such as
``(a || b) ? -inf : x`` it can keep one divergent branch region per statement instead of a
predicated select (sm90-probe ISSUE-017). The LLVM backend already evaluates both operands.
Operands that could fault or be undefined when the guard excludes them keep the short-circuit
form: global and shared loads, division or shifts by a variable, impure calls.
"""

import re

import numpy as np
import pytest

import tvm_tirx
import tvm_tirx.testing
from tvm_tirx.script import tirx as T
from tvm_tirx.testing import env


def _compile(func, arch="sm_90a"):
    target = tvm_tirx.target.Target({"kind": "cuda", "arch": arch})
    mod = tvm_tirx.compile(tvm_tirx.IRModule({"main": func}), target=target, pipeline="tirx")
    src = mod.mod.imports[0].inspect_source()
    return mod, src[src.find("main_kernel(") :]


def _mask_kernel(op):
    # B3's mask shape: per element, two compares of register values, joined by Or / And.
    @T.function
    def main(out: T.Tensor((128,), "float32"), x: T.Tensor((128,), "float32"), n: T.int32):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=128))
        tid = T.cuda.thread_idx("x")
        row = T.alloc_local((1,), "int32")
        lim = T.alloc_local((1,), "int32")
        row[0] = tid // 4
        lim[0] = n - 1
        col = tid % 32
        if T.constexpr(op == "or"):
            masked = T.Or(col >= lim[0], col > row[0])
        else:
            masked = T.And(col >= lim[0], col > row[0])
        out[tid] = T.Select(masked, T.float32(-1e30), x[tid])

    return main


@pytest.mark.parametrize("op, sym", [("or", "|"), ("and", "&")])
def test_register_compares_use_bitwise(op, sym):
    _, src = _compile(_mask_kernel(op))
    assert f") {sym} (" in src, src
    assert "||" not in src and "&&" not in src, src


def _guard_kernel(kind):
    @T.function
    def main(out: T.Tensor((64,), "int32"), a: T.Tensor((32,), "int32"), n: T.int32, d: T.int32):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=64))
        tid = T.cuda.thread_idx("x")
        smem = T.alloc_shared((32,), "int32")
        regs = T.alloc_local((4,), "int32")
        if tid < 32:
            smem[tid] = a[tid]
        regs[0] = tid
        regs[1] = tid
        regs[2] = tid
        regs[3] = tid
        T.cuda.cta_sync()
        if T.constexpr(kind == "global"):
            ok = T.And(tid < n, a[T.min(tid, 63)] > 0)
        elif T.constexpr(kind == "shared"):
            ok = T.And(tid < n, smem[T.min(tid, 63)] > 0)
        elif T.constexpr(kind == "divide"):
            ok = T.And(d != 0, T.truncdiv(tid, d) > 1)
        elif T.constexpr(kind == "shift"):
            ok = T.And(d < 32, T.shift_left(tid, d) > 1)
        else:  # a register array at a variable index
            ok = T.And(tid < 4, regs[tid] >= 0)
        out[tid] = T.Select(ok, T.int32(1), T.int32(0))

    return main


GUARDS = ["global", "shared", "divide", "shift", "local_variable_index"]


@pytest.mark.parametrize("kind", GUARDS)
def test_guards_keep_short_circuit(kind):
    _, src = _compile(_guard_kernel(kind))
    assert "&&" in src, src


@pytest.mark.parametrize("kind", ["shift_const", "divide_const"])
def test_constant_divisor_and_shift_use_bitwise(kind):
    @T.function
    def main(out: T.Tensor((64,), "int32"), n: T.int32):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=64))
        tid = T.cuda.thread_idx("x")
        if T.constexpr(kind == "shift_const"):
            ok = T.And(tid < n, T.shift_left(tid, 3) > 8)
        else:
            ok = T.And(tid < n, T.truncdiv(tid, 3) > 2)
        out[tid] = T.Select(ok, T.int32(1), T.int32(0))

    _, src = _compile(main)
    assert re.search(r"\) & \(", src), src
    assert "&&" not in src, src


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda_compute(9), reason="needs an sm_90 or newer GPU")
@pytest.mark.parametrize("op", ["or", "and"])
def test_mask_runs(op):
    mod, _ = _compile(_mask_kernel(op), env.cuda_arch())
    dev = tvm_tirx.cuda(0)
    x = np.arange(128, dtype="float32")
    n = 20
    out = tvm_tirx.runtime.tensor(np.zeros(128, "float32"), device=dev)
    mod(out, tvm_tirx.runtime.tensor(x, device=dev), n)
    tid = np.arange(128)
    col, row = tid % 32, tid // 4
    if op == "or":
        masked = (col >= n - 1) | (col > row)
    else:
        masked = (col >= n - 1) & (col > row)
    np.testing.assert_array_equal(out.numpy(), np.where(masked, np.float32(-1e30), x))


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda_compute(9), reason="needs an sm_90 or newer GPU")
@pytest.mark.parametrize("kind", GUARDS)
def test_guards_run(kind):
    mod, _ = _compile(_guard_kernel(kind), env.cuda_arch())
    dev = tvm_tirx.cuda(0)
    a = np.arange(1, 33, dtype="int32")
    out = tvm_tirx.runtime.tensor(np.zeros(64, "int32"), device=dev)
    mod(out, tvm_tirx.runtime.tensor(a, device=dev), 32, 0)
    tid = np.arange(64)
    expected = {
        "global": tid < 32,
        "shared": tid < 32,
        "divide": np.zeros(64, bool),
        "shift": (tid << 0) > 1,
        "local_variable_index": tid < 4,
    }[kind]
    np.testing.assert_array_equal(out.numpy(), expected.astype("int32"))


if __name__ == "__main__":
    tvm_tirx.testing.main()
