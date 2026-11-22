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
"""StmtSimplify keeps a value the program bound once (sm90-probe ISSUE-023).

The analyzer substitutes pure bindings into their uses so the rewrite rules see the structure;
StmtSimplify then folds verbatim copies of an in-scope binding's value back into its variable.
Only non-trivial bindings used at least twice take part.
"""

import numpy as np
import pytest

import tvm_tirx
import tvm_tirx.testing
from tvm_tirx.script import tirx as T
from tvm_tirx.testing import env


def _body(func):
    mod = tvm_tirx.tirx.transform.StmtSimplify()(tvm_tirx.IRModule({"main": func}))
    return str(mod["main"].body)


def _cuda(func, arch="sm_90a"):
    target = tvm_tirx.target.Target({"kind": "cuda", "arch": arch})
    mod = tvm_tirx.compile(tvm_tirx.IRModule({"main": func}), target=target, pipeline="tirx")
    src = mod.mod.imports[0].inspect_source()
    return mod, src[src.rfind("__global__") :]


@T.function
def _e8m0_let(w_in: T.Tensor((1,), "int64"), out: T.Tensor((8,), "uint16")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=1))
    w = w_in[0]
    for i in T.unroll(8):
        x: T.let = T.Cast("uint16", T.shift_right(w, T.int64(8) * i))
        e: T.let = x & T.uint16(0xFF)
        out[i] = T.Select(
            e == T.uint16(255),
            T.uint16(0x7FFF),
            T.Select(e == T.uint16(0), T.uint16(0x40), (x << T.uint16(7)) & T.uint16(0x7F80)),
        )


@T.function
def _e8m0_python(w_in: T.Tensor((1,), "int64"), out: T.Tensor((8,), "uint16")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=1))
    w = w_in[0]
    for i in T.unroll(8):
        x = T.Cast("uint16", T.shift_right(w, T.int64(8) * i))
        e = x & T.uint16(0xFF)
        out[i] = T.Select(
            e == T.uint16(255),
            T.uint16(0x7FFF),
            T.Select(e == T.uint16(0), T.uint16(0x40), (x << T.uint16(7)) & T.uint16(0x7F80)),
        )


@pytest.mark.parametrize("func", [_e8m0_let, _e8m0_python], ids=["let", "python"])
def test_bound_values_are_printed_once(func):
    _, src = _cuda(func)
    # One shift per byte: in the binding of x, not again at every use.
    assert src.count(">> (int64_t)8)") == 1, src
    assert "(e_1 == (ushort)255)" in src, src
    assert "(x_1 << (ushort)7)" in src, src


def test_single_use_binding_is_unchanged():
    @T.function
    def main(a: T.Tensor((1,), "int32"), out: T.Tensor((1,), "int32")):
        v = a[0]
        x: T.let = v * 3 + 1
        out[0] = x

    assert "out[0] = v * 3 + 1" in _body(main)


def test_simplification_through_the_binding_is_kept():
    @T.function
    def main(a: T.Tensor((1,), "int32"), out: T.Tensor((2,), "int32")):
        n = a[0]
        x: T.let = n * 4
        out[0] = x // 4
        out[1] = x + 1

    body = _body(main)
    assert "out[0] = n" in body, body  # simplified through the substituted value
    assert "out[1] = x + 1" in body, body  # and the other use folded back


@T.function
def _branch_scope(a: T.Tensor((2,), "int32"), out: T.Tensor((3,), "int32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=1))
    p = a[0]
    q = a[1]
    if p > 0:
        y: T.let = p * q + 7
        out[0] = y
        out[1] = y * 2
    out[2] = p * q + 7  # y is out of scope here


def test_binding_is_not_folded_out_of_its_scope():
    body = _body(_branch_scope)
    tail = body[body.rindex("out[2]") :]
    assert "y" not in tail.split("\n")[0], body


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda_compute(9), reason="needs an sm_90 or newer GPU")
def test_values_on_gpu():
    dev = tvm_tirx.cuda(0)
    w = np.array([0x00FF0001_7F80FF00], "int64")
    raw = w.view("uint64")[0]
    ref = []
    for i in range(8):
        b = (int(raw) >> (8 * i)) & 0xFF
        ref.append(0x7FFF if b == 0xFF else 0x40 if b == 0 else (b << 7) & 0x7F80)
    for func in (_e8m0_let, _e8m0_python):
        mod, _ = _cuda(func, env.cuda_arch())
        out = tvm_tirx.runtime.tensor(np.zeros(8, "uint16"), device=dev)
        mod(tvm_tirx.runtime.tensor(w, device=dev), out)
        np.testing.assert_array_equal(out.numpy(), np.array(ref, "uint16"))
    mod, _ = _cuda(_branch_scope, env.cuda_arch())
    out = tvm_tirx.runtime.tensor(np.zeros(3, "int32"), device=dev)
    mod(tvm_tirx.runtime.tensor(np.array([3, 5], "int32"), device=dev), out)
    np.testing.assert_array_equal(out.numpy(), [22, 44, 22])


if __name__ == "__main__":
    tvm_tirx.testing.main()
