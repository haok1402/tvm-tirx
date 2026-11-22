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
"""A sign test on a uint32 reinterpreted as int32 survives codegen (sm90-probe ISSUE-011)."""

import re

import numpy as np
import pytest

import tvm_tirx
from tvm_tirx.script import tirx as T
from tvm_tirx.testing import env


def _kernel():
    # DeepGEMM's grouped layout: an `int` entry read as raw bits, reinterpreted, tested for the
    # padding marker -1 (is_computation_valid) and clamped before it indexes memory
    # (get_global_idx's max(0, ...)).
    @T.function
    def main(A: T.Tensor((2,), "uint32"), C: T.Tensor((8,), "int32"), B: T.Tensor((2, 2), "int32")):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=2))
        tx = T.cuda.thread_idx("x")
        if T.Cast("int32", A[tx]) >= 0:
            B[tx, 0] = 1
        else:
            B[tx, 0] = 0
        B[tx, 1] = C[T.min(T.max(T.Cast("int32", A[tx]), 0), 7)]

    return main


def _compile(arch="sm_90a"):
    target = tvm_tirx.target.Target({"kind": "cuda", "arch": arch})
    mod = tvm_tirx.compile(tvm_tirx.IRModule({"main": _kernel()}), target=target, pipeline="tirx")
    src = mod.mod.imports[0].inspect_source()
    return mod, src[src.find("main_kernel(") :]


def test_sign_test_and_clamp_are_kept():
    _, body = _compile()
    # Before the fix, `int32(A[tx]) >= 0` folded to true (B[tx, 0] = 1 unconditionally, no
    # branch) and the clamp folded to C_ptr[(int)A_ptr[...]].
    assert re.search(r"0 <= \(\(int\)A_ptr|\(\(int\)A_ptr\[[^]]*\]\) >= 0", body), body
    assert re.search(r"max\(\(\(int\)A_ptr\[[^]]*\]\), 0\)", body), body


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="needs a CUDA GPU")
def test_sign_test_and_clamp_on_gpu():
    mod, _ = _compile(env.cuda_arch())
    dev = tvm_tirx.cuda(0)
    a = tvm_tirx.runtime.tensor(np.array([0xFFFFFFFF, 7], "uint32"), device=dev)
    c = tvm_tirx.runtime.tensor(np.arange(10, 18, dtype="int32"), device=dev)
    b = tvm_tirx.runtime.tensor(np.full((2, 2), 42, "int32"), device=dev)
    mod(a, c, b)
    # 0xFFFFFFFF is -1: skipped, and clamped to index 0.
    np.testing.assert_array_equal(b.numpy(), [[0, 10], [1, 17]])


if __name__ == "__main__":
    tvm_tirx.testing.main()
