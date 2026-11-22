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
"""NaN tests written as comparisons survive to the GPU (sm90-probe ISSUE-013)."""

import numpy as np
import pytest

import tvm_tirx
import tvm_tirx.testing
from tvm_tirx.script import tirx as T
from tvm_tirx.testing import env

N = 32


def _kernel(dtype, backend_config=None):
    @T.function
    def main(
        A: T.Tensor((N,), dtype),
        Y: T.Tensor((N,), dtype),
        B: T.Tensor((N,), "int32"),
        C: T.Tensor((N,), "int32"),
        D: T.Tensor((N,), "int32"),
        E: T.Tensor((N,), "int32"),
        F: T.Tensor((N,), "int32"),
    ):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=N), backend_config=backend_config)
        i = T.cuda.thread_idx("x")
        B[i] = T.Select(A[i] != A[i], 1, 0)
        C[i] = T.Select(A[i] == A[i], 1, 0)
        D[i] = T.Select(T.isnan(A[i]), 1, 0)
        E[i] = T.Select(T.Not(A[i] < Y[i]), 1, 0)
        # FlashMLA's row-sum guard: (rL == 0 || rL != rL) ? 1 : rL.
        F[i] = T.Select(T.Or(A[i] == T.Cast(dtype, 0), A[i] != A[i]), 1, 0)

    return main


def _compile(dtype, arch="sm_90a", backend_config=None):
    target = tvm_tirx.target.Target({"kind": "cuda", "arch": arch})
    mod = tvm_tirx.compile(
        tvm_tirx.IRModule({"main": _kernel(dtype, backend_config)}), target=target, pipeline="tirx"
    )
    src = mod.mod.imports[0].inspect_source()
    return mod, src[src.find("main_kernel(") :]


@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
def test_nan_tests_reach_cuda(dtype):
    _, body = _compile(dtype)
    assert "!=" in body, body
    assert body.count("==") >= 2, body
    assert "!(" in body, body


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="needs a CUDA GPU")
@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
@pytest.mark.parametrize("fast_math", [True, False], ids=["fast_math", "no_fast_math"])
def test_nan_tests_on_gpu(dtype, fast_math):
    torch = pytest.importorskip("torch")
    config = None if fast_math else {"cuda": {"nvrtc": [], "nvcc": []}}
    mod, _ = _compile(dtype, env.cuda_arch(), config)
    a = torch.zeros(N, dtype=getattr(torch, dtype), device="cuda")
    a[1] = float("nan")
    a[2] = 3.0
    y = torch.ones(N, dtype=getattr(torch, dtype), device="cuda")
    outs = [torch.full((N,), -1, dtype=torch.int32, device="cuda") for _ in range(5)]
    mod(a, y, *outs)
    b, c, d, e, f = (o.cpu().numpy() for o in outs)
    # Element 0 is 0.0, element 1 NaN, element 2 3.0 (> y = 1).
    np.testing.assert_array_equal(b[:3], [0, 1, 0])  # x != x
    np.testing.assert_array_equal(c[:3], [1, 0, 1])  # x == x
    np.testing.assert_array_equal(d[:3], [0, 1, 0])  # isnan
    np.testing.assert_array_equal(e[:3], [0, 1, 1])  # !(x < y)
    np.testing.assert_array_equal(f[:3], [1, 1, 0])  # x == 0 || x != x


if __name__ == "__main__":
    tvm_tirx.testing.main()
