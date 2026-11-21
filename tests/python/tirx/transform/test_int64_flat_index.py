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
"""Flat tensor indices switch to int64 for spans of 2^30 elements or more.

int32 index arithmetic is kept within |v| < 2^30 (NarrowDataType), because rewrites may form
sums of two index terms. A tensor with a static span of more than 2^30 elements is therefore
indexed in int64. A symbolic shape with int32 dimensions keeps int32 indices: it asserts that
its span is at most 2^30, which ``tirx.enable_debug`` checks at the call.
"""

import re

import numpy as np
import pytest

import tvm_tirx
from tvm_tirx.script import tirx as T
from tvm_tirx.testing import env


def _fill(M, N, dtype="int8"):
    @T.function
    def main(A: T.Tensor((M, N), dtype)):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=M, block=256))
        bx = T.cuda.block_idx("x")
        tx = T.cuda.thread_idx("x")
        for j in T.serial(N // 256):
            A[bx, j * 256 + tx] = T.Cast(dtype, bx % 100 + j % 7)

    return main


def _symbolic_fill(dim_dtype):
    m = T.dynamic("m", "int32")
    n = T.dynamic("n", dim_dtype)

    @T.function
    def main(A: T.Tensor((m, n), "int8")):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=m, block=256))
        bx = T.cuda.block_idx("x")
        tx = T.cuda.thread_idx("x")
        for j in T.serial(n // 256):
            A[bx, j * 256 + tx] = T.Cast("int8", bx % 100)

    return main


def _compile(func, arch="sm_90a", config=None):
    target = tvm_tirx.target.Target({"kind": "cuda", "arch": arch})
    with tvm_tirx.transform.PassContext(config=config or {}):
        mod = tvm_tirx.compile(tvm_tirx.IRModule({"main": func}), target=target, pipeline="tirx")
    return mod, mod.mod.imports[0].inspect_source()


def _store_index(src):
    body = src[src.find("main_kernel(") :]
    return re.search(r"A_ptr\[(.*)\] =", body).group(1)


@pytest.mark.parametrize(
    "shape, wide",
    [
        ((4096, 4096), False),
        ((32768, 32768), False),  # 2^30 elements: every index is below 2^30
        ((32769, 32768), True),  # 2^30 + 2^15
        ((65537, 32768), True),  # 2^31 + 2^15
    ],
)
def test_static_span_selects_index_type(shape, wide):
    index = _store_index(_compile(_fill(*shape))[1])
    assert ("int64_t" in index) == wide, index


@pytest.mark.parametrize("dim_dtype", ["int32", "int64"])
def test_symbolic_shape_keeps_declared_index_type(dim_dtype):
    index = _store_index(_compile(_symbolic_fill(dim_dtype))[1])
    assert ("int64_t" in index) == (dim_dtype == "int64"), index


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="needs a CUDA GPU")
def test_store_beyond_int32_elements():
    """A (65537, 32768) int8 tensor (2^31 + 2^15 elements, 2 GiB) is written correctly."""
    M, N = 65537, 32768
    mod, _ = _compile(_fill(M, N), arch=env.cuda_arch())
    A = tvm_tirx.runtime.tensor(np.full((M, N), -1, dtype="int8"), device=tvm_tirx.cuda(0))
    mod(A)
    out = A.numpy()
    j = np.arange(N) // 256
    for bx in (0, 65535, 65536):
        np.testing.assert_array_equal(out[bx], (bx % 100 + j % 7).astype("int8"))


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="needs a CUDA GPU")
def test_debug_checks_symbolic_int32_span():
    mod, _ = _compile(
        _symbolic_fill("int32"), arch=env.cuda_arch(), config={"tirx.enable_debug": True}
    )
    small = tvm_tirx.runtime.tensor(np.zeros((4, 1024), dtype="int8"), device=tvm_tirx.cuda(0))
    mod(small)
    assert (small.numpy() == (np.arange(4) % 100)[:, None]).all()
    large = tvm_tirx.runtime.empty((32769, 32768), "int8", device=tvm_tirx.cuda(0))
    with pytest.raises(ValueError, match="spans more than 2\\^30 elements"):
        mod(large)


if __name__ == "__main__":
    tvm_tirx.testing.main()
