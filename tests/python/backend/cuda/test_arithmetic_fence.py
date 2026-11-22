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
"""T.cuda.arithmetic_fence: pin the association of a sum (sm90-probe ISSUE-016).

nvcc/NVRTC reassociate integer sums freely. In an address such as
``row * stride + kv_offset + lane_term`` they may group the loop-invariant terms first, which puts
the per-thread term inside the warp-uniform part and moves it off the uniform datapath. No C
spelling of the sum prevents that. ``arithmetic_fence(x)`` returns ``x`` through an empty asm, which
emits no instruction but which the compiler cannot look through.
"""

import re

import numpy as np
import pytest

import tvm_tirx
import tvm_tirx.testing
from tvm_tirx.backend.config import parse_backend_config
from tvm_tirx.backend.cuda.compiler import compile_source
from tvm_tirx.script import tirx as T
from tvm_tirx.testing import env

TYPES = {
    "int32": ("int", "r"),
    "uint32": ("unsigned int", "r"),
    "int64": ("long long", "l"),
    "uint64": ("unsigned long long", "l"),
    "float32": ("float", "f"),
    "float64": ("double", "d"),
}


def _compile(func, arch="sm_90a"):
    target = tvm_tirx.target.Target({"kind": "cuda", "arch": arch})
    mod = tvm_tirx.compile(tvm_tirx.IRModule({"main": func}), target=target, pipeline="tirx")
    return mod, mod.mod.imports[0].inspect_source()


def _ptx(mod):
    imported = mod.mod.imports[0]
    config = parse_backend_config(imported.inspect_source("backend_config"))
    config["cuda"]["target_format"] = "ptx"
    return compile_source(imported.inspect_source(), config).binary.decode()


def _ptx_instructions(ptx):
    # Instruction lines of the kernel body, without the inline-asm markers and labels.
    body = ptx[ptx.find(".entry") :]
    return [
        line.strip()
        for line in body.splitlines()
        if re.match(r"\s+[a-z@]", line) and not line.strip().startswith("//")
    ]


def _cancel(dtype, fence):
    # (a + c) - c: nvcc/NVRTC (and TIRx's own simplifier) fold it to a unless the sum is fenced.
    @T.function
    def main(out: T.Tensor((1,), dtype), a_in: T.Tensor((1,), dtype), c_in: T.Tensor((1,), dtype)):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=1))
        a = a_in[0]
        c = c_in[0]
        if T.constexpr(fence):
            out[0] = T.cuda.arithmetic_fence(a + c) - c
        else:
            out[0] = (a + c) - c

    return main


@pytest.mark.parametrize("dtype", list(TYPES))
def test_helper_is_empty_asm(dtype):
    c_type, constraint = TYPES[dtype]
    _, src = _compile(_cancel(dtype, fence=True))
    helper = f"tvm_builtin_cuda_arithmetic_fence_{dtype}"
    assert f"__forceinline__ __device__ {c_type} {helper}({c_type} value) {{" in src, src
    assert f'asm("" : "+{constraint}"(value));' in src, src
    # Not volatile and no clobber: the compiler may still hoist and CSE it.
    assert "volatile" not in src[src.find(helper) : src.find("return value;", src.find(helper))]


@pytest.mark.parametrize("dtype", ["float16", "bfloat16", "int8", "uint16", "bool"])
def test_unsupported_dtype(dtype):
    @T.function
    def main(out: T.Tensor((1,), dtype), x: T.Tensor((1,), dtype)):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=1))
        out[0] = T.cuda.arithmetic_fence(x[0])

    with pytest.raises(Exception, match="arithmetic_fence supports"):
        _compile(main)


def test_result_dtype_follows_operand():
    for dtype in TYPES:
        x = tvm_tirx.tirx.Var("x", dtype)
        assert str(T.cuda.arithmetic_fence(x).ty) == dtype


@pytest.mark.parametrize("dtype", ["int32", "uint64"])
def test_blocks_reassociation(dtype):
    # Without the fence, (a + c) - c is folded; with it, the add and the sub both survive to PTX.
    plain = _ptx_instructions(_ptx(_compile(_cancel(dtype, fence=False))[0]))
    fenced = _ptx_instructions(_ptx(_compile(_cancel(dtype, fence=True))[0]))
    assert not any(i.startswith(("add.", "sub.")) for i in plain), plain
    assert any(i.startswith("add.") for i in fenced), fenced
    assert any(i.startswith("sub.") for i in fenced), fenced


def test_emits_no_instruction():
    # The fence itself adds nothing to the PTX: a fenced copy has the instructions of a plain copy.
    def copy(fence):
        @T.function
        def main(out: T.Tensor((1,), "uint64"), x: T.Tensor((1,), "uint64")):
            T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=1))
            if T.constexpr(fence):
                out[0] = T.cuda.arithmetic_fence(x[0])
            else:
                out[0] = x[0]

        return main

    plain = _ptx_instructions(_ptx(_compile(copy(False))[0]))
    fenced = _ptx_instructions(_ptx(_compile(copy(True))[0]))
    assert len(plain) == len(fenced), (plain, fenced)


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda_compute(9), reason="needs an sm_90 or newer GPU")
@pytest.mark.parametrize("dtype", list(TYPES))
def test_value_is_unchanged(dtype):
    mod, _ = _compile(_cancel(dtype, fence=True), env.cuda_arch())
    dev = tvm_tirx.cuda(0)
    a = np.array([7], dtype)
    c = np.array([5], dtype)
    out = tvm_tirx.runtime.tensor(np.zeros(1, dtype), device=dev)
    mod(out, tvm_tirx.runtime.tensor(a, device=dev), tvm_tirx.runtime.tensor(c, device=dev))
    assert out.numpy()[0] == (a + c - c)[0]


if __name__ == "__main__":
    tvm_tirx.testing.main()
