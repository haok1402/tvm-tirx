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
"""`ld.param` / `ld.const`: read a `__grid_constant__` parameter through the constant bank.

A generic `ld` of a kernel parameter's address is correct but lowers to `LD.E` (sm90-probe
ISSUE-018); CUDA's struct copy of the parameter is `ld.param` / `LDC`. The address is a
state-space address, from `cvta.to.param.u64`.
"""

import re
import subprocess
import tempfile

import numpy as np
import pytest

import tvm_tirx
import tvm_tirx.testing
from tvm_tirx.backend.config import parse_backend_config
from tvm_tirx.backend.cuda.compiler import compile_source
from tvm_tirx.script import tirx as T
from tvm_tirx.testing import env


def _copy_kernel(route):
    # Each of 8 threads copies 16 bytes of the 128-byte tensor map parameter to out.
    @T.function
    def main(A: T.Tensor((256, 64), "float16"), out: T.Tensor((32,), "uint32")):
        A_map: T.let[T.handle("tensormap")] = T.stack_alloca("tensormap", 1)
        T.cuda.tensormap_encode_tiled(
            A_map, A.data, 64, 256, 128, 64, 1, 1, 1, descriptor_dtype="float16", rank=2
        )
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
        tid = T.cuda.thread_idx("x")
        if tid < 8:
            w = T.alloc_local((4,), "uint32")
            offset = T.Cast("uint64", tid) * T.uint64(16)
            if T.constexpr(route == "param"):
                base = T.local_scalar("uint64")
                T.ptx.cvta.to.param.u64(base, T.reinterpret(T.address_of(A_map), ty="handle"))
                T.ptx.ld.param.v4.u32(w[0], w[1], w[2], w[3], base + offset)
            else:
                generic = T.reinterpret(T.address_of(A_map), ty="uint64") + offset
                T.ptx.ld.v4.u32(w[0], w[1], w[2], w[3], generic)
            for k in T.serial(4):
                out[tid * 4 + k] = w[k]

    return main


def _compile(func, arch="sm_90a"):
    target = tvm_tirx.target.Target({"kind": "cuda", "arch": arch})
    return tvm_tirx.compile(tvm_tirx.IRModule({"main": func}), target=target, pipeline="tirx")


def _device_source(mod):
    for imported in mod.mod.imports:
        if imported.kind == "cuda" or "__global__" in imported.inspect_source():
            return imported
    raise AssertionError("no CUDA module")


def _sass_loads(mod):
    imported = _device_source(mod)
    config = parse_backend_config(imported.inspect_source("backend_config"))
    config["cuda"]["target_format"] = "cubin"
    binary = compile_source(imported.inspect_source(), config).binary
    with tempfile.NamedTemporaryFile(suffix=".cubin") as f:
        f.write(binary)
        f.flush()
        sass = subprocess.run(
            ["cuobjdump", "-sass", f.name], capture_output=True, text=True, check=True
        ).stdout
    return sass


def test_ld_param_renders():
    src = _device_source(_compile(_copy_kernel("param"))).inspect_source()
    assert "cvta.to.param.u64 %0, %1;" in src, src
    assert "ld.param.v4.u32 {%0, %1, %2, %3}, [%4];" in src, src


@pytest.mark.skipif(not env.has_cuda(), reason="needs the CUDA toolkit")
def test_ld_param_is_a_constant_bank_load():
    # The parameter route reads through the constant bank; the generic route through LD.E.
    param = _sass_loads(_compile(_copy_kernel("param")))
    generic = _sass_loads(_compile(_copy_kernel("generic")))
    assert not re.search(r"\bLD\.E", param), param
    assert re.search(r"\bLDC\b", param), param
    assert re.search(r"\bLD\.E\.128\b", generic), generic


@pytest.mark.parametrize(
    "spelling, match",
    [
        ("ld.volatile.param.u32", "not valid on .param"),
        ("ld.relaxed.cta.param.u32", "not valid on .param"),
        ("ld.acquire.gpu.const.u32", "not valid on .const"),
        ("ld.param.L2::cache_hint.u32", "not valid on .param"),
        ("ld.param::func.u32", "func"),
    ],
)
def test_invalid_forms_are_rejected(spelling, match):
    with pytest.raises((ValueError, AttributeError, KeyError), match=match):

        @T.function
        def main(out: T.Tensor((1,), "uint32")):
            T.device_entry()
            v = T.local_scalar("uint32")
            T.ptx[spelling](v, T.uint64(0))
            out[0] = v


def test_generic_pointer_is_rejected():
    # A generic pointer is not a .param address: ld.param on one would read the wrong location.
    with pytest.raises(ValueError, match=r"cvta\.to\.param\.u64"):

        @T.function
        def main(out: T.Tensor((1,), "uint32")):
            T.device_entry()
            v = T.local_scalar("uint32")
            T.ptx.ld.param.u32(v, out.ptr_to([0]))
            out[0] = v


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda_compute(9), reason="needs an sm_90 or newer GPU")
def test_ld_param_reads_the_parameter():
    dev = tvm_tirx.cuda(0)
    a = tvm_tirx.runtime.tensor(np.zeros((256, 64), "float16"), device=dev)
    outs = {}
    for route in ("param", "generic"):
        mod = _compile(_copy_kernel(route), env.cuda_arch())
        out = tvm_tirx.runtime.tensor(np.zeros(32, "uint32"), device=dev)
        mod(a, out)
        outs[route] = out.numpy()
    assert outs["param"].any()
    np.testing.assert_array_equal(outs["param"], outs["generic"])


if __name__ == "__main__":
    tvm_tirx.testing.main()
