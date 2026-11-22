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
"""NVRTC modules carry no libcu++ payload (sm90-probe ISSUE-024).

`<cuda/std/limits>` and `<cuda/std/type_traits>` put ~500 constexpr statics into every module as
`.nv.global` / `.nv.constant4` data, which costs launch time. The NVRTC preamble defines INFINITY
as a builtin and the generated header defines std::is_same itself; libcu++'s traits come in only
when the source needs one of the others.
"""

import math
import struct

import numpy as np
import pytest

import tvm_tirx
import tvm_tirx.testing
from tvm_tirx.backend.config import parse_backend_config
from tvm_tirx.backend.cuda.compiler import compile_source
from tvm_tirx.script import tirx as T
from tvm_tirx.testing import env

INF_SRC = "__device__ __forceinline__ float tirx_test_inf() { return INFINITY; }\n"
TRAIT_SRC = (
    "template <class T, typename std::enable_if<std::is_integral<T>::value, int>::type = 0>\n"
    "__device__ __forceinline__ T tirx_test_twice(T x) { return x + x; }\n"
)


def _payload(cubin):
    """Bytes in .nv.global, .nv.global.init and .nv.constant4 of an ELF cubin."""
    shoff = struct.unpack_from("<Q", cubin, 0x28)[0]
    shentsize, shnum, shstrndx = struct.unpack_from("<HHH", cubin, 0x3A)
    secs = [struct.unpack_from("<IIQQQQIIQQ", cubin, shoff + i * shentsize) for i in range(shnum)]
    strtab = secs[shstrndx][4]
    sizes = {}
    for sec in secs:
        name = cubin[strtab + sec[0] : cubin.index(b"\0", strtab + sec[0])].decode()
        if name in (".nv.global", ".nv.global.init", ".nv.constant4"):
            sizes[name] = sizes.get(name, 0) + sec[5]
    return sizes


def _kernel(extra):
    # fp16 + fp8 values pull in the header helpers that use std::is_same_v; warp_max's identity is
    # -INFINITY.
    @T.function
    def main(out: T.Tensor((32,), "float32"), x: T.Tensor((32,), "float16")):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
        tid = T.cuda.thread_idx("x")
        v = T.Cast("float32", T.Cast("float8_e4m3fn", x[tid]))
        if T.constexpr(extra == "inf"):
            out[tid] = T.cuda.warp_max(v) + T.cuda.func_call("tirx_test_inf", INF_SRC, ty="float32")
        elif T.constexpr(extra == "trait"):
            out[tid] = T.Cast(
                "float32", T.cuda.func_call("tirx_test_twice", tid, TRAIT_SRC, ty="int32")
            )
        else:
            out[tid] = T.cuda.warp_max(v)

    return main


def _compile(func, arch="sm_90a"):
    target = tvm_tirx.target.Target({"kind": "cuda", "arch": arch})
    return tvm_tirx.compile(tvm_tirx.IRModule({"main": func}), target=target, pipeline="tirx")


def _cubin(mod):
    imported = mod.mod.imports[0]
    config = parse_backend_config(imported.inspect_source("backend_config"))
    config["cuda"]["target_format"] = "cubin"
    return compile_source(imported.inspect_source(), config).binary


def test_header_has_no_libcuxx_traits():
    src = _compile(_kernel("plain")).mod.imports[0].inspect_source()
    rtc = src[src.index("#ifdef __CUDACC_RTC__") : src.index("#else")]
    assert "#include <cuda/std/type_traits>" not in rtc, rtc
    assert "#include <cuda/std/limits>" not in rtc, rtc
    assert "is_same_v" in rtc, rtc


@pytest.mark.skipif(not env.has_cuda(), reason="needs the CUDA toolkit")
@pytest.mark.parametrize("extra", ["plain", "inf"])
def test_module_has_no_payload(extra):
    sizes = _payload(_cubin(_compile(_kernel(extra))))
    assert sum(sizes.values()) == 0, sizes


CUDA_STD_SRC = {
    "is_same_v": (
        "__device__ __forceinline__ int tirx_test_cuda_std(int x) "
        "{ return cuda::std::is_same_v<int, int> ? x : -x; }\n"
    ),
    "numeric_limits": (
        "__device__ __forceinline__ int tirx_test_cuda_std(int x) "
        "{ return x < cuda::std::numeric_limits<int>::max() ? x : 0; }\n"
    ),
}


@pytest.mark.skipif(not env.has_cuda(), reason="needs the CUDA toolkit")
@pytest.mark.parametrize("facility", list(CUDA_STD_SRC))
def test_cuda_std_names_still_compile(facility):
    # Source naming a cuda::std:: facility gets libcu++'s traits and limits back.
    @T.function
    def main(out: T.Tensor((32,), "int32")):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
        tid = T.cuda.thread_idx("x")
        out[tid] = T.cuda.func_call("tirx_test_cuda_std", tid, CUDA_STD_SRC[facility], ty="int32")

    mod = _compile(main)
    _cubin(mod)
    if env.has_cuda_compute(9):
        out = tvm_tirx.runtime.tensor(np.zeros(32, "int32"), device=tvm_tirx.cuda(0))
        _compile(main, env.cuda_arch())(out)
        np.testing.assert_array_equal(out.numpy(), np.arange(32))


@pytest.mark.skipif(not env.has_cuda(), reason="needs the CUDA toolkit")
def test_other_std_traits_still_compile():
    # Source naming another std:: trait gets libcu++'s traits (and their payload) back.
    _cubin(_compile(_kernel("trait")))


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda_compute(9), reason="needs an sm_90 or newer GPU")
def test_infinity_value():
    mod = _compile(_kernel("inf"), env.cuda_arch())
    dev = tvm_tirx.cuda(0)
    out = tvm_tirx.runtime.tensor(np.zeros(32, "float32"), device=dev)
    mod(out, tvm_tirx.runtime.tensor(np.ones(32, "float16"), device=dev))
    assert all(math.isinf(v) and v > 0 for v in out.numpy())


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda_compute(9), reason="needs an sm_90 or newer GPU")
def test_warp_max_with_builtin_infinity():
    mod = _compile(_kernel("plain"), env.cuda_arch())
    dev = tvm_tirx.cuda(0)
    x = np.linspace(-3, 3, 32).astype("float16")
    out = tvm_tirx.runtime.tensor(np.zeros(32, "float32"), device=dev)
    mod(out, tvm_tirx.runtime.tensor(x, device=dev))
    ref = np.float32(np.max(x.astype("float32")))
    np.testing.assert_allclose(out.numpy(), np.full(32, ref), rtol=0.07)


if __name__ == "__main__":
    tvm_tirx.testing.main()
