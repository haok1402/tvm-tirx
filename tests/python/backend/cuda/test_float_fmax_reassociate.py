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
"""NaN-ignoring T.fmax / T.fmin and opt-in float reassociation (sm90-probe ISSUE-009)."""

import re
import shutil
import subprocess
import tempfile

import numpy as np
import pytest

import tvm_tirx
from tvm_tirx.backend.cuda.compiler import compile_source
from tvm_tirx.script import tirx as T
from tvm_tirx.testing import env


def _binary_kernel(fn, dtype="float32"):
    @T.function
    def main(a: T.Tensor((64,), dtype), b: T.Tensor((64,), dtype), out: T.Tensor((64,), dtype)):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=64))
        tx = T.cuda.thread_idx("x")
        out[tx] = fn(a[tx], b[tx])

    return main


def _softmax_kernel(reassociate):
    @T.function
    def main(
        s: T.Tensor((64,), "float32"), m: T.Tensor((1,), "float32"), out: T.Tensor((64,), "float32")
    ):
        if T.constexpr(reassociate):
            T.func_attr({"tirx.float_reassociate": True})
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=64))
        tx = T.cuda.thread_idx("x")
        out[tx] = T.exp2((s[tx] - m[0]) * T.float32(0.18033688))

    return main


def _compile(func, arch="sm_90a", config=None):
    target = tvm_tirx.target.Target({"kind": "cuda", "arch": arch})
    with tvm_tirx.transform.PassContext(config=config or {}):
        mod = tvm_tirx.compile(tvm_tirx.IRModule({"main": func}), target=target, pipeline="tirx")
    src = mod.mod.imports[0].inspect_source()
    return mod, src[src.find("main_kernel(") :]


def _sass_ops(src, arch="sm_90a"):
    """Opcode histogram of the kernel's SASS (NVRTC with the backend's flags)."""
    if shutil.which("cuobjdump") is None:
        pytest.skip("needs cuobjdump")
    binary = compile_source(src, {"cuda": {"arch": arch}}).binary
    with tempfile.NamedTemporaryFile(suffix=".cubin") as f:
        f.write(binary)
        f.flush()
        text = subprocess.run(["cuobjdump", "-sass", f.name], capture_output=True, text=True).stdout
    ops = {}
    for line in text.splitlines():
        m = re.match(r"\s*/\*[0-9a-f]+\*/\s+(?:@!?P\w+\s+)?([A-Z0-9]+)", line)
        if m:
            ops[m.group(1)] = ops.get(m.group(1), 0) + 1
    return ops


def _full_source(func, **kwargs):
    target = tvm_tirx.target.Target({"kind": "cuda", "arch": "sm_90a"})
    with tvm_tirx.transform.PassContext(config=kwargs.get("config", {})):
        mod = tvm_tirx.compile(tvm_tirx.IRModule({"main": func}), target=target, pipeline="tirx")
    return mod.mod.imports[0].inspect_source()


@pytest.mark.parametrize(
    "dtype, fn, symbol",
    [
        ("float32", T.fmax, "fmaxf("),
        ("float32", T.fmin, "fminf("),
        ("float64", T.fmax, "fmax("),
        ("float16", T.fmax, "__hmax("),
        ("bfloat16", T.fmin, "__hmin("),
    ],
)
def test_fmax_lowers_to_one_call(dtype, fn, symbol):
    _, body = _compile(_binary_kernel(fn, dtype))
    assert symbol in body, body
    assert "?" not in body and "!=" not in body, body


def test_fmax_is_one_fmnmx():
    max_ops = _sass_ops(_full_source(_binary_kernel(T.max)))
    fmax_ops = _sass_ops(_full_source(_binary_kernel(T.fmax)))
    # T.max keeps NaN propagation: compare + NaN test + select.
    assert max_ops.get("FSEL", 0) >= 1 and max_ops.get("FMNMX", 0) == 0, max_ops
    assert fmax_ops.get("FMNMX", 0) == 1 and fmax_ops.get("FSEL", 0) == 0, fmax_ops


def test_fmax_rejects_integers():
    with pytest.raises(TypeError, match="floating-point"):
        T.fmax(T.int32(1), T.int32(2))


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="needs a CUDA GPU")
@pytest.mark.parametrize("fn", ["fmax", "fmin", "max", "min"])
def test_nan_semantics(fn):
    mod, _ = _compile(_binary_kernel(getattr(T, fn)), arch=env.cuda_arch())
    rng = np.random.default_rng(0)
    a = rng.standard_normal(64).astype("float32")
    b = rng.standard_normal(64).astype("float32")
    a[:8] = np.nan
    b[4:12] = np.nan
    dev = tvm_tirx.cuda(0)
    out = tvm_tirx.runtime.tensor(np.zeros(64, "float32"), device=dev)
    mod(tvm_tirx.runtime.tensor(a, device=dev), tvm_tirx.runtime.tensor(b, device=dev), out)
    expected = {
        "fmax": np.fmax,  # NaN-ignoring
        "fmin": np.fmin,
        "max": np.maximum,  # NaN-propagating, unchanged
        "min": np.minimum,
    }[fn](a, b)
    np.testing.assert_array_equal(out.numpy(), expected)


@pytest.mark.skipif(not env.has_llvm(), reason="needs LLVM")
def test_fmax_on_llvm():
    @T.function
    def main(
        a: T.Tensor((4,), "float32"), b: T.Tensor((4,), "float32"), out: T.Tensor((4,), "float32")
    ):
        for i in range(4):
            out[i] = T.fmax(a[i], b[i])

    mod = tvm_tirx.compile(main, target="llvm")
    a = np.array([np.nan, 1, 2, np.nan], "float32")
    b = np.array([0, np.nan, 3, np.nan], "float32")
    out = tvm_tirx.runtime.tensor(np.zeros(4, "float32"))
    mod(tvm_tirx.runtime.tensor(a), tvm_tirx.runtime.tensor(b), out)
    np.testing.assert_array_equal(out.numpy(), np.fmax(a, b))


def test_reassociation_is_off_by_default():
    _, body = _compile(_softmax_kernel(False))
    assert "- m_ptr[0]) * " in body, body


@pytest.mark.parametrize("via", ["attr", "config"])
def test_reassociation_distributes(via):
    config = {"tirx.float_reassociate": True} if via == "config" else None
    _, body = _compile(_softmax_kernel(via == "attr"), config=config)
    # (s - m) * c -> s * c - m * c: nvcc contracts it into one FFMA.
    assert re.search(r"\(s_ptr\[[^]]*\] \* [^()]*\) - \(m_ptr\[0\] \* ", body), body


def test_reassociated_exponent_is_ffma():
    default = _sass_ops(_full_source(_softmax_kernel(False)))
    fast = _sass_ops(_full_source(_softmax_kernel(True)))
    assert default.get("FADD", 0) == 1 and default.get("FMUL", 0) >= 1, default
    assert fast.get("FFMA", 0) >= 1 and fast.get("FADD", 0) == 0, fast


@pytest.mark.parametrize("dtype", ["int32", "int64"])
def test_reassociation_leaves_integers(dtype):
    def kernel(reassociate):
        @T.function
        def main(a: T.Tensor((64,), dtype), out: T.Tensor((64,), dtype)):
            if T.constexpr(reassociate):
                T.func_attr({"tirx.float_reassociate": True})
            T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=64))
            tx = T.cuda.thread_idx("x")
            out[tx] = (a[tx] - a[0]) * a[1]

        return main

    # Integer arithmetic is exact, and the pass never touches it.
    assert _compile(kernel(True))[1] == _compile(kernel(False))[1]


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="needs a CUDA GPU")
def test_reassociated_softmax_is_close():
    rng = np.random.default_rng(1)
    s = (rng.standard_normal(64) * 8).astype("float32")
    m = np.array([s.max()], "float32")
    dev = tvm_tirx.cuda(0)
    outs = []
    for reassociate in (False, True):
        mod, _ = _compile(_softmax_kernel(reassociate), arch=env.cuda_arch())
        out = tvm_tirx.runtime.tensor(np.zeros(64, "float32"), device=dev)
        mod(tvm_tirx.runtime.tensor(s, device=dev), tvm_tirx.runtime.tensor(m, device=dev), out)
        outs.append(out.numpy())
    expected = np.exp2((s.astype("float64") - m[0]) * 0.18033688)
    for out in outs:
        np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-7)


if __name__ == "__main__":
    tvm_tirx.testing.main()
