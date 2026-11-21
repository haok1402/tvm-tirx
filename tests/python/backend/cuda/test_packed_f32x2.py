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
"""Arch-portable packed f32x2 helpers (T.cuda.f32x2_<op>_<rnd>[_ftz])."""

import os

import numpy as np
import pytest

import tvm_tirx
from tvm_tirx.backend.cuda.lang import packed
from tvm_tirx.script import tirx as T
from tvm_tirx.testing import env

OPS = ("add", "sub", "mul", "fma")
VARIANTS = [
    (op, rnd, ftz) for op in OPS for rnd in ("rn", "rz", "rm", "rp") for ftz in (False, True)
]
N = 256


def _name(op, rnd, ftz):
    return f"{op}_{rnd}" + ("_ftz" if ftz else "")


def _kernel(op, rnd, ftz):
    """out[i] = helper(A[i], B[i] (, C[i])); ref[i] = the same op lane by lane through
    scalar T.ptx instructions with the same rounding and ftz."""
    helper = lambda *xs: packed.f32x2(op, *xs, rnd=rnd, ftz=ftz)  # noqa: E731
    scalar = getattr(getattr(T.ptx, op), rnd)
    scalar = getattr(scalar, "ftz") if ftz else scalar
    scalar = scalar.f32

    # fmt: off
    @T.function
    def main(A: T.Tensor((N,), "uint64"), B: T.Tensor((N,), "uint64"), C: T.Tensor((N,), "uint64"),
             out: T.Tensor((N,), "uint64"), ref: T.Tensor((N, 2), "float32")):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=N))
        i = T.cuda.thread_idx("x")
        if T.constexpr(op == "fma"):
            out[i] = helper(A[i], B[i], C[i])
        else:
            out[i] = helper(A[i], B[i])
        lane = T.alloc_local((2,), "float32")
        if T.constexpr(op == "fma"):
            scalar(lane[0], T.cuda.float2_x(A[i]), T.cuda.float2_x(B[i]), T.cuda.float2_x(C[i]))
            scalar(lane[1], T.cuda.float2_y(A[i]), T.cuda.float2_y(B[i]), T.cuda.float2_y(C[i]))
        else:
            scalar(lane[0], T.cuda.float2_x(A[i]), T.cuda.float2_x(B[i]))
            scalar(lane[1], T.cuda.float2_y(A[i]), T.cuda.float2_y(B[i]))
        ref[i, 0] = lane[0]
        ref[i, 1] = lane[1]
    # fmt: on
    return main


def _source(func, arch):
    previous = os.environ.get("TVM_TIRX_COMPILE_FORCE_FALLBACK")
    os.environ["TVM_TIRX_COMPILE_FORCE_FALLBACK"] = "1"
    try:
        target = tvm_tirx.target.Target({"kind": "cuda", "arch": arch})
        mod = tvm_tirx.compile(tvm_tirx.IRModule({"main": func}), target=target, pipeline="tirx")
    finally:
        if previous is None:
            del os.environ["TVM_TIRX_COMPILE_FORCE_FALLBACK"]
        else:
            os.environ["TVM_TIRX_COMPILE_FORCE_FALLBACK"] = previous
    return mod.mod.imports[0].inspect_source()


def test_argument_checks():
    with pytest.raises(ValueError, match="op must be"):
        packed.helper_source("div")
    with pytest.raises(ValueError, match="rounding"):
        packed.helper_source("add", "rx")
    with pytest.raises(ValueError, match="takes 3 operands"):
        packed.f32x2("fma", 1, 2)


def test_helper_has_packed_and_scalar_paths():
    src = _source(_kernel("mul", "rz", True), "sm_90a")
    assert src.count("unsigned long long tvm_builtin_f32x2_mul_rz_ftz(") == 1  # one definition
    assert "tvm_builtin_f32x2_mul_rz_ftz(" in src
    assert "#if __CUDA_ARCH__ >= 1000" in src
    assert 'asm("mul.rz.ftz.f32x2 %0, %1, %2;"' in src
    assert 'asm("mul.rz.ftz.f32 %0, %1, %2;" : "=f"(r.x)' in src
    src = _source(_kernel("fma", "rn", False), "sm_90a")
    assert 'asm("fma.rn.f32x2 %0, %1, %2, %3;"' in src


@pytest.mark.skipif(not env.has_cuda(), reason="needs NVRTC")
@pytest.mark.parametrize("op", OPS)
def test_same_source_compiles_packed_on_sm100_and_scalar_on_sm90(op):
    from tvm_tirx.backend.cuda.compiler import compile_source

    src = _source(_kernel(op, "rn", False), "sm_90a")
    ptx = {
        arch: compile_source(src, {"cuda": {"arch": arch, "target_format": "ptx"}}).binary.decode()
        for arch in ("sm_90a", "sm_100a")
    }
    assert f"{op}.rn.f32x2" in ptx["sm_100a"]
    assert f"{op}.rn.f32x2" not in ptx["sm_90a"]
    assert f"{op}.rn.f32" in ptx["sm_90a"]


def _inputs(rng):
    import torch

    f = rng.standard_normal((3, N, 2)).astype(np.float32) * 4
    # Denormals, signed zeros, and ties, so ftz and the rounding mode matter.
    f[0, :16, 0] = np.float32(1e-40)
    f[1, :16, 0] = np.float32(-3.0)
    f[0, 16:32, 1] = np.float32(-0.0)
    f[0, 32:48] = np.float32(1.0 + 2.0**-23)
    f[1, 32:48] = np.float32(1.0 + 2.0**-22)
    return [torch.from_numpy(x.copy()).cuda().view(torch.uint64).reshape(N) for x in f], f


@pytest.mark.skipif(not env.has_cuda_compute(8), reason="needs a CUDA GPU")
@pytest.mark.parametrize("op, rnd, ftz", VARIANTS, ids=[_name(*v) for v in VARIANTS])
def test_bit_exact_against_scalar_ptx(op, rnd, ftz):
    """On sm_100+ the packed instruction runs; elsewhere the scalar path. Both must
    equal the scalar PTX instruction lane by lane, bit for bit."""
    import torch

    (a, b, c), f = _inputs(np.random.default_rng(0))
    out = torch.zeros(N, dtype=torch.float64, device="cuda").view(torch.uint64)
    ref = torch.zeros(N, 2, dtype=torch.float32, device="cuda")
    target = tvm_tirx.target.Target("cuda")
    mod = tvm_tirx.compile(
        tvm_tirx.IRModule({"main": _kernel(op, rnd, ftz)}), target=target, pipeline="tirx"
    )
    mod(a, b, c, out, ref)
    torch.cuda.synchronize()
    got = out.view(torch.float32).reshape(N, 2).cpu().numpy()
    np.testing.assert_array_equal(got.view(np.uint32), ref.cpu().numpy().view(np.uint32))
    if rnd == "rn" and not ftz and op in ("add", "mul"):
        expected = (f[0] + f[1]) if op == "add" else (f[0] * f[1])
        np.testing.assert_array_equal(got.view(np.uint32), expected.view(np.uint32))
