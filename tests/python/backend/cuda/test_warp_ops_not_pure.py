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
"""Warp collectives and time-varying special-register reads are not pure (sm90-probe ISSUE-021).

A pure call may be substituted for the variable it is bound to, i.e. re-evaluated at each use.
For `__activemask` that use can sit in a divergent branch with fewer lanes; for `any_sync`, in a
branch where not every lane in the mask runs it; for two `%clock` reads, `t1 - t0` folds to 0.
"""

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
    return mod, src[src.rfind("__global__") :]


@T.function
def _activemask(out: T.Tensor((32,), "uint32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    tid = T.cuda.thread_idx("x")
    m: T.let = T.cuda.__activemask()  # all 32 lanes are active here
    out[tid] = T.uint32(0)
    if tid < 16:
        out[tid] = m


@T.function
def _any_sync(out: T.Tensor((32,), "int32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    tid = T.cuda.thread_idx("x")
    v: T.let = T.cuda.any_sync(T.uint32(0xFFFFFFFF), tid == 0)  # all 32 lanes vote
    out[tid] = 0
    if tid < 16:
        out[tid] = v


@T.function
def _clock64(out: T.Tensor((1,), "int64"), a: T.Tensor((1024,), "float32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=1))
    t0: T.let = T.cuda.clock64()
    acc = T.local_scalar("float32", T.float32(0))
    for i in T.serial(1024):
        acc = acc + a[i]
    a[0] = acc
    t1: T.let = T.cuda.clock64()
    out[0] = t1 - t0


def test_activemask_is_not_rematerialized():
    _, src = _compile(_activemask)
    assert src.count("__activemask()") == 1, src


def test_any_sync_is_not_moved_into_the_branch():
    _, src = _compile(_any_sync)
    assert src.count("any_sync(") == 1, src
    assert src.index("any_sync(") < src.index("< 16"), src


def test_clock64_reads_are_not_merged():
    _, src = _compile(_clock64)
    assert src.count("clock64") >= 2, src
    assert "out_ptr[0] = (int64_t)0" not in src and "out_ptr[0] = 0" not in src, src


@pytest.mark.parametrize("reg", ["clock", "%clock", "clock_hi", "clock64", "globaltimer", "pm0"])
def test_mov_sreg_rejects_time_varying_registers(reg):
    with pytest.raises(ValueError, match="time-varying register"):
        T.cuda.mov_sreg(32, reg)


def test_mov_sreg_keeps_static_registers():
    @T.function
    def main(out: T.Tensor((32,), "int32")):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
        tid = T.cuda.thread_idx("x")
        out[tid] = T.cuda.mov_sreg(32, "laneid")

    target = tvm_tirx.target.Target({"kind": "cuda", "arch": "sm_90a"})
    mod = tvm_tirx.compile(tvm_tirx.IRModule({"main": main}), target=target, pipeline="tirx")
    assert "mov.u32 %0, %laneid;" in mod.mod.imports[0].inspect_source()


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda_compute(9), reason="needs an sm_90 or newer GPU")
def test_activemask_value():
    mod, _ = _compile(_activemask, env.cuda_arch())
    out = tvm_tirx.runtime.tensor(np.zeros(32, "uint32"), device=tvm_tirx.cuda(0))
    mod(out)
    expected = np.where(np.arange(32) < 16, 0xFFFFFFFF, 0).astype("uint32")
    np.testing.assert_array_equal(out.numpy(), expected)


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda_compute(9), reason="needs an sm_90 or newer GPU")
def test_any_sync_value():
    mod, _ = _compile(_any_sync, env.cuda_arch())
    out = tvm_tirx.runtime.tensor(np.zeros(32, "int32"), device=tvm_tirx.cuda(0))
    mod(out)
    expected = np.where(np.arange(32) < 16, 1, 0).astype("int32")
    np.testing.assert_array_equal(out.numpy() != 0, expected != 0)


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda_compute(9), reason="needs an sm_90 or newer GPU")
def test_clock64_interval_is_positive():
    mod, _ = _compile(_clock64, env.cuda_arch())
    dev = tvm_tirx.cuda(0)
    out = tvm_tirx.runtime.tensor(np.zeros(1, "int64"), device=dev)
    mod(out, tvm_tirx.runtime.tensor(np.ones(1024, "float32"), device=dev))
    assert out.numpy()[0] > 0


if __name__ == "__main__":
    tvm_tirx.testing.main()
