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
"""CUDA launch-index registers: value ranges for the analyzer and CSE-able reads."""

import pytest

import tvm_tirx
from tvm_tirx import ir
from tvm_tirx.script import tirx as T
from tvm_tirx.sym import Analyzer

_INT32 = ir.PrimType("int32")


def _axis_call(name, axis):
    return ir.Call(ir.Op.get(f"tirx.cuda.{name}"), [ir.StringImm(axis)], ty=_INT32)


def _sreg_call(register):
    return ir.Call(ir.Op.get("tirx.cuda.mov_sreg"), [ir.IntImm("int32", 32), register], ty=_INT32)


@pytest.mark.parametrize(
    "call, low, high",
    [
        (_axis_call("block_idx", "x"), 0, 2**31 - 2),
        (_axis_call("block_idx", "y"), 0, 65534),
        (_axis_call("thread_idx", "x"), 0, 1023),
        (_axis_call("thread_idx", "z"), 0, 63),
        (_axis_call("cluster_id", "x"), 0, 2**31 - 2),
        (_axis_call("cluster_cta_id", "x"), 0, 15),
        (_axis_call("grid_dim", "x"), 1, 2**31 - 1),
        (_axis_call("block_dim", "x"), 1, 1024),
        (_axis_call("cluster_dim", "y"), 1, 16),
        (_sreg_call("clusterid.x"), 0, 2**31 - 2),
        (_sreg_call("cluster_ctarank"), 0, 15),
        (_sreg_call("laneid"), 0, 31),
    ],
)
def test_launch_register_bounds(call, low, high):
    bound = Analyzer().const_int_bound(call)
    assert (bound.min_value, bound.max_value) == (low, high)


def test_unknown_register_is_unbounded():
    bound = Analyzer().const_int_bound(_sreg_call("clock"))
    assert (bound.min_value, bound.max_value) == (-(2**31), 2**31 - 1)


def test_launch_config_enforces_the_assumed_cluster_cap():
    """cluster_cta_id is bounded by 16 CTAs per cluster, so a larger cluster is rejected."""
    T.cuda.LaunchConfig(grid=64, block=128, cluster=16)
    T.cuda.LaunchConfig(grid=(8, 8), block=128, cluster=(4, 4))
    with pytest.raises(ValueError, match="at most 16 CTAs"):
        T.cuda.LaunchConfig(grid=64, block=128, cluster=32)
    with pytest.raises(ValueError, match="at most 16 CTAs"):
        T.cuda.LaunchConfig(grid=(8, 8), block=128, cluster=(4, 8))
    with pytest.raises(ValueError, match="at most 16 CTAs"):
        T.cuda.LaunchConfig(grid=(8, 8), block=128, cluster=(2, 1), preferred_cluster=(8, 4))


def _source(func):
    target = tvm_tirx.target.Target({"kind": "cuda", "arch": "sm_90a"})
    mod = tvm_tirx.compile(tvm_tirx.IRModule({"main": func}), target=target, pipeline="tirx")
    return mod.mod.imports[0].inspect_source()


def _cluster_tiles():
    @T.function
    def main(out: T.Tensor((16 * 2, 28), "int32"), clock: T.Tensor((1,), "int32")):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=132, block=128, cluster=2))
        cid = T.cuda.cluster_id("x")
        rank = T.cuda.cluster_cta_id("x")
        for j in T.serial(7):
            t = cid + j * 66
            if t < 448:
                out[(t // 28) * 2 + rank, t % 28] = t
        clock[0] = T.cuda.mov_sreg(32, "clock")

    return main


def test_cluster_register_reads_are_not_volatile():
    """Registers fixed for the CTA's lifetime are read without volatile, so nvcc CSEs them."""
    src = _source(_cluster_tiles())
    assert 'asm("mov.u32 %0, %clusterid.x;"' in src
    assert 'asm("mov.u32 %0, %cluster_ctaid.x;"' in src
    assert 'asm volatile("mov.u32 %0, %clock;"' in src


def test_launch_index_division_has_no_sign_fixup():
    """A non-negative launch index divides with plain truncdiv / truncmod.

    Without a range, floordiv and floormod of an index expression lower with a negative-operand
    correction (``(r >> 31)``) or a large positive shift (sm90-probe ISSUE-001).
    """
    src = _source(_cluster_tiles())
    body = src[src.find("main_kernel(") :]
    assert ">> 31" not in body
    assert "/ 28)" in body and "% 28)" in body


if __name__ == "__main__":
    tvm_tirx.testing.main()
