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
"""Warp-collective results stay where they are computed (sm90-probe ISSUE-008).

A full-mask shuffle in the short-circuit operand of ``||`` runs on a subset of the warp, which
is undefined behavior and hangs on H200. The value of ``broadcast_warpgroup_id()`` must be
computed once, where it is called, and only read in later expressions.
"""

import re

import numpy as np
import pytest

import tvm_tirx
from tvm_tirx.script import tirx as T
from tvm_tirx.testing import env


def _mask(tx, tail):
    # A Python helper composes expressions without the parser's assignment binding, as
    # tirx-lite kernels do: cw is one Python value with two uses.
    cw = T.cuda.broadcast_warpgroup_id() - 1
    # The first operand differs across the lanes of a warp, so only some lanes evaluate the
    # second operand of ||.
    return T.Or(tx % 8 >= tail, cw * 64 + tx % 64 > 70), cw


def _masked_kernel(via_helper):
    @T.function
    def main(out: T.Tensor((384,), "int32"), tail: T.int32):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=384))
        tx = T.cuda.thread_idx("x")
        if tx >= 128:
            if T.constexpr(via_helper):
                masked, cw = _mask(tx, tail)
                out[tx] = T.Select(masked, -1, cw)
            else:
                # The call written directly in the short-circuit operand.
                out[tx] = T.Select(
                    T.Or(tx % 8 >= tail, (T.cuda.broadcast_warpgroup_id() - 1) * 64 + tx % 64 > 70),
                    -1,
                    tx // 128 - 1,
                )
        else:
            out[tx] = -2

    return main


def _compile(func, arch="sm_90a"):
    target = tvm_tirx.target.Target({"kind": "cuda", "arch": arch})
    mod = tvm_tirx.compile(tvm_tirx.IRModule({"main": func}), target=target, pipeline="tirx")
    return mod, mod.mod.imports[0].inspect_source()


@pytest.mark.parametrize("via_helper", [False, True])
def test_broadcast_is_bound_outside_short_circuit(via_helper):
    _, src = _compile(_masked_kernel(via_helper))
    body = src[src.find("main_kernel(") :]
    shuffles = [line.strip() for line in body.splitlines() if "__shfl_sync" in line]
    # One shuffle, in an assignment of its own, never inside a || / && / ?: expression.
    assert len(shuffles) == 1, shuffles
    assert re.match(r"int \w+ = .*__shfl_sync", shuffles[0]), shuffles[0]
    assert "||" not in shuffles[0] and "?" not in shuffles[0], shuffles[0]


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="needs a CUDA GPU")
@pytest.mark.parametrize("via_helper", [False, True])
def test_broadcast_in_short_circuit_runs(via_helper):
    mod, _ = _compile(_masked_kernel(via_helper), arch=env.cuda_arch())
    out = tvm_tirx.runtime.tensor(np.zeros(384, dtype="int32"), device=tvm_tirx.cuda(0))
    tail = 3
    mod(out, tail)
    tx = np.arange(384)
    cw = tx // 128 - 1
    expected = np.where((tx % 8 >= tail) | (cw * 64 + tx % 64 > 70), -1, cw)
    expected[:128] = -2
    np.testing.assert_array_equal(out.numpy(), expected)


if __name__ == "__main__":
    tvm_tirx.testing.main()
