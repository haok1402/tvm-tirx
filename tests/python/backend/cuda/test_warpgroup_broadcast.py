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
"""T.cuda.broadcast_warpgroup_id: a warp-uniform warpgroup index at its call site."""

import numpy as np
import pytest

import tvm_tirx
from tvm_tirx.script import tirx as T
from tvm_tirx.testing import env


def _consumer_index():
    @T.function
    def main(out: T.Tensor((384,), "int32")):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=384))
        tx = T.cuda.thread_idx("x")
        if tx >= 128:
            # Warp-converged: the condition is uniform within each warp.
            wg = T.cuda.broadcast_warpgroup_id() - 1
            out[tx] = wg
        else:
            out[tx] = -1

    return main


def _source(func, arch="sm_90a"):
    target = tvm_tirx.target.Target({"kind": "cuda", "arch": arch})
    mod = tvm_tirx.compile(tvm_tirx.IRModule({"main": func}), target=target, pipeline="tirx")
    return mod, mod.mod.imports[0].inspect_source()


def test_broadcast_warpgroup_id_shuffles_at_the_call_site():
    _, src = _source(_consumer_index())
    body = src[src.find("main_kernel(") :]
    branch = body.find("if (128 <= ")
    if branch < 0:
        branch = body.find("if (")
    shuffle = body.find("__shfl_sync")
    assert branch >= 0 and shuffle > branch, body
    assert ">> 7" in body[shuffle : body.find(";", shuffle)]
    # No kernel-entry warp index is bound for it.
    assert "warp_id_in_cta" not in body


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="needs a CUDA GPU")
def test_broadcast_warpgroup_id_values():
    mod, _ = _source(_consumer_index(), arch=env.cuda_arch())
    out = tvm_tirx.runtime.tensor(np.zeros(384, dtype="int32"), device=tvm_tirx.cuda(0))
    mod(out)
    expected = np.array([-1] * 128 + [0] * 128 + [1] * 128, dtype="int32")
    np.testing.assert_array_equal(out.numpy(), expected)


if __name__ == "__main__":
    tvm_tirx.testing.main()
