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
is undefined behavior and hangs on H200. The value of ``broadcast_warpgroup_id()``, and of the
other value-returning warp collectives, must be computed once, where it is called, and only
read in later expressions.
"""

import re
import threading
from concurrent.futures import ThreadPoolExecutor

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


def _ballot_kernel():
    def mask(tx):
        ballot = T.cuda.ballot_sync(T.uint32(0xFFFFFFFF), tx % 3 == 0)
        return T.Or(tx % 8 >= 3, ballot > T.uint32(5)), ballot

    @T.function
    def main(out: T.Tensor((32,), "uint32")):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
        tx = T.cuda.thread_idx("x")
        masked, ballot = mask(tx)
        out[tx] = T.Select(masked, T.uint32(0), ballot)

    return main


def test_ballot_is_bound_outside_short_circuit():
    _, src = _compile(_ballot_kernel())
    body = src[src.find("main_kernel(") :]
    ballots = [line.strip() for line in body.splitlines() if "ballot_sync(" in line]
    assert len(ballots) == 1, ballots
    assert re.match(r"uint \w+ = .*ballot_sync\(", ballots[0]), ballots[0]


def _shuffle_mask(tx):
    value = T.gpu_warp_shuffle(T.uint32(0xFFFFFFFF), tx * 3, 0, 32, 32)
    return T.Or(tx % 8 >= 3, value > 5), value


def test_warp_shuffle_is_bound_outside_short_circuit():
    @T.function
    def main(out: T.Tensor((32,), "int32")):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
        tx = T.cuda.thread_idx("x")
        masked, value = _shuffle_mask(tx)
        out[tx] = T.Select(masked, 0, value)

    _, src = _compile(main)
    body = src[src.find("main_kernel(") :]
    shuffles = [line.strip() for line in body.splitlines() if "__shfl_sync" in line]
    assert len(shuffles) == 1, shuffles
    assert re.match(r"int \w+ = .*__shfl_sync", shuffles[0]), shuffles[0]


@pytest.mark.parametrize(
    "collective, symbol",
    [
        (lambda tx: T.cuda.__shfl_xor_sync(T.uint32(0xFFFFFFFF), tx * 3, 1, 32), "__shfl_xor_sync"),
        (lambda tx: T.cuda.warp_sum(tx * 3), "warp_reduce"),
    ],
)
def test_cuda_collectives_are_bound(collective, symbol):
    def helper(tx):
        value = collective(tx)
        return T.Or(tx % 8 >= 3, value > 5), value

    @T.function
    def main(out: T.Tensor((32,), "int32")):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
        tx = T.cuda.thread_idx("x")
        masked, value = helper(tx)
        out[tx] = T.Select(masked, 0, value)

    _, src = _compile(main)
    body = src[src.find("main_kernel(") :]
    calls = [line.strip() for line in body.splitlines() if symbol in line]
    assert len(calls) == 1, calls
    assert re.match(rf"int \w+ = .*{symbol}", calls[0]), calls[0]


@pytest.mark.parametrize(
    "construct",
    [
        lambda: T.cuda.broadcast_warpgroup_id(),
        lambda: T.cuda.__shfl_sync(T.uint32(0xFFFFFFFF), 1, 0, 32),
        lambda: T.gpu_warp_shuffle(T.uint32(0xFFFFFFFF), 1, 0, 32, 32),
    ],
)
def test_collective_outside_kernel_body_raises(construct):
    # e.g. precomputed in a Python kernel factory, then used in the body: the unbound call
    # would be re-emitted at each use.
    with pytest.raises(RuntimeError, match="inside the kernel body"):
        construct()


def test_call_site_record_is_per_thread():
    # The record that lets `x: T.let[...] = collective(...)` reuse the call-site binding must
    # not be visible to (or consumed by) a kernel being built on another thread.
    from tvm_tirx.tirx.script.ir_builder import op as _op

    bound = object()
    _op._CALL_SITE_BINDING.last = bound
    seen = []
    other = threading.Thread(target=lambda: seen.append(_op.take_call_site_binding(bound)))
    other.start()
    other.join()
    assert seen == [False]
    assert _op.take_call_site_binding(bound)
    assert not _op.take_call_site_binding(bound)


def test_concurrent_builds_bind_once():
    def build(_):
        @T.function
        def main(a: T.Tensor((32,), "int32")):
            T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
            tx = T.cuda.thread_idx("x")
            x: T.let[T.int32] = T.cuda.__shfl_sync(T.uint32(0xFFFFFFFF), a[tx], 0, 32)
            a[tx] = x + 1

        return main.script()

    with ThreadPoolExecutor(8) as pool:
        scripts = list(pool.map(build, range(32)))
    assert len(set(scripts)) == 1, scripts[0]
    assert scripts[0].count("__shfl_sync") == 1, scripts[0]


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
