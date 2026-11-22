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
"""A blocking mbarrier wait as one asm retry loop (sm90-probe ISSUE-012).

A TIRx `while` loop around the single-shot `mbarrier.try_wait` is a C loop. nvcc/NVRTC do not
unroll an enclosing `#pragma unroll` loop with a runtime trip count whose body holds another
loop. `T.cuda.mbarrier_wait_parity` keeps the retry inside the asm, as cutlass does.
"""

import numpy as np
import pytest

import tvm_tirx
import tvm_tirx.testing
from tvm_tirx.backend.config import parse_backend_config
from tvm_tirx.backend.cuda.compiler import compile_source
from tvm_tirx.script import tirx as T
from tvm_tirx.testing import env

CUTLASS_WAIT = (
    '"mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1, %2; \\n\\t"',
    '"@P1 bra DONE; \\n\\t"',
    '"bra     LAB_WAIT; \\n\\t"',
)


def _kernel(form):
    # One thread arrives on the barrier and waits for its phase once per iteration of an
    # unrolled loop with a runtime trip count (the shape of a producer loop over KV blocks).
    @T.function
    def main(out: T.Tensor((1,), "uint32"), n: T.int32):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
        bar = T.shared_scalar("uint64")
        if T.cuda.thread_idx("x") == 0:
            T.ptx["mbarrier.init.shared::cta.b64"](T.address_of(bar), T.uint32(1))
            T.ptx.fence.mbarrier_init.release.cluster()
            acc = T.local_scalar("uint32", T.uint32(0))
            for i in T.serial(T.Cast("uint32", n), unroll=True):
                T.ptx["mbarrier.arrive.shared::cta.b64"](T.address_of(bar))
                if T.constexpr(form == "while"):
                    done = T.local_scalar("uint32", T.uint32(0))
                    while done == T.uint32(0):
                        T.ptx["mbarrier.try_wait.parity.shared::cta.b64"](
                            done, T.address_of(bar), i & T.uint32(1), T.uint32(0x989680)
                        )
                else:
                    T.cuda.mbarrier_wait_parity(T.address_of(bar), i & T.uint32(1))
                acc = acc + i
            out[0] = acc

    return main


def _compile(form, arch="sm_90a"):
    target = tvm_tirx.target.Target({"kind": "cuda", "arch": arch})
    mod = tvm_tirx.compile(
        tvm_tirx.IRModule({"main": _kernel(form)}), target=target, pipeline="tirx"
    )
    return mod, mod.mod.imports[0].inspect_source()


def _ptx(mod):
    imported = mod.mod.imports[0]
    config = parse_backend_config(imported.inspect_source("backend_config"))
    config["cuda"]["target_format"] = "ptx"
    return compile_source(imported.inspect_source(), config).binary.decode()


def test_wait_is_cutlass_asm():
    _, src = _compile("asm")
    for line in CUTLASS_WAIT:
        assert line in src, src
    assert "while" not in src[src.find("main_kernel(") :], src


def test_qualifiers():
    @T.function
    def main(out: T.Tensor((1,), "uint32")):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
        bar = T.shared_scalar("uint64")
        T.cuda.mbarrier_wait_parity(T.address_of(bar), 0, 100, sem="acquire", scope="cluster")
        out[0] = T.uint32(1)

    target = tvm_tirx.target.Target({"kind": "cuda", "arch": "sm_90a"})
    mod = tvm_tirx.compile(tvm_tirx.IRModule({"main": main}), target=target, pipeline="tirx")
    src = mod.mod.imports[0].inspect_source()
    assert "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 P1" in src, src


@pytest.mark.parametrize("sem, scope", [("acquire", ""), ("", "cta"), ("weak", "cta")])
def test_qualifiers_are_validated(sem, scope):
    with pytest.raises(ValueError, match="mbarrier_wait_parity"):
        T.cuda.mbarrier_wait_parity(T.uint32(0), 0, sem=sem, scope=scope)


def test_enclosing_loop_unrolls():
    # The C loop around the single-shot try_wait keeps the enclosing loop rolled: one try_wait
    # site. The asm retry loop leaves it free to unroll, as nvcc does for cutlass.
    while_sites = _ptx(_compile("while")[0]).count("mbarrier.try_wait")
    asm_sites = _ptx(_compile("asm")[0]).count("mbarrier.try_wait")
    assert while_sites == 1, while_sites
    assert asm_sites > 1, asm_sites


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda_compute(9), reason="needs an sm_90 or newer GPU")
@pytest.mark.parametrize("form", ["while", "asm"])
def test_wait_runs(form):
    mod, _ = _compile(form, env.cuda_arch())
    out = tvm_tirx.runtime.tensor(np.zeros(1, "uint32"), device=tvm_tirx.cuda(0))
    n = 37
    mod(out, n)
    assert out.numpy()[0] == n * (n - 1) // 2


if __name__ == "__main__":
    tvm_tirx.testing.main()
