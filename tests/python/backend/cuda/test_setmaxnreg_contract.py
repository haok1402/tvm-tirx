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
"""setmaxnreg launch contract: a pinned entry register count, and the entry-register formula."""

import os
import re
import shutil
import subprocess
import tempfile

import pytest

import tvm_tirx
from tvm_tirx.backend.cuda.verify import SetMaxNRegContractError, entry_registers
from tvm_tirx.script import tirx as T


def _ws_kernel(block=384, attrs=None, use_setmaxnreg=True):
    kw = {} if attrs is None else {"kernel_attrs": T.cuda.KernelAttributes(**attrs)}

    # fmt: off
    @T.function
    def main(A: T.Tensor((1024,), "float32")):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=block), **kw)
        tx = T.cuda.thread_idx("x")
        if tx < 128:
            if T.constexpr(use_setmaxnreg):
                T.ptx["setmaxnreg.dec.sync.aligned.u32"](40)
            A[tx] = A[tx] + 1.0
        else:
            if T.constexpr(use_setmaxnreg):
                T.ptx["setmaxnreg.inc.sync.aligned.u32"](232)
            A[tx] = A[tx] * 2.0
    # fmt: on
    return main


def _compile(func, arch="sm_90a", fallback=True):
    previous = os.environ.get("TVM_TIRX_COMPILE_FORCE_FALLBACK")
    if fallback:
        os.environ["TVM_TIRX_COMPILE_FORCE_FALLBACK"] = "1"
    try:
        target = tvm_tirx.target.Target({"kind": "cuda", "arch": arch})
        return tvm_tirx.compile(tvm_tirx.IRModule({"main": func}), target=target, pipeline="tirx")
    finally:
        if fallback:
            if previous is None:
                del os.environ["TVM_TIRX_COMPILE_FORCE_FALLBACK"]
            else:
                os.environ["TVM_TIRX_COMPILE_FORCE_FALLBACK"] = previous


def test_entry_registers_formula():
    assert entry_registers(384, 1) == 168
    assert entry_registers(384, 2) == 80
    assert entry_registers(320, 1) == 168
    assert entry_registers(640, 1) == 96
    assert entry_registers(128, 1) == 255
    assert entry_registers(256, 1, inc_targets=(232,)) == 232
    assert entry_registers(384, None, 168) == 168
    assert entry_registers(384) is None


@pytest.mark.parametrize("arch", ["sm_90a", "sm_100a"])
def test_unpinned_setmaxnreg_is_rejected(arch):
    with pytest.raises(SetMaxNRegContractError, match="min_blocks_per_sm=1"):
        _compile(_ws_kernel(), arch)


@pytest.mark.parametrize("arch", ["sm_90a", "sm_100a"])
@pytest.mark.parametrize("attrs", [{"min_blocks_per_sm": 1}, {"max_registers_per_thread": 168}])
def test_pinned_setmaxnreg_compiles(arch, attrs):
    _compile(_ws_kernel(attrs=attrs), arch)


def test_kernels_without_setmaxnreg_are_unaffected():
    _compile(_ws_kernel(use_setmaxnreg=False))


def _ptxas_log_and_sass(func, arch):
    from tvm_tirx.backend.cuda.backend_config import compiler_options, resolve_backend_config
    from tvm_tirx.support import nvcc

    src = _compile(func, arch).mod.imports[0].inspect_source()
    options = compiler_options(resolve_backend_config({"arch": arch}))
    binary, log = nvcc._compile_cuda_nvrtc(src, "cubin", arch, options)
    sass = ""
    if shutil.which("cuobjdump"):
        with tempfile.NamedTemporaryFile(suffix=".cubin") as f:
            f.write(bytes(binary))
            f.flush()
            sass = subprocess.run(
                ["cuobjdump", "-sass", f.name], capture_output=True, text=True
            ).stdout
    return log, sass


@pytest.mark.parametrize("arch", ["sm_90a", "sm_100a"])
@pytest.mark.parametrize(
    "block, min_blocks",
    [(128, 1), (128, 3), (192, 2), (256, 1), (256, 2), (320, 1), (384, 1), (384, 2), (512, 3),
     (640, 1)],
)  # fmt: skip
def test_entry_registers_matches_ptxas(arch, block, min_blocks):
    """The formula equals ptxas's "Used N registers" with setmaxnreg present, and
    ptxas keeps setmaxnreg (no C7508, USETMAXREG in SASS). Needs NVRTC, not a GPU."""
    pytest.importorskip("cuda.bindings.nvrtc")
    log, sass = _ptxas_log_and_sass(_ws_kernel(block, {"min_blocks_per_sm": min_blocks}), arch)
    used = int(re.search(r"Used (\d+) registers", log).group(1))
    # warpgroup 0 decreases to 40; the others (if any) increase to 232
    inc = (232,) if block > 128 else ()
    assert used == entry_registers(block, min_blocks, inc_targets=inc)
    assert "C7508" not in log
    if sass:
        assert "SETMAXREG" in sass
