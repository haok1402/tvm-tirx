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
"""nvcc builds architecture-specific targets without a base PTX pass (sm90-probe ISSUE-014).

``nvcc --fatbin -arch sm_90a`` also compiles a portable ``compute_90`` PTX, which rejects
``wgmma.fence`` and every other sm_90a-only instruction. No GPU is needed.
"""

import re
import shutil
import subprocess
import tempfile

import numpy as np
import pytest

import tvm_tirx
import tvm_tirx.testing
from tvm_tirx.backend.cuda.compiler import compile_source
from tvm_tirx.script import tirx as T
from tvm_tirx.support.nvcc import _nvcc_arch_args
from tvm_tirx.testing import env

pytestmark = pytest.mark.skipif(shutil.which("nvcc") is None, reason="needs nvcc")

WGMMA_FENCE = (
    'extern "C" __global__ void k(float* a) { asm volatile("wgmma.fence.sync.aligned;"); '
    "a[0] = 1.f; }"
)
PLAIN = 'extern "C" __global__ void k(float* a) { a[0] = 1.f; }'


@pytest.mark.parametrize(
    "arch, target_format, args",
    [
        ("sm_90a", "fatbin", ["-gencode", "arch=compute_90a,code=[sm_90a,compute_90a]"]),
        ("sm_100f", "fatbin", ["-gencode", "arch=compute_100f,code=[sm_100f,compute_100f]"]),
        ("sm_90", "fatbin", ["-arch", "sm_90"]),
        ("sm_90a", "cubin", ["-arch", "sm_90a"]),
        ("sm_90a", "ptx", ["-arch", "sm_90a"]),
    ],
)
def test_arch_args(arch, target_format, args):
    assert _nvcc_arch_args(arch, target_format) == args


def _fatbin_contents(binary):
    with tempfile.NamedTemporaryFile(suffix=".fatbin") as f:
        f.write(binary)
        f.flush()
        listing = subprocess.run(
            ["cuobjdump", "-lelf", "-lptx", f.name], capture_output=True, text=True, check=True
        ).stdout
    return sorted(re.findall(r"\.(sm_\w+)\.(cubin|ptx)", listing))


@pytest.mark.parametrize("target_format", ["fatbin", "cubin", "ptx"])
def test_wgmma_fence_builds_with_nvcc(target_format):
    config = {"cuda": {"compiler": "nvcc", "arch": "sm_90a", "target_format": target_format}}
    result = compile_source(WGMMA_FENCE, config)
    assert result.binary
    if target_format == "fatbin" and shutil.which("cuobjdump"):
        # The sm_90a code and its PTX, and no base sm_90 PTX.
        assert _fatbin_contents(result.binary) == [("sm_90a", "cubin"), ("sm_90a", "ptx")]


@pytest.mark.parametrize("arch", ["sm_100a", "sm_100f"])
def test_other_arch_specific_fatbins_have_no_base_ptx(arch):
    if not shutil.which("cuobjdump"):
        pytest.skip("needs cuobjdump")
    config = {"cuda": {"compiler": "nvcc", "arch": arch, "target_format": "fatbin"}}
    result = compile_source(PLAIN, config)
    contents = _fatbin_contents(result.binary)
    # One cubin (nvcc lists a family-specific sm_100f cubin as sm_100), and only the
    # target's own PTX: no base compute_100 PTX.
    assert [kind for _, kind in contents].count("cubin") == 1, contents
    assert [name for name, kind in contents if kind == "ptx"] == [arch], contents


def _wgmma_fence_module(arch="sm_90a"):
    config = {"cuda": {"compiler": "nvcc", "arch": arch}}

    @T.function
    def k(A: T.Tensor((32,), "float32")):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=128), backend_config=config)
        T.ptx.wgmma.fence.sync.aligned()
        A[0] = T.float32(1)

    target = tvm_tirx.target.Target({"kind": "cuda", "arch": arch})
    return tvm_tirx.compile(tvm_tirx.IRModule({"main": k}), target=target, pipeline="tirx")


def test_tirx_kernel_with_wgmma_fence_compiles_with_nvcc():
    assert _wgmma_fence_module().mod.imports[0].inspect_source()


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda() or env.cuda_arch() != "sm_90a", reason="needs an sm_90a GPU")
def test_nvcc_fatbin_runs():
    module = _wgmma_fence_module()
    a = tvm_tirx.runtime.tensor(np.zeros(32, "float32"), device=tvm_tirx.cuda(0))
    module(a)
    assert a.numpy()[0] == 1


if __name__ == "__main__":
    tvm_tirx.testing.main()
