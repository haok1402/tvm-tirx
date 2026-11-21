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
"""ptxas performance remarks: parsing, warn/error/ignore policy, and real NVRTC compiles."""

import os
import warnings

import pytest

import tvm_tirx
from tvm_tirx.backend.cuda.backend_config import validate_backend_config
from tvm_tirx.backend.cuda.perf_remarks import (
    PerfRemark,
    PerfRemarkError,
    TIRxPerfWarning,
    parse_compiler_log,
    report_perf_remarks,
)
from tvm_tirx.script import tirx as T
from tvm_tirx.testing import env

# Logs as printed by CUDA 13.2 NVRTC (ptxas -v) for sm_90a.
_SERIALIZED_LOG = "\n".join(
    [
        'tvm_kernels.cu(12): warning #177-D: variable "bx" was declared but never referenced',
        "ptxas info    : (C7517) wgmma.wait_group is injected in around line 646 by compiler"
        " to allow use of registers defined by GMMA in function 'main_kernel'",
        "ptxas info    : (C7511) Potential Performance Loss: wgmma.mma_async instructions are"
        " serialized due to insufficient register resources for the wgmma pipeline in the"
        " function 'main_kernel' in line 650",
        "ptxas info    : 1100 bytes gmem",
        "ptxas info    : Compiling entry function 'main_kernel' for 'sm_90a'",
        "ptxas info    : Function properties for main_kernel",
        "ptxas         .     0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads",
        "ptxas info    : Used 90 registers, used 0 barriers, 24576 bytes smem",
    ]
)
_SPILL_LOG = """\
ptxas warning : Local memory used for function 'main_kernel', size of stack frame: 504 bytes
ptxas info    : Compiling entry function 'main_kernel' for 'sm_90a'
ptxas info    : Function properties for main_kernel
ptxas         .     504 bytes stack frame, 964 bytes spill stores, 964 bytes spill loads
ptxas info    : Used 32 registers, used 0 barriers, 504 bytes cumulative stack size
"""
_STACK_LOG = """\
ptxas info    : Function properties for helper_fn
    1072 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Function properties for main_kernel
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
"""
_CLEAN_LOG = """\
ptxas info    : Compiling entry function 'main_kernel' for 'sm_90a'
ptxas info    : Function properties for main_kernel
ptxas         .     0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 90 registers, used 0 barriers, 24576 bytes smem
"""


def test_parse_wgmma_serialization_remarks():
    remarks = parse_compiler_log(_SERIALIZED_LOG)
    assert [(r.kind, r.code, r.function) for r in remarks] == [
        ("ptxas", "C7517", "main_kernel"),
        ("ptxas", "C7511", "main_kernel"),
    ]
    assert "serialized due to insufficient register resources" in remarks[1].message
    assert str(remarks[1]).startswith("[C7511] in main_kernel: Potential Performance Loss")


def test_parse_spills_and_stack_frames():
    (spill,) = parse_compiler_log(_SPILL_LOG)
    assert spill == PerfRemark(
        kind="spill",
        message="964 bytes spill stores, 964 bytes spill loads (504 bytes stack frame)",
        function="main_kernel",
    )
    (stack,) = parse_compiler_log(_STACK_LOG)
    assert (stack.kind, stack.function) == ("stack", "helper_fn")
    assert "1072 bytes stack frame" in stack.message


def test_parse_clean_and_empty_logs():
    assert parse_compiler_log(_CLEAN_LOG) == ()
    assert parse_compiler_log("") == ()
    assert parse_compiler_log(None) == ()


def test_report_policy():
    remarks = parse_compiler_log(_SERIALIZED_LOG)
    with pytest.warns(TIRxPerfWarning, match=r"\[C7511\] in main_kernel"):
        report_perf_remarks(remarks, "warn")
    with pytest.raises(PerfRemarkError, match=r"2 performance remark\(s\) for sm_90a") as info:
        report_perf_remarks(remarks, "error", arch="sm_90a")
    assert info.value.remarks == remarks
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        report_perf_remarks(remarks, "ignore")
        report_perf_remarks((), "error")
    with pytest.raises(ValueError, match="perf_remarks"):
        report_perf_remarks(remarks, "loud")


def test_backend_config_validates_perf_remarks():
    assert validate_backend_config({"perf_remarks": "error"}) == {"perf_remarks": "error"}
    with pytest.raises(ValueError, match="perf_remarks"):
        validate_backend_config({"perf_remarks": "loud"})
    with pytest.raises(TypeError):
        validate_backend_config({"perf_remarks": True})


# ---------------------------------------------------------------------------
# Real compiles. They need NVRTC, not a GPU: every kernel is compile-only, and
# sm_90a is a valid NVRTC target on any machine with the CUDA toolkit.
# ---------------------------------------------------------------------------

_N = 128
_ACC = _N // 2
_MMA = f"wgmma.mma_async.sync.aligned.m64n{_N}k16.f32.bf16.bf16"


def _wgmma_chain(hazard, k_steps=2, backend_config=None):
    """One warpgroup and one WGMMA chain of ``k_steps`` MMAs. With ``hazard``, a
    non-wgmma instruction writes an accumulator between the first two MMAs, which
    forces ptxas to serialize the chain."""

    # fmt: off
    @T.function
    def main(Out: T.Tensor((128, _ACC), "float32")):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=128), backend_config=backend_config)
        tx = T.cuda.thread_idx("x")
        A_smem = T.alloc_shared((64, 64), "bfloat16", align=1024)
        B_smem = T.alloc_shared((_N, 64), "bfloat16", align=1024)
        descA: T.uint64
        descB: T.uint64
        acc = T.alloc_tensor(ty_args=[T.Tensor((_ACC,), "float32", scope="local")])
        for i in T.serial(0, _ACC):
            acc[i] = T.float32(0.0)
        T.cuda.wgmma.encode_matrix_descriptor(T.address_of(descA), A_smem.data, 1, 64, 3)  # noqa: F821
        T.cuda.wgmma.encode_matrix_descriptor(T.address_of(descB), B_smem.data, 1, 64, 3)  # noqa: F821
        T.ptx.wgmma.fence.sync.aligned()
        for k in T.unroll(0, k_steps):
            T.ptx[_MMA](*[acc[i] for i in range(_ACC)], descA + T.uint64(2 * k), descB + T.uint64(2 * k), True, 1, 1, 0, 0)  # noqa: E501, F821
            if T.constexpr(hazard):
                if k == 0:
                    acc[0] = acc[0] + T.Cast("float32", tx >> 31)
        T.ptx.wgmma.commit_group.sync.aligned()
        T.ptx.wgmma.wait_group.sync.aligned(0)
        for i in T.serial(0, _ACC):
            Out[tx, i] = acc[i]
    # fmt: on
    return main


def _register_hog(min_blocks_per_sm):
    """128 live floats per thread. 16 blocks of 128 threads cap ptxas at 32 registers."""

    # fmt: off
    @T.function
    def main(A: T.Tensor((128, 128), "float32"), Out: T.Tensor((128, 128), "float32")):
        T.device_entry(
            launch=T.cuda.LaunchConfig(grid=1, block=128),
            kernel_attrs=T.cuda.KernelAttributes(min_blocks_per_sm=min_blocks_per_sm),
        )
        tx = T.cuda.thread_idx("x")
        v = T.alloc_local((128,), "float32")
        for i in T.unroll(0, 128):
            v[i] = A[tx, i]
        for _ in T.serial(0, 4):
            for i in T.unroll(0, 128):
                v[i] = v[i] * v[(i + 1) % 128] + v[(i + 7) % 128]
        for i in T.unroll(0, 128):
            Out[tx, i] = v[i]
    # fmt: on
    return main


def _cuda_source(func):
    previous = os.environ.get("TVM_TIRX_COMPILE_FORCE_FALLBACK")
    os.environ["TVM_TIRX_COMPILE_FORCE_FALLBACK"] = "1"
    try:
        target = tvm_tirx.target.Target({"kind": "cuda", "arch": "sm_90a"})
        mod = tvm_tirx.compile(tvm_tirx.IRModule({"main": func}), target=target, pipeline="tirx")
    finally:
        if previous is None:
            del os.environ["TVM_TIRX_COMPILE_FORCE_FALLBACK"]
        else:
            os.environ["TVM_TIRX_COMPILE_FORCE_FALLBACK"] = previous
    return mod.mod.imports[0].inspect_source()


def _compile(func, **cuda):
    from tvm_tirx.backend.cuda.compiler import compile_source

    return compile_source(_cuda_source(func), {"cuda": {"arch": "sm_90a", **cuda}})


requires_nvrtc = pytest.mark.skipif(not env.has_cuda(), reason="needs NVRTC (CUDA toolkit)")


@requires_nvrtc
def test_serialized_wgmma_chain_warns_and_raises():
    with pytest.warns(TIRxPerfWarning, match=r"\[C75\d\d\] in main_kernel"):
        result = _compile(_wgmma_chain(hazard=True))
    assert any(r.code and r.code.startswith("C75") for r in result.remarks)
    assert "(C75" in result.log
    with pytest.raises(PerfRemarkError, match=r"serialized"):
        _compile(_wgmma_chain(hazard=True), perf_remarks="error")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert _compile(_wgmma_chain(hazard=True), perf_remarks="ignore").remarks


@requires_nvrtc
@pytest.mark.parametrize("k_steps", [2, 4])
def test_clean_wgmma_chain_has_no_remarks(k_steps):
    with warnings.catch_warnings():
        warnings.simplefilter("error", TIRxPerfWarning)
        result = _compile(_wgmma_chain(hazard=False, k_steps=k_steps), perf_remarks="error")
    assert result.remarks == ()
    assert "Function properties for main_kernel" in result.log


@requires_nvrtc
def test_register_spills_are_reported():
    with pytest.warns(TIRxPerfWarning, match=r"\[spill\] in main_kernel: \d+ bytes spill stores"):
        result = _compile(_register_hog(min_blocks_per_sm=16))
    assert [r.kind for r in result.remarks] == ["spill"]
    with warnings.catch_warnings():
        warnings.simplefilter("error", TIRxPerfWarning)
        assert _compile(_register_hog(min_blocks_per_sm=1)).remarks == ()


@requires_nvrtc
def test_error_mode_through_tvm_tirx_compile():
    """The CUDA module factory compiles through the same path; error mode stops the build."""
    func = _wgmma_chain(hazard=True, backend_config={"cuda": {"perf_remarks": "error"}})
    target = tvm_tirx.target.Target({"kind": "cuda", "arch": "sm_90a"})
    with pytest.raises(Exception, match=r"PerfRemarkError: ptxas reported"):
        tvm_tirx.compile(tvm_tirx.IRModule({"main": func}), target=target, pipeline="tirx")
