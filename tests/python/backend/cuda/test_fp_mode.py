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
"""Per-kernel floating-point mode: backend_config["cuda"]["fp_mode"] (sm90-probe ISSUE-010)."""

import numpy as np
import pytest

import tvm_tirx
import tvm_tirx.testing
from tvm_tirx.backend.config import merge_backend_configs, parse_backend_config
from tvm_tirx.backend.cuda.backend_config import (
    compiler_options,
    fp_mode_flags,
    resolve_backend_config,
    validate_backend_config,
)
from tvm_tirx.script import tirx as T
from tvm_tirx.testing import env

PRECISE = ["--ftz=false", "--prec-div=true", "--prec-sqrt=true", "--fmad=true"]


@pytest.mark.parametrize(
    "mode, flags",
    [
        ("fast", ["--use_fast_math"]),
        ("precise", PRECISE),
        ({"fast_math": True, "ftz": False}, ["--use_fast_math", "--ftz=false"]),
        ({"fmad": False}, ["--fmad=false"]),
        ({"ftz": True, "prec_div": False}, ["--ftz=true", "--prec-div=false"]),
    ],
)
def test_fp_mode_flags(mode, flags):
    assert fp_mode_flags(mode) == flags


@pytest.mark.parametrize(
    "mode, error",
    [
        ("approximate", ValueError),
        ({"flush": True}, ValueError),
        ({"ftz": 1}, TypeError),
        (["fast"], TypeError),
    ],
)
def test_fp_mode_is_validated(mode, error):
    with pytest.raises(error):
        validate_backend_config({"fp_mode": mode})


def test_default_flags_are_unchanged():
    # No fp_mode anywhere: the argument lists decide, as before.
    config = resolve_backend_config({"arch": "sm_90a"})
    assert "fp_mode" not in config
    assert compiler_options(config)[0] == "--use_fast_math"
    legacy = resolve_backend_config({"arch": "sm_90a", "nvrtc": ["--use_fast_math", "--ftz=false"]})
    assert compiler_options(legacy)[:2] == ["--use_fast_math", "--ftz=false"]


def test_fp_mode_composes_with_the_lists():
    # The default lists' --use_fast_math gives way; other arguments stay.
    config = resolve_backend_config({"arch": "sm_90a", "fp_mode": "precise"})
    assert config["nvrtc"] == config["nvcc"] == []
    assert compiler_options(config)[:4] == PRECISE
    # A caller's list set for another reason keeps its other arguments.
    caller = {"cuda": {"nvrtc": ["--use_fast_math", "-lineinfo"]}}
    kernel = {"cuda": {"fp_mode": "precise"}}
    merged = merge_backend_configs(caller, kernel)["cuda"]
    assert merged["nvrtc"] == ["-lineinfo"]
    options = compiler_options(resolve_backend_config({**merged, "arch": "sm_90a"}))
    assert options[:5] == [*PRECISE, "-lineinfo"]
    assert "--use_fast_math" not in options


@pytest.mark.parametrize(
    "outer, inner",
    [
        # Two explicit modes that differ.
        ({"fp_mode": "fast"}, {"fp_mode": "precise"}),
        # An inner list that spells FP flags under an outer fp_mode.
        ({"fp_mode": "precise"}, {"nvrtc": ["--use_fast_math"]}),
        ({"fp_mode": "precise"}, {"nvcc": ["-ftz=true"]}),
    ],
)
def test_conflicts_are_errors(outer, inner):
    with pytest.raises(ValueError, match="fp_mode"):
        merge_backend_configs({"cuda": outer}, {"cuda": inner})


def test_fp_mode_and_fp_flags_in_one_config_are_an_error():
    with pytest.raises(ValueError, match="fp_mode"):
        validate_backend_config({"fp_mode": "precise", "nvrtc": ["--ftz=false"]})


def test_equal_modes_agree():
    merged = merge_backend_configs(
        {"cuda": {"fp_mode": "fast"}}, {"cuda": {"fp_mode": {"fast_math": True}}}
    )
    assert fp_mode_flags(merged["cuda"]["fp_mode"]) == ["--use_fast_math"]


def _scale_product_kernel(entry_config=None):
    # DeepGEMM's promotion: out += sa * sb * acc. With sa * sb about 1e-40 (an f32 subnormal),
    # fast math (-ftz=true) flushes the product to 0.
    @T.function
    def main(
        sa: T.Tensor((64,), "float32"),
        sb: T.Tensor((64,), "float32"),
        acc: T.Tensor((64,), "float32"),
        out: T.Tensor((64,), "float32"),
    ):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=64), backend_config=entry_config)
        tx = T.cuda.thread_idx("x")
        out[tx] = out[tx] + sa[tx] * sb[tx] * acc[tx]

    return main


def _target(arch, caller=None):
    target = {"kind": "cuda", "arch": arch}
    if caller is not None:
        target["backend_config"] = {"cuda": caller}
    return target


def _ptx(module):
    from tvm_tirx.backend.cuda.compiler import compile_source

    imported = module.mod.imports[0]
    config = parse_backend_config(imported.inspect_source("backend_config"))
    config["cuda"]["target_format"] = "ptx"
    return config, compile_source(imported.inspect_source(), config).binary.decode()


@pytest.mark.parametrize(
    "caller", [None, {"nvrtc": ["--use_fast_math", "--generate-line-info"]}], ids=["", "caller"]
)
def test_precise_kernel_emits_no_ftz(caller):
    kernel = _scale_product_kernel({"cuda": {"fp_mode": "precise"}})
    module = tvm_tirx.compile(kernel, target=_target("sm_90a", caller))
    config, ptx = _ptx(module)
    assert config["cuda"]["fp_mode"] == "precise"
    assert ".ftz" not in ptx, ptx
    if caller is not None:
        assert "--generate-line-info" in config["cuda"]["nvrtc"]
    # The default still flushes.
    _, fast_ptx = _ptx(tvm_tirx.compile(_scale_product_kernel(), target=_target("sm_90a", caller)))
    assert "mul.ftz.f32" in fast_ptx or "fma.rn.ftz.f32" in fast_ptx


def test_caller_fp_mode_conflicting_with_kernel_is_an_error():
    # The error names both sources and both modes.
    kernel = _scale_product_kernel({"cuda": {"fp_mode": "precise"}})
    with pytest.raises(Exception) as info:
        tvm_tirx.compile(kernel, target=_target("sm_90a", {"fp_mode": "fast"}))
    message = str(info.value)
    assert "the device_entry backend_config sets 'precise'" in message, message
    assert "the target's backend_config sets 'fast'" in message, message


def test_function_fp_mode_conflicting_with_entry_is_an_error():
    @T.function
    def main(a: T.Tensor((64,), "float32")):
        T.func_attr({"backend_config": '{"cuda":{"fp_mode":"fast"}}'})
        T.device_entry(
            launch=T.cuda.LaunchConfig(grid=1, block=64),
            backend_config={"cuda": {"fp_mode": "precise"}},
        )
        tx = T.cuda.thread_idx("x")
        a[tx] = a[tx] * T.float32(0.5)

    with pytest.raises(Exception) as info:
        tvm_tirx.compile(main, target=_target("sm_90a"))
    message = str(info.value)
    assert "the device_entry backend_config sets 'precise'" in message, message
    assert "the function's backend_config attribute sets 'fast'" in message, message


def test_target_fp_mode_conflicting_with_function_is_an_error():
    @T.function
    def main(a: T.Tensor((64,), "float32")):
        T.func_attr({"backend_config": '{"cuda":{"fp_mode":"precise"}}'})
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=64))
        tx = T.cuda.thread_idx("x")
        a[tx] = a[tx] * T.float32(0.5)

    with pytest.raises(Exception) as info:
        tvm_tirx.compile(main, target=_target("sm_90a", {"fp_mode": "fast"}))
    message = str(info.value)
    assert "the function's backend_config attribute sets 'precise'" in message, message
    assert "the target's backend_config sets 'fast'" in message, message


def test_fp_flags_under_an_inherited_mode_name_both_sources():
    with pytest.raises(ValueError) as info:
        merge_backend_configs(
            {"cuda": {"fp_mode": "precise"}}, {"cuda": {"nvrtc": ["--use_fast_math"]}}
        )
    assert "'--use_fast_math'" in str(info.value) and "'precise'" in str(info.value)


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="needs a CUDA GPU")
@pytest.mark.parametrize("caller", [None, {"nvrtc": ["--generate-line-info"]}], ids=["", "caller"])
def test_denormal_products_match_reference_bitwise(caller):
    rng = np.random.default_rng(0)
    sa = (rng.uniform(1, 2, 64) * 1e-20).astype("float32")
    sb = (rng.uniform(1, 2, 64) * 1e-20).astype("float32")
    acc = rng.uniform(1, 4, 64).astype("float32")
    dev = tvm_tirx.cuda(0)
    # The reference without fast math: f32 multiply, then FMA (nvcc's default --fmad=true).
    product = sa * sb
    expected = np.array(
        [np.float32(np.float64(p) * np.float64(a)) for p, a in zip(product, acc)], "float32"
    )
    assert np.all(product != 0) and np.all(np.abs(product) < np.finfo("float32").tiny)

    def run(entry_config):
        module = tvm_tirx.compile(
            _scale_product_kernel(entry_config), target=_target(env.cuda_arch(), caller)
        )
        out = tvm_tirx.runtime.tensor(np.zeros(64, "float32"), device=dev)
        args = [tvm_tirx.runtime.tensor(x, device=dev) for x in (sa, sb, acc)]
        module(*args, out)
        return out.numpy()

    np.testing.assert_array_equal(run({"cuda": {"fp_mode": "precise"}}), expected)
    if caller is None:
        # The default flushes the subnormal product.
        np.testing.assert_array_equal(run(None), np.zeros(64, "float32"))


if __name__ == "__main__":
    tvm_tirx.testing.main()
