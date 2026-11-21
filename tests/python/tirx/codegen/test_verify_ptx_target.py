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
"""VerifyPTXTarget and the measured per-variant arch floors (CPU only: nothing here runs NVRTC)."""

import pytest

import tvm_tirx
from tvm_tirx.backend.cuda.ptx import floors
from tvm_tirx.backend.cuda.ptx.floors_generated import ARCHS, FLOORS
from tvm_tirx.backend.cuda.ptx.gen_floors import measured_families
from tvm_tirx.backend.cuda.ptx.table import TABLE, variants
from tvm_tirx.backend.cuda.transform import PTXTargetError, VerifyPTXTarget
from tvm_tirx.script import tirx as T


def _target(arch):
    return tvm_tirx.target.Target({"kind": "cuda", "arch": arch})


@pytest.fixture
def no_nvrtc(monkeypatch):
    """Fail if device code reaches the CUDA compiler: the rejection must come first."""
    from tvm_tirx.backend.cuda import compiler  # pylint: disable=import-outside-toplevel

    def unreachable(*_args, **_kwargs):
        raise AssertionError("device code reached NVRTC before VerifyPTXTarget rejected it")

    monkeypatch.setattr(compiler, "compile_source", unreachable)


def test_floor_data_is_consistent_with_the_table():
    assert len(set(ARCHS)) == len(ARCHS) and len(ARCHS) < 64
    assert set(FLOORS) <= set(measured_families(TABLE))
    for family, (slots, table) in FLOORS.items():
        entry = TABLE[family]  # every family in the data still exists
        assert all(0 <= i < len(entry.slots) for i in slots), family
        for key, mask in table.items():
            assert len(key) == len(slots) and 0 <= mask < 1 << len(ARCHS), (family, key)
            for i, token in zip(slots, key):
                slot = entry.slots[i]
                assert token in slot.choices or (slot.optional and token == ""), (family, key)


@pytest.mark.parametrize("family", sorted(FLOORS))
def test_every_family_is_decided_for_sm90a(family):
    """A measured variant is either accepted on sm_90a or rejected there with a reason."""
    entry = TABLE[family]
    decided = 0
    for tokens in variants(entry):
        archs = floors.accepted_archs(entry, tokens)
        if archs is None:
            continue  # rejected by its own cert_arch too (a harness artifact): unchecked
        decided += 1
        reason = floors.rejection(entry, tokens, "sm_90a")
        if "sm_90a" in archs:
            assert reason is None
        else:  # e.g. sm_100a-only: rejected, saying what it needs
            assert reason is not None and "requires" in reason
    assert decided


@pytest.mark.parametrize(
    "family, accepted, rejected",
    [
        ("wgmma_bf16_ss", {"sm_90a"}, {"sm_80", "sm_89", "sm_90"}),
        ("setmaxnreg", {"sm_90a"}, {"sm_80", "sm_89", "sm_90"}),
        ("tcgen05_fence", set(), {"sm_80", "sm_89", "sm_90", "sm_90a"}),
        ("tcgen05_mma_ss", set(), {"sm_80", "sm_89", "sm_90", "sm_90a"}),
    ],
)
def test_measured_floors_match_the_isa(family, accepted, rejected):
    """Spot checks of the measured data against known PTX ISA targets."""
    entry = TABLE[family]
    tokens = next(iter(variants(entry)))
    archs = floors.accepted_archs(entry, tokens)
    assert archs is not None and accepted <= archs and not rejected & archs, sorted(archs)


def test_unknown_arch_or_family_is_unchecked():
    entry = TABLE["tcgen05_fence"]
    tokens = next(iter(variants(entry)))
    for arch in ("sm_100a", "sm_86", "sm_999"):  # not a checked target
        assert not floors.checked(arch) and floors.rejection(entry, tokens, arch) is None


def test_families_without_an_nvrtc_witness_are_not_measured():
    """sm_107*-certified families: NVRTC cannot compile their cert_arch, so nothing vouches for
    a rejection; they stay unchecked."""
    sm107 = [name for name, e in TABLE.items() if (e.cert_arch or "").startswith("sm_107")]
    assert sm107
    assert not set(sm107) & set(measured_families(TABLE)) and not set(sm107) & set(FLOORS)
    entry = TABLE[sm107[0]]
    assert floors.rejection(entry, next(iter(variants(entry))), "sm_90a") is None


@pytest.mark.parametrize("family", ["mbarrier_init", "fence_proxy", "ld", "ld_vec"])
def test_families_certified_at_sm90_or_older_are_not_measured(family):
    """cert_arch is the maximum floor over a family: at or below sm_90 it runs on every sm_90+."""
    entry = TABLE[family]
    assert entry.cert_arch in (None, "sm_75", "sm_80", "sm_90")
    assert family not in measured_families(TABLE) and family not in FLOORS
    tokens = next(iter(variants(entry)))
    assert floors.rejection(entry, tokens, "sm_80") is None


def _setmaxnreg_kernel():
    @T.function
    def kernel(A: T.Tensor((128,), "float32")):
        T.device_entry(
            launch=T.cuda.LaunchConfig(grid=1, block=128),
            kernel_attrs=T.cuda.KernelAttributes(min_blocks_per_sm=1),
        )
        T.ptx["setmaxnreg.inc.sync.aligned.u32"](240)  # sm_90a and later a/f targets only
        A[T.cuda.thread_idx("x")] = T.float32(1)

    return kernel


def _tcgen05_kernel():
    @T.function
    def kernel(A: T.Tensor((128,), "float32")):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=128))
        T.ptx.tcgen05.fence__before_thread_sync()  # sm_100 family only
        A[T.cuda.thread_idx("x")] = T.float32(1)

    return kernel


@pytest.mark.parametrize("arch", ["sm_90", "sm_80"])
def test_rejected_before_nvrtc_with_the_source_line(arch, no_nvrtc):
    mod = tvm_tirx.IRModule({"main": _setmaxnreg_kernel()})
    with pytest.raises(PTXTargetError) as info:
        tvm_tirx.compile(mod, target=_target(arch), pipeline="tirx")
    message = str(info.value)
    assert "setmaxnreg.inc.sync.aligned.u32" in message and f"the target is {arch}" in message
    assert "test_verify_ptx_target.py:" in message  # the TIRx source location of the call
    assert "requires one of sm_90a" in message  # names the arch that does accept it


def test_sm100_instruction_compiled_for_sm90a_is_rejected(no_nvrtc):
    mod = tvm_tirx.IRModule({"main": _tcgen05_kernel()})
    with pytest.raises(PTXTargetError, match=r"tcgen05\.fence::before_thread_sync"):
        tvm_tirx.compile(mod, target=_target("sm_90a"), pipeline="tirx")


def test_the_pass_does_not_modify_the_module():
    mod = tvm_tirx.tirx.transform.BindTarget(_target("sm_90a"))(
        tvm_tirx.IRModule({"main": _setmaxnreg_kernel()})
    )
    assert VerifyPTXTarget()(mod).same_as(mod)


def test_call_site_chains_name_the_kernel_line():
    """A call traced inside a helper carries a CallSiteLoc chain (innermost frame first)."""
    from tvm_tirx.backend.cuda.transform.verify_ptx_target import _location
    from tvm_tirx.ir import SourceLoc, SourceName
    from tvm_tirx.ir.location import CallSiteLoc, UnknownLoc

    def line(name, n):
        return SourceLoc(SourceName(name), n, 1, n, 2)

    class Node:
        def __init__(self, loc):
            self.loc = loc

    chain = CallSiteLoc(line("helper.py", 10), CallSiteLoc(line("mid.py", 20), line("k.py", 30)))
    assert _location(Node(chain)) == "helper.py:10:1 <- mid.py:20:1 <- k.py:30:1"
    assert _location(Node(line("k.py", 7))) == "k.py:7:1"
    assert _location(Node(UnknownLoc())) is None
