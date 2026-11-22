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
"""VerifyPTXTarget: reject PTX the target arch cannot run, before NVRTC, at its TIRx source line.

Every ``tirx.ptx.<family>`` call in a CUDA device function is decoded as codegen decodes it
(``ptx.engine._make_codegen``: the modifier tokens sit just before the trailing flag marker). It is
checked against the measured per-variant floors in ``ptx/floors_nvrtc_*.py``. Without this, a
kernel written for sm_100 and compiled for sm_90a fails deep inside NVRTC, as a PTX line number with
no TIRx location; with it, it fails here, naming the instruction and the TIRx source line.

The check only rejects what the production compile path rejects today: an instruction, token
combination or target arch the floor data does not cover is not checked.
"""

from __future__ import annotations

import tvm_ffi

from tvm_tirx.ir import Call, Stmt
from tvm_tirx.ir.location import CallSiteLoc
from tvm_tirx.ir.transform import module_pass
from tvm_tirx.tirx import Function


@tvm_ffi.register_error
class PTXTargetError(ValueError):
    """A PTX instruction the compilation target cannot run, with its TIRx source location.

    Registered with tvm-ffi so it keeps its type when raised inside the pass pipeline.
    """


def _source_line(loc) -> str | None:
    source = getattr(loc, "source_name", None)
    line = getattr(loc, "start_line", None)
    if source is None or line is None:
        return None
    column = getattr(loc, "start_column", None)
    return f"{getattr(source, 'name', source)}:{line}" + (f":{column}" if column else "")


def _location(node) -> str | None:
    """``file:line:col`` of a node's source location, or None if it has none.

    A call traced inside a helper carries a call-site chain (innermost frame first); it is shown
    as ``helper.py:10 <- kernel.py:42``, ending at the kernel's own line.
    """
    loc, lines = getattr(node, "loc", None), []
    while isinstance(loc, CallSiteLoc):
        lines.append(_source_line(loc.callee))
        loc = loc.caller
    lines.append(_source_line(loc))
    lines = [line for line in lines if line]
    if not lines:
        return None
    return " <- ".join(lines if len(lines) <= 4 else [*lines[:2], "...", *lines[-1:]])


def _instruction(entry, tokens) -> str:
    mnemonic = entry.mnemonic or entry.ptx_name
    return ".".join([mnemonic, *(t for t in tokens if t)])


def _target_arch(func) -> str | None:
    target = func.attrs.get("target") if func.attrs is not None else None
    if target is None or getattr(target, "kind", None) is None:
        return None
    if str(target.kind.name) != "cuda":
        return None
    arch = target.attrs.get("arch")
    return None if arch is None else str(arch)


def find_violations(func, arch):
    """``[(location, instruction, reason)]`` for the PTX calls in ``func`` that ``arch`` rejects."""
    from tvm_tirx.backend.cuda.codegen.utils import parse_str  # pylint: disable=import-outside-toplevel
    from tvm_tirx.backend.cuda.ptx import floors  # pylint: disable=import-outside-toplevel
    from tvm_tirx.backend.cuda.ptx.table import TABLE  # pylint: disable=import-outside-toplevel

    if not floors.checked(arch):
        return []
    found = {}  # ordered, without repeats
    # Calls built by lowering (tile patterns) often carry no location of their own: report the
    # innermost enclosing statement or call that does, i.e. the TIRx line the call came from.
    enclosing = ["<unknown location>"]

    def check(call):
        name = getattr(call.op, "name", "")
        if not name.startswith("tirx.ptx."):
            return
        entry = TABLE.get(name.removeprefix("tirx.ptx."))
        if entry is None:
            return
        n_slots = len(entry.slots)
        args = list(call.args)
        if len(args) < n_slots + 1:
            return
        try:
            tokens = [parse_str(a) for a in args[len(args) - n_slots - 1 : -1]]
        except Exception:
            return
        reason = floors.rejection(entry, tokens, arch)
        if reason is not None:
            found.setdefault(
                (_location(call) or enclosing[-1], _instruction(entry, tokens), reason)
            )

    def visit(node, visitor):
        if isinstance(node, Call):
            check(node)
        location = _location(node)
        if location is not None:
            enclosing.append(location)
        try:
            return visitor.default_visit(node)
        finally:
            if location is not None:
                enclosing.pop()

    tvm_ffi.structural_visit(func.body, [((Stmt, Call), visit)])
    return list(found)


def VerifyPTXTarget():
    """Raise :class:`PTXTargetError` for PTX instructions the device target arch cannot run.

    Run on device functions after lowering (in ``finalize_device_passes``), so every PTX call that
    reaches codegen is seen. It does not modify the module.
    """

    @module_pass(opt_level=0, name="VerifyPTXTarget")
    def _pass(mod, _ctx):
        errors = []
        for gvar, func in mod.functions.items():
            if not isinstance(func, Function):
                continue
            arch = _target_arch(func)
            if arch is None:
                continue
            for location, instruction, reason in find_violations(func, arch):
                errors.append(
                    f"{location}: `{instruction}` {reason}; the target is {arch} "
                    f"(in {gvar.name_hint})"
                )
        if errors:
            shown = errors[:20] + ([f"... and {len(errors) - 20} more"] if len(errors) > 20 else [])
            raise PTXTargetError(
                "PTX instructions the target architecture cannot run:\n  " + "\n  ".join(shown)
            )
        return mod

    return _pass
