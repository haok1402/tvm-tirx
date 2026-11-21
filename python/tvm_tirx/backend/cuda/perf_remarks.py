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
"""Performance remarks from the CUDA device compiler's log.

ptxas reports problems that leave a kernel correct but slow: WGMMA pipelines it had
to serialize (``(C75xx)`` remarks such as C7511/C7514/C7517), register spills, and
local-memory stack frames. NVRTC and nvcc return them in the compile log, which
:func:`tvm_tirx.backend.cuda.compiler.compile_source` now parses into
:class:`PerfRemark` objects. ``backend_config={"cuda": {"perf_remarks": ...}}`` picks
what happens next: ``"warn"`` (default) emits a :class:`TIRxPerfWarning` per remark,
``"error"`` raises :class:`PerfRemarkError`, and ``"ignore"`` only records them on the
:class:`~tvm_tirx.backend.cuda.compiler.CompilationResult`.

Spill and stack-frame sizes come from ptxas ``-v`` (on by default in the CUDA
backend config). ``(C75xx)`` remarks are printed without ``-v``.
"""

from __future__ import annotations

import dataclasses
import re
import warnings

PERF_REMARK_MODES = ("warn", "error", "ignore")

_CODE = re.compile(r"\((C75\d\d)\)\s*(.*)")
_FUNCTION_IN_MESSAGE = re.compile(r"function '([^']+)'")
_PROPERTIES_FOR = re.compile(r"Function properties for (\S+)")
_PROPERTIES = re.compile(
    r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads"
)


class TIRxPerfWarning(UserWarning):
    """A performance remark reported by the CUDA device compiler."""


class PerfRemarkError(RuntimeError):
    """Raised for performance remarks when ``perf_remarks`` is ``"error"``."""

    def __init__(self, remarks, arch=None):
        self.remarks = tuple(remarks)
        where = f" for {arch}" if arch else ""
        lines = "\n".join(f"  - {remark}" for remark in self.remarks)
        super().__init__(
            f"PerfRemarkError: ptxas reported {len(self.remarks)} performance remark(s){where} "
            f"with perf_remarks='error':\n{lines}"
        )


@dataclasses.dataclass(frozen=True)
class PerfRemark:
    """One performance remark.

    ``kind`` is ``"ptxas"`` for a ``(C75xx)`` remark (``code`` holds its id),
    ``"spill"`` for register spills, or ``"stack"`` for a non-zero stack frame
    without spills. ``function`` is the kernel or device function, when ptxas
    names it.
    """

    kind: str
    message: str
    function: str | None = None
    code: str | None = None

    def __str__(self):
        tag = self.code or self.kind
        where = f" in {self.function}" if self.function else ""
        return f"[{tag}]{where}: {self.message}"


def parse_compiler_log(log: str) -> tuple[PerfRemark, ...]:
    """Extract performance remarks from an NVRTC or nvcc compile log."""
    remarks = []
    function = None
    for line in (log or "").splitlines():
        code = _CODE.search(line)
        if code:
            named = _FUNCTION_IN_MESSAGE.search(code.group(2))
            remarks.append(
                PerfRemark(
                    kind="ptxas",
                    message=" ".join(code.group(2).split()),
                    function=named.group(1) if named else None,
                    code=code.group(1),
                )
            )
            continue
        header = _PROPERTIES_FOR.search(line)
        if header:
            function = header.group(1)
            continue
        props = _PROPERTIES.search(line)
        if props:
            stack, stores, loads = (int(value) for value in props.groups())
            if stores or loads:
                remarks.append(
                    PerfRemark(
                        kind="spill",
                        message=(
                            f"{stores} bytes spill stores, {loads} bytes spill loads "
                            f"({stack} bytes stack frame)"
                        ),
                        function=function,
                    )
                )
            elif stack:
                remarks.append(
                    PerfRemark(
                        kind="stack",
                        message=f"{stack} bytes stack frame (local memory)",
                        function=function,
                    )
                )
    return tuple(remarks)


def report_perf_remarks(remarks, mode="warn", *, arch=None):
    """Apply a ``perf_remarks`` policy to parsed remarks."""
    if mode not in PERF_REMARK_MODES:
        raise ValueError(f"perf_remarks must be one of {PERF_REMARK_MODES}, got {mode!r}")
    if not remarks or mode == "ignore":
        return
    if mode == "error":
        raise PerfRemarkError(remarks, arch)
    for remark in remarks:
        warnings.warn(f"ptxas {remark}", TIRxPerfWarning, stacklevel=3)
