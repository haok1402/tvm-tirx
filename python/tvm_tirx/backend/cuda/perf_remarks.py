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
``"error"`` raises :class:`PerfRemarkError`, and ``"ignore"`` only records them.

The remarks are structured end to end:

* :func:`~tvm_tirx.backend.cuda.compiler.compile_source` returns them on its
  :class:`~tvm_tirx.backend.cuda.compiler.CompilationResult`;
* :func:`tvm_tirx.compile` collects the remarks of every CUDA device compile it
  triggers into ``Executable.perf_remarks``. In error mode it raises a
  :class:`PerfRemarkError` whose ``remarks`` holds them, even though the compile
  itself runs inside the C++ module factory.

Users who run with ``-W error`` (warnings as errors) should set
``perf_remarks="ignore"`` or filter :class:`TIRxPerfWarning`; otherwise a kernel
with a remark fails to compile.

Spill and stack-frame sizes come from ptxas ``-v`` (on by default in the CUDA
backend config). ``(C75xx)`` remarks are printed without ``-v``.
"""

from __future__ import annotations

import contextlib
import contextvars
import dataclasses
import re
import warnings

PERF_REMARK_MODES = ("warn", "error", "ignore")

_CODE = re.compile(r"\((C75\d\d)\)\s*(.*)")
_FUNCTION_IN_MESSAGE = re.compile(r"function '([^']+)'")
_COMPILING = re.compile(r"Compiling entry function '([^']+)' for '(sm_\w+)'")
_PROPERTIES_FOR = re.compile(r"Function properties for (\S+)")
_PROPERTIES = re.compile(
    r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads"
)


class TIRxPerfWarning(UserWarning):
    """A performance remark reported by the CUDA device compiler."""


class PerfRemarkError(RuntimeError):
    """Raised for performance remarks when ``perf_remarks`` is ``"error"``.

    ``remarks`` holds the :class:`PerfRemark` records. :func:`tvm_tirx.compile`
    re-raises this type with its remarks even when the compile ran inside the
    C++ CUDA module factory.
    """

    def __init__(self, remarks_or_message=(), arch=None):
        if isinstance(remarks_or_message, str):  # rebuilt from an FFI error message
            super().__init__(remarks_or_message)
            self.remarks = ()
            return
        self.remarks = tuple(remarks_or_message)
        where = f" for {arch}" if arch else ""
        lines = "\n".join(f"  - {remark}" for remark in self.remarks)
        super().__init__(
            f"ptxas reported {len(self.remarks)} performance remark(s){where} "
            f"with perf_remarks='error':\n{lines}"
        )


@dataclasses.dataclass(frozen=True)
class PerfRemark:
    """One performance remark.

    ``kind`` is ``"ptxas"`` for a ``(C75xx)`` remark (``code`` holds its id),
    ``"spill"`` for register spills, or ``"stack"`` for a non-zero stack frame
    without spills. ``function`` is the kernel or device function and ``arch``
    the target it was compiled for, when the log says so (a multi-arch nvcc build
    reports each arch separately).
    """

    kind: str
    message: str
    function: str | None = None
    code: str | None = None
    arch: str | None = None

    def __str__(self):
        tag = self.code or self.kind
        where = f" in {self.function}" if self.function else ""
        arch = f" ({self.arch})" if self.arch else ""
        return f"[{tag}]{where}{arch}: {self.message}"


def parse_compiler_log(log: str, arch: str | None = None) -> tuple[PerfRemark, ...]:
    """Extract performance remarks from an NVRTC or nvcc compile log.

    ptxas prints a target's ``(C75xx)`` remarks before that target's
    ``Compiling entry function 'f' for 'sm_XX'`` headers, and its spill and stack
    sizes after them. So a ``(C75xx)`` remark takes the arch of the next header,
    and a spill or stack remark the function and arch of the previous one. A
    ``(C75xx)`` remark that names no function (C7508) is attributed only when the
    log has a single entry function. ``arch`` is the default when the log has no
    headers (e.g. without ``ptxas -v``).
    """
    lines = (log or "").splitlines()
    headers = [
        (i, m.group(1), m.group(2))
        for i, line in enumerate(lines)
        if (m := _COMPILING.search(line))
    ]
    entry_functions = {name for _, name, _ in headers}
    only_function = next(iter(entry_functions)) if len(entry_functions) == 1 else None

    def next_header_arch(i):
        return next((a for j, _, a in headers if j > i), arch)

    def prev_header_arch(i):
        return next((a for j, _, a in reversed(headers) if j < i), arch)

    remarks = []
    function = None
    for i, line in enumerate(lines):
        code = _CODE.search(line)
        if code:
            named = _FUNCTION_IN_MESSAGE.search(code.group(2))
            remarks.append(
                PerfRemark(
                    kind="ptxas",
                    message=" ".join(code.group(2).split()),
                    function=named.group(1) if named else only_function,
                    code=code.group(1),
                    arch=next_header_arch(i),
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
                message = (
                    f"{stores} bytes spill stores, {loads} bytes spill loads "
                    f"({stack} bytes stack frame)"
                )
                remarks.append(PerfRemark("spill", message, function, arch=prev_header_arch(i)))
            elif stack:
                message = f"{stack} bytes stack frame (local memory)"
                remarks.append(PerfRemark("stack", message, function, arch=prev_header_arch(i)))
    return tuple(remarks)


# -- collection across the C++ module factory --------------------------------------
#
# tvm_tirx.compile reaches compile_source through a C++ callback, which keeps only
# an exception's message. The callback runs synchronously on the caller's thread,
# so a context variable carries the structured remarks (and the error-mode failure)
# back to the Python caller. Collectors nest: each active one sees every remark.

_COLLECTORS: contextvars.ContextVar[tuple] = contextvars.ContextVar(
    "tvm_tirx_perf_remark_collectors", default=()
)


@dataclasses.dataclass(eq=False)
class RemarkCollector:
    """Remarks recorded by every CUDA device compile while the collector is active."""

    remarks: list = dataclasses.field(default_factory=list)
    error: PerfRemarkError | None = None


@contextlib.contextmanager
def collect_perf_remarks():
    """Collect the remarks of the device compiles run inside this ``with`` block.

    Blocks may nest (``tvm_tirx.compile`` opens one of its own); every active
    collector records every remark.
    """
    collector = RemarkCollector()
    token = _COLLECTORS.set((*_COLLECTORS.get(), collector))
    try:
        yield collector
    finally:
        _COLLECTORS.reset(token)


def _active_collectors():
    return _COLLECTORS.get()


def report_perf_remarks(remarks, mode="warn", *, arch=None):
    """Apply a ``perf_remarks`` policy to parsed remarks, and record them for any
    active :func:`collect_perf_remarks` block."""
    if mode not in PERF_REMARK_MODES:
        raise ValueError(f"perf_remarks must be one of {PERF_REMARK_MODES}, got {mode!r}")
    for collector in _active_collectors():
        collector.remarks.extend(remarks)
    if not remarks or mode == "ignore":
        return
    if mode == "error":
        error = PerfRemarkError(remarks, arch)
        for collector in _active_collectors():
            collector.error = error
        raise error
    for remark in remarks:
        warnings.warn(f"ptxas {remark}", TIRxPerfWarning, stacklevel=3)


def _register_ffi_error():
    """Let tvm-ffi rebuild PerfRemarkError (type and message) on paths without a collector."""
    try:
        import tvm_ffi  # pylint: disable=import-outside-toplevel

        tvm_ffi.register_error("PerfRemarkError", PerfRemarkError)
    except Exception:  # pylint: disable=broad-except
        pass


_register_ffi_error()
