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
"""Launch-contract checks on CUDA device functions before code generation."""

from __future__ import annotations

from tvm_ffi import structural_walk

from tvm_tirx import ir

KERNEL_ATTRS = "cuda.kernel_attrs"
_REGISTER_BOUNDS = ("min_blocks_per_sm", "max_registers_per_thread")
REGISTER_FILE = 65536  # 32-bit registers per SM, sm_90 and sm_100 families
MAX_REGISTERS_PER_THREAD = 255


class SetMaxNRegContractError(ValueError):
    """A kernel uses ``setmaxnreg`` without a pinned entry register count."""


def entry_registers(
    threads: int,
    min_blocks_per_sm: int | None = None,
    max_registers_per_thread: int | None = None,
    inc_targets=(),
):
    """Per-thread registers at kernel entry when the count is pinned, else None.

    ``max_registers_per_thread`` (``__maxnreg__``) pins it directly. With
    ``min_blocks_per_sm`` (``__launch_bounds__(threads, min_blocks)``), ptxas fits
    ``min_blocks`` CTAs into each of the SM's 4 sub-partitions (16K registers
    each), so the cap is ``floor8(512 / ceil(warps * min_blocks / 4))``, at most
    255. If that reaches 256 and the kernel raises registers with
    ``setmaxnreg.inc``, ptxas uses the largest ``inc`` target instead.

    Measured with CUDA 13.2 ptxas for 128 to 640 threads x 1 to 3 blocks:
    384 x 1 -> 168, 384 x 2 -> 80, 320 x 1 -> 168, 256 x 2 -> 128, 640 x 1 -> 96,
    128 x 1 -> 255, 256 x 1 with inc 232 -> 232.

    The result times ``threads`` is the pool ``setmaxnreg`` redistributes. The
    warpgroups' targets must fit in it, or ``setmaxnreg.inc`` waits forever.
    """
    if max_registers_per_thread is not None:
        return int(max_registers_per_thread)
    if min_blocks_per_sm is None:
        return None
    warps = -(-threads // 32)
    per_subpartition = -(-(warps * min_blocks_per_sm) // 4)
    cap = 512 // per_subpartition
    if cap >= 256:
        return max(inc_targets) if inc_targets else MAX_REGISTERS_PER_THREAD
    return cap // 8 * 8


def _setmaxnreg_calls(func):
    calls = []

    def visit(call):
        op = call.op
        if isinstance(op, ir.Op) and op.name.startswith("tirx.ptx.setmaxnreg"):
            calls.append(call)

    structural_walk(func, (ir.Call, visit))
    return calls


def verify_setmaxnreg_launch_contract(mod):
    """Reject kernels whose ``setmaxnreg`` ptxas would silently drop.

    ptxas only honours ``setmaxnreg`` when the register count at kernel entry is
    pinned by ``.minnctapersm`` or ``.maxnreg``. A kernel with only
    ``__launch_bounds__(threads)`` (TIRx's default) gets remark C7508,
    "'setmaxnreg' ignored; unable to determine register count at entry", and the
    instruction disappears from SASS. The warp-specialized register split then
    never happens. This holds on sm_90a and sm_100a alike.
    """
    for gvar, func in mod.functions.items():
        calls = _setmaxnreg_calls(func)
        if not calls:
            continue
        attrs = func.attrs.get(KERNEL_ATTRS) if func.attrs is not None else None
        if attrs is not None and any(key in attrs for key in _REGISTER_BOUNDS):
            continue
        loc = getattr(calls[0], "loc", None)
        where = f" (first use: {loc})" if loc is not None and str(loc) not in ("", "None") else ""
        raise SetMaxNRegContractError(
            f"kernel {gvar.name_hint!r} uses setmaxnreg{where} but does not pin its entry register "
            "count, so ptxas would ignore it (C7508) and drop it from SASS. Add "
            "kernel_attrs=T.cuda.KernelAttributes(min_blocks_per_sm=1) (or "
            "max_registers_per_thread=...) to T.device_entry."
        )
