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
"""Per-variant arch floors of the PTX table: which CUDA archs accept an instruction variant.

The data is measured (``gen_floors.py`` compiles every variant through the production NVRTC path
at each arch of ``floors_generated.ARCHS``), so a check against it rejects exactly what that
compile path rejects today, and nothing else:

* only families whose ``cert_arch`` (the maximum floor over their variants) is newer than sm_90 are
  measured: every other family runs on all sm_90+ targets, from the table's own metadata. So an
  sm_90-only variant of such a family (e.g. a ``.cluster``-scoped ``ld``) compiled for sm_80 or
  sm_89 is not caught here and still fails later, in NVRTC;
* a family, a token combination or a target arch the data does not cover is **unchecked**;
* a family can override the generated data with ``InstructionEntry.floor`` where the ISA is stricter
  than ptxas (none does today);
* the data holds for the toolchain it was measured with (``floors_generated.TOOLCHAIN``: the NVRTC
  version and the CUDA runtime headers NVRTC includes). Under any other toolchain the check is
  skipped, with one warning, rather than risk rejecting what that toolchain accepts.
"""

from __future__ import annotations

import warnings
from functools import cache

from .floors_generated import ARCHS, FLOORS, TOOLCHAIN
from .toolchain import current_toolchain


def accepted_archs(entry, tokens) -> frozenset[str] | None:
    """The archs that accept ``entry`` with modifier ``tokens``, or None when that is unknown."""
    if entry.floor is not None:
        from .table import mods  # pylint: disable=import-outside-toplevel

        return entry.floor(mods(entry, list(tokens)))
    data = FLOORS.get(entry.name)
    if data is None:
        return None
    slots, table = data
    mask = table.get(tuple(tokens[i] for i in slots))
    return None if mask is None else _archs(mask)


@cache
def _archs(mask: int) -> frozenset[str]:
    return frozenset(arch for bit, arch in enumerate(ARCHS) if mask >> bit & 1)


@cache
def toolchain_matches() -> bool:
    """Whether this process's NVRTC and CUDA headers are the ones the floors were measured with."""
    current = current_toolchain()
    if current == TOOLCHAIN:
        return True
    warnings.warn(
        f"VerifyPTXTarget is disabled: the PTX arch floors were measured with {TOOLCHAIN}, but "
        f"this toolchain is {current}. Regenerate them with "
        "`python -m tvm_tirx.backend.cuda.ptx.gen_floors`.",
        stacklevel=2,
    )
    return False


def checked(arch: str) -> bool:
    """Whether the floor data covers ``arch`` (and applies to this toolchain) at all."""
    return arch in ARCHS and toolchain_matches()


def rejection(entry, tokens, arch: str) -> str | None:
    """Why ``arch`` cannot run this variant, or None if it can (or the data does not say)."""
    if not checked(arch):
        return None
    archs = accepted_archs(entry, tokens)
    if archs is None or arch in archs:
        return None
    ordered = [a for a in ARCHS if a in archs]
    if ordered:
        return f"requires one of {', '.join(ordered)}"
    newer = f" such as {entry.cert_arch}" if entry.cert_arch else ""
    return f"requires a newer target{newer} (rejected on every checked arch: {', '.join(ARCHS)})"
