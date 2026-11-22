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
at each checked arch), so a check against it rejects exactly what that compile path rejects today,
and nothing else:

* only families whose ``cert_arch`` (the maximum floor over their variants) is newer than sm_90 are
  measured: every other family runs on all sm_90+ targets, from the table's own metadata. So an
  sm_90-only variant of such a family (e.g. a ``.cluster``-scoped ``ld``) compiled for sm_80 or
  sm_89 is not caught here and still fails later, in NVRTC;
* a family, a token combination or a target arch the data does not cover is **unchecked**;
* a family can override the generated data with ``InstructionEntry.floor`` where the ISA is stricter
  than ptxas (none does today);
* the data holds for the toolchain it was measured with: the NVRTC version and the CUDA runtime
  headers NVRTC includes. Each measured toolchain has its own generated data module,
  ``floors_nvrtc_<major>_<minor>.py`` (its ``TOOLCHAIN``, ``ARCHS`` and ``FLOORS``), and the check
  uses the one that matches this process. Under a toolchain with no data it is skipped, with one
  warning, rather than risk rejecting what that toolchain accepts.
"""

from __future__ import annotations

import importlib
import os
import pkgutil
import warnings
from functools import cache

from .toolchain import current_toolchain

#: Module-name prefix of the generated data modules, one per measured toolchain.
DATA_PREFIX = "floors_nvrtc_"


@cache
def datasets() -> tuple:
    """Every recorded data module (``floors_nvrtc_<major>_<minor>``), in name order."""
    names = sorted(
        info.name
        for info in pkgutil.iter_modules([os.path.dirname(__file__)])
        if info.name.startswith(DATA_PREFIX)
    )
    return tuple(importlib.import_module(f"{__package__}.{name}") for name in names)


@cache
def dataset():
    """The data module measured with this process's NVRTC and CUDA headers, or None."""
    current = current_toolchain()
    for data in datasets():
        if data.TOOLCHAIN == current:
            return data
    recorded = ", ".join(str(data.TOOLCHAIN) for data in datasets()) or "none"
    warnings.warn(
        f"VerifyPTXTarget is disabled: the PTX arch floors were measured with {recorded}, but "
        f"this toolchain is {current}. Record it with "
        "`python -m tvm_tirx.backend.cuda.ptx.gen_floors`, which writes its own "
        "floors_nvrtc_<major>_<minor>.py next to the others.",
        stacklevel=2,
    )
    return None


def accepted_archs(entry, tokens) -> frozenset[str] | None:
    """The archs that accept ``entry`` with modifier ``tokens``, or None when that is unknown."""
    if entry.floor is not None:
        from .table import mods  # pylint: disable=import-outside-toplevel

        return entry.floor(mods(entry, list(tokens)))
    data = dataset()
    family = None if data is None else data.FLOORS.get(entry.name)
    if family is None:
        return None
    slots, table = family
    mask = table.get(tuple(tokens[i] for i in slots))
    return None if mask is None else _archs(data.ARCHS, mask)


@cache
def _archs(archs: tuple, mask: int) -> frozenset[str]:
    return frozenset(arch for bit, arch in enumerate(archs) if mask >> bit & 1)


def toolchain_matches() -> bool:
    """Whether floor data was recorded for this process's NVRTC and CUDA headers."""
    return dataset() is not None


def checked(arch: str) -> bool:
    """Whether the floor data covers ``arch`` (and applies to this toolchain) at all."""
    data = dataset()
    return data is not None and arch in data.ARCHS


def rejection(entry, tokens, arch: str) -> str | None:
    """Why ``arch`` cannot run this variant, or None if it can (or the data does not say)."""
    if not checked(arch):
        return None
    archs = accepted_archs(entry, tokens)
    if archs is None or arch in archs:
        return None
    archs_checked = dataset().ARCHS
    ordered = [a for a in archs_checked if a in archs]
    if ordered:
        return f"requires one of {', '.join(ordered)}"
    newer = f" such as {entry.cert_arch}" if entry.cert_arch else ""
    every = ", ".join(archs_checked)
    return f"requires a newer target{newer} (rejected on every checked arch: {every})"
