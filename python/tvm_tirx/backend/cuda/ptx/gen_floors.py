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
"""Generate ``floors_generated.py``: which CUDA archs accept each PTX table variant.

    python -m tvm_tirx.backend.cuda.ptx.gen_floors [--jobs N] [--families a,b] [--out PATH]

Scope. ``InstructionEntry.cert_arch`` is the maximum floor over a family's variants (the table
certification test assembles every variant there). A family with ``cert_arch`` of sm_90 or older
(or None: certified at sm_90) therefore runs on every sm_90+ target and needs no data for them; it
is left unchecked (its per-variant floors below sm_90 are not measured). Only the families whose
``cert_arch`` is newer than sm_90 *and* one NVRTC can target are measured
(:func:`measured_families`), at the :data:`CHECKED_ARCHS`. A family certified at sm_107* (NVRTC
13.4 cannot target it) would have no witness, so it stays unchecked.

For each measured family and every legal modifier combination (``table.variants``), one
representative rendering is compiled through the production compile path
(``support.nvcc.compile_cuda`` with NVRTC, cubin) at each checked arch, and at the family's own
``cert_arch`` as a witness. A variant the witness rejects too is a harness artifact, not an arch
floor, and stays unchecked. Legality depends on the modifier tokens, not on register dtypes,
predication or immediate values, so one rendering per token combination suffices.

Measured, not derived: ptxas is the ground truth for what compiles today, including the ``a``/``f``
suffix rules. ``floors.py`` turns the result into a check that can only reject what this toolchain
rejects. Regenerate after changing the table or the NVRTC version.

Each helper is wrapped in its own ``__global__`` certification kernel, so a ptxas error's PTX line
number identifies the failing variant: the ``.entry`` that contains the line. Failing variants are
removed and the rest recompiled until the batch is clean.
"""

from __future__ import annotations

import argparse
import itertools
import json
import multiprocessing
import os
import re
import sys
import time

# Targets the check covers; the order is the bit order of the generated masks. Other targets
# (sm_86/87/88, sm_100 and later) are unchecked.
CHECKED_ARCHS = ("sm_80", "sm_89", "sm_90", "sm_90a")
# Archs NVRTC 13.4 accepts as cubin targets, i.e. usable as a witness (sm_107* is not among them).
NVRTC_ARCHS = frozenset(
    ("sm_75", "sm_80", "sm_86", "sm_87", "sm_88", "sm_89", "sm_90", "sm_90a", "sm_100", "sm_100a",
     "sm_100f", "sm_103", "sm_103a", "sm_103f", "sm_110", "sm_110a", "sm_110f", "sm_120", "sm_120a",
     "sm_120f", "sm_121", "sm_121a", "sm_121f")
)  # fmt: skip
# cert_arch values at or below the oldest target that matters: such a family runs on all sm_90+.
_UNMEASURED_CERT_ARCHS = (None, "sm_75", "sm_80", "sm_90")


def measured_families(table):
    """The families measured: ``cert_arch`` newer than sm_90, and one NVRTC can target.

    A family certified at an arch NVRTC cannot target (sm_107*) has no witness, so a rejection
    could not be told apart from a harness artifact. It stays unchecked.
    """
    return sorted(
        name
        for name, e in table.items()
        if e.cert_arch not in _UNMEASURED_CERT_ARCHS and e.cert_arch in NVRTC_ARCHS
    )


def _archs_for(entry):
    witness = entry.cert_arch
    return CHECKED_ARCHS + ((witness,) if witness not in CHECKED_ARCHS else ())


# NVRTC has no host <cstdint>; its preamble (support.nvcc) already provides the integer types.
CERT_PRELUDE = "#include <cuda_fp16.h>\n#include <cuda_bf16.h>"
MAX_SOURCE_CHARS = 256 * 1024
_SIGNATURE_RE = re.compile(
    r"\A__forceinline__ __device__ void (?P<helper>[A-Za-z_]\w*)\((?P<params>[^)]*)\) \{"
)
_PARAM_RE = re.compile(r"(?P<type>.+?)(?P<reference>&)?\s+(?P<name>__[A-Za-z_]\w*)\Z")


def certification_kernel(helper, helper_source, kernel_index):
    """Keep a helper inline and call it from a retained, production-shaped kernel.

    Merely placing an unreferenced ``__forceinline__`` helper in a translation
    unit does not certify it: nvcc may discard it before ptxas sees its body.
    The old workaround stripped ``__forceinline__`` and retained the resulting
    device function with RDC.  That changes the device ABI and is not an
    equivalent compilation shape for wide inline-asm operands.

    A global caller is retained without RDC and forces the original helper to
    inline, which is the shape emitted code uses.  Pointer arguments come from
    the runtime sink so address instructions are not specialized to null.
    Register inputs are locals, avoiding an artificial device ABI boundary;
    ordinary values and raw 32-bit addresses are loaded through the sink so
    they keep the runtime data flow of generated calls.  Predicate and b128
    inputs remain local because crossing either through the certification ABI
    creates a tool-only compilation shape.  Written values are loaded and
    stored through the sink so non-volatile output instructions cannot be
    optimized away before ptxas.
    """
    match = _SIGNATURE_RE.match(helper_source)
    if match is None or match.group("helper") != helper:
        raise ValueError(f"cannot parse certification signature for {helper}")

    declarations = []
    arguments = []
    stores = []
    params = match.group("params")
    parsed_params = params.split(", ") if params else ()
    for param_index, param in enumerate(parsed_params):
        param_match = _PARAM_RE.fullmatch(param)
        if param_match is None:
            raise ValueError(f"cannot parse {helper} parameter: {param}")
        c_type = param_match.group("type")
        param_name = param_match.group("name")
        local = f"__cert_arg{param_index}"
        input_offset = 16 * param_index
        if c_type.endswith("*"):
            declarations.append(f"  {c_type} {local} = __cert_sink + {input_offset};")
        elif param_match.group("reference"):
            declarations.append(
                f"  {c_type} {local} = "
                f"*reinterpret_cast<const {c_type}*>(__cert_sink + {input_offset});"
            )
        elif c_type in ("__int128_t", "__uint128_t") or param_name == "__pred":
            declarations.append(f"  {c_type} {local}{{}};")
        else:
            declarations.append(
                f"  {c_type} {local} = "
                f"*reinterpret_cast<const {c_type}*>(__cert_sink + {input_offset});"
            )
        arguments.append(local)
        if param_match.group("reference"):
            # Every C binding is at most 16 bytes.  Distinct, aligned locations
            # keep all outputs observable without imposing a constraint letter
            # of our own on the value being certified.
            stores.append(
                f"  *reinterpret_cast<{c_type}*>(__cert_sink + "
                f"{16 * (len(parsed_params) + len(stores))}) = {local};"
            )

    kernel = [
        f'extern "C" __global__ void __ptx_cert_{kernel_index}(char* __cert_sink) {{',
        *declarations,
        f"  {helper}({', '.join(arguments)});",
        *stores,
        "}",
    ]
    return "\n".join((helper_source, *kernel))


_PTXAS_LINE_RE = re.compile(r"ptx input, line (\d+); error")
_ENTRY_RE = re.compile(r"^\.(?:visible )?\.entry (__ptx_cert_\d+)\(")


def representatives(entry):
    """One rendering per distinct modifier-token combination: ``{tokens: rendering}``."""
    from tvm_tirx.backend.cuda.ptx.table import renderings  # pylint: disable=import-outside-toplevel

    reps = {}
    for tokens, dtypes, predicated, imms, sinks in renderings(entry):
        reps.setdefault(tuple(tokens), (tuple(tokens), predicated, dtypes, imms, sinks))
    return reps


def _compile(source, arch, fmt):
    from tvm_tirx.script import tirx as T  # pylint: disable=import-outside-toplevel
    from tvm_tirx.support import nvcc  # pylint: disable=import-outside-toplevel

    config = T.cuda.BackendConfig(arch=arch, target_format=fmt, compiler="nvrtc")
    return nvcc.compile_cuda(source, backend_config={"cuda": config})


def _batch_source(items):
    return "\n".join((CERT_PRELUDE, *(src for _, src in items)))


def _accepted(items, arch):
    """Keys of the ``items`` (``[(key, kernel_source)]``) that ``arch`` accepts.

    Kernels are renumbered by position each round, so a failing ``.entry`` name is its position.
    Errors that cannot be attributed to a kernel (a front-end error, or no line in the log) fall
    back to halving the batch.
    """
    items = [_renumber(item, pos) for pos, item in enumerate(items)]
    while items:
        source = _batch_source(items)
        try:
            _compile(source, arch, "cubin")
            return [key for key, _ in items]
        except Exception as err:
            lines = {int(n) for n in _PTXAS_LINE_RE.findall(str(err))}
        failed = set()
        if lines:
            try:
                ptx = _compile(source, arch, "ptx")
                failed = _entries_at(bytes(ptx).decode(), lines)
            except Exception:
                failed = set()
        if not failed:
            if len(items) == 1:
                return []
            half = len(items) // 2
            return _accepted(items[:half], arch) + _accepted(items[half:], arch)
        survivors = [item for pos, item in enumerate(items) if f"__ptx_cert_{pos}" not in failed]
        items = [_renumber(item, pos) for pos, item in enumerate(survivors)]
    return []


def _renumber(item, pos):
    key, src = item
    return key, re.sub(r"__ptx_cert_\d+\(", f"__ptx_cert_{pos}(", src)


def _entries_at(ptx, lines):
    """Names of the ``.entry`` functions containing the given 1-based PTX lines."""
    current, owner = None, {}
    for number, text in enumerate(ptx.splitlines(), start=1):
        match = _ENTRY_RE.match(text.strip())
        if match:
            current = match.group(1)
        owner[number] = current
    return {owner[n] for n in lines if owner.get(n)}


def _job(args):
    job_id, chunk = args
    family, _, arch = job_id
    from tvm_tirx.backend.cuda.ptx.render import render_variant  # pylint: disable=import-outside-toplevel
    from tvm_tirx.backend.cuda.ptx.table import TABLE  # pylint: disable=import-outside-toplevel

    entry, start = TABLE[family], time.time()
    items = []
    for pos, rendering in enumerate(chunk):
        _, helper, src = render_variant(entry, *rendering)
        items.append((rendering[0], certification_kernel(helper, src, pos)))
    ok = [list(key) for key in _accepted(items, arch)]
    return job_id, ok, time.time() - start


def _chunks(family, reps):
    """Split a family's representatives into source-size-bounded chunks."""
    from tvm_tirx.backend.cuda.ptx.render import render_variant  # pylint: disable=import-outside-toplevel
    from tvm_tirx.backend.cuda.ptx.table import TABLE  # pylint: disable=import-outside-toplevel

    entry = TABLE[family]
    chunk, size = [], len(CERT_PRELUDE)
    for rendering in reps.values():
        _, _, src = render_variant(entry, *rendering)
        n = len(src) + 400
        if chunk and size + n > MAX_SOURCE_CHARS:
            yield chunk
            chunk, size = [], len(CERT_PRELUDE)
        chunk.append(rendering)
        size += n
    if chunk:
        yield chunk


def _project(masks, n_slots):
    """The smallest set of slot positions that determines every variant's mask.

    Returns ``(slots, table)``: ``table`` maps the tokens at ``slots`` to the arch mask. A mask of
    None (unknown) is a value of its own here, so the projection never merges an unknown variant
    into a known one; the caller drops the None entries.
    """
    for size in range(n_slots + 1):
        for slots in itertools.combinations(range(n_slots), size):
            table = {}
            for tokens, mask in masks.items():
                key = tuple(tokens[i] for i in slots)
                if table.setdefault(key, mask) != mask:
                    break
            else:
                return slots, table
    raise AssertionError("unreachable: all slots determine the mask")


def generate(families=None, jobs=8, log=sys.stderr, cache=None):
    """Measure every family's variants; ``cache`` (a JSON-lines path) makes a rerun resume."""
    from tvm_tirx.backend.cuda.ptx.table import TABLE  # pylint: disable=import-outside-toplevel

    names = sorted(families or measured_families(TABLE))
    work, reps_by = [], {}
    for name in names:
        reps_by[name] = representatives(TABLE[name])
        for index, chunk in enumerate(_chunks(name, reps_by[name])):
            work.extend(((name, index, arch), chunk) for arch in _archs_for(TABLE[name]))
    n_variants = sum(len(r) for r in reps_by.values())
    print(f"gen_floors: {len(names)} families, {n_variants} variants, {len(work)} jobs", file=log)
    masks = {name: dict.fromkeys(reps_by[name], 0) for name in names}
    witnessed = {name: set() for name in names}

    def record(job_id, ok):
        family, _, arch = job_id
        if arch not in CHECKED_ARCHS:  # the witness
            witnessed[family].update(tuple(key) for key in ok)
            return
        bit = 1 << CHECKED_ARCHS.index(arch)
        for key in ok:
            masks[family][tuple(key)] |= bit

    finished = set()
    if cache and os.path.exists(cache):
        with open(cache, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                job_id = (row["family"], row["chunk"], row["arch"])
                if job_id[0] in masks and job_id not in finished:
                    finished.add(job_id)
                    record(job_id, row["ok"])
    todo = [(job_id, chunk) for job_id, chunk in work if job_id not in finished]
    print(f"gen_floors: {len(work) - len(todo)} jobs from cache, {len(todo)} to run", file=log)
    t0 = time.time()
    out = open(cache, "a", encoding="utf-8") if cache else None
    try:
        with multiprocessing.get_context("spawn").Pool(jobs) as pool:
            for done, (job_id, ok, seconds) in enumerate(pool.imap_unordered(_job, todo), 1):
                record(job_id, ok)
                if out:
                    family, chunk, arch = job_id
                    row = {"family": family, "chunk": chunk, "arch": arch, "ok": ok, "s": seconds}
                    out.write(json.dumps(row) + "\n")
                    out.flush()
                if done % 200 == 0:
                    print(f"gen_floors: {done}/{len(todo)} jobs, {time.time() - t0:.0f}s", file=log)
    finally:
        if out:
            out.close()
    result = {}
    for name in names:
        has_witness = len(_archs_for(TABLE[name])) > len(CHECKED_ARCHS)  # sm_90a: its own witness
        values = {
            key: (mask if mask or not has_witness or key in witnessed[name] else None)
            for key, mask in masks[name].items()
        }
        slots, table = _project(values, len(TABLE[name].slots))
        table = {key: mask for key, mask in table.items() if mask is not None}
        if table:
            result[name] = (slots, table)
    return result


def _literal(value):
    """``repr`` with double quotes (PTX tokens never contain a quote), as ruff format writes it."""
    return repr(value).replace("'", '"')


def write(result, path):
    license_header = []
    with open(__file__, encoding="utf-8") as f:
        for line in f:  # this module's own ASF header
            if not line.startswith("#"):
                break
            license_header.append(line.rstrip("\n"))
    lines = [
        *license_header,
        "# ruff: noqa: E501  (one family per line, so the data stays greppable)",
        "# Generated by tvm_tirx.backend.cuda.ptx.gen_floors -- do not edit by hand.",
        "# For each PTX family: (slot positions, {tokens at those slots: arch mask}).",
        "# Bit i of a mask is ARCHS[i]: the production compile path (NVRTC) accepts the variant.",
        "# A family, token combination or target arch absent here is unchecked.",
        "",
        f"ARCHS = {_literal(tuple(CHECKED_ARCHS))}",
        "",
        "# fmt: off",
        "FLOORS = {",
    ]
    for name in sorted(result):
        slots, table = result[name]
        body = ", ".join(f"{_literal(key)}: {mask:#x}" for key, mask in sorted(table.items()))
        lines.append(f"    {_literal(name)}: ({_literal(slots)}, {{{body}}}),")
    lines += ["}", "# fmt: on"]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument(
        "--cache", default=None, help="JSON-lines job cache; a rerun resumes from it"
    )
    parser.add_argument("--families", default=None, help="comma-separated subset (default: all)")
    parser.add_argument(
        "--out", default=None, help="output path (default: ptx/floors_generated.py)"
    )
    args = parser.parse_args(argv)
    out = args.out or os.path.join(os.path.dirname(__file__), "floors_generated.py")
    families = args.families.split(",") if args.families else None
    result = generate(families, jobs=args.jobs, cache=args.cache)
    write(result, out)
    print(f"gen_floors: wrote {len(result)} families to {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
