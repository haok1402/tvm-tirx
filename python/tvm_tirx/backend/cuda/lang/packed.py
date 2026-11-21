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
"""Arch-portable packed ``f32x2`` arithmetic.

``add/sub/mul/fma{.rnd}{.ftz}.f32x2`` exist only on sm_100 and newer. ptxas
rejects them elsewhere ("Feature 'mul.f32x2' requires .target sm_100 or
higher"), so a kernel that spells ``T.ptx.mul.rz.ftz.f32x2`` cannot be compiled
for Hopper. These helpers take and return the packed pair as a ``uint64`` (the
``.b64`` register the ISA uses). They compile to:

* the packed instruction when the device code is built for sm_100 or newer;
* two scalar instructions with the same rounding and ftz otherwise (e.g. sm_90a).

Packed f32x2 arithmetic is defined lane by lane, so both paths give bit-identical
results, and one kernel source serves every arch. The choice is made by the CUDA
preprocessor (``__CUDA_ARCH__``) for the arch being compiled, so the same TIRx
module can be built for several targets. Example::

    pk = packed.f32x2_mul(a, b, rnd="rz", ftz=True)   # was T.ptx.mul.rz.ftz.f32x2(pk, a, b)
    lo, hi = T.cuda.float2_x(pk), T.cuda.float2_y(pk)
"""

from __future__ import annotations

from tvm_tirx.backend.cuda.op import func_call

ROUNDINGS = ("rn", "rz", "rm", "rp")
_OPS = {"add": 2, "sub": 2, "mul": 2, "fma": 3}


def helper_name(op: str, rnd: str = "rn", ftz: bool = False) -> str:
    return f"tvm_builtin_f32x2_{op}_{rnd}" + ("_ftz" if ftz else "")


def helper_source(op: str, rnd: str = "rn", ftz: bool = False) -> str:
    """The CUDA device function behind :func:`f32x2` for one variant."""
    if op not in _OPS:
        raise ValueError(f"f32x2 op must be one of {sorted(_OPS)}, got {op!r}")
    if rnd not in ROUNDINGS:
        raise ValueError(f"f32x2 rounding must be one of {ROUNDINGS}, got {rnd!r}")
    srcs = ("a", "b", "c")[: _OPS[op]]
    mods = f".{rnd}" + (".ftz" if ftz else "")
    operands = ", ".join(f"%{i + 1}" for i in range(len(srcs)))
    params = ", ".join(f"unsigned long long {x}" for x in srcs)
    packed_in = ", ".join(f'"l"({x})' for x in srcs)
    unpack = "".join(f"  float2 {x}2 = *reinterpret_cast<float2*>(&{x});\n" for x in srcs)
    lanes = "".join(
        f'  asm("{op}{mods}.f32 %0, {operands};" : "=f"(r.{lane}) : '
        + ", ".join(f'"f"({x}2.{lane})' for x in srcs)
        + ");\n"
        for lane in ("x", "y")
    )
    return (
        f"\n__forceinline__ __device__ unsigned long long {helper_name(op, rnd, ftz)}("
        f"{params}) {{\n"
        "#if __CUDA_ARCH__ >= 1000\n"
        "  unsigned long long d;\n"
        f'  asm("{op}{mods}.f32x2 %0, {operands};" : "=l"(d) : {packed_in});\n'
        "  return d;\n"
        "#else\n"
        f"{unpack}"
        "  float2 r;\n"
        f"{lanes}"
        "  return *reinterpret_cast<unsigned long long*>(&r);\n"
        "#endif\n"
        "}\n"
    )


def f32x2(op, *operands, rnd="rn", ftz=False):
    """``op{.rnd}{.ftz}.f32x2`` on packed ``uint64`` operands, portable across archs.

    ``op`` is ``"add"``, ``"sub"``, ``"mul"`` (two operands) or ``"fma"`` (three:
    ``a * b + c``). Returns the packed ``uint64`` result.
    """
    if op not in _OPS:
        raise ValueError(f"f32x2 op must be one of {sorted(_OPS)}, got {op!r}")
    if len(operands) != _OPS[op]:
        raise ValueError(f"f32x2 {op} takes {_OPS[op]} operands, got {len(operands)}")
    return func_call(helper_name(op, rnd, ftz), *operands, helper_source(op, rnd, ftz), ty="uint64")


def f32x2_add(a, b, *, rnd="rn", ftz=False):
    return f32x2("add", a, b, rnd=rnd, ftz=ftz)


def f32x2_sub(a, b, *, rnd="rn", ftz=False):
    return f32x2("sub", a, b, rnd=rnd, ftz=ftz)


def f32x2_mul(a, b, *, rnd="rn", ftz=False):
    return f32x2("mul", a, b, rnd=rnd, ftz=ftz)


def f32x2_fma(a, b, c, *, rnd="rn", ftz=False):
    return f32x2("fma", a, b, c, rnd=rnd, ftz=ftz)
