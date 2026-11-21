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
"""Differential test of the simplifiers on int32 index expressions with large constants.

Each random expression is evaluated exactly (Python integers) at points where every
intermediate of the original fits int32. Its simplified form is evaluated the way hardware
computes it, with int32 wrap-around, and must give the same value without dividing by zero.
Constants near powers of two up to 2^30 make constant products and sums overflow int32,
which is where rewrite and canonical rules must not fold.
"""

import random

import pytest

from tvm_tirx import ir, tirx
from tvm_tirx.sym import Analyzer

INT32_MIN, INT32_MAX = -(2**31), 2**31 - 1
# Near powers of two, a LowerIntrin shift coefficient (24970741 = ceil(2^30 / 43)) and its
# product with 43, and sqrt(2^31).
LARGE = [2**k + d for k in (16, 20, 24, 28, 29, 30) for d in (-1, 0, 1)]
LARGE += [24970741, 43 * 24970741, 46341]


class _Invalid(Exception):
    pass


def _trunc_div(a, b):
    quotient = abs(a) // abs(b)
    return quotient if (a >= 0) == (b > 0) else -quotient


_OPS = {
    "Add": lambda a, b: a + b,
    "Sub": lambda a, b: a - b,
    "Mul": lambda a, b: a * b,
    "FloorDiv": lambda a, b: a // b,
    "FloorMod": lambda a, b: a % b,
    "Div": _trunc_div,
    "Mod": lambda a, b: a - _trunc_div(a, b) * b,
    "Min": min,
    "Max": max,
}


def _evaluate(expr, env, wrap):
    kind = type(expr).__name__
    if kind == "Var":
        return env[expr.name]
    if kind == "IntImm":
        return int(expr.value)
    if kind == "Cast":
        return _evaluate(expr.value, env, wrap)
    a, b = _evaluate(expr.a, env, wrap), _evaluate(expr.b, env, wrap)
    if kind in ("FloorDiv", "FloorMod", "Div", "Mod") and b == 0:
        raise ZeroDivisionError(str(expr))
    value = _OPS[kind](a, b)
    if not INT32_MIN <= value <= INT32_MAX:
        if not wrap:
            raise _Invalid()
        value = (value - INT32_MIN) % 2**32 + INT32_MIN
    return value


def _expressions(seed, count):
    rng = random.Random(seed)
    x, y = tirx.Var("x", "int32"), tirx.Var("y", "int32")

    def constant():
        value = rng.choice(LARGE) if rng.random() < 0.5 else rng.randint(1, 300)
        return value if rng.random() < 0.7 else -value

    def divisor():
        return ir.IntImm("int32", abs(constant()))

    def generate(depth):
        if depth == 0 or rng.random() < 0.2:
            return rng.choice([x, y, x, ir.IntImm("int32", constant())])
        op = rng.choice(
            ["add", "sub", "mul", "fdiv", "fmod", "tdiv", "tmod", "min", "max", "mulc", "fdivc"]
        )
        a = generate(depth - 1)
        if op == "mulc":
            return a * ir.IntImm("int32", constant())
        if op == "fdivc":
            return tirx.floordiv(a, divisor())
        b = divisor() if op in ("fdiv", "fmod", "tdiv", "tmod") else generate(depth - 1)
        return {
            "add": lambda: a + b,
            "sub": lambda: a - b,
            "mul": lambda: a * b,
            "fdiv": lambda: tirx.floordiv(a, b),
            "fmod": lambda: tirx.floormod(a, b),
            "tdiv": lambda: tirx.truncdiv(a, b),
            "tmod": lambda: tirx.truncmod(a, b),
            "min": lambda: tirx.Min(a, b),
            "max": lambda: tirx.Max(a, b),
        }[op]()

    for _ in range(count):
        expr = generate(rng.randint(2, 4))
        points = [
            {
                "x": rng.randint(INT32_MIN, INT32_MAX)
                if rng.random() < 0.5
                else rng.randint(-5000, 5000),
                "y": rng.randint(-5000, 5000),
            }
            for _ in range(24)
        ]
        cases = []
        for point in points:
            try:
                cases.append((point, _evaluate(expr, point, wrap=False)))
            except _Invalid:
                pass
        if cases:
            yield expr, cases


@pytest.mark.parametrize("method", ["rewrite_simplify", "canonical_simplify", "simplify"])
def test_simplified_value_matches_exact_value(method):
    mismatches = []
    for expr, cases in _expressions(seed=20261121, count=1500):
        try:
            simplified = getattr(Analyzer(), method)(expr)
        except ValueError:
            # A literal that cannot be represented is reported, not miscompiled.
            continue
        for point, expected in cases:
            try:
                actual = _evaluate(simplified, point, wrap=True)
            except ZeroDivisionError:
                actual = "division by zero"
            if actual != expected:
                mismatches.append(f"{expr} -> {simplified} at {point}: {actual} != {expected}")
                break
    assert not mismatches, "\n".join(mismatches[:10])
