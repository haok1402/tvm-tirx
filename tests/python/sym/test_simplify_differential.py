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


def _evaluate(expr, env, wrap, limit=2**31):
    """Evaluate expr exactly, or with int32 wrap-around when wrap is set.

    Exact evaluation rejects (raises _Invalid) any intermediate outside int32, or with a
    magnitude of limit or more when a smaller domain limit is given.
    """
    kind = type(expr).__name__
    if kind == "Var":
        return env[expr.name]
    if kind == "IntImm":
        return int(expr.value)
    if kind == "Cast":
        return _evaluate(expr.value, env, wrap, limit)
    a, b = _evaluate(expr.a, env, wrap, limit), _evaluate(expr.b, env, wrap, limit)
    if kind in ("FloorDiv", "FloorMod", "Div", "Mod") and b == 0:
        raise ZeroDivisionError(str(expr))
    value = _OPS[kind](a, b)
    if not wrap and limit < 2**31 and not -limit < value < limit:
        raise _Invalid()
    if not INT32_MIN <= value <= INT32_MAX:
        if not wrap:
            raise _Invalid()
        value = (value - INT32_MIN) % 2**32 + INT32_MIN
    return value


def _expressions(seed, count, limit=2**31):
    """Random expressions with the points where they are evaluated.

    With the default limit, points draw x anywhere in int32 half of the time, else small, and
    keep the points where every exact intermediate fits int32. With a domain limit, constants
    have a magnitude below limit, points favor values near +-limit, and every exact
    intermediate must have a magnitude below limit.
    """
    rng = random.Random(seed)
    x, y = tirx.Var("x", "int32"), tirx.Var("y", "int32")
    large = (
        LARGE if limit >= 2**31 else [v for v in LARGE if v < limit] + [limit - 1, limit // 2 + 1]
    )

    def constant():
        value = rng.choice(large) if rng.random() < 0.5 else rng.randint(1, 300)
        return value if rng.random() < 0.7 else -value

    def domain_value():
        kind = rng.random()
        if kind < 0.4:
            return rng.randint(-limit + 1, limit - 1)
        if kind < 0.7:
            return rng.choice([-1, 1]) * (limit - rng.randint(1, 2000))
        return rng.randint(-5000, 5000)

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
        if limit >= 2**31:
            points = [
                {
                    "x": rng.randint(INT32_MIN, INT32_MAX)
                    if rng.random() < 0.5
                    else rng.randint(-5000, 5000),
                    "y": rng.randint(-5000, 5000),
                }
                for _ in range(24)
            ]
        else:
            points = [{"x": domain_value(), "y": domain_value()} for _ in range(24)]
        cases = []
        for point in points:
            try:
                cases.append((point, _evaluate(expr, point, wrap=False, limit=limit)))
            except _Invalid:
                pass
        if cases:
            yield expr, cases


@pytest.mark.parametrize("method", ["rewrite_simplify", "canonical_simplify", "simplify"])
def test_simplified_value_matches_exact_value(method):
    mismatches = []
    for expr, cases in _expressions(seed=20261121, count=1500):
        # A canonical form whose literals overflow falls back to the rewrite rules, so no
        # simplifier raises on these inputs.
        simplified = getattr(Analyzer(), method)(expr)
        for point, expected in cases:
            try:
                actual = _evaluate(simplified, point, wrap=True)
            except ZeroDivisionError:
                actual = "division by zero"
            if actual != expected:
                mismatches.append(f"{expr} -> {simplified} at {point}: {actual} != {expected}")
                break
    assert not mismatches, "\n".join(mismatches[:10])


@pytest.mark.parametrize("method", ["rewrite_simplify", "canonical_simplify", "simplify"])
def test_index_domain_values_are_exact(method):
    """In the int32 index domain, every simplified value is exact.

    The domain: every constant, and every intermediate value of the input expression, has a
    magnitude below 2^30. A rewrite forms a new term as the sum or difference of two existing
    subexpressions, e.g. min(c, x) - x -> min(c - x, 0) or max(a, b) + c -> max(a + c, b + c),
    so one rewrite step stays within int32 there. Chains of steps are not proven; this test
    checks them. An in-bounds index of a tensor with at most 2^30 elements, written as a sum of
    non-negative terms, is in this domain; larger tensors are indexed in int64.
    """
    mismatches = []
    for expr, cases in _expressions(seed=20261122, count=1500, limit=2**30):
        try:
            simplified = getattr(Analyzer(), method)(expr)
        except ValueError:
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


_X, _Y = tirx.Var("x", "int32"), tirx.Var("y", "int32")
_BOUNDARY_CASES = {
    # min(c, x) - x -> min(c - x, 0): c - x is a new term.
    "min_minus_inside": (tirx.Min(ir.IntImm("int32", 2**30 - 1), _X) - _X, {"x": -(2**30) + 1}),
    "min_minus_constant_outside": (
        tirx.Min(ir.IntImm("int32", 2**30 + 1), _X) - _X,
        {"x": -(2**30) + 1},
    ),
    # x + max(y + x, c - x) -> max(y + x + x, c): (y + x) + x is a new term.
    "max_plus_inside": (
        _X + tirx.Max(_Y + _X, ir.IntImm("int32", 203) - _X),
        {"x": -(2**29) + 100, "y": -(2**29) + 100},
    ),
    "max_plus_intermediate_outside": (
        _X + tirx.Max(_Y + _X, ir.IntImm("int32", 203) - _X),
        {"x": -(2**30) + 100, "y": -(2**30) + 100},
    ),
}


@pytest.mark.parametrize(
    "case",
    [
        "min_minus_inside",
        pytest.param(
            "min_minus_constant_outside",
            marks=pytest.mark.xfail(
                strict=True, reason="out of contract: the constant 2^30 + 1 is outside the domain"
            ),
        ),
        "max_plus_inside",
        pytest.param(
            "max_plus_intermediate_outside",
            marks=pytest.mark.xfail(
                strict=True,
                reason="out of contract: the input intermediate y + x is -2^31 + 200",
            ),
        ),
    ],
)
def test_index_domain_boundary(case):
    """At the domain boundary: exact inside, and a known wrong value just outside."""
    expr, point = _BOUNDARY_CASES[case]
    simplified = Analyzer().rewrite_simplify(expr)
    assert _evaluate(simplified, point, wrap=True) == _evaluate(expr, point, wrap=False)
