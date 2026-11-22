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
"""Comparison identities that fail for NaN are not applied to floats (sm90-probe ISSUE-013).

IEEE 754: with a NaN operand, `x == x` is false, `x != x` is true, every ordered comparison is
false, and the negation of an ordered comparison is true. `x != x` is the standard NaN test.
"""

import pytest
import tvm_ffi

import tvm_tirx
import tvm_tirx.testing
from tvm_tirx import tirx
from tvm_tirx.script import tirx as T
from tvm_tirx.sym import Analyzer

FLOATS = ["float32", "float16", "bfloat16", "float64", "float8_e4m3fn"]
INTS = ["int32", "int64", "int8", "uint32"]


def _vars(dtype):
    return tirx.Var("x", dtype), tirx.Var("y", dtype)


def _float_cases(x, y, dtype):
    # Each would fold to a constant, or to a form with different NaN semantics, if the
    # integer identities applied.
    return {
        "x == x": x == x,
        "x != x": x != x,
        "!(x < y)": tirx.Not(x < y),
        "!(x <= y)": tirx.Not(x <= y),
        "!(x > y)": tirx.Not(x > y),
        "!(x >= y)": tirx.Not(x >= y),
        "x <= y || y < x": tirx.Or(x <= y, y < x),
        "y < x || x <= y": tirx.Or(y < x, x <= y),
        "x < y || y < x": tirx.Or(x < y, y < x),
        "x == 0 || x != x": tirx.Or(x == tirx.const(0, dtype), x != x),
    }


@pytest.mark.parametrize("dtype", FLOATS)
def test_float_nan_sensitive_identities_are_kept(dtype):
    x, y = _vars(dtype)
    analyzer = Analyzer()
    for name, expr in _float_cases(x, y, dtype).items():
        simplified = analyzer.simplify(expr)
        assert not isinstance(simplified, tvm_tirx.ir.IntImm), (name, simplified)
        # No rewrite into a form with different NaN semantics: the result is the same
        # comparison, possibly re-spelled through `!=` / `>` / `>=`.
        assert "x" in str(simplified), (name, simplified)
    # The structure that carries the NaN behaviour survives.
    assert tvm_ffi.structural_equal(analyzer.simplify(x != x), x != x)
    assert tvm_ffi.structural_equal(analyzer.simplify(x <= y), x <= y)
    not_lt = analyzer.simplify(tirx.Not(x < y))
    assert isinstance(not_lt, tirx.Not), not_lt


@pytest.mark.parametrize("dtype", INTS)
def test_integer_identities_still_fold(dtype):
    x, y = _vars(dtype)
    analyzer = Analyzer()
    assert tvm_ffi.structural_equal(analyzer.simplify(x == x), T.bool(True))
    assert tvm_ffi.structural_equal(analyzer.simplify(x != x), T.bool(False))
    assert tvm_ffi.structural_equal(analyzer.simplify(tirx.Not(x < y)), y <= x)
    assert tvm_ffi.structural_equal(analyzer.simplify(tirx.Not(x <= y)), y < x)
    assert tvm_ffi.structural_equal(analyzer.simplify(tirx.Or(x <= y, y < x)), T.bool(True))


def test_bool_identities_still_fold():
    b = tirx.Var("b", "bool")
    assert tvm_ffi.structural_equal(Analyzer().simplify(b == b), T.bool(True))


def test_float_select_on_nan_test_is_kept():
    x = tirx.Var("x", "float32")
    guarded = T.Select(tirx.Or(x == T.float32(0), x != x), T.float32(1), x)
    simplified = Analyzer().simplify(guarded)
    assert "x != x" in str(simplified) or "!(x == x)" in str(simplified), simplified


if __name__ == "__main__":
    tvm_tirx.testing.main()
