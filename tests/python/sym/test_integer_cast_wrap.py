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
"""Integer casts from or to unsigned types wrap (sm90-probe ISSUE-011).

``int32(x)`` for ``x: uint32`` reinterprets the bits: ``0xFFFFFFFF`` becomes ``-1``. The
analyzer used to intersect the source range with the target's, so ``int32(x) >= 0`` folded to
true and ``max(int32(x), 0)`` to ``int32(x)``.
"""

import pytest
import tvm_ffi

import tvm_tirx
import tvm_tirx.testing
from tvm_tirx import tirx
from tvm_tirx.script import tirx as T
from tvm_tirx.sym import Analyzer, ConstIntBound, ProofStrength

NEG_INF = ConstIntBound.NEG_INF
POS_INF = ConstIntBound.POS_INF


def _bound(expr, analyzer=None):
    b = (analyzer or Analyzer()).const_int_bound(expr)
    return (b.min_value, b.max_value)


def test_issue_repro_does_not_fold():
    x = tirx.Var("x", "uint32")
    v = x.astype("int32")
    analyzer = Analyzer()
    assert not tvm_ffi.structural_equal(analyzer.simplify(v >= 0), T.bool(True))
    assert not analyzer.can_prove(v >= 0)
    assert not analyzer.can_prove(v >= 0, ProofStrength.SYMBOLIC_BOUND)
    clamp = analyzer.simplify(T.max(v, 0))
    assert not tvm_ffi.structural_equal(clamp, v), clamp


@pytest.mark.parametrize(
    "source, target, expected",
    [
        # Reinterpreting casts: the source range does not fit, so it does not carry over.
        ("uint32", "int32", (-(2**31), 2**31 - 1)),
        ("uint64", "int64", (NEG_INF, POS_INF)),
        ("uint32", "int16", (-(2**15), 2**15 - 1)),
        ("int32", "uint32", (0, 2**32 - 1)),
        ("uint32", "uint16", (0, 2**16 - 1)),
        # Casts that keep every value keep the range.
        ("uint8", "int32", (0, 255)),
        ("uint32", "int64", (0, 2**32 - 1)),
        ("int8", "int32", (-128, 127)),
        ("bool", "int32", (0, 1)),
    ],
)
def test_cast_bound_of_full_range(source, target, expected):
    assert _bound(tirx.Var("x", source).astype(target)) == expected


def test_cast_bound_keeps_fitting_range():
    analyzer = Analyzer()
    x = tirx.Var("x", "int32")
    assert _bound((tirx.floormod(x, 7) + 7).astype("uint32"), analyzer) == (7, 13)
    # truncmod(x, 3) is in [-2, 2]: uint32(-1) == 0xFFFFFFFF.
    assert _bound(tirx.truncmod(x, 3).astype("uint32"), analyzer) == (0, 2**32 - 1)


def test_signed_narrowing_keeps_no_overflow_assumption():
    # Index arithmetic is assumed not to overflow: an int64 index cast to int32 keeps the part
    # of its range that int32 holds (unchanged behaviour).
    n = tirx.Var("n", "int64")
    analyzer = Analyzer()
    context = tvm_tirx.sym.ConstraintContext()
    context.append(tvm_tirx.sym.PredicateConstraint(n >= 0))
    analyzer.sync_constraint_context(context)
    assert _bound(n.astype("int32"), analyzer) == (0, 2**31 - 1)
    assert analyzer.can_prove(n.astype("int32") >= 0)


def test_cast_to_bool_is_nonzero_test():
    x = tirx.Var("x", "uint32")
    assert _bound((x + T.uint32(2)).astype("bool")) == (0, 1)


def test_small_unsigned_cast_still_folds():
    x = tirx.Var("x", "uint8")
    assert Analyzer().can_prove(x.astype("int32") >= 0)


def test_modular_set_after_wrap():
    analyzer = Analyzer()
    x = tirx.Var("x", "uint32")
    # int32(3 * x) wraps modulo 2^32, which 3 does not divide.
    assert analyzer.modular_set((x * T.uint32(3)).astype("int32")).coeff == 1
    # A power-of-two stride survives the wrap.
    m = analyzer.modular_set((x * T.uint32(16) + T.uint32(4)).astype("int32"))
    assert (m.coeff, m.base) == (16, 4)
    # No wrap, no change.
    y = tirx.Var("y", "int32")
    m = analyzer.modular_set((tirx.floormod(y, 5) * 3).astype("uint32"))
    assert m.coeff == 3


def test_int_set_of_wrapping_cast():
    analyzer = Analyzer()
    i = tirx.Var("i", "int32")
    s = analyzer.int_set((i - 1).astype("uint32"), {i: tvm_tirx.sym.IntervalSet(0, 4)})
    # i - 1 is -1 at i = 0: the cast covers 0xFFFFFFFF, not [uint32(-1), 3].
    assert s.is_everything()
    s = analyzer.int_set((i + 1).astype("uint32"), {i: tvm_tirx.sym.IntervalSet(0, 4)})
    assert analyzer.can_prove_equal(s.min_value, 1) and analyzer.can_prove_equal(s.max_value, 5)


def test_z3_models_wrap():
    analyzer = Analyzer()
    if not analyzer.is_z3_enabled:
        pytest.skip("Z3 prover is disabled in this build")
    x = tirx.Var("x", "uint32")
    v = x.astype("int32")
    assert not analyzer.can_prove(v >= 0)
    # Only a wrapping model proves these.
    assert analyzer.can_prove(tirx.Or(x < T.uint32(2**31), v < 0))
    assert analyzer.can_prove(tirx.Or(x != T.uint32(2**32 - 1), v == -1))
    y = tirx.Var("y", "int32")
    assert analyzer.can_prove(tirx.Or(y >= 0, y.astype("uint32") >= T.uint32(2**31)))


if __name__ == "__main__":
    tvm_tirx.testing.main()
