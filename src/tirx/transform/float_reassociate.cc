/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file float_reassociate.cc
 * \brief Opt-in floating-point distribution of a multiplier over a sum.
 *
 * With the function attribute or PassContext option ``tirx.float_reassociate`` set, rewrite
 * ``(a +- b) * c`` and ``c * (a +- b)`` on floating-point values into ``a * c +- b * c`` when
 * ``c`` is a constant, a variable or a load (cheap to repeat). The backend compiler can then
 * contract ``a * c +- (b * c)`` into one FMA and compute a shared ``b * c`` once: for the
 * softmax exponent ``exp2((s - m) * c)`` over many ``s`` with one row maximum ``m``, each
 * element costs one FFMA instead of FADD + FMUL.
 *
 * The rewrite changes rounding (it is a reassociation, not a contraction of the expression as
 * written), so it is off by default. It never applies to integers.
 */
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/prim/op.h>
#include <tvm/ir/prim/vector_expr.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <utility>

namespace tvm {
namespace tirx {

namespace {

bool IsFloat(const PrimExpr& e) {
  return e.ty().MatchesCode(DLDataTypeCode::kDLFloat, DLDataTypeCode::kDLBfloat);
}

// A multiplier that is cheap and safe to evaluate twice.
bool IsRepeatable(const PrimExpr& e) {
  if (e.as<FloatImmObj>() || e.as<VarObj>() || e.as<TensorLoadObj>()) return true;
  if (const auto* b = e.as<prim::BroadcastObj>()) return IsRepeatable(b->value);
  return false;
}

class FloatReassociator : public StmtExprMutator {
 public:
  using StmtExprMutator::Mutate;
  using StmtExprMutator::Mutate_;

  UnchangedOr<PrimExpr> Mutate_(const prim::MulObj* op, InplaceMode inplace_mode) final {
    PrimExpr result =
        StmtExprMutator::Mutate_(op, inplace_mode).ValueOrUnchanged(ffi::GetRef<PrimExpr>(op));
    const auto* mul = result.as<prim::MulObj>();
    if (mul == nullptr || !IsFloat(result)) return result;
    PrimExpr distributed = Distribute(mul->a, mul->b);
    if (distributed.same_as(result)) return result;
    return distributed;
  }

 private:
  // (a +- b) * c -> a * c +- b * c, recursively over the sum; anything else stays a product.
  static PrimExpr Distribute(const PrimExpr& x, const PrimExpr& c) {
    if (!IsRepeatable(c)) {
      if (IsRepeatable(x) && (c.as<prim::AddObj>() || c.as<prim::SubObj>())) {
        return Distribute(c, x);
      }
      return prim::Mul(x, c);
    }
    if (const auto* add = x.as<prim::AddObj>()) {
      return prim::Add(Distribute(add->a, c), Distribute(add->b, c));
    }
    if (const auto* sub = x.as<prim::SubObj>()) {
      return prim::Sub(Distribute(sub->a, c), Distribute(sub->b, c));
    }
    return prim::Mul(x, c);
  }
};

}  // namespace

namespace transform {

Pass FloatReassociate() {
  auto pass_func = [](Function f, IRModule m, PassContext ctx) {
    bool enabled = f->GetAttr<bool>("tirx.float_reassociate")
                       .value_or(ctx->GetConfig<bool>("tirx.float_reassociate").value_or(false));
    if (!enabled) return f;
    auto* n = f.CopyOnWrite();
    n->body = FloatReassociator().Mutate(n->body, InplaceMode::kAllow).ValueOrUnchanged(n->body);
    return f;
  };
  return CreateFunctionPass(pass_func, 0, "tirx.FloatReassociate");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tirx.transform.FloatReassociate", FloatReassociate);
  ::tvm::transform::PassContext::RegisterConfigOption<bool>("tirx.float_reassociate");
}

}  // namespace transform
}  // namespace tirx
}  // namespace tvm
