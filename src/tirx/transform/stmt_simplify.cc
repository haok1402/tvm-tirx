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
 * \file stmt_simplify.cc
 * \brief Statement simplifier based on analyzer
 */

#include <tvm/ffi/cast.h>
#include <tvm/ffi/extra/structural_hash.h>
#include <tvm/ffi/extra/structural_mutate.h>
#include <tvm/ffi/extra/structural_visit.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/prim/expr.h>
#include <tvm/ir/prim/op.h>
#include <tvm/sym/analyzer.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/op/annotation.h>
#include <tvm/tirx/op/memory.h>
#include <tvm/tirx/stmt_functor_with_constraint_context.h>
#include <tvm/tirx/transform.h>

#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace tvm {
namespace tirx {
using namespace tvm::prim;

namespace {

class StmtSimplifier : public StmtExprMutatorWithConstraintContext {
 public:
  using StmtExprMutatorWithConstraintContext::Mutate;
  using StmtExprMutatorWithConstraintContext::Mutate_;
  explicit StmtSimplifier(const sym::ConstraintContext& context)
      : StmtExprMutatorWithConstraintContext(context) {}

  Function Run(Function func);

 protected:
  using Parent = StmtExprMutatorWithConstraintContext;

  UnchangedOr<ffi::Any> Mutate(ffi::AnyView input, InplaceMode inplace_mode) final;

  UnchangedOr<Stmt> Mutate_(const ForObj* op, InplaceMode inplace_mode) final;

  UnchangedOr<Stmt> Mutate_(const BindObj* op, InplaceMode inplace_mode) override;

  UnchangedOr<Stmt> Mutate_(const IfObj* op, InplaceMode inplace_mode) override;

  // eliminate useless stores
  UnchangedOr<Stmt> Mutate_(const TensorStoreObj* op, InplaceMode inplace_mode) override;

 private:
  bool ArrayDeepEqual(const ffi::Array<PrimExpr>& lhs, const ffi::Array<PrimExpr>& rhs);

  /* \brief Internal utility for checking conditionals
   *
   * Substitutes any known Bind values and then simplifies with the analyzer.
   */
  ffi::Optional<bool> ProveCondition(PrimExpr condition);

  // Pure Bind values kept for substitution into assert conditions.
  // Grows monotonically under SSA — no scope-based cleanup required.
  ffi::Map<Var, PrimExpr> non_inlined_bindings_;
  /* \brief Fold copies of bound values back into their variables.
   *
   * The analyzer substitutes every pure binding into its uses, which gives the rewrite rules the
   * structure they need but also undoes a value the program bound once: `x = e; ... x ... x`
   * prints as a dead `x` and `e` at every use (sm90-probe ISSUE-023). After simplification,
   * replace each verbatim copy of an in-scope binding's (substituted) value with the variable.
   * Only bindings whose value is non-trivial (not a constant or a variable, as CanInlineLet) and
   * whose variable the input uses at least twice take part, so single-use bindings print as
   * before.
   */
  PrimExpr FoldBoundValues(const PrimExpr& expr, const PrimExpr& original);

  struct ExprHash {
    size_t operator()(const PrimExpr& e) const { return ffi::StructuralHash::Hash(e); }
  };
  // Set while a TensorStore's indices are simplified: indices are left unfolded (see
  // Mutate_(TensorStoreObj)).
  bool fold_disabled_{false};
  // Variables bound to a pure, non-trivial value that the function uses at least twice.
  std::unordered_set<const VarObj*> foldable_vars_;
  // Each foldable binding's value as the analyzer substitutes it, keyed by the variable.
  std::unordered_map<const VarObj*, PrimExpr> substituted_values_;
};

// A binding that is left to the compiler: a constant, a variable (or a cast of one), or a value
// computed only from constants and special registers (threadIdx.x & 31, lane_id(), blockIdx.y * 4,
// ...). The last kind is cheap to rematerialize from S2R; holding it in a variable for the
// whole scope instead changes register allocation (e.g. +13 % SASS in one sm_100 registry
// kernel) without removing duplicated work.
bool IsTrivialBinding(const PrimExpr& value) {
  if (prim::IsConstNumber(value) || value.as<PrimVar>()) return true;
  if (const auto* cast = value.as<prim::CastObj>(); cast && cast->value.as<PrimVar>()) return true;
  bool has_var = false;
  ffi::StructuralVisit(
      value,
      [&](const VarObj*,
          ffi::StructuralVisitorObj*) -> ffi::Expected<ffi::Optional<ffi::VisitInterrupt>> {
        has_var = true;
        return ffi::VisitInterrupt();
      },
      [](const TypeObj*, ffi::StructuralVisitorObj*)
          -> ffi::Expected<ffi::Optional<ffi::VisitInterrupt>> { return std::nullopt; });
  return !has_var;
}

Function StmtSimplifier::Run(Function func) {
  AddParamConstraints(func);
  {
    // Count the uses (not the definitions) of every variable.
    std::unordered_map<const VarObj*, int> uses;
    ffi::StructuralVisit(
        func->body,
        [&](const VarObj* var, ffi::StructuralVisitorObj* visitor)
            -> ffi::Expected<ffi::Optional<ffi::VisitInterrupt>> {
          if (visitor->def_region_kind() == kTVMFFIDefRegionKindNone) ++uses[var];
          return std::nullopt;
        },
        [](const TypeObj*, ffi::StructuralVisitorObj*)
            -> ffi::Expected<ffi::Optional<ffi::VisitInterrupt>> { return std::nullopt; });
    ffi::StructuralVisit(
        func->body,
        [&](const BindObj* bind, ffi::StructuralVisitorObj* visitor)
            -> ffi::Expected<ffi::Optional<ffi::VisitInterrupt>> {
          if (auto value = bind->value.as<PrimExpr>()) {
            const PrimExpr& v = value.value();
            bool trivial = IsTrivialBinding(v);
            if (!trivial && v.ty().IsScalar() && uses[bind->var.get()] >= 2 &&
                SideEffect(v) <= CallEffectKind::kPure) {
              foldable_vars_.insert(bind->var.get());
            }
          }
          return visitor->DefaultVisitExpected(bind);
        },
        [](const TypeObj*, ffi::StructuralVisitorObj*)
            -> ffi::Expected<ffi::Optional<ffi::VisitInterrupt>> { return std::nullopt; });
  }
  auto* n = func.CopyOnWrite();
  // Shared string literals no longer enter the primitive analyzer.  Inline their
  // SSA bindings so existing literal arguments remain constants during lowering.
  ffi::Map<Var, StringImm> string_bindings;
  n->body = ffi::StructuralMap<ffi::WalkOrder::kPostOrder>(
                std::move(n->body),
                [&](const Bind& bind) -> ffi::UnchangedOr<ffi::Any> {
                  if (!bind.defined()) return ffi::Unchanged();
                  if (auto value = bind->value.as<StringImm>()) {
                    string_bindings.Set(bind->var, *value);
                    return ffi::Any(Evaluate(0));
                  }
                  return ffi::Unchanged();
                },
                [&](const Var& var) -> ffi::UnchangedOr<ffi::Any> {
                  if (auto value = string_bindings.Get(var)) return ffi::Any(*value);
                  return ffi::Unchanged();
                })
                .as_or_throw<SeqStmt>();
  n->body = Mutate(n->body, InplaceMode::kAllow)
                .as_or_throw<UnchangedOr<SeqStmt>>()
                .ValueOrUnchanged(n->body);
  return func;
}

UnchangedOr<ffi::Any> StmtSimplifier::Mutate(ffi::AnyView input, InplaceMode inplace_mode) {
  if (input.as<TensorType>()) {
    return ffi::Unchanged();
  }
  if (def_region_kind() != kTVMFFIDefRegionKindNone && input.as<Var>()) {
    return Parent::Mutate(input, inplace_mode);
  }
  // The annotation's structural hook records a premise for following statements.
  // Its void result is a PrimExpr too, but arithmetic simplification alone would
  // bypass that hook and lose the scope-wide guarantee.
  if (auto call = input.as<Call>(); call && call.value()->op.same_as(assume_always_op())) {
    return Parent::Mutate(input, inplace_mode);
  }
  if (auto expr = input.as<PrimExpr>()) {
    PrimExpr simplified = GetSyncedAnalyzer()->Simplify(*expr);
    if (!foldable_vars_.empty() && !fold_disabled_) simplified = FoldBoundValues(simplified, *expr);
    if (simplified.same_as(*expr) || prim::ExprDeepEqual()(simplified, *expr)) {
      return ffi::Unchanged();
    }
    return simplified;
  }
  return Parent::Mutate(input, inplace_mode);
}

UnchangedOr<Stmt> StmtSimplifier::Mutate_(const ForObj* op, InplaceMode inplace_mode) {
  return Parent::Mutate_(op, inplace_mode);
}

UnchangedOr<Stmt> StmtSimplifier::Mutate_(const BindObj* op, InplaceMode inplace_mode) {
  if (const auto* call = op->value.as<CallObj>()) {
    // Preserve tensor type metadata; only declaration data is simplified.
    if (call->op.same_as(tirx::alloc_tensor_op())) return ffi::Unchanged();
    if (call->op.same_as(tirx::decl_tensor_op())) {
      // The Call and its arguments may be shared even when the Bind is writable.
      auto data = this->Mutate(call->args[0], InplaceMode::kDisallow);
      if (data.UnchangedOrSameAs(call->args[0])) return ffi::Unchanged();
      auto value = ffi::make_object<CallObj>(*call);
      value->args.Set(0, std::move(data).ValueUnchecked());
      if (inplace_mode == InplaceMode::kAllow) {
        const_cast<BindObj*>(op)->value = Expr(std::move(value));
        return ffi::Unchanged();
      }
      auto copy = ffi::make_object<BindObj>(*op);
      copy->value = Expr(std::move(value));
      return Stmt(std::move(copy));
    }
  }
  auto prim_value = op->value.as<PrimExpr>();
  if (!prim_value) {
    return Parent::Mutate_(op, inplace_mode);
  }
  PrimExpr value =
      this->Mutate(prim_value.value(), inplace_mode).ValueOrUnchanged(prim_value.value());
  // Keep the statement: the shared sequence collector records its transformed
  // binding for following siblings after this callback returns.
  if (SideEffect(value) <= CallEffectKind::kPure) {
    // Record the binding so we can substitute it into assert conditions
    // (see Mutate_(const AssertStmtObj*, InplaceMode)).  Under SSA each var is
    // bound exactly once, so the map grows monotonically without key
    // conflicts.  No scope-based cleanup is needed because vars bound
    // in inner scopes are only referenced within those scopes; stale
    // entries are harmless and never consulted again.
    non_inlined_bindings_.Set(op->var, value);
  }

  if (value.same_as(op->value)) {
    return ffi::Unchanged();
  } else {
    if (inplace_mode == InplaceMode::kAllow) {
      auto* n = const_cast<BindObj*>(op);
      n->value = std::move(value);
      return ffi::Unchanged();
    }
    auto n = ffi::make_object<BindObj>(*op);
    n->value = std::move(value);
    return Stmt(n);
  }
}

UnchangedOr<Stmt> StmtSimplifier::Mutate_(const IfObj* op, InplaceMode inplace_mode) {
  if (ffi::Optional<bool> cond = ProveCondition(op->condition)) {
    if (cond.value()) {
      return this->Mutate(op->then_case, inplace_mode).ValueOrUnchanged(op->then_case);
    } else if (op->else_case) {
      return this->Mutate(op->else_case.value(), inplace_mode)
          .ValueOrUnchanged(op->else_case.value());
    } else {
      return Evaluate(0);
    }
  } else {
    return Parent::Mutate_(op, inplace_mode);
  }
}

UnchangedOr<Stmt> StmtSimplifier::Mutate_(const TensorStoreObj* op, InplaceMode inplace_mode) {
  // Indices keep the substituted form: later passes and the CUDA codegen prove alignment and
  // vector access patterns on them (e.g. `i * 4` in a Ramp base) without the bindings in scope.
  // Only the stored value is folded.
  // The store may be mutated in place, so keep the value as written for FoldBoundValues.
  PrimExpr original_value = op->value;
  fold_disabled_ = true;
  TensorStore store = Parent::Mutate_(op, inplace_mode)
                          .ValueOrUnchanged(ffi::GetRef<Stmt>(op))
                          .as_or_throw<TensorStore>();
  fold_disabled_ = false;
  if (!foldable_vars_.empty()) {
    PrimExpr folded = FoldBoundValues(store->value, original_value);
    if (!folded.same_as(store->value)) {
      auto n = ffi::make_object<TensorStoreObj>(*store.get());
      n->value = std::move(folded);
      store = Stmt(std::move(n)).as_or_throw<TensorStore>();
    }
  }
  if (const TensorLoadObj* load = store->value.as<TensorLoadObj>()) {
    TensorVar tensor = load->source.as_or_throw<tvm::tirx::TensorVar>();
    if (tensor.same_as(store->dest.as_or_throw<TensorVar>()) &&
        ArrayDeepEqual(load->indices, store->indices)) {
      return Evaluate(0);
    }
  }
  return store;
}

bool StmtSimplifier::ArrayDeepEqual(const ffi::Array<PrimExpr>& lhs,
                                    const ffi::Array<PrimExpr>& rhs) {
  if (lhs.size() != rhs.size()) {
    return false;
  }
  for (size_t i = 0; i < lhs.size(); i++) {
    if (!prim::ExprDeepEqual()(lhs[i], rhs[i])) {
      return false;
    }
  }
  return true;
}

PrimExpr StmtSimplifier::FoldBoundValues(const PrimExpr& expr, const PrimExpr& original) {
  if (expr.as<IntImmObj>() || expr.as<FloatImmObj>() || expr.as<VarObj>()) return expr;
  // Undo only the analyzer's substitution: fold the bindings that `original` referenced,
  // directly or through another binding's value. An expression the program wrote out itself
  // stays as written, even if it equals some binding's value.
  std::unordered_map<const VarObj*, PrimExpr> in_scope;
  constraints->ForEach([&](const sym::Constraint& constraint) {
    if (const auto* binding = constraint.as<sym::BindingConstraintObj>()) {
      if (foldable_vars_.count(binding->var.get()))
        in_scope.insert_or_assign(binding->var.get(), binding->value);
    }
  });
  if (in_scope.empty()) return expr;
  std::unordered_set<const VarObj*> referenced;
  std::vector<PrimExpr> worklist{original};
  while (!worklist.empty()) {
    PrimExpr e = worklist.back();
    worklist.pop_back();
    ffi::StructuralVisit(
        e,
        [&](const VarObj* var,
            ffi::StructuralVisitorObj*) -> ffi::Expected<ffi::Optional<ffi::VisitInterrupt>> {
          auto it = in_scope.find(var);
          if (it != in_scope.end() && referenced.insert(var).second) worklist.push_back(it->second);
          return std::nullopt;
        },
        [](const TypeObj*, ffi::StructuralVisitorObj*)
            -> ffi::Expected<ffi::Optional<ffi::VisitInterrupt>> { return std::nullopt; });
  }
  if (referenced.empty()) return expr;
  // The in-scope foldable bindings, value -> variable. The context is the live scope, so a
  // variable bound in a branch that has ended cannot be folded into code after it.
  std::unordered_map<PrimExpr, Var, ExprHash, prim::ExprDeepEqual> by_value;
  constraints->ForEach([&](const sym::Constraint& constraint) {
    const auto* binding = constraint.as<sym::BindingConstraintObj>();
    if (!binding || !referenced.count(binding->var.get())) return;
    auto it = substituted_values_.find(binding->var.get());
    if (it == substituted_values_.end()) {
      PrimExpr value = GetSyncedAnalyzer()->Simplify(binding->value);
      it = substituted_values_.emplace(binding->var.get(), value).first;
    }
    if (prim::IsConstNumber(it->second) || it->second.as<VarObj>()) return;
    by_value.emplace(it->second, binding->var);
  });
  if (by_value.empty()) return expr;
  // Vector values, vector index structure (Ramp / Broadcast) and loads keep the substituted
  // form: later passes and the codegen prove alignment and access patterns on them. Mask them
  // with placeholder variables, fold the rest, then put them back. (A pre-order callback that
  // returns a node as its own replacement does not stop the walk from descending into it.)
  std::unordered_map<const VarObj*, PrimExpr> holes;
  PrimExpr masked = ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(
                        expr,
                        [&](const PrimExpr& e) -> ffi::Expected<ffi::UnchangedOr<ffi::Any>> {
                          if (e.as<VarObj>() || e.as<IntImmObj>() || e.as<FloatImmObj>())
                            return ffi::Unchanged();
                          if (!e.ty().IsScalar() || e.as<prim::RampObj>() ||
                              e.as<prim::BroadcastObj>() || e.as<TensorLoadObj>()) {
                            PrimVar hole("fold_hole", e.ty());
                            holes.emplace(hole.get(), e);
                            return ffi::Any(hole);
                          }
                          return ffi::Unchanged();
                        })
                        .as_or_throw<PrimExpr>();
  // Pre-order, so the largest matching subexpression is folded first.
  PrimExpr folded = ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(
                        masked,
                        [&](const PrimExpr& e) -> ffi::Expected<ffi::UnchangedOr<ffi::Any>> {
                          if (e.as<VarObj>() || e.as<IntImmObj>() || e.as<FloatImmObj>()) {
                            return ffi::Unchanged();
                          }
                          auto it = by_value.find(e);
                          if (it != by_value.end()) return ffi::Any(it->second);
                          return ffi::Unchanged();
                        })
                        .as_or_throw<PrimExpr>();
  if (folded.same_as(masked)) return expr;
  if (holes.empty()) return folded;
  return ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(
             folded,
             [&](const Var& var) -> ffi::Expected<ffi::UnchangedOr<ffi::Any>> {
               auto it = holes.find(var.get());
               if (it != holes.end()) return ffi::Any(it->second);
               return ffi::Unchanged();
             })
      .as_or_throw<PrimExpr>();
}

ffi::Optional<bool> StmtSimplifier::ProveCondition(PrimExpr condition) {
  auto f_substitute = [this](const Var& var) -> ffi::Expected<ffi::UnchangedOr<ffi::Any>> {
    if (auto repl = non_inlined_bindings_.Get(var)) return ffi::Any(*std::move(repl));
    return ffi::Unchanged();
  };
  condition = ffi::StructuralMap<ffi::WalkOrder::kPreOrder>(condition, f_substitute)
                  .as_or_throw<PrimExpr>();
  condition = GetSyncedAnalyzer()->Simplify(condition);
  if (const auto* as_int = condition.as<IntImmObj>()) {
    return as_int->value != 0;
  } else {
    return std::nullopt;
  }
}

}  // namespace

Function StmtSimplify(Function func, const sym::Analyzer& analyzer) {
  return ffi::make_object<StmtSimplifier>(analyzer->GetConstraintContext())->Run(std::move(func));
}

namespace transform {

Pass StmtSimplify() {
  auto pass_func = [](Function f, IRModule m, PassContext ctx) {
    sym::Analyzer analyzer;
    return tirx::StmtSimplify(std::move(f), analyzer);
  };
  return CreateFunctionPass(pass_func, 0, "tirx.StmtSimplify");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tirx.transform.StmtSimplify", StmtSimplify);
}

}  // namespace transform
}  // namespace tirx
}  // namespace tvm
