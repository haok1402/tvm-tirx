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
 * \file backend/cuda/op/target_builtin.cc
 *
 *  builtin intrinsic operators specific to CUDA target.
 */
#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/ffi/function.h>
#include <tvm/ir/expr.h>
#include <tvm/ir/op.h>
#include <tvm/ir/prim/op.h>
#include <tvm/runtime/base.h>
#include <tvm/runtime/memory_scope.h>
#include <tvm/sym/analyzer.h>
#include <tvm/tirx/op/attr.h>
#include <tvm/tirx/type.h>

#include <string>
#include <unordered_map>

namespace tvm {

namespace tirx {
namespace builtin {

namespace {
void RegisterDeviceIntrinsics();

Type InferTypeMovSreg(const CallObj* call) {
  TVM_FFI_CHECK(!call->args.empty(), TypeError) << "mov_sreg expects a bit width";
  const auto* bits = call->args[0].as<IntImmObj>();
  TVM_FFI_CHECK(bits, TypeError) << "mov_sreg expects a constant bit width";
  return PrimType::Int(static_cast<int64_t>(bits->value));
}

Type InferTypeDeclTmem(const CallObj* call) {
  static const auto scope_tmem = runtime::MemoryScope::Get("tmem");
  TVM_FFI_CHECK_EQ(call->args.size(), 1U, ValueError) << "decl_tmem expects one address operand";
  TVM_FFI_CHECK_EQ(call->ty_args.size(), 1U, ValueError)
      << "decl_tmem expects exactly one TensorType argument";
  auto type = call->ty_args[0].as_or_throw<TensorType>();
  TVM_FFI_CHECK(type->scope == scope_tmem, ValueError) << "decl_tmem requires a tmem tensor type";
  auto addr_type = call->args[0]->ty.as_or_throw<PrimType>();
  TVM_FFI_CHECK(addr_type.IsScalar() &&
                    addr_type.MatchesCode(DLDataTypeCode::kDLInt, DLDataTypeCode::kDLUInt) &&
                    addr_type.bits() == 32,
                TypeError)
      << "decl_tmem address must be a scalar 32-bit integer";
  return type;
}

void ValidateDeclTmem(const CallObj* call) {
  Type inferred = InferTypeDeclTmem(call);
  TVM_FFI_CHECK(ffi::StructuralEqual()(call->ty, inferred), TypeError)
      << "decl_tmem result must match its TensorType argument";
  TVM_FFI_CHECK(!call->attrs.has_value(), ValueError) << "decl_tmem does not accept attributes";
}

// Hardware limits of the CUDA launch indices, per axis x, y, z: grid dimensions up to
// 2^31 - 1, 65535 and 65535; block dimensions up to 1024, 1024 and 64; at most 16 CTAs per
// cluster. They give the analyzer a sign and a range for index arithmetic, e.g. so that
// floordiv(block_idx, c) lowers to truncdiv without a negative-operand fix-up.
constexpr int64_t kGridLimit[3] = {(int64_t{1} << 31) - 1, 65535, 65535};
constexpr int64_t kBlockLimit[3] = {1024, 1024, 64};
// Per-architecture assumption: at most 16 CTAs per cluster, the non-portable maximum of
// sm_90 and sm_100 (8 is portable). FConstIntBound sees only the call, not the target, so the
// cap cannot come from it; LaunchConfig enforces the same cap (MAX_CLUSTER_CTAS in
// backend/cuda/launch/_impl.py) so that a TIRx launch never exceeds this bound. Raise both
// together for an architecture with larger clusters.
constexpr int64_t kClusterLimit = 16;

ffi::Optional<sym::ConstIntBound> CudaRegisterBound(const std::string& name) {
  // name is a PTX special register such as "ctaid.x" or "cluster_nctarank".
  auto axis = [&]() -> int {
    char last = name.empty() ? 0 : name.back();
    return name.size() > 2 && name[name.size() - 2] == '.' && last >= 'x' && last <= 'z'
               ? last - 'x'
               : -1;
  };
  auto base = name.substr(0, name.rfind('.'));
  int dim = axis();
  if (dim >= 0) {
    if (base == "tid") return sym::ConstIntBound(0, kBlockLimit[dim] - 1);
    if (base == "ntid") return sym::ConstIntBound(1, kBlockLimit[dim]);
    if (base == "ctaid" || base == "clusterid") return sym::ConstIntBound(0, kGridLimit[dim] - 1);
    if (base == "nctaid" || base == "nclusterid") return sym::ConstIntBound(1, kGridLimit[dim]);
    if (base == "cluster_ctaid") return sym::ConstIntBound(0, kClusterLimit - 1);
    if (base == "cluster_nctaid") return sym::ConstIntBound(1, kClusterLimit);
    return std::nullopt;
  }
  if (name == "laneid") return sym::ConstIntBound(0, 31);
  if (name == "cluster_ctarank") return sym::ConstIntBound(0, kClusterLimit - 1);
  if (name == "cluster_nctarank") return sym::ConstIntBound(1, kClusterLimit);
  return std::nullopt;
}

ffi::Optional<sym::ConstIntBound> ConstIntBoundCudaAxis(const CallObj* call) {
  static const std::unordered_map<std::string, std::string> registers = {
      {"tirx.cuda.block_idx", "ctaid"},           {"tirx.cuda.thread_idx", "tid"},
      {"tirx.cuda.cluster_id", "clusterid"},      {"tirx.cuda.cluster_cta_id", "cluster_ctaid"},
      {"tirx.cuda.grid_dim", "nctaid"},           {"tirx.cuda.block_dim", "ntid"},
      {"tirx.cuda.cluster_dim", "cluster_nctaid"}};
  const auto* axis = call->args.empty() ? nullptr : call->args[0].as<StringImmObj>();
  auto it = registers.find(std::string(call->op.as<OpObj>()->name));
  if (!axis || it == registers.end()) return std::nullopt;
  return CudaRegisterBound(it->second + "." + std::string(axis->value));
}

ffi::Optional<sym::ConstIntBound> ConstIntBoundMovSreg(const CallObj* call) {
  const auto* name = call->args.size() > 1 ? call->args[1].as<StringImmObj>() : nullptr;
  if (!name) return std::nullopt;
  return CudaRegisterBound(std::string(name->value));
}

Type InferTypeCudaAxis(const CallObj* call) {
  TVM_FFI_CHECK_EQ(call->args.size(), 1U, ValueError);
  const auto* axis = call->args[0].as<StringImmObj>();
  TVM_FFI_CHECK(axis && (axis->value == "x" || axis->value == "y" || axis->value == "z"),
                ValueError)
      << "CUDA index axis must be x, y, or z";
  return PrimType::Int(32);
}

template <size_t N>
static Type InferTypeReturnArgType(const CallObj* call) {
  TVM_FFI_CHECK_GT(call->args.size(), N, ValueError)
      << "Return type inference requires argument " << N;
  return call->args[N]->ty;
}

}  // namespace

void RegisterCudaTargetBuiltins() {
  static bool registered = false;
  if (registered) return;
  registered = true;

  OpDef("tirx.cuda.decl_tmem")
      .signature(sig::arg<IntExpr>("addr", "Encoded TMEM base address in physical columns."),
                 sig::ty_arg<TensorType>("tensor_type"))
      .set_native_fn_attr<&InferTypeDeclTmem>(tvm::op_attr::kInferType)
      .set_validator(ffi::reflection::NativeFunctionView<void(
                         const CallObj*)>::FromNative<&ValidateDeclTmem>(),
                     true)
      .set_attr(tvm::op_attr::kCallEffectKind, CallEffectKind::kPure)
      .expose();

  OpDef("tirx.cuda.bmma_sync")
      .set_attr(tvm::op_attr::kFixedReturnType, PrimType::Void())
      .signature(sig::arg("fragment_d", "The D fragment."),
                 sig::arg<IntExpr>("index_d", "The D fragment index."),
                 sig::arg("fragment_a", "The A fragment."),
                 sig::arg<IntExpr>("index_a", "The A fragment index."),
                 sig::arg("fragment_b", "The B fragment."),
                 sig::arg<IntExpr>("index_b", "The B fragment index."),
                 sig::arg("fragment_c", "The C fragment."),
                 sig::arg<IntExpr>("index_c", "The C fragment index."))
      .set_attr(tvm::op_attr::kCallEffectKind, CallEffectKind::kOpaque)
      .expose();

  // Raw legacy cp.async form emitted by InjectPTXAsyncCopy (and round-tripped by
  // the T.s_tir.cp_async_raw.legacy 6-arg surface). It carries the element dtype in Call.dtype
  // and prints it with ty=; user-issued copies go through T.ptx instead.
  OpDef("tirx.s_tir.cp_async_raw")
      .signature(sig::arg("dst_ptr", "The destination pointer."),
                 sig::arg<IntExpr>("dst_offset", "The destination offset."),
                 sig::arg("src_ptr", "The source pointer."),
                 sig::arg<IntExpr>("src_offset", "The source offset."),
                 sig::arg<IntExpr>("cp_size", "The copy size."), sig::var_args("args"))
      .set_attr(tvm::op_attr::kCallEffectKind, CallEffectKind::kOpaque)
      .expose();

  OpDef("tirx.cuda.timer_init")
      .signature(sig::arg("profiler_buffer", "The profiler buffer."),
                 sig::arg("profiler_tag", "The profiler tag."),
                 sig::arg<IntExpr>("profiler_write_offset", "The profiler write offset."),
                 sig::arg<IntExpr>("num_groups", "The number of groups."),
                 sig::arg<IntExpr>("group_id", "The group identifier."))
      .set_attr(tvm::op_attr::kCallEffectKind, CallEffectKind::kOpaque)
      .set_attr(tvm::op_attr::kFixedReturnType, PrimType::Void())
      .expose();

  OpDef("tirx.cuda.timer_start")
      .signature(sig::arg("event_type", "The event type."),
                 sig::arg("profiler_buffer", "The profiler buffer."),
                 sig::arg("profiler_tag", "The profiler tag."),
                 sig::arg<IntExpr>("profiler_write_offset", "The profiler write offset."),
                 sig::arg<IntExpr>("profiler_write_stride", "The profiler write stride."),
                 sig::arg("leader_cond", "The leader condition."))
      .set_attr(tvm::op_attr::kCallEffectKind, CallEffectKind::kOpaque)
      .set_attr(tvm::op_attr::kFixedReturnType, PrimType::Void())
      .expose();

  OpDef("tirx.cuda.timer_end")
      .signature(sig::arg("event_type", "The event type."),
                 sig::arg("profiler_buffer", "The profiler buffer."),
                 sig::arg("profiler_tag", "The profiler tag."),
                 sig::arg<IntExpr>("profiler_write_offset", "The profiler write offset."),
                 sig::arg<IntExpr>("profiler_write_stride", "The profiler write stride."),
                 sig::arg("leader_cond", "The leader condition."))
      .set_attr(tvm::op_attr::kCallEffectKind, CallEffectKind::kOpaque)
      .set_attr(tvm::op_attr::kFixedReturnType, PrimType::Void())
      .expose();

  OpDef("tirx.cuda.timer_finalize")
      .signature(sig::arg("profiler_buffer", "The profiler buffer."),
                 sig::arg("profiler_tag", "The profiler tag."),
                 sig::arg<IntExpr>("profiler_write_offset", "The profiler write offset."),
                 sig::arg<IntExpr>("profiler_write_stride", "The profiler write stride."),
                 sig::arg("leader_cond", "The leader condition."))
      .set_attr(tvm::op_attr::kCallEffectKind, CallEffectKind::kOpaque)
      .set_attr(tvm::op_attr::kFixedReturnType, PrimType::Void())
      .expose();

  RegisterDeviceIntrinsics();
}

namespace {

TVM_FFI_NO_INLINE void RegisterDeviceIntrinsicAttrs(OpDef& def, CallEffectKind effect_kind) {
  def.set_attr(tvm::op_attr::kCallEffectKind, effect_kind);
}

template <typename... Specs>
OpDef& RegisterDeviceIntrinsic(OpDef&& def, CallEffectKind effect_kind, const Specs&... specs) {
  def.signature(specs...);
  RegisterDeviceIntrinsicAttrs(def, effect_kind);
  return def;
}

void RegisterDeviceIntrinsics() {
  for (const char* name : {"block_idx", "thread_idx", "cluster_id", "cluster_cta_id", "grid_dim",
                           "block_dim", "cluster_dim"}) {
    RegisterDeviceIntrinsic(OpDef(std::string("tirx.cuda.") + name), CallEffectKind::kPure,
                            sig::arg<StringImm>("axis"))
        .set_native_fn_attr<&InferTypeCudaAxis>(tvm::op_attr::kInferType)
        .set_native_fn_attr<&ConstIntBoundCudaAxis>(sym::op_attr::kConstIntBound)
        .expose();
  }
  for (const char* name : {"linear_thread_id", "warp_id", "lane_id", "warpgroup_id",
                           "warp_in_warpgroup", "thread_in_warpgroup", "cta_pair_id"}) {
    RegisterDeviceIntrinsic(OpDef(std::string("tirx.cuda.") + name), CallEffectKind::kPure)
        .set_attr(tvm::op_attr::kFixedReturnType, PrimType::Int(32))
        .expose();
  }
  OpDef("tirx.cuda.warpgroup_id",
        "The index of the calling thread's warpgroup (128 threads) in the CTA.\n\n"
        "It is computed once at kernel entry from a warp shuffle of the warp index. To keep "
        "arithmetic on it in uniform registers inside warp-specialized code (e.g. WGMMA shared-"
        "memory descriptors), use broadcast_warpgroup_id there instead.");
  // A cross-lane broadcast at its call site: opaque, so that no pass moves it into or out of
  // the warp-converged code where the caller placed it.
  RegisterDeviceIntrinsic(
      OpDef("tirx.cuda.broadcast_warpgroup_id",
            "The warpgroup index broadcast from lane 0 of the calling warp, at the call site.\n\n"
            "Equals warpgroup_id(). nvcc knows the result is warp-uniform, so arithmetic on it "
            "stays in uniform registers; e.g. WGMMA shared-memory descriptors derived from it need "
            "no R2UR per instruction. Unlike warpgroup_id(), which is computed at kernel entry, "
            "the broadcast runs where it is called, e.g. after setmaxnreg in a consumer branch.\n\n"
            "Contract: call only from warp-converged code, where all 32 lanes of the warp are "
            "active (it is a full-mask __shfl_sync). Calling it under a lane-divergent condition, "
            "such as lane_id() == 0, is undefined behavior. The script constructor binds the "
            "result to a variable at the call site, so the shuffle runs once, there, and later "
            "uses of the value (e.g. in a short-circuit T.Or operand) read the variable."),
      CallEffectKind::kOpaque)
      .set_attr(tvm::op_attr::kFixedReturnType, PrimType::Int(32))
      .expose();
  RegisterDeviceIntrinsic(OpDef("tirx.cuda.any_sync"), CallEffectKind::kPure,
                          sig::arg<IntExpr>("mask"), sig::arg("pred"))
      .set_attr(tvm::op_attr::kFixedReturnType, PrimType::Int(32))
      .expose();
  RegisterDeviceIntrinsic(OpDef("tirx.cuda.atomic_add"), CallEffectKind::kOpaque,
                          sig::arg("res_addr"), sig::arg("value"))
      .set_native_fn_attr<&InferTypeReturnArgType<1>>(tvm::op_attr::kInferType)
      .expose();
  RegisterDeviceIntrinsic(OpDef("tirx.cuda.atomic_cas"), CallEffectKind::kOpaque, sig::arg("ptr"),
                          sig::arg("old_val"), sig::arg("new_val"))
      .set_native_fn_attr<&InferTypeReturnArgType<1>>(tvm::op_attr::kInferType)
      .expose();
  RegisterDeviceIntrinsic(OpDef("tirx.cuda.wait_until"), CallEffectKind::kOpaque, sig::arg("dst"),
                          sig::arg("ptr"), sig::arg("condition"), sig::arg("scope"),
                          sig::arg("space"), sig::arg("ptx_type"), sig::arg<IntExpr>("backoff_ns"))
      .set_attr(tvm::op_attr::kFixedReturnType, PrimType::Void())
      .expose();
  RegisterDeviceIntrinsic(OpDef("tirx.cuda.ballot_sync"), CallEffectKind::kOpaque,
                          sig::arg<IntExpr>("mask"), sig::arg("pred"))
      .set_attr(tvm::op_attr::kFixedReturnType, PrimType::UInt(32))
      .expose();
  RegisterDeviceIntrinsic(OpDef("tirx.cuda.bfloat1622float2"), CallEffectKind::kOpaque,
                          sig::arg("packed"))
      .set_attr(tvm::op_attr::kFixedReturnType, PrimType::UInt(64))
      .expose();
  RegisterDeviceIntrinsic(OpDef("tirx.cuda.bfloat162float"), CallEffectKind::kOpaque,
                          sig::arg("src"))
      .set_attr(tvm::op_attr::kFixedReturnType, PrimType::Float(32))
      .expose();
  RegisterDeviceIntrinsic(OpDef("tirx.cuda.clock64"), CallEffectKind::kOpaque)
      .set_attr(tvm::op_attr::kFixedReturnType, PrimType::UInt(64))
      .expose();
  RegisterDeviceIntrinsic(OpDef("tirx.cuda.cluster_sync"), CallEffectKind::kOpaque)
      .set_attr(tvm::op_attr::kFixedReturnType, PrimType::Void())
      .expose();
  RegisterDeviceIntrinsic(OpDef("tirx.cuda.cta_reduce"), CallEffectKind::kOpaque, sig::arg("value"),
                          sig::arg("op"), sig::arg<IntExpr>("num_warps"), sig::arg("scratch"))
      .set_native_fn_attr<&InferTypeReturnArgType<0>>(tvm::op_attr::kInferType)
      .expose();
  RegisterDeviceIntrinsic(OpDef("tirx.cuda.cta_sync"), CallEffectKind::kOpaque)
      .set_attr(tvm::op_attr::kFixedReturnType, PrimType::Void())
      .expose();
  RegisterDeviceIntrinsic(OpDef("tirx.cuda.cvta_generic_to_shared"), CallEffectKind::kOpaque,
                          sig::arg("ptr"))
      .set_attr(tvm::op_attr::kFixedReturnType, PrimType::UInt(32))
      .expose();
  RegisterDeviceIntrinsic(OpDef("tirx.cuda.elect_sync"), CallEffectKind::kOpaque)
      .set_attr(tvm::op_attr::kFixedReturnType, PrimType::UInt(32))
      .expose();
  RegisterDeviceIntrinsic(OpDef("tirx.cuda.fadd2_rn"), CallEffectKind::kOpaque, sig::arg("a"),
                          sig::arg("b"))
      .set_attr(tvm::op_attr::kFixedReturnType, PrimType::UInt(64))
      .expose();
  RegisterDeviceIntrinsic(OpDef("tirx.cuda.fdividef"), CallEffectKind::kPure, sig::arg("x"),
                          sig::arg("y"))
      .set_attr(tvm::op_attr::kFixedReturnType, PrimType::Float(32))
      .expose();
  RegisterDeviceIntrinsic(OpDef("tirx.cuda.ffs_u32"), CallEffectKind::kOpaque,
                          sig::arg<IntExpr>("value"))
      .set_attr(tvm::op_attr::kFixedReturnType, PrimType::Int(32))
      .expose();
  RegisterDeviceIntrinsic(OpDef("tirx.cuda.float22bfloat162_rn"), CallEffectKind::kOpaque,
                          sig::arg("v0"), sig::arg("v1"))
      .set_attr(tvm::op_attr::kFixedReturnType, PrimType::UInt(32))
      .expose();
  RegisterDeviceIntrinsic(OpDef("tirx.cuda.float22bfloat162_rn_from_float2"),
                          CallEffectKind::kOpaque, sig::arg("packed"))
      .set_attr(tvm::op_attr::kFixedReturnType, PrimType::UInt(32))
      .expose();
  RegisterDeviceIntrinsic(OpDef("tirx.cuda.float22half2"), CallEffectKind::kOpaque, sig::arg("dst"),
                          sig::arg("src"))
      .set_attr(tvm::op_attr::kFixedReturnType, PrimType::Void())
      .expose();
  RegisterDeviceIntrinsic(OpDef("tirx.cuda.float2_x"), CallEffectKind::kOpaque, sig::arg("packed"))
      .set_attr(tvm::op_attr::kFixedReturnType, PrimType::Float(32))
      .expose();
  RegisterDeviceIntrinsic(OpDef("tirx.cuda.float2_y"), CallEffectKind::kOpaque, sig::arg("packed"))
      .set_attr(tvm::op_attr::kFixedReturnType, PrimType::Float(32))
      .expose();
  RegisterDeviceIntrinsic(OpDef("tirx.cuda.float8tohalf8"), CallEffectKind::kOpaque,
                          sig::arg("src_addr"), sig::arg("dst_addr"))
      .set_attr(tvm::op_attr::kFixedReturnType, PrimType::Void())
      .expose();
  RegisterDeviceIntrinsic(OpDef("tirx.cuda.float_as_uint"), CallEffectKind::kOpaque, sig::arg("x"))
      .set_attr(tvm::op_attr::kFixedReturnType, PrimType::UInt(32))
      .expose();
  RegisterDeviceIntrinsic(OpDef("tirx.cuda.fmul2_rn"), CallEffectKind::kOpaque, sig::arg("a"),
                          sig::arg("b"))
      .set_attr(tvm::op_attr::kFixedReturnType, PrimType::UInt(64))
      .expose();
  RegisterDeviceIntrinsic(OpDef("tirx.cuda.fp8x4_e4m3_from_float4"), CallEffectKind::kOpaque,
                          sig::arg("x"), sig::arg("y"), sig::arg("z"), sig::arg("w"))
      .set_attr(tvm::op_attr::kFixedReturnType, PrimType::UInt(32))
      .expose();
  RegisterDeviceIntrinsic(OpDef("tirx.cuda.func_call"), CallEffectKind::kOpaque,
                          sig::arg("func_name"), sig::var_args("args"))
      .set_attr(tvm::op_attr::kFixedReturnType, PrimType::Void())
      .expose();
  RegisterDeviceIntrinsic(OpDef("tirx.cuda.get_tmem_addr"), CallEffectKind::kOpaque,
                          sig::arg("addr"), sig::arg<IntExpr>("row_offset"),
                          sig::arg<IntExpr>("col_offset"))
      .set_attr(tvm::op_attr::kFixedReturnType, PrimType::UInt(32))
      .expose();
  RegisterDeviceIntrinsic(OpDef("tirx.cuda.grid_sync"), CallEffectKind::kOpaque)
      .set_attr(tvm::op_attr::kFixedReturnType, PrimType::Void())
      .expose();
  RegisterDeviceIntrinsic(OpDef("tirx.cuda.half2float"), CallEffectKind::kOpaque, sig::arg("src"))
      .set_attr(tvm::op_attr::kFixedReturnType, PrimType::Float(32))
      .expose();
  RegisterDeviceIntrinsic(OpDef("tirx.cuda.half8tofloat8"), CallEffectKind::kOpaque,
                          sig::arg("src_addr"), sig::arg("dst_addr"))
      .set_attr(tvm::op_attr::kFixedReturnType, PrimType::Void())
      .expose();
  RegisterDeviceIntrinsic(OpDef("tirx.cuda.hmax2"), CallEffectKind::kOpaque, sig::arg("a"),
                          sig::arg("b"))
      .set_attr(tvm::op_attr::kFixedReturnType, PrimType::UInt(32))
      .expose();
  RegisterDeviceIntrinsic(OpDef("tirx.cuda.hmin2"), CallEffectKind::kOpaque, sig::arg("a"),
                          sig::arg("b"))
      .set_attr(tvm::op_attr::kFixedReturnType, PrimType::UInt(32))
      .expose();
  RegisterDeviceIntrinsic(OpDef("tirx.cuda.make_float2"), CallEffectKind::kOpaque, sig::arg("x"),
                          sig::arg("y"))
      .set_attr(tvm::op_attr::kFixedReturnType, PrimType::UInt(64))
      .expose();
  RegisterDeviceIntrinsic(OpDef("tirx.cuda.mbarrier_wait"), CallEffectKind::kOpaque,
                          sig::arg("bar"), sig::arg<IntExpr>("phase"))
      .set_attr(tvm::op_attr::kFixedReturnType, PrimType::Void())
      .expose();
  // Blocking parity wait as one asm retry loop (cutlass ClusterBarrier::wait); the Python
  // wrapper T.cuda.mbarrier_wait_parity supplies the defaults for ticks, sem and scope.
  RegisterDeviceIntrinsic(OpDef("tirx.cuda.mbarrier_wait_parity"), CallEffectKind::kOpaque,
                          sig::arg("bar"), sig::arg<IntExpr>("phase"), sig::arg<IntExpr>("ticks"),
                          sig::arg<StringImm>("sem"), sig::arg<StringImm>("scope"))
      .set_attr(tvm::op_attr::kFixedReturnType, PrimType::Void())
      .expose();
  RegisterDeviceIntrinsic(OpDef("tirx.cuda.mbarrier_wait_acquire_cluster"), CallEffectKind::kOpaque,
                          sig::arg("bar"), sig::arg<IntExpr>("phase"))
      .set_attr(tvm::op_attr::kFixedReturnType, PrimType::Void())
      .expose();
  RegisterDeviceIntrinsic(OpDef("tirx.cuda.mov_sreg"), CallEffectKind::kPure,
                          sig::arg<IntExpr>("bits"), sig::arg("reg_name"))
      .set_native_fn_attr<&InferTypeMovSreg>(tvm::op_attr::kInferType)
      .set_native_fn_attr<&ConstIntBoundMovSreg>(sym::op_attr::kConstIntBound)
      .expose();
  RegisterDeviceIntrinsic(OpDef("tirx.cuda.nano_sleep"), CallEffectKind::kOpaque,
                          sig::arg<IntExpr>("time"))
      .set_attr(tvm::op_attr::kFixedReturnType, PrimType::Void())
      .expose();
  RegisterDeviceIntrinsic(OpDef("tirx.cuda.printf"), CallEffectKind::kOpaque, sig::arg("fmt"),
                          sig::var_args("args"))
      .set_attr(tvm::op_attr::kFixedReturnType, PrimType::Void())
      .expose();
  RegisterDeviceIntrinsic(OpDef("tirx.cuda.reduce_add_sync_u32"), CallEffectKind::kOpaque,
                          sig::arg<IntExpr>("mask"), sig::arg<IntExpr>("value"))
      .set_attr(tvm::op_attr::kFixedReturnType, PrimType::UInt(32))
      .expose();
  RegisterDeviceIntrinsic(OpDef("tirx.cuda.reduce_min_sync_u32"), CallEffectKind::kOpaque,
                          sig::arg<IntExpr>("mask"), sig::arg<IntExpr>("value"))
      .set_attr(tvm::op_attr::kFixedReturnType, PrimType::UInt(32))
      .expose();
  RegisterDeviceIntrinsic(OpDef("tirx.cuda.runtime_instr_desc"), CallEffectKind::kOpaque,
                          sig::arg("desc"), sig::arg<IntExpr>("sf_id"))
      .set_attr(tvm::op_attr::kFixedReturnType, PrimType::Void())
      .expose();
  RegisterDeviceIntrinsic(OpDef("tirx.cuda.smem_addr_from_uint64"), CallEffectKind::kOpaque,
                          sig::arg<IntExpr>("cluster_addr"))
      .set_attr(tvm::op_attr::kFixedReturnType, PrimType::UInt(32))
      .expose();
  RegisterDeviceIntrinsic(OpDef("tirx.cuda.syncthreads_and"), CallEffectKind::kOpaque,
                          sig::arg("cond"))
      .set_attr(tvm::op_attr::kFixedReturnType, PrimType::Int(64))
      .expose();
  RegisterDeviceIntrinsic(OpDef("tirx.cuda.syncthreads_or"), CallEffectKind::kOpaque,
                          sig::arg("cond"))
      .set_attr(tvm::op_attr::kFixedReturnType, PrimType::Int(64))
      .expose();
  RegisterDeviceIntrinsic(OpDef("tirx.cuda.thread_fence"), CallEffectKind::kOpaque)
      .set_attr(tvm::op_attr::kFixedReturnType, PrimType::Void())
      .expose();
  RegisterDeviceIntrinsic(OpDef("tirx.cuda.thread_rank"), CallEffectKind::kPure)
      .set_attr(tvm::op_attr::kFixedReturnType, PrimType::Int(32))
      .expose();
  RegisterDeviceIntrinsic(OpDef("tirx.cuda.trap_when_assert_failed"), CallEffectKind::kOpaque,
                          sig::arg("cond"))
      .set_attr(tvm::op_attr::kFixedReturnType, PrimType::Void())
      .expose();
  RegisterDeviceIntrinsic(OpDef("tirx.cuda.uint_as_float"), CallEffectKind::kOpaque,
                          sig::arg<IntExpr>("bits"))
      .set_attr(tvm::op_attr::kFixedReturnType, PrimType::Float(32))
      .expose();
  RegisterDeviceIntrinsic(OpDef("tirx.cuda.warp_reduce"), CallEffectKind::kOpaque,
                          sig::arg("value"), sig::arg("op"), sig::arg<IntExpr>("width"))
      .set_native_fn_attr<&InferTypeReturnArgType<0>>(tvm::op_attr::kInferType)
      .expose();
  RegisterDeviceIntrinsic(OpDef("tirx.cuda.warp_sync"), CallEffectKind::kOpaque)
      .set_attr(tvm::op_attr::kFixedReturnType, PrimType::Void())
      .expose();
  RegisterDeviceIntrinsic(OpDef("tirx.cuda.warpgroup_sync"), CallEffectKind::kOpaque,
                          sig::arg<IntExpr>("bar_no"))
      .set_attr(tvm::op_attr::kFixedReturnType, PrimType::Void())
      .expose();
  RegisterDeviceIntrinsic(OpDef("tirx.cuda.wgmma_encode_matrix_descriptor"),
                          CallEffectKind::kOpaque, sig::arg("desc"), sig::arg("addr"),
                          sig::arg<IntExpr>("ldo"), sig::arg<IntExpr>("sdo"),
                          sig::arg<IntExpr>("swizzle"))
      .set_attr(tvm::op_attr::kFixedReturnType, PrimType::Void())
      .expose();
  RegisterDeviceIntrinsic(OpDef("tirx.cuda.wgmma_noop_barrier"), CallEffectKind::kOpaque,
                          sig::arg("reg"))
      .set_attr(tvm::op_attr::kFixedReturnType, PrimType::Void())
      .expose();
  RegisterDeviceIntrinsic(OpDef("tirx.nvshmem.barrier_all"), CallEffectKind::kOpaque)
      .set_attr(tvm::op_attr::kFixedReturnType, PrimType::Void())
      .expose();
  RegisterDeviceIntrinsic(OpDef("tirx.nvshmem.fence"), CallEffectKind::kOpaque)
      .set_attr(tvm::op_attr::kFixedReturnType, PrimType::Void())
      .expose();
  RegisterDeviceIntrinsic(OpDef("tirx.nvshmem.getmem_nbi"), CallEffectKind::kOpaque,
                          sig::arg("dst"), sig::arg("src"), sig::arg<IntExpr>("nelems"),
                          sig::arg<IntExpr>("pe"))
      .set_attr(tvm::op_attr::kFixedReturnType, PrimType::Void())
      .expose();
  RegisterDeviceIntrinsic(OpDef("tirx.nvshmem.getmem_nbi_block"), CallEffectKind::kOpaque,
                          sig::arg("dst"), sig::arg("src"), sig::arg<IntExpr>("nelems"),
                          sig::arg<IntExpr>("pe"))
      .set_attr(tvm::op_attr::kFixedReturnType, PrimType::Void())
      .expose();
  RegisterDeviceIntrinsic(OpDef("tirx.nvshmem.getmem_nbi_warp"), CallEffectKind::kOpaque,
                          sig::arg("dst"), sig::arg("src"), sig::arg<IntExpr>("nelems"),
                          sig::arg<IntExpr>("pe"))
      .set_attr(tvm::op_attr::kFixedReturnType, PrimType::Void())
      .expose();
  RegisterDeviceIntrinsic(OpDef("tirx.nvshmem.my_pe"), CallEffectKind::kOpaque)
      .set_attr(tvm::op_attr::kFixedReturnType, PrimType::Int(32))
      .expose();
  RegisterDeviceIntrinsic(OpDef("tirx.nvshmem.n_pes"), CallEffectKind::kOpaque)
      .set_attr(tvm::op_attr::kFixedReturnType, PrimType::Int(32))
      .expose();
  RegisterDeviceIntrinsic(OpDef("tirx.nvshmem.putmem_nbi"), CallEffectKind::kOpaque,
                          sig::arg("dst"), sig::arg("src"), sig::arg<IntExpr>("nelems"),
                          sig::arg<IntExpr>("pe"))
      .set_attr(tvm::op_attr::kFixedReturnType, PrimType::Void())
      .expose();
  RegisterDeviceIntrinsic(OpDef("tirx.nvshmem.putmem_nbi_block"), CallEffectKind::kOpaque,
                          sig::arg("dst"), sig::arg("src"), sig::arg<IntExpr>("nelems"),
                          sig::arg<IntExpr>("pe"))
      .set_attr(tvm::op_attr::kFixedReturnType, PrimType::Void())
      .expose();
  RegisterDeviceIntrinsic(OpDef("tirx.nvshmem.putmem_nbi_warp"), CallEffectKind::kOpaque,
                          sig::arg("dst"), sig::arg("src"), sig::arg<IntExpr>("nelems"),
                          sig::arg<IntExpr>("pe"))
      .set_attr(tvm::op_attr::kFixedReturnType, PrimType::Void())
      .expose();
  RegisterDeviceIntrinsic(OpDef("tirx.nvshmem.putmem_signal_nbi"), CallEffectKind::kOpaque,
                          sig::arg("dst"), sig::arg("src"), sig::arg<IntExpr>("nelems"),
                          sig::arg("sig_addr"), sig::arg<IntExpr>("signal"), sig::arg("sig_op"),
                          sig::arg<IntExpr>("pe"))
      .set_attr(tvm::op_attr::kFixedReturnType, PrimType::Void())
      .expose();
  RegisterDeviceIntrinsic(OpDef("tirx.nvshmem.putmem_signal_nbi_block"), CallEffectKind::kOpaque,
                          sig::arg("dst"), sig::arg("src"), sig::arg<IntExpr>("nelems"),
                          sig::arg("sig_addr"), sig::arg<IntExpr>("signal"), sig::arg("sig_op"),
                          sig::arg<IntExpr>("pe"))
      .set_attr(tvm::op_attr::kFixedReturnType, PrimType::Void())
      .expose();
  RegisterDeviceIntrinsic(OpDef("tirx.nvshmem.putmem_signal_nbi_warp"), CallEffectKind::kOpaque,
                          sig::arg("dst"), sig::arg("src"), sig::arg<IntExpr>("nelems"),
                          sig::arg("sig_addr"), sig::arg<IntExpr>("signal"), sig::arg("sig_op"),
                          sig::arg<IntExpr>("pe"))
      .set_attr(tvm::op_attr::kFixedReturnType, PrimType::Void())
      .expose();
  RegisterDeviceIntrinsic(OpDef("tirx.nvshmem.quiet"), CallEffectKind::kOpaque)
      .set_attr(tvm::op_attr::kFixedReturnType, PrimType::Void())
      .expose();
  RegisterDeviceIntrinsic(OpDef("tirx.nvshmem.signal_op"), CallEffectKind::kOpaque,
                          sig::arg("sig_addr"), sig::arg<IntExpr>("signal"), sig::arg("sig_op"),
                          sig::arg<IntExpr>("pe"))
      .set_attr(tvm::op_attr::kFixedReturnType, PrimType::Void())
      .expose();
  RegisterDeviceIntrinsic(OpDef("tirx.nvshmem.wait_until"), CallEffectKind::kOpaque,
                          sig::arg("ivar"), sig::arg("cmp"), sig::arg<IntExpr>("cmp_value"),
                          sig::arg("type"))
      .set_attr(tvm::op_attr::kFixedReturnType, PrimType::Void())
      .expose();
}

}  // namespace

TVM_FFI_STATIC_INIT_BLOCK() { RegisterCudaTargetBuiltins(); }

}  // namespace builtin
}  // namespace tirx
}  // namespace tvm
