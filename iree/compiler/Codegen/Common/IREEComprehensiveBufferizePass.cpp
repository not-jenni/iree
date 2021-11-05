// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- IREEComprehensiveBufferizePass.cpp.cpp - -------------------------===//
//
// Wrapper pass to use MLIRs ComprehensiveBufferization pass.
//
//===----------------------------------------------------------------------===//

#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Codegen/Transforms/Transforms.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Flow/IR/FlowTypes.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Dialect/Shape/IR/ShapeOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/ErrorHandling.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Linalg/ComprehensiveBufferize/ComprehensiveBufferize.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/BufferUtils.h"
#include "mlir/Transforms/Passes.h"

#define DEBUG_TYPE "iree-codegen-linalg-bufferize"

namespace mlir {
template <typename TensorType>
static MemRefType getMemrefTypeForTensor(TensorType tensorType,
                                         MemRefLayoutAttrInterface layout = {},
                                         Attribute memorySpace = {}) {
  return MemRefType::get(tensorType.getShape(), tensorType.getElementType(),
                         layout, memorySpace);
}

namespace iree_compiler {

namespace {
/// Pass to convert from tensor based ops to memref based ops.
class IREEComprehensiveBufferizePass
    : public IREEComprehensiveBufferizeBase<IREEComprehensiveBufferizePass> {
 public:
  explicit IREEComprehensiveBufferizePass(
      linalg::AllocationCallbacks allocationFn)
          : allocationFn(allocationFn) {}
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithmeticDialect, IREE::Util::UtilDialect,
                    linalg::LinalgDialect, memref::MemRefDialect,
                    scf::SCFDialect, StandardOpsDialect, tensor::TensorDialect,
                    vector::VectorDialect>();
    linalg::registerBufferizableOpInterfaceExternalModels(registry);
  }
  void runOnOperation() override;

 private:
  linalg::AllocationCallbacks allocationFn;
};
}  // namespace

static bool isaTensor(Type t) { return t.isa<TensorType>(); };

/// Stitch comprehensive bufferization inside of IREE by proceeding as follows:
///   1. a. Bufferizes InterfaceBindingSubspanOp optimistically
///      b. Insert a memref::TensorLoad to serve as glue between the buffer and
///         tensor worlds.
///      c. Record aliasInfo of memref::TensorLoad manually
///      d. Record inplaceability of memref::TensorLoad manually
///      e. Record the bufferization of memref::TensorLoad manually
///   2. Rewrite all Flow::Dispatch::TensorLoad ops as Tensor::ExtractSliceOp
///      that comprehensive bufferization understands.
///   3. Specifically select the ops we want to bufferize / skip. In the future,
///      this may be better specified with a BufferizationOpInterface.
///   4. Perform analysis and bufferization on the ops.
void IREEComprehensiveBufferizePass::runOnOperation() {
  ModuleOp moduleOp = getOperation();
  MLIRContext *context = &getContext();

  for (auto funcOp : moduleOp.getOps<FuncOp>()) {
    OpBuilder b(context);

    // 1. First go over all hal.interface.binding.subspan ops and create
    // counterparts working with memrefs.
    BlockAndValueMapping bvm, tensorLoads;
    linalg::BufferizationAliasInfo aliasInfo(funcOp);
    // These are used until late, erase on scoped exit.
    SmallVector<Operation *> toEraseLate;
    auto scopeGuard = llvm::make_scope_exit([&]() {
      for (Operation *op : llvm::reverse(toEraseLate)) op->erase();
    });
    funcOp.walk([&](IREE::HAL::InterfaceBindingSubspanOp op) {
      auto shapedType =
          op.getResult().getType().dyn_cast<IREE::Flow::DispatchTensorType>();
      if (!shapedType || !shapedType.hasRank()) return;
      OpBuilder::InsertionGuard g(b);
      b.setInsertionPoint(op);
      // 1.a. Just change the result type of the InterfaceBindingSubspanOp to
      // from the base buffer.
      auto memRefType = getMemrefTypeForTensor(shapedType);
      auto baseBuffer = b.create<IREE::HAL::InterfaceBindingSubspanOp>(
          op->getLoc(), memRefType, op.binding(), op.byte_offset(),
          op.byte_length(), op.dynamic_dims());
      bvm.map(op, baseBuffer);

      // This op does not operate on core tensor types and has half-side
      // effecting semantics. It cannot be added to BufferizationAliasInfo.
      // Instead:
      // 1.b. Insert a memref::TensorLoad to serve as glue between the buffer
      // and tensor worlds.
      Value tensor = b.create<memref::TensorLoadOp>(op->getLoc(), baseBuffer);
      // 1.c. Insert a new entry manually into the existing aliasInfo.
      aliasInfo.createAliasInfoEntry(op.result());
      aliasInfo.createAliasInfoEntry(tensor);
      tensorLoads.map(op.result(), tensor);
      // 1.d. Mark tensors that bufferize to writeable memory as such.
      if (shapedType.getAccess() != IREE::Flow::TensorAccess::ReadOnly) {
        aliasInfo.setBufferizesToWritableMemory(tensor);
      }
      // 1.e. Save tensor -> baseBuffer into BVM.
      bvm.map(tensor, baseBuffer);

      // Drop the original op that is now bufferized.
      toEraseLate.push_back(op);
    });

    // 2. Rewrite all Flow::Dispatch::TensorLoad ops as Tensor::ExtractSliceOp.
    funcOp.walk<WalkOrder::PostOrder>([&](IREE::Flow::DispatchTensorLoadOp op) {
      OpBuilder b(op);
      Value v = b.create<tensor::ExtractSliceOp>(
          op->getLoc(), op.result().getType().cast<RankedTensorType>(),
          tensorLoads.lookup(op.source()), op.getMixedOffsets(),
          op.getMixedSizes(), op.getMixedStrides());
      // Insert a new entry manually into the existing aliasInfo.
      aliasInfo.createAliasInfoEntry(v);
      op.result().replaceAllUsesWith(v);
      toEraseLate.push_back(op);
    });
    funcOp.walk<WalkOrder::PostOrder>(
        [&](IREE::Flow::DispatchTensorStoreOp op) {
          OpBuilder b(op);
          Value v = b.create<tensor::InsertSliceOp>(
              op->getLoc(), op.value(), tensorLoads.lookup(op.target()),
              op.getMixedOffsets(), op.getMixedSizes(), op.getMixedStrides());
          // Insert a new entry manually into the existing aliasInfo.
          aliasInfo.createAliasInfoEntry(v);
          toEraseLate.push_back(op);
        });

    // TODO: Visit all the operations that return `tensor`s that are not handled
    // by comprehensive bufferize.

    // 3. Specifically select the ops we want to bufferize / skip. In the
    // future, this may be better specified with a BufferizationOpInterface.
    DominanceInfo domInfo(funcOp);
    SmallVector<Operation *> ops;
    ops.reserve(funcOp.body().front().getOperations().size());
    WalkResult opsSelected =
        funcOp.body().walk([&](Operation *op) -> WalkResult {
          if (isa<IREE::HAL::InterfaceBindingSubspanOp,
                  IREE::Flow::DispatchTensorLoadOp,
                  IREE::Flow::DispatchTensorStoreOp>(op)) {
            return WalkResult::advance();
          }
          if (llvm::none_of(op->getOperandTypes(), isaTensor) &&
              llvm::none_of(op->getResultTypes(), isaTensor)) {
            return WalkResult::advance();
          }
          if (op->getParentOfType<linalg::LinalgOp>())
            return WalkResult::advance();
          // TODO: if we want to bufferize function calls, we need FuncOp
          // and to pass a proper bufferizedFunctionTypes.
          if (isa<CallOpInterface>(op)) {
            return static_cast<LogicalResult>(op->emitError(
                "CallOpInterface bufferization not supported in IREE"));
          }
          ops.push_back(op);
          return WalkResult::advance();
        });

    // 4. Perform inplaceability analysis of `ops`.
    if (opsSelected.wasInterrupted() ||
        failed(linalg::inPlaceAnalysis(ops, aliasInfo, domInfo))) {
      return signalPassFailure();
    }

    // 5. Perform bufferization.
    for (Operation *op : ops) {
      if (failed(linalg::bufferizeOp(op, bvm, aliasInfo, allocationFn,
                                     /*bufferizedFunctionTypes=*/nullptr))) {
        return signalPassFailure();
      }
    }
  }
}

// TODO: pass this to comprehensive bufferize.
static Value defaultAllocationFn(OpBuilder &builder, Location loc,
                                 ArrayRef<int64_t> staticShape,
                                 Type elementType,
                                 ArrayRef<Value> dynamicSizes) {
  auto allocationType = MemRefType::get(staticShape, elementType);
  return builder.create<memref::AllocOp>(loc, allocationType, dynamicSizes);
}

std::unique_ptr<OperationPass<ModuleOp>> createIREEComprehensiveBufferizePass(
    linalg::AllocationCallbacks allocationFns) {
  return std::make_unique<IREEComprehensiveBufferizePass>(allocationFns);
}

void addIREEComprehensiveBufferizePasses(
    OpPassManager &passManager, linalg::AllocationCallbacks allocationFns) {
  passManager.addPass(createIREEComprehensiveBufferizePass(allocationFns));
  passManager.addPass(memref::createResolveShapedTypeResultDimsPass());
  passManager.addNestedPass<FuncOp>(createCanonicalizerPass());
  passManager.addNestedPass<FuncOp>(createCSEPass());
  passManager.addNestedPass<FuncOp>(createCleanupBufferAllocViewPass());
}

}  // namespace iree_compiler
}  // namespace mlir