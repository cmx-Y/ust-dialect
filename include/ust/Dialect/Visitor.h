#ifndef UST_DIALECT_HLSCPP_VISITOR_H
#define UST_DIALECT_HLSCPP_VISITOR_H

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "llvm/ADT/TypeSwitch.h"

namespace mlir {
namespace ust {

/// This class is a visitor for SSACFG operation nodes.
template <typename ConcreteType, typename ResultType, typename... ExtraArgs>
class HLSCppVisitorBase {
public:
  ResultType dispatchVisitor(Operation *op, ExtraArgs... args) {
    auto *thisCast = static_cast<ConcreteType *>(this);
    return TypeSwitch<Operation *, ResultType>(op)
        .template Case<
            // Float binary expressions.
            arith::CmpFOp, arith::AddFOp, arith::SubFOp, arith::MulFOp,

            // Integer binary expressions.
            arith::CmpIOp, arith::AddIOp, arith::SubIOp, arith::MulIOp,

            // Memref-related statements.
            memref::LoadOp, memref::StoreOp,

            // Bufferization operations
            bufferization::ToMemrefOp, bufferization::ToTensorOp,

            // Special operations.
            arith::ConstantOp, func::ReturnOp,

            //SparseTensor operations
            sparse_tensor::ToPositionsOp, sparse_tensor::ToCoordinatesOp,
            sparse_tensor::ToValuesOp,

            // SCF statements.
            scf::ForOp, scf::IfOp, scf::ParallelOp, scf::ReduceOp,
            scf::ReduceReturnOp, scf::YieldOp, scf::WhileOp, scf::ConditionOp>(
            [&](auto opNode) -> ResultType {
              return thisCast->visitOp(opNode, args...);
            })
        .Default([&](auto opNode) -> ResultType {
          return thisCast->visitInvalidOp(op, args...);
        });
  }

  /// This callback is invoked on any invalid operations.
  ResultType visitInvalidOp(Operation *op, ExtraArgs... args) {
    op->emitOpError("is unsupported operation.");
    abort();
  }

  /// This callback is invoked on any operations that are not handled by the
  /// concrete visitor.
  ResultType visitUnhandledOp(Operation *op, ExtraArgs... args) {
    return ResultType();
  }

#define HANDLE(OPTYPE)                                                         \
  ResultType visitOp(OPTYPE op, ExtraArgs... args) {                           \
    return static_cast<ConcreteType *>(this)->visitUnhandledOp(op, args...);   \
  }

  // Float binary expressions.
  HANDLE(arith::CmpFOp);
  HANDLE(arith::AddFOp);
  HANDLE(arith::SubFOp);
  HANDLE(arith::MulFOp);

  // Integer binary expressions.
  HANDLE(arith::CmpIOp);
  HANDLE(arith::AddIOp);
  HANDLE(arith::SubIOp);
  HANDLE(arith::MulIOp);

  // Memref-related statements.
  HANDLE(memref::LoadOp);
  HANDLE(memref::StoreOp);

  // Bufferization operations
  HANDLE(bufferization::ToMemrefOp);
  HANDLE(bufferization::ToTensorOp);

  // Special operations.
  HANDLE(arith::ConstantOp);
  HANDLE(func::ReturnOp);

  // SparseTensor operations
  HANDLE(sparse_tensor::ToPositionsOp);
  HANDLE(sparse_tensor::ToCoordinatesOp);
  HANDLE(sparse_tensor::ToValuesOp);

  // SCF statements.
  HANDLE(scf::ForOp);
  HANDLE(scf::IfOp);
  HANDLE(scf::ParallelOp);
  HANDLE(scf::ReduceOp);
  HANDLE(scf::ReduceReturnOp);
  HANDLE(scf::YieldOp);
  HANDLE(scf::WhileOp);
  HANDLE(scf::ConditionOp);

#undef HANDLE
};
} // namespace ust
} // namespace mlir

#endif // UST_DIALECT_HLSCPP_VISITOR_H