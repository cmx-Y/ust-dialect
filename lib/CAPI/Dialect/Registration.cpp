#include "ust-c/Dialect/Registration.h"
#include "ust/Transforms/Passes.h"

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Func/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Transforms/Passes.h"

#include "mlir/InitAllDialects.h"

void ustMlirRegisterAllDialects(MlirContext context) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::func::FuncDialect,
                  mlir::arith::ArithDialect, mlir::tensor::TensorDialect,
                  mlir::affine::AffineDialect, mlir::math::MathDialect,
                  mlir::memref::MemRefDialect, mlir::pdl::PDLDialect,
                  mlir::transform::TransformDialect>();
  unwrap(context)->appendDialectRegistry(registry);
  unwrap(context)->loadAllAvailableDialects();
}

void ustMlirRegisterAllPasses() {
  // General passes
  mlir::registerTransformsPasses();

  // Conversion passes
  mlir::registerConversionPasses();

  // Dialect passes
  mlir::affine::registerAffinePasses();
  mlir::arith::registerArithPasses();
  mlir::LLVM::registerLLVMPasses();
  mlir::memref::registerMemRefPasses();
  mlir::registerLinalgPasses();
  mlir::registerTransformsPasses();

  mlir::ust::registerUSTPasses();
}