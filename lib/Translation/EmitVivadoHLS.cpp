#include "ust/Translation/EmitVivadoHLS.h"
#include "ust/Dialect/Visitor.h"
#include "ust/Translation/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AffineExprVisitor.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "llvm/Support/raw_ostream.h"

#include <iostream>

using namespace mlir;
using namespace ust;

//===----------------------------------------------------------------------===//
// ModuleEmitter Class Declaration
//===----------------------------------------------------------------------===//

namespace {
class ModuleEmitter : public USTEmitterBase {
public:
  using operand_range = Operation::operand_range;
  explicit ModuleEmitter(USTEmitterState &state) : USTEmitterBase(state) {}

  /// Top-level MLIR module emitter.
  void emitModule(ModuleOp module);

private:

  void emitFunction(func::FuncOp func);
  void emitHostFunction(func::FuncOp func);
};
} // namespace

void ModuleEmitter::emitFunction(func::FuncOp func) {

  // Emit function signature.
  os << "void " << func.getName() << "(\n";
  addIndent();

  reduceIndent();
  os << "\n) {";

  // Emit function body.
  addIndent();

  reduceIndent();
  os << "}\n";

  // An empty line.
  os << "\n";
}

void ModuleEmitter::emitHostFunction(func::FuncOp func) {
  if (func.getBlocks().size() != 1)
    emitError(func, "has zero or more than one basic blocks.");

  os << "/// This is top function.\n";

  // Emit function signature.
  os << "int main(int argc, char **argv) {\n";
  addIndent();

  os << "  return 0;\n";
  reduceIndent();
  os << "}\n";

  // An empty line.
  os << "\n";
}

/// Top-level MLIR module emitter.
void ModuleEmitter::emitModule(ModuleOp module) {
  std::string device_header = R"XXX(
//===------------------------------------------------------------*- C++ -*-===//
//
// Automatically generated file for High-level Synthesis (HLS).
//
//===----------------------------------------------------------------------===//
#include <algorithm>
#include <ap_axi_sdata.h>
#include <ap_fixed.h>
#include <ap_int.h>
#include <hls_math.h>
#include <hls_stream.h>
#include <math.h>
#include <stdint.h>
using namespace std;
)XXX";

  std::string host_header = R"XXX(
//===------------------------------------------------------------*- C++ -*-===//
//
// Automatically generated file for host
//
//===----------------------------------------------------------------------===//
// standard C/C++ headers
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <time.h>

// vivado hls headers
#include "kernel.h"
#include <ap_fixed.h>
#include <ap_int.h>
#include <hls_stream.h>

#include <ap_axi_sdata.h>
#include <ap_fixed.h>
#include <ap_int.h>
#include <hls_math.h>
#include <hls_stream.h>
#include <math.h>
#include <stdint.h>

)XXX";

  if (module.getName().has_value() && module.getName().value() == "host") {
    os << host_header;
    for (auto op : module.getOps<func::FuncOp>()) {
      if (op.getName() == "main")
        emitHostFunction(op);
      else
        emitFunction(op);
    }
  } else {
    os << device_header;
    for (auto &op : *module.getBody()) {
      if (auto func = dyn_cast<func::FuncOp>(op))
        emitFunction(func);
      // else if (auto cst = dyn_cast<memref::GlobalOp>(op))
      //   emitGlobal(cst);
      else
        emitError(&op, "is unsupported operation.");
    }
  }
}

//===----------------------------------------------------------------------===//
// Entry of ust-translate
//===----------------------------------------------------------------------===//

LogicalResult ust::emitVivadoHLS(ModuleOp module, llvm::raw_ostream &os) {
  USTEmitterState state(os);
  ModuleEmitter(state).emitModule(module);
  return failure(state.encounteredError);
}

void ust::registerEmitVivadoHLSTranslation() {
  static TranslateFromMLIRRegistration toVivadoHLS(
      "emit-vivado-hls", "Emit Vivado HLS", emitVivadoHLS,
      [&](DialectRegistry &registry) {
        // clang-format off
        registry.insert<
          mlir::scf::SCFDialect,
          mlir::func::FuncDialect,
          mlir::sparse_tensor::SparseTensorDialect,
          mlir::arith::ArithDialect,
          bufferization::BufferizationDialect
        >();
        // clang-format on
      });
}