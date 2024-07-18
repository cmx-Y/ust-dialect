
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include "ust/Bindings/Python/USTModule.h"
#include "ust-c/Translation/EmitVivadoHLS.h"
#include "ust/Transforms/Passes.h"
#include "mlir-c/Bindings/Python/Interop.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"
#include "mlir/CAPI/IR.h"
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/PDL/IR/PDLOps.h"
#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#include "llvm-c/ErrorHandling.h"
#include "llvm/Support/Signals.h"

namespace py = pybind11;
using namespace mlir::python::adaptors;

using namespace mlir;
using namespace mlir::python;
using namespace ust;

//===----------------------------------------------------------------------===//
// Customized Python classes
//===----------------------------------------------------------------------===//

// PybindUtils.h
class PyFileAccumulator {
public:
  PyFileAccumulator(const pybind11::object &fileObject, bool binary)
      : pyWriteFunction(fileObject.attr("write")), binary(binary) {}

  void *getUserData() { return this; }

  MlirStringCallback getCallback() {
    return [](MlirStringRef part, void *userData) {
      pybind11::gil_scoped_acquire acquire;
      PyFileAccumulator *accum = static_cast<PyFileAccumulator *>(userData);
      if (accum->binary) {
        // Note: Still has to copy and not avoidable with this API.
        pybind11::bytes pyBytes(part.data, part.length);
        accum->pyWriteFunction(pyBytes);
      } else {
        pybind11::str pyStr(part.data,
                            part.length); // Decodes as UTF-8 by default.
        accum->pyWriteFunction(pyStr);
      }
    };
  }

private:
  pybind11::object pyWriteFunction;
  bool binary;
};

//===----------------------------------------------------------------------===//
// Pass APIs
//===----------------------------------------------------------------------===//

static bool loopBoundConstant(MlirModule &mlir_mod) {
  py::gil_scoped_release();
  auto mod = unwrap(mlir_mod);
  return applyLoopBoundConstant(mod);
}

//===----------------------------------------------------------------------===//
// Emission APIs
//===----------------------------------------------------------------------===//

static bool emitVivadoHls(MlirModule &mod, py::object fileObject) {
  PyFileAccumulator accum(fileObject, false);
  py::gil_scoped_release();
  return mlirLogicalResultIsSuccess(
      mlirEmitVivadoHls(mod, accum.getCallback(), accum.getUserData()));
}

//===----------------------------------------------------------------------===//
// UST Python module definition
//===----------------------------------------------------------------------===//

PYBIND11_MODULE(_ust, m) {
  m.doc() = "UST Python Native Extension";
  llvm::sys::PrintStackTraceOnErrorSignal(/*argv=*/"");
  LLVMEnablePrettyStackTrace();

  auto ust_m = m.def_submodule("ust");

  // register dialects
  ust_m.def(
      "register_dialect",
      [](MlirContext context) {
        mlir::DialectRegistry registry;
        // mlir::ust::registerTransformDialectExtension(registry);
        unwrap(context)->appendDialectRegistry(registry);
      },
      py::arg("context") = py::none());

  // Apply transform to a design.
  ust_m.def("apply_transform", [](MlirModule &mlir_mod) {
    ModuleOp module = unwrap(mlir_mod);

    // Simplify the loop structure after the transform.
    PassManager pm(module.getContext());
    pm.addNestedPass<func::FuncOp>(
        mlir::affine::createSimplifyAffineStructuresPass());
    pm.addPass(createCanonicalizerPass());
    if (failed(pm.run(module)))
      throw py::value_error("failed to apply the post-transform optimization");
  });

  // Pass APIs.
  ust_m.def("loop_bound_constant", &loopBoundConstant);

  // Codegen APIs.
  ust_m.def("emit_vhls", &emitVivadoHls);

}
