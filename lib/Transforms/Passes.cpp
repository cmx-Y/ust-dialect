#include "ust/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Pass/PassOptions.h"

using namespace mlir;
using namespace mlir::ust;
//===----------------------------------------------------------------------===//
// Pass registration
//===----------------------------------------------------------------------===//

namespace {
struct USTPipelineOptions : public PassPipelineOptions<USTPipelineOptions> {
  Option<unsigned> iPosSize{
      *this, "pos-size", llvm::cl::init(2),
      llvm::cl::desc("Specify the size of the position tensor.")};
  Option<unsigned> iCrdSize{
      *this, "crd-size", llvm::cl::init(2),
      llvm::cl::desc("Specify the size of the coordinate tensor.")};
  Option<unsigned> iValSize{
      *this, "val-size", llvm::cl::init(2),
      llvm::cl::desc("Specify the size of the value tensor.")};
};
} // namespace

void ust::registerUSTPipeline() {
  PassPipelineRegistration<USTPipelineOptions>(
      "ust-pipeline",
      "Compile TOSA (from Torch-MLIR) to HLS C++ with ScaleFlow",
      [](OpPassManager &pm, const USTPipelineOptions &opts) {

        pm.addPass(ust::createSetSparseInfoPass(opts.iPosSize, opts.iCrdSize,
                                                opts.iValSize));

      });
}

void mlir::ust::registerUSTPasses() {
    ::registerPasses();
    ::registerUSTPipeline();
    }