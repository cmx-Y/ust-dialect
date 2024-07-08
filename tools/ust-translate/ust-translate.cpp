#include "ust/Translation/EmitVivadoHLS.h"
#include "mlir/InitAllTranslations.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"

int main(int argc, char **argv) {
  mlir::registerAllTranslations();
  mlir::ust::registerEmitVivadoHLSTranslation();

  return failed(mlir::mlirTranslateMain(
      argc, argv, "UST MLIR Dialect Translation Tool"));
}
