
#include "ust/Translation/EmitVivadoHLS.h"
#include "ust-c/Translation/EmitVivadoHLS.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"
#include "mlir/CAPI/Utils.h"

using namespace mlir;
using namespace ust;

MlirLogicalResult mlirEmitVivadoHls(MlirModule module,
                                    MlirStringCallback callback,
                                    void *userData) {
  mlir::detail::CallbackOstream stream(callback, userData);
  return wrap(emitVivadoHLS(unwrap(module), stream));
}