#ifndef UST_TRANSLATION_EMITVIVADOHLS_H
#define UST_TRANSLATION_EMITVIVADOHLS_H

#include "mlir/IR/BuiltinOps.h"

namespace mlir {
namespace ust {

LogicalResult emitVivadoHLS(ModuleOp module, llvm::raw_ostream &os);
void registerEmitVivadoHLSTranslation();

} // namespace ust
} // namespace mlir

#endif // UST_TRANSLATION_EMITVIVADOHLS_H