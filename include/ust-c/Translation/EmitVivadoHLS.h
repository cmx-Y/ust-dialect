#ifndef UST_C_TRANSLATION_EMITVIVADOHLS_H
#define UST_C_TRANSLATION_EMITVIVADOHLS_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_CAPI_EXPORTED MlirLogicalResult mlirEmitVivadoHls(MlirModule module,
                                                    MlirStringCallback callback,
                                                    void *userData);

#ifdef __cplusplus
}
#endif

#endif // UST_C_TRANSLATION_EMITVIVADOHLS_H
