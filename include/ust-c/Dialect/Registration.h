
#ifndef UST_MLIR_C_REGISTRATION_H
#define UST_MLIR_C_REGISTRATION_H

#include "mlir/CAPI/IR.h"
#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

/** Registers all dialects with a context.
 * This is needed before creating IR for these Dialects.
 */
MLIR_CAPI_EXPORTED void ustMlirRegisterAllDialects(MlirContext context);

/** Registers all passes for symbolic access with the global registry. */
MLIR_CAPI_EXPORTED void ustMlirRegisterAllPasses();

#ifdef __cplusplus
}
#endif

#endif // UST_MLIR_C_REGISTRATION_H