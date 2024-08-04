#ifndef UST_DIALECT
#define UST_DIALECT

include "mlir/IR/OpBase.td"

def UST_Dialect : Dialect {
  let name = "ust";
  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
  let cppNamespace = "::mlir::ust";
}

#endif // UST_DIALECT