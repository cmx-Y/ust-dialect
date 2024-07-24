
#ifndef UST_ATTRS
#define UST_ATTRS

include "ust/Dialect/USTOps.td"
include "mlir/IR/EnumAttr.td"


def SparseInfo : UST_Attr<"SparseInfo"> {
  let summary = "Loop information";
  let description = [{
    This attribute includes loop information used by estimator and optimizer,
    including flattened trip count, iteration latency, and minimum initial
    interval (II).
  }];

  let hasCustomAssemblyFormat = 1;
  let mnemonic = "s";
  let parameters = (ins "int64_t":$posSize);
}

#endif // UST_ATTRS