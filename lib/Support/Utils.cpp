#include "ust/Support/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
using namespace mlir;
using namespace ust;

//===----------------------------------------------------------------------===//
// HLSCpp attribute utils
//===----------------------------------------------------------------------===//

std::vector<std::string> ust::split_names(const std::string &arg_names) {
  std::stringstream ss(arg_names);
  std::vector<std::string> args;
  while (ss.good()) {
    std::string substr;
    getline(ss, substr, ',');
    args.push_back(substr);
  }
  return args;
}