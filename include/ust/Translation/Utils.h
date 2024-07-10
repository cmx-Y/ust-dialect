#ifndef UST_TRANSLATION_UTILS_H
#define UST_TRANSLATION_UTILS_H

#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/IR/AffineExprVisitor.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "llvm/Support/raw_ostream.h"
#include "ust/Translation/EmitVivadoHLS.h"

using namespace mlir;
using namespace ust;

//===----------------------------------------------------------------------===//
// Base Classes
//===----------------------------------------------------------------------===//

/// This class maintains the mutable state that cross-cuts and is shared by the
/// various emitters.
class USTEmitterState {
public:
  explicit USTEmitterState(raw_ostream &os) : os(os) {}

  // The stream to emit to.
  raw_ostream &os;

  bool encounteredError = false;
  unsigned currentIndent = 0;

  // This table contains all declared values.
  DenseMap<Value, SmallString<8>> nameTable;
  std::map<std::string, int> nameConflictCnt;

private:
  USTEmitterState(const USTEmitterState &) = delete;
  void operator=(const USTEmitterState &) = delete;
};

/// This is the base class for all of the HLSCpp Emitter components.
class USTEmitterBase {
public:
  explicit USTEmitterBase(USTEmitterState &state)
      : state(state), os(state.os) {}

  InFlightDiagnostic emitError(Operation *op, const Twine &message) {
    state.encounteredError = true;
    return op->emitError(message);
  }

  raw_ostream &indent() { return os.indent(state.currentIndent); }

  void addIndent() { state.currentIndent += 2; }
  void reduceIndent() { state.currentIndent -= 2; }

  // All of the mutable state we are maintaining.
  USTEmitterState &state;

  // The stream to emit to.
  raw_ostream &os;

  /// Value name management methods.
  SmallString<8> addName(Value val, bool isPtr = false, std::string name = "");

  SmallString<8> getName(Value val);

  bool isDeclared(Value val) {
    if (getName(val).empty()) {
      return false;
    } else
      return true;
  }

private:
  USTEmitterBase(const USTEmitterBase &) = delete;
  void operator=(const USTEmitterBase &) = delete;
};

void fixUnsignedType(Value &result, bool isUnsigned);
void fixUnsignedType(memref::GlobalOp &op, bool isUnsigned);

#endif // UST_TRANSLATION_UTILS_H