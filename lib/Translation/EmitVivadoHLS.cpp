#include "ust/Translation/EmitVivadoHLS.h"
#include "ust/Dialect/Visitor.h"
#include "ust/Translation/Utils.h"
#include "ust/Support/Utils.h"

#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AffineExprVisitor.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "llvm/Support/raw_ostream.h"

#include <iostream>

using namespace mlir;
using namespace ust;

//===----------------------------------------------------------------------===//
// Utils
//===----------------------------------------------------------------------===//

// used for determine whether to generate C++ default types or ap_(u)int
static bool BIT_FLAG = false;

static SmallString<16> getTypeName(Type valType) {
  if (auto arrayType = valType.dyn_cast<ShapedType>())
    valType = arrayType.getElementType();

  // Handle float types.
  if (valType.isa<Float32Type>())
    return SmallString<16>("float");
  else if (valType.isa<Float64Type>())
    return SmallString<16>("double");

  // Handle integer types.
  else if (valType.isa<IndexType>())
    return SmallString<16>("int");
  else if (auto intType = valType.dyn_cast<IntegerType>()) {
    if (intType.getWidth() == 1) {
      if (!BIT_FLAG)
        return SmallString<16>("bool");
      else
        return SmallString<16>("ap_uint<1>");
    } else {
      std::string signedness = "";
      if (intType.getSignedness() == IntegerType::SignednessSemantics::Unsigned)
        signedness = "u";
      if (!BIT_FLAG) {
        switch (intType.getWidth()) {
        case 8:
        case 16:
        case 32:
        case 64:
          return SmallString<16>(signedness + "int" +
                                 std::to_string(intType.getWidth()) + "_t");
        default:
          return SmallString<16>("ap_" + signedness + "int<" +
                                 std::to_string(intType.getWidth()) + ">");
        }
      } else {
        return SmallString<16>("ap_" + signedness + "int<" +
                               std::to_string(intType.getWidth()) + ">");
      }
    }
  }

  // Handle (custom) fixed point types.
  else
    assert(1 == 0 && "Got unsupported type.");

  return SmallString<16>();
}

static SmallString<16> getTypeName(Value val) {
  // Handle memref, tensor, and vector types.
  auto valType = val.getType();
  return getTypeName(valType);
}

//===----------------------------------------------------------------------===//
// ModuleEmitter Class Declaration
//===----------------------------------------------------------------------===//

namespace {
class ModuleEmitter : public USTEmitterBase {
public:
  using operand_range = Operation::operand_range;
  explicit ModuleEmitter(USTEmitterState &state) : USTEmitterBase(state) {}

  /// SCF statement emitters.
  void emitScfFor(scf::ForOp op);
  void emitScfYield(scf::YieldOp op);
  void emitScfWhile(scf::WhileOp op);

  /// Special operation emitters.
  void emitConstant(arith::ConstantOp op);

  /// Memref-related statement emitters.
  void emitLoad(memref::LoadOp op);
  void emitStore(memref::StoreOp op);

  /// Bufferization-related statement emitters.
  void emitToMemref(bufferization::ToMemrefOp op);

  /// Sparse tensor-related statement emitters.
  void emitToPositions(sparse_tensor::ToPositionsOp op);
  void emitToCoordinates(sparse_tensor::ToCoordinatesOp op);
  void emitToValues(sparse_tensor::ToValuesOp op);

  /// Standard operation emitters.
  void emitBinary(Operation *op, const char *syntax);
  
  /// Top-level MLIR module emitter.
  void emitModule(ModuleOp module);

private:
  /// C++ component emitters.
  void emitValue(Value val, unsigned rank = 0, bool isPtr = false,
                 std::string name = "");
  void emitInfoAndNewLine(Operation *op);
  void emitBlock(Block &block);

  void emitFunction(func::FuncOp func);
  void emitHostFunction(func::FuncOp func);
};
} // namespace

//===----------------------------------------------------------------------===//
// StmtVisitor, ExprVisitor, and PragmaVisitor Classes
//===----------------------------------------------------------------------===//

namespace {
class StmtVisitor : public HLSCppVisitorBase<StmtVisitor, bool> {
public:
  StmtVisitor(ModuleEmitter &emitter) : emitter(emitter) {}

  using HLSCppVisitorBase::visitOp;
  /// SCF statements.
  bool visitOp(scf::ForOp op) { return emitter.emitScfFor(op), true; };
  bool visitOp(scf::ParallelOp op) { return true; };
  bool visitOp(scf::ReduceOp op) { return true; };
  bool visitOp(scf::ReduceReturnOp op) { return true; };
  bool visitOp(scf::YieldOp op) { return emitter.emitScfYield(op), true;};
  bool visitOp(scf::WhileOp op) { return emitter.emitScfWhile(op), true;};
  bool visitOp(scf::ConditionOp op) { return true; };

  /// Memref-related statements.
  bool visitOp(memref::LoadOp op) { return emitter.emitLoad(op), true;}
  bool visitOp(memref::StoreOp op) { return true; }

  /// Bufferization-related statements.
  bool visitOp(bufferization::ToMemrefOp op) { return emitter.emitToMemref(op), true; }
  bool visitOp(bufferization::ToTensorOp op) { return true; }

  /// Sparse tensor-related statements.
  bool visitOp(sparse_tensor::ToPositionsOp op) { return emitter.emitToPositions(op), true; }
  bool visitOp(sparse_tensor::ToCoordinatesOp op) { return emitter.emitToCoordinates(op), true; }
  bool visitOp(sparse_tensor::ToValuesOp op) { return emitter.emitToValues(op), true; }

private:
  ModuleEmitter &emitter;
};
} // namespace

namespace {
class ExprVisitor : public HLSCppVisitorBase<ExprVisitor, bool> {
public:
  ExprVisitor(ModuleEmitter &emitter) : emitter(emitter) {}

  using HLSCppVisitorBase::visitOp;

  /// Float binary expressions.
  bool visitOp(arith::CmpFOp op) {return true; }
  bool visitOp(arith::AddFOp op) { return emitter.emitBinary(op, "+"), true; }
  bool visitOp(arith::SubFOp op) { return emitter.emitBinary(op, "-"), true; }
  bool visitOp(arith::MulFOp op) { return emitter.emitBinary(op, "*"), true; }

  /// Integer binary expressions.
  bool visitOp(arith::CmpIOp op) { return true; }
  bool visitOp(arith::AddIOp op) { return emitter.emitBinary(op, "+"), true; }
  bool visitOp(arith::SubIOp op) { return emitter.emitBinary(op, "-"), true; }
  bool visitOp(arith::MulIOp op) { return emitter.emitBinary(op, "*"), true; }


   /// Special operations.
  bool visitOp(arith::ConstantOp op) { return emitter.emitConstant(op), true; }
  bool visitOp(func::ReturnOp op) { return true; }

private:
  ModuleEmitter &emitter;
};
} // namespace

/// C++ component emitters.
void ModuleEmitter::emitValue(Value val, unsigned rank, bool isPtr,
                              std::string name) {
  assert(!(rank && isPtr) && "should be either an array or a pointer.");

  // Value has been declared before or is a constant number.
  if (isDeclared(val)) {
    os << getName(val);
    for (unsigned i = 0; i < rank; ++i)
      os << "[iv" << i << "]";
    return;
  }

  os << getTypeName(val) << " ";

  if (name == "") {
    // Add the new value to nameTable and emit its name.
    os << addName(val, isPtr);
    for (unsigned i = 0; i < rank; ++i)
      os << "[iv" << i << "]";
  } else {
    os << addName(val, isPtr, name);
  }
}

void ModuleEmitter::emitInfoAndNewLine(Operation *op) {
  os << "\t//";
  // Print line number.
  if (auto loc = op->getLoc().dyn_cast<FileLineColLoc>())
    os << " L" << loc.getLine();

  // // Print schedule information.
  // if (auto timing = getTiming(op))
  //   os << ", [" << timing.getBegin() << "," << timing.getEnd() << ")";

  // // Print loop information.
  // if (auto loopInfo = getLoopInfo(op))
  //   os << ", iterCycle=" << loopInfo.getIterLatency()
  //      << ", II=" << loopInfo.getMinII();

  os << "\n";
}

/// MLIR component and HLS C++ pragma emitters.
void ModuleEmitter::emitBlock(Block &block) {
  for (auto &op : block) {
    if (ExprVisitor(*this).dispatchVisitor(&op))
      continue;

    if (StmtVisitor(*this).dispatchVisitor(&op))
      continue;

    emitError(&op, "can't be correctly emitted.");
  }
}

void ModuleEmitter::emitConstant(arith::ConstantOp op) {
  // This indicates the constant type is scalar (float, integer, or bool).
  if (isDeclared(op.getResult()))
    return;

  if (auto denseAttr = op.getValue().dyn_cast<DenseElementsAttr>()) {
    indent();
    Value result = op.getResult(); // memref
    fixUnsignedType(result, op->hasAttr("unsigned"));
    //emitArrayDecl(result);
    os << " = {";
    auto type = op.getResult().getType().cast<ShapedType>().getElementType();

    unsigned elementIdx = 0;
    for (auto element : denseAttr.getValues<Attribute>()) {
      if (type.isF32()) {
        auto value = element.cast<FloatAttr>().getValue().convertToFloat();
        if (std::isfinite(value))
          os << value;
        else if (value > 0)
          os << "INFINITY";
        else
          os << "-INFINITY";

      } else if (type.isF64()) {
        auto value = element.cast<FloatAttr>().getValue().convertToDouble();
        if (std::isfinite(value))
          os << value;
        else if (value > 0)
          os << "INFINITY";
        else
          os << "-INFINITY";

      } else if (type.isInteger(1))
        os << element.cast<BoolAttr>().getValue();
      else if (type.isIntOrIndex())
        os << element.cast<IntegerAttr>().getValue();
      else
        emitError(op, "array has unsupported element type.");

      if (elementIdx++ != denseAttr.getNumElements() - 1)
        os << ", ";
    }
    os << "};";
    emitInfoAndNewLine(op);
  } else
    emitError(op, "has unsupported constant type.");
}

// TODO: Support emmiting pipeline loop for bufferization::ToMemrefOp
void ModuleEmitter::emitToMemref(bufferization::ToMemrefOp op) {
  indent();
  auto tensor = op.getTensor().getType().dyn_cast<RankedTensorType>();
  if (tensor) {
    os << getTypeName(tensor.getElementType()) << " ";
  }
  os << addName(op.getResult(), false, "");
  for (auto &shape : tensor.getShape())
    os << "[" << shape << "]";
  os << ";\n";
}

// TODO: How to decide pos array's size
void ModuleEmitter::emitToPositions(sparse_tensor::ToPositionsOp op) {
  indent();
  auto tensor = op.getTensor().getType().dyn_cast<RankedTensorType>();
  if (tensor) {
    os << getTypeName(tensor.getElementType()) << " ";
  }
  os << addName(op.getResult(), false, "");
  os << "[3];\n";
}

void ModuleEmitter::emitToCoordinates(sparse_tensor::ToCoordinatesOp op) {
  indent();
  auto tensor = op.getTensor().getType().dyn_cast<RankedTensorType>();
  if (tensor) {
    os << getTypeName(tensor.getElementType()) << " ";
  }
  os << addName(op.getResult(), false, "");
  os << "[8];\n";
}

void ModuleEmitter::emitToValues(sparse_tensor::ToValuesOp op) {
  indent();
  auto tensor = op.getTensor().getType().dyn_cast<RankedTensorType>();
  if (tensor) {
    os << getTypeName(tensor.getElementType()) << " ";
  }
  os << addName(op.getResult(), false, "");
  os << "[8];\n";
}

void ModuleEmitter::emitScfFor(scf::ForOp op) {
  indent();
  os << "for (";
  auto iterVar = op.getInductionVar();

  // Emit lower bound.
  emitValue(iterVar);
  os << " = ";
  emitValue(op.getLowerBound());
  os << "; ";

  // Emit upper bound.
  emitValue(iterVar);
  os << " < ";
  emitValue(op.getUpperBound());
  os << "; ";

  // Emit increase step.
  emitValue(iterVar);
  os << " += ";
  emitValue(op.getStep());
  os << ") {";
  emitInfoAndNewLine(op);

  addIndent();
  emitBlock(*op.getBody());
  reduceIndent();

  indent();
  os << "}\n";
}

void ModuleEmitter::emitLoad(memref::LoadOp op) {
  indent();
  Value result = op.getResult();
  fixUnsignedType(result, op->hasAttr("unsigned"));
  emitValue(result);
  os << " = ";
  auto memref = op.getMemRef();
  emitValue(memref);
  auto attr = memref.getType().dyn_cast<MemRefType>().getMemorySpace();
  if (attr &&
      attr.cast<StringAttr>().getValue().str().substr(0, 6) == "stream") {
    auto attr_str = attr.cast<StringAttr>().getValue().str();
    int S_index = attr_str.find("S"); // spatial
    int T_index = attr_str.find("T"); // temporal
    if (S_index != -1 && T_index != -1) {
      auto st_str = attr_str.substr(S_index, T_index - S_index + 1);
      std::reverse(st_str.begin(), st_str.end());
      auto indices = op.getIndices();
      st_str = st_str.substr(0, indices.size());
      std::reverse(st_str.begin(), st_str.end());
      for (unsigned i = 0; i < indices.size(); ++i) {
        if (st_str[i] == 'S') {
          os << "[";
          emitValue(indices[i]);
          os << "]";
        }
      }
    }
    os << ".read(); // ";
    emitValue(memref); // comment
  }
  for (auto index : op.getIndices()) {
    os << "[";
    emitValue(index);
    os << "]";
  }
  os << ";";
  emitInfoAndNewLine(op);
}

void ModuleEmitter::emitBinary(Operation *op, const char *syntax) {
  // auto rank = emitNestedLoopHead(op->getResult(0));
  indent();
  Value result = op->getResult(0);
  fixUnsignedType(result, op->hasAttr("unsigned"));
  emitValue(result);
  os << " = ";
  emitValue(op->getOperand(0));
  os << " " << syntax << " ";
  emitValue(op->getOperand(1));
  os << ";";
  emitInfoAndNewLine(op);
  // emitNestedLoopTail(rank);
}

void ModuleEmitter::emitScfYield(scf::YieldOp op) {
  if (op.getNumOperands() == 0)
    return;

  // For now, only and scf::If operations will use scf::Yield to return
  // generated values.
  if (auto parentOp = dyn_cast<scf::IfOp>(op->getParentOp())) {
    unsigned resultIdx = 0;
    for (auto result : parentOp.getResults()) {
      // unsigned rank = emitNestedLoopHead(result);
      indent();
      emitValue(result);
      os << " = ";
      emitValue(op.getOperand(resultIdx++));
      os << ";";
      emitInfoAndNewLine(op);
      //emitNestedLoopTail(rank);
    }
  }
}

void ModuleEmitter::emitScfWhile(scf::WhileOp op) {
  indent();
  os << "while (";
  emitBlock(op.getBefore().front());
  os << ") {";
  emitInfoAndNewLine(op);

  addIndent();
  emitBlock(op.getAfter().front());
  reduceIndent();

  indent();
  os << "}\n";
}


void ModuleEmitter::emitFunction(func::FuncOp func) {

  if (func.getBlocks().empty())
    // This is a declaration.
    return;

  if (func.getBlocks().size() > 1)
    emitError(func, "has more than one basic blocks.");

  // Emit function signature.
  os << "void " << func.getName() << "(\n";
  addIndent();

  // This vector is to record all ports of the function.
  SmallVector<Value, 8> portList;

  // Emit input arguments.
  unsigned argIdx = 0;
  std::vector<std::string> input_args;
  if (func->hasAttr("inputs")) {
    std::string input_names =
        func->getAttr("inputs").cast<StringAttr>().getValue().str();
    input_args = split_names(input_names);
  }
  std::string output_names;
  if (func->hasAttr("outputs")) {
    output_names = func->getAttr("outputs").cast<StringAttr>().getValue().str();
    // suppose only one output
    input_args.push_back(output_names);
  }
  std::string itypes = "";
  if (func->hasAttr("itypes"))
    itypes = func->getAttr("itypes").cast<StringAttr>().getValue().str();
  else {
    for (unsigned i = 0; i < func.getNumArguments(); ++i)
      itypes += "x";
  }
  for (auto &arg : func.getArguments()) {
    indent();
    fixUnsignedType(arg, itypes[argIdx] == 'u');
    if (arg.getType().isa<ShapedType>()) {
      auto tensor = arg.getType().dyn_cast<RankedTensorType>();
      if (tensor) {
        os << getTypeName(tensor.getElementType()) << " ";
        os << addName(arg, false, "");
        for (auto &shape : tensor.getShape())
          os << "[" << shape << "]";
      }
    } else {
      if (input_args.size() == 0) {
        emitValue(arg);
      } else {
        emitValue(arg, 0, false, input_args[argIdx]);
      }
    }

    portList.push_back(arg);
    if (argIdx++ != func.getNumArguments() - 1)
      os << ",\n";
  }

  // Emit results.
  auto args = func.getArguments();
  std::string otypes = "";
  if (func->hasAttr("otypes"))
    otypes = func->getAttr("otypes").cast<StringAttr>().getValue().str();
  else {
    for (unsigned i = 0; i < func.getNumArguments(); ++i)
      otypes += "x";
  }
  if (auto funcReturn =
          dyn_cast<func::ReturnOp>(func.front().getTerminator())) {
    unsigned idx = 0;
    for (auto result : funcReturn.getOperands()) {
      if (std::find(args.begin(), args.end(), result) == args.end()) {
        os << ",\n";
        indent();

        // TODO: a known bug, cannot return a value twice, e.g. return %0, %0
        // : index, index. However, typically this should not happen.
        fixUnsignedType(result, otypes[idx] == 'u');
        if (result.getType().isa<ShapedType>()) {
          auto tensor = result.getType().dyn_cast<RankedTensorType>();
          if (tensor) {
            os << getTypeName(tensor.getElementType()) << " ";
            os << addName(result, false, "");
            for (auto &shape : tensor.getShape())
              os << "[" << shape << "]";
          }
        } else {
          // In Vivado HLS, pointer indicates the value is an output.
          if (output_names != "")
            emitValue(result, /*rank=*/0, /*isPtr=*/true);
          else
            emitValue(result, /*rank=*/0, /*isPtr=*/true, output_names);
        }

        portList.push_back(result);
      }
      idx += 1;
    }
  } else
    emitError(func, "doesn't have a return operation as terminator.");

  reduceIndent();
  os << "\n) {";
  emitInfoAndNewLine(func);

  // Emit function body.
  addIndent();

  emitBlock(func.front());

  reduceIndent();
  os << "}\n";

  // An empty line.
  os << "\n";
}

void ModuleEmitter::emitHostFunction(func::FuncOp func) {
  if (func.getBlocks().size() != 1)
    emitError(func, "has zero or more than one basic blocks.");

  os << "/// This is top function.\n";

  // Emit function signature.
  os << "int main(int argc, char **argv) {\n";
  addIndent();

  os << "  return 0;\n";
  reduceIndent();
  os << "}\n";

  // An empty line.
  os << "\n";
}

/// Top-level MLIR module emitter.
void ModuleEmitter::emitModule(ModuleOp module) {
  std::string device_header = R"XXX(
//===------------------------------------------------------------*- C++ -*-===//
//
// Automatically generated file for High-level Synthesis (HLS).
//
//===----------------------------------------------------------------------===//
#include <algorithm>
#include <ap_axi_sdata.h>
#include <ap_fixed.h>
#include <ap_int.h>
#include <hls_math.h>
#include <hls_stream.h>
#include <math.h>
#include <stdint.h>
using namespace std;
)XXX";

  std::string host_header = R"XXX(
//===------------------------------------------------------------*- C++ -*-===//
//
// Automatically generated file for host
//
//===----------------------------------------------------------------------===//
// standard C/C++ headers
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <time.h>

// vivado hls headers
#include "kernel.h"
#include <ap_fixed.h>
#include <ap_int.h>
#include <hls_stream.h>

#include <ap_axi_sdata.h>
#include <ap_fixed.h>
#include <ap_int.h>
#include <hls_math.h>
#include <hls_stream.h>
#include <math.h>
#include <stdint.h>

)XXX";

  if (module.getName().has_value() && module.getName().value() == "host") {
    os << host_header;
    for (auto op : module.getOps<func::FuncOp>()) {
      if (op.getName() == "main")
        emitHostFunction(op);
      else
        emitFunction(op);
    }
  } else {
    os << device_header;
    for (auto &op : *module.getBody()) {
      if (auto func = dyn_cast<func::FuncOp>(op))
        emitFunction(func);
      // else if (auto cst = dyn_cast<memref::GlobalOp>(op))
      //   emitGlobal(cst);
      else
        emitError(&op, "is unsupported operation.");
    }
  }
}

//===----------------------------------------------------------------------===//
// Entry of ust-translate
//===----------------------------------------------------------------------===//

LogicalResult ust::emitVivadoHLS(ModuleOp module, llvm::raw_ostream &os) {
  USTEmitterState state(os);
  ModuleEmitter(state).emitModule(module);
  return failure(state.encounteredError);
}

void ust::registerEmitVivadoHLSTranslation() {
  static TranslateFromMLIRRegistration toVivadoHLS(
      "emit-vivado-hls", "Emit Vivado HLS", emitVivadoHLS,
      [&](DialectRegistry &registry) {
        // clang-format off
        registry.insert<
          mlir::scf::SCFDialect,
          mlir::func::FuncDialect,
          mlir::sparse_tensor::SparseTensorDialect,
          mlir::arith::ArithDialect,
          bufferization::BufferizationDialect
        >();
        // clang-format on
      });
}