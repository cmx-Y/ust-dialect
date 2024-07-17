[English](#en) | [中文](#cn)

<span id="en">Add Python Binding</span>
===========================


<span id="cn">添加 Python Binding</span>
===========================
以 `emit_vhls` 为例，简单介绍下如何将 Pass、Codegen 等相关的 C++ 代码暴露到 Python 层面。

在 `include/ust-c/Translation/EmitVivadoHLS.h` 中声明 c wrapper：
```c
MLIR_CAPI_EXPORTED MlirLogicalResult mlirEmitVivadoHls(MlirModule module,
                                                    MlirStringCallback callback,
                                                    void *userData);
```

在 `lib/CAPI/Translation/EmitVivadoHLS.cpp` 中对 c wrapper 进行具体定义：
```c++
MlirLogicalResult mlirEmitVivadoHls(MlirModule module,
                                    MlirStringCallback callback,
                                    void *userData) {
  mlir::detail::CallbackOstream stream(callback, userData);
  return wrap(emitVivadoHLS(unwrap(module), stream));
}
```
该 wrapper 就是用 `wrap` 函数对 `emitVivadoHLS` 函数进行包装，回顾一下 `emitVivadoHLS` 函数:
```c++
LogicalResult ust::emitVivadoHLS(ModuleOp module, llvm::raw_ostream &os) {
  USTEmitterState state(os);
  ModuleEmitter(state).emitModule(module);
  return failure(state.encounteredError);
}
```
在该函数中，我们具体实现了如何根据一个 mlir module 生成对应的 hls 代码，即 Codegen 具体过程。

以上步骤均完成后，需要在 `lib/Bindings/Python/USTModule.cpp` 中再对 `mlirEmitVivadoHls` 进行一次 wrap 并声明：
```c++
static bool emitVivadoHls(MlirModule &mod, py::object fileObject) {
  PyFileAccumulator accum(fileObject, false);
  py::gil_scoped_release();
  return mlirLogicalResultIsSuccess(
      mlirEmitVivadoHls(mod, accum.getCallback(), accum.getUserData()));
}

PYBIND11_MODULE(_ust, m) {
  ...
  // Codegen APIs.
  ust_m.def("emit_vhls", &emitVivadoHls);
  ...
}
```