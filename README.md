# UST Dialect
## Building
### Dependency
- gcc 9.5.0
- cmake >= 3.22

First of all, you should clone this repository.
```git
git clone https://github.com/cmx-Y/ust-dialect.git
git submodule update --init --recursive
```
### Build llvm
```bash
cd ust-dialect/externals/llvm-project
mkdir build && cd build
cmake -G "Unix Makefiles" ../llvm \
   -DLLVM_ENABLE_PROJECTS=mlir \
   -DLLVM_BUILD_EXAMPLES=ON \
   -DLLVM_TARGETS_TO_BUILD="X86" \
   -DCMAKE_BUILD_TYPE=Release \
   -DLLVM_ENABLE_ASSERTIONS=ON \
   -DLLVM_INSTALL_UTILS=ON
make -j8
# Export the LLVM build directory
export LLVM_BUILD_DIR=$(pwd)
```

### Build ust-dialect
```bash
cd ust-dialect
mkdir build && cd build
cmake -G "Unix Makefiles" .. \
   -DMLIR_DIR=$LLVM_BUILD_DIR/lib/cmake/mlir \
   -DLLVM_EXTERNAL_LIT=$LLVM_BUILD_DIR/bin/llvm-lit \
   -DPYTHON_BINDING=OFF \
   -DOPENSCOP=OFF
make -j8
```

## Examples
```bash
cd ust-dialect/build
./bin/ust-opt ../test/SparseTensor/csr_spmv.mlir
./bin/ust-translate ../test/SparseTensor/csr_spmv.mlir -emit-vivado-hls
```