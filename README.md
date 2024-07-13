# UST Dialect
## Building
### Dependency
- gcc 9.5.0
- cmake >= 3.22
- conda 23.1.0 (if build llvm with python binding)

First of all, you should clone this repository.
```git
git clone https://github.com/cmx-Y/ust-dialect.git
git submodule update --init --recursive
```
### Build llvm
- Without Python binding
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
- With Python binding
```bash
# Use conda to create ust-dev environment
conda create -n ust-dev python=3.8
conda activate ust-dev
cd ust-dialect
pip install -r externals/llvm-project/mlir/python/requirements.txt

# Build mlir
cd ./externals/llvm-project/
mkdir build && cd build
cmake -G "Unix Makefiles" ../llvm \
   -DLLVM_ENABLE_PROJECTS=mlir \
   -DLLVM_BUILD_EXAMPLES=ON \
   -DLLVM_TARGETS_TO_BUILD="host" \
   -DCMAKE_BUILD_TYPE=Release \
   -DLLVM_ENABLE_ASSERTIONS=ON \
   -DLLVM_INSTALL_UTILS=ON \
   -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
   -DPython3_EXECUTABLE=`which python3`
make -j8

# Export the LLVM build directory
export LLVM_BUILD_DIR=$(pwd)
```

### Build ust-dialect
- Without Python binding
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
- With Python binding
```bash
conda activate ust-dev
cd ust-dialect
mkdir build && cd build
cmake -G "Unix Makefiles" .. \
   -DMLIR_DIR=$LLVM_BUILD_DIR/lib/cmake/mlir \
   -DLLVM_EXTERNAL_LIT=$LLVM_BUILD_DIR/bin/llvm-lit \
   -DPYTHON_BINDING=ON \
   -DOPENSCOP=OFF \
   -DPython3_EXECUTABLE=`which python3` \
   -DCMAKE_CXX_FLAGS="-Wfatal-errors -std=c++17"
make -j8

# Export the generated UST-MLIR Python library
export PYTHONPATH=$(pwd)/tools/ust/python_packages/ust_core:${PYTHONPATH}
```

## Examples
```bash
cd ust-dialect/build
./bin/ust-opt ../test/SparseTensor/examples/csr_spmv.mlir
./bin/ust-translate ../test/SparseTensor/examples/csr_spmv.mlir -emit-vivado-hls
```