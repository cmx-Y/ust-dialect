// $ cd ust-dialect/build
// $ ./bin/ust-translate ../test/SparseTensor/examples/csr_spmv.mlir -emit-vivado-hls \
//   > ../test/SparseTensor/codegen_output/csr_spmv_codegen.cpp

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
void matvec(
  double v0[32][64],
  double v1[64],
  double v2[32],
  double v3[32]
) {	// L2
  double v4[3];
  double v5[8];
  double v6[8];
  double v7[64];
  double v8[32];
  for (int v9 = 0; v9 < 32; v9 += 1) {	// L11
    double v10 = v8[v9];	// L12
    int v11 = v4[v9];	// L13
    int v12 = v9 + 1;	// L14
    int v13 = v4[v12];	// L15
    for (int v14 = v11; v14 < v13; v14 += 1) {	// L16
      int v15 = v5[v14];	// L17
      double v16 = v6[v14];	// L18
      double v17 = v7[v15];	// L19
      double v18 = v16 * v17;	// L20
      double v19 = double v20 + v18;	// L21
    }
  }
}

