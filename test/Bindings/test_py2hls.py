
import io
from ust_mlir.ir import *
from ust_mlir.dialects import sparse_tensor as st
from ust_mlir.dialects import func
from ust_mlir.dialects.linalg.opdsl import lang as dsl
from ust_mlir import passmanager
from ust_mlir.passmanager import PassManager
from ust_mlir.dialects import ust as ust_d


@dsl.linalg_structured_op
def matvec_dsl(
    A=dsl.TensorDef(dsl.T, dsl.S.M, dsl.S.K),
    B=dsl.TensorDef(dsl.T, dsl.S.K),
    C=dsl.TensorDef(dsl.T, dsl.S.M, output=True),
):
    """
    Performs matrix-vector multiplication using DSL.

    Args:
        A (dsl.TensorDef): Input matrix of shape (M, K).
        B (dsl.TensorDef): Input vector of shape (K).
        C (dsl.TensorDef): Output vector of shape (M).

    Returns:
        None
    """
    C[dsl.D.m] += A[dsl.D.m, dsl.D.k] * B[dsl.D.k]

def build_SpMV(attr: st.EncodingAttr):
    """
    Builds a SpMV (Sparse Matrix-Vector Multiplication) module.

    Args:
        attr (st.EncodingAttr): The encoding attribute.

    Returns:
        Module: The created module.
    """
    module = Module.create()
    f64 = F64Type.get()
    # a is a 2D sparese tensor of shape (32, 64)
    # b is a 1D dense tensor of shape (64)
    a = RankedTensorType.get([32, 64], f64, attr)
    b = RankedTensorType.get([64], f64)
    c = RankedTensorType.get([32], f64)
    arguments = [a, b, c]
    with InsertionPoint(module.body):

        @func.FuncOp.from_py_func(*arguments)
        def matvec(*args):
            return matvec_dsl(args[0], args[1], outs=[args[2]])

    return module

def test_py2hls():
    """
    This function tests the conversion of Python code to HLS code using the ust-dialect library.

    It performs the following steps:
    1. Creates a context and sets the location to unknown.
    2. Defines a level list and an ordering using the AffineMap.get_permutation method.
    3. Creates an encoding attribute using the level and ordering.
    4. Builds the SpMV module using the encoding attribute.
    5. Sets up a pass manager and adds two passes: "sparse-reinterpret-map" and "sparsification".
    6. Runs the passes on the module operation.
    7. Creates a string buffer and emits the VHLS code using ust_d.emit_vhls.
    8. If the code generation is successful, prints the HLS code.
    9. Otherwise, raises a RuntimeError.

    """
    with Context() as ctx, Location.unknown():
        level = [st.LevelType.dense, st.LevelType.compressed]
        ordering = AffineMap.get_permutation([0, 1])
        attr = st.EncodingAttr.get(
            level, ordering, ordering, 0, 0
        )
        module = build_SpMV(attr)
        pm = PassManager()
        pm.add("sparse-reinterpret-map")
        pm.add("sparsification")
        pm.run(module.operation)
        buf = io.StringIO()
        res = ust_d.emit_vhls(module, buf)
        if res:
            buf.seek(0)
            hls_code = buf.read()
            print(hls_code)
        else:
            raise RuntimeError("HLS codegen failed")

if __name__ == "__main__":
    test_py2hls()