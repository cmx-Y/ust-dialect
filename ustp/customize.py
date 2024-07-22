
import io
from ust_mlir.ir import (
    Module,
    Context,
    Location,
)
from ust_mlir.dialects import ust as ust_d


class Schedule:
    def __init__(
            self,
            module,
        ):
        self.module = module

    def build(self, target=None):
        if target is None or target == "llvm":
            return
        if target == "hls":
            buf = io.StringIO()
            res = ust_d.emit_vhls(self.module, buf)
            if res:
                buf.seek(0)
                hls_code = buf.read()
                print(hls_code)
            else:
                raise RuntimeError("HLS codegen failed")
            return
        raise NotImplementedError(f"Target {target} is not supported")
    

def customize():
    with Context() as ctx, Location.unknown():
        module = Module.create()
        sch = Schedule(module)
        return sch