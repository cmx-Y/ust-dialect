
import os
import sys
from typing import List

import numpy as np
from ust_mlir.dialects import affine, arith, builtin
from ust_mlir.dialects import math, memref, scf, func, tensor
from ust_mlir.exceptions import *

def get_line_number(frame=0):
    fr = sys._getframe(frame + 1) # +1 to ignore this function call
    return (os.path.basename(fr.f_code.co_filename), fr.f_lineno)

class USTFlags(object):
    def __init__(self):
        self.BUILD_INPLACE = False
        self.BIT_OP = False

    def enable_build_inplace(self):
        self.BUILD_INPLACE = True

    def disable_build_inplace(self):
        self.BUILD_INPLACE = False

    def is_build_inplace(self):
        return self.BUILD_INPLACE

    def reset(self):
        self.BUILD_INPLACE = False


flags = USTFlags()
enable_build_inplace = flags.enable_build_inplace
disable_build_inplace = flags.disable_build_inplace
is_build_inplace = flags.is_build_inplace
reset_build_inplace = flags.reset


def is_floating_point_type(dtype):
    return isinstance(dtype, (F16Type, F32Type, F64Type))


def get_floating_point_width(dtype):
    if F16Type.isinstance(dtype):
        return 16
    elif F32Type.isinstance(dtype):
        return 32
    elif F64Type.isinstance(dtype):
        return 64


def is_integer_type(dtype):
    return isinstance(dtype, IntegerType)


def is_unsigned_type(dtype):
    return isinstance(dtype, IntegerType) and dtype.is_unsigned


def is_signed_type(dtype):
    return isinstance(dtype, IntegerType) and dtype.is_signless


def is_fixed_type(dtype):
    return isinstance(dtype, (hcl_d.FixedType, hcl_d.UFixedType))


def is_signed_fixed_type(dtype):
    return isinstance(dtype, hcl_d.FixedType)


def is_unsigned_fixed_type(dtype):
    return isinstance(dtype, hcl_d.UFixedType)


def is_index_type(dtype):
    return isinstance(dtype, IndexType)


def is_struct_type(dtype):
    return isinstance(dtype, hcl_d.StructType)


def get_mlir_type(dtype):
    """
    Get MLIR type from string.
    Note that the returned type is for ExprOp creation intead of ExprOp.build().
    This is because signedness infomation is preserved.
    i.e. "uint8" is returned as unsigned type instead of signless type. 
    @param: dtype: string or MLIR type
    """
    if (
        is_integer_type(dtype)
        or is_floating_point_type(dtype)
        or is_fixed_type(dtype)
        or is_index_type(dtype)
        or is_struct_type(dtype)
    ):
        return dtype
    elif isinstance(dtype, str):
        if dtype[0:5] == "index":
            return IndexType.get()
        elif dtype[0:3] == "int":
            return IntegerType.get_signless(int(dtype[3:]))
        elif dtype[0:4] == "uint":
            return IntegerType.get_unsigned(int(dtype[4:]))
        elif dtype[0:5] == "float":
            if dtype[5:] == "16":
                return F16Type.get()
            elif dtype[5:] == "32":
                return F32Type.get()
            elif dtype[5:] == "64":
                return F64Type.get()
            else:
                raise DTypeError(f"Not supported floating point type: {dtype}")
        else:
            raise DTypeError("Unrecognized data type: {}".format(dtype))
    else:
        raise DTypeError(
            "Unrecognized data type format: {} of Type({})".format(
                dtype, type(dtype))
        )


def get_concrete_type(dtype):
    if IntegerType.isinstance(dtype):
        return IntegerType(dtype)
    elif F16Type.isinstance(dtype):
        return F16Type(dtype)
    elif F32Type.isinstance(dtype):
        return F32Type(dtype)
    elif F64Type.isinstance(dtype):
        return F64Type(dtype)
    elif IndexType.isinstance(dtype):
        return IndexType(dtype)
    else:
        raise DTypeError("Unrecognized data type: {}".format(dtype))


def get_bitwidth(dtype):
    if IntegerType.isinstance(dtype):
        return dtype.width
    elif F16Type.isinstance(dtype):
        return 16
    elif F32Type.isinstance(dtype):
        return 32
    elif F64Type.isinstance(dtype):
        return 64
    else:
        raise DTypeError("Unrecognized data type: {}".format(dtype))


def print_mlir_type(dtype):
    """ Print MLIR type to C/HLSC types
    @param dtype: MLIR type
    """
    if is_floating_point_type(dtype):
        if dtype.width == 32:
            return "float"
        elif dtype.width == 64:
            return "double"
        else:
            raise DTypeError("Not supported data type: {}".format(dtype))
    elif is_integer_type(dtype):
        if isinstance(dtype, IndexType) or dtype.is_signed or dtype.is_signless:
            if dtype.width == 32:
                return "int"
            elif dtype.width == 64:
                return "long int"
            elif dtype.width == 1:
                return "bool"
            else:
                return "ap_int<{}>".format(dtype.width)
        elif dtype.is_unsigned:
            if dtype.width == 32:
                return "unsigned int"
            elif dtype.width == 64:
                return "unsigned long int"
            elif dtype.width == 1:
                return "bool"
            else:
                return "ap_uint<{}>".format(dtype.width)
    elif is_fixed_type(dtype):
        if isinstance(dtype, hcl_d.FixedType):
            return "ap_fixed<{}, {}>".format(dtype.width, dtype.frac)
        elif isinstance(dtype, hcl_d.UFixedType):
            return "ap_ufixed<{}, {}>".format(dtype.width, dtype.frac)
        else:
            raise DTypeError("Not supported data type: {}".format(dtype))
    elif is_struct_type(dtype):
        raise HCLNotImplementedError("struct type printing to be implemented")
    else:
        raise DTypeError("Not supported data type: {}".format(dtype))


def mlir_type_to_str(dtype):
    """ Build HeteroCL-compatible type string from MLIR type
    @param dtype: MLIR type
    """
    if is_signed_type(dtype):
        return "int{}".format(get_bitwidth(dtype))
    elif is_unsigned_type(dtype):
        return "uint{}".format(get_bitwidth(dtype))
    elif is_floating_point_type(dtype):
        return "float{}".format(get_bitwidth(dtype))
    elif is_signed_fixed_type(dtype):
        if dtype.frac == 0:
            return "int{}".format(dtype.width)
        return "fixed{}_{}".format(dtype.width, dtype.frac)
    elif is_unsigned_fixed_type(dtype):
        if dtype.frac == 0:
            return "uint{}".format(dtype.width)
        return "ufixed{}_{}".format(dtype.width, dtype.frac)
    elif is_struct_type(dtype):
        type_str = "Struct("
        for ft in dtype.field_types:
            type_str += mlir_type_to_str(ft) + ", "
        type_str = type_str[:-2] + ")"
        return type_str
    else:
        raise DTypeError("Unrecognized data type: {}".format(dtype))


def get_signless_type(dtype):
    if is_integer_type(dtype):
        return IntegerType.get_signless(get_bitwidth(dtype))
    elif is_struct_type(dtype):
        new_field_types = []
        for field_type in dtype.field_types:
            field_type = get_concrete_type(field_type)
            if is_integer_type(field_type):
                new_field_types.append(
                    get_signless_type(field_type)
                )
            elif is_struct_type(field_type):
                new_field_types.append(get_signless_type(field_type))
            else:
                new_field_types.append(field_type)
        dtype = hcl_d.StructType.get(new_field_types)
        return dtype
    else:
        return dtype

def is_all_field_int(dtype):
    """ Check if a struct type has all integer fields
    """
    if not is_struct_type(dtype):
        return False
    dtype = get_concrete_type(dtype)
    for field_type in dtype.field_types:
        field_type = get_concrete_type(field_type)
        if is_struct_type(field_type):
            if not is_all_field_int(field_type):
                return False
        elif not is_integer_type(field_type):
            return False
    return True

class USTMLIRInsertionPoint(object):
    def __init__(self):
        self.ip_stack = []

    def clear(self):
        self.ip_stack = []

    def get(self):
        return self.ip_stack[-1]

    def get_global(self):
        return self.ip_stack[0]

    def save(self, ip):
        if not isinstance(ip, InsertionPoint):
            ip = InsertionPoint(ip)
        self.ip_stack.append(ip)

    def restore(self):
        return self.ip_stack.pop()


GlobalInsertionPoint = USTMLIRInsertionPoint()


def floating_point_error(op_name):
    return DTypeError("{} does not support floating point inputs".format(op_name))


def get_ust_op(expr, dtype=None):
    if isinstance(expr, (int, float)):
        if dtype == None:
            if isinstance(expr, int):
                if expr < 0xFFFFFFFF:
                    return ConstantOp(IntegerType.get_signless(32), expr)
                else:
                    return ConstantOp(IntegerType.get_signless(64), expr)
            elif isinstance(expr, float):
                return ConstantOp(F32Type.get(), expr)
        else:
            return ConstantOp(dtype, expr)
    else:
        if dtype != None and dtype != expr.dtype:
            expr = CastOp(expr, dtype)
        return expr


def get_type_rank(dtype):
    """
    We always cast lower rank types to higher rank types.
    Base rank 1 (lowest): integer and fixed point types
    Base rank 2: index type
    Base rank 3 (highest): float types
    Types with larger dynamic range should have higher ranks.
    """
    if is_integer_type(dtype):
        base = 0
        width = dtype.width
        if width > 2048:
            raise DTypeError("Cannot support integer width larger than 2048")
        base += width
        return base
    elif is_fixed_type(dtype):
        base = 0
        width = dtype.width
        frac = dtype.frac
        return base + (width - frac)
    elif is_index_type(dtype):  # width 32
        base = 2049
        return base
    elif is_floating_point_type(dtype):
        base = 10000
        if isinstance(dtype, F16Type):
            base += 1
        elif isinstance(dtype, F32Type):
            base += 2
        elif isinstance(dtype, F64Type):
            base += 3
        else:
            raise DTypeError(
                "Unrecognized floating point type: {}".format(dtype))
        return base
    else:
        raise DTypeError("Unrecognized type: {}".format(dtype))


def cast_types(lhs, rhs):
    """
    Cast types for binary operations
    lhs always has higher rank than rhs
    Implementation based on
    https://en.cppreference.com/w/c/language/conversion
    """
    ltype = lhs.dtype
    rtype = rhs.dtype
    # 1) If one operand is long double (omitted)
    # 2) Otherwise, if lhs is double
    if isinstance(ltype, F64Type):
        # integer or real floating type to double
        res_type = F64Type.get()
        DTypeWarning("Casting value {} from {} to {}".format(
            rhs, rtype, res_type)).log()
        return lhs, CastOp(rhs, res_type)
    # 3) Otherwise, if lhs is float
    elif isinstance(ltype, F32Type):
        # integer type to float
        res_type = F32Type.get()
        DTypeWarning("Casting value {} from {} to {}".format(
            rhs, rtype, res_type)).log()
        return lhs, CastOp(rhs, res_type)
    # 4) Otherwise, if lhs is integer.
    elif isinstance(ltype, (IntegerType, IndexType)):
        # 4.1) lhs is int or index, rhs is int of lower rank, rhs gets promoted
        if isinstance(rtype, IntegerType):
            res_type = ltype
            DTypeWarning("Casting value {} from {} to {}".format(
                rhs, rtype, res_type)).log()
            return lhs, CastOp(rhs, res_type)
        # 4.2) lhs is index, rhs is also index, nothing to do
        elif isinstance(rtype, IndexType):
            return lhs, rhs
        # 4.3) lhs is int or index, rhs is fixed point of lower rank
        # e.g. Int(100) + Fixed(3, 2) -> Fixed(100 + 2, 2)
        elif is_signed_fixed_type(rtype):
            res_type = hcl_d.FixedType.get(
                ltype.width + rtype.frac, rtype.frac)
            DTypeWarning("Casting value {} from {} to {}".format(
                lhs, ltype, res_type)).log()
            DTypeWarning("Casting value {} from {} to {}".format(
                rhs, rtype, res_type)).log()
            return CastOp(lhs, res_type), CastOp(rhs, res_type)
        # 4.4) lhs is int or index, rhs is unsigned fixed point of lower rank
        # e.g. Int(100) + UFixed(3, 2) -> UFixed(100 + 2, 2)
        elif is_unsigned_fixed_type(rtype):
            res_type = hcl_d.UFixedType.get(
                ltype.width + rtype.frac, rtype.frac)
            DTypeWarning("Casting value {} from {} to {}".format(
                lhs, ltype, res_type)).log()
            DTypeWarning("Casting value {} from {} to {}".format(
                rhs, rtype, res_type)).log()
            return CastOp(lhs, res_type), CastOp(rhs, res_type)
        else:
            # unexpected type
            raise DTypeError("Unexpected type: {}".format(rtype))
    # 5) Otherwise, if lhs is fixed type.
    elif is_fixed_type(ltype):
        # 5.1) lhs is fixed point, rhs is integer or fixed point of lower rank, cast rhs to lhs
        if is_integer_type(rtype) or is_fixed_type(rtype):
            res_type = ltype
            DTypeWarning("Casting value {} from {} to {}".format(
                rhs, rtype, res_type)).log()
            return lhs, CastOp(rhs, res_type)
        else:
            # unexpected type
            raise DTypeError("Unexpected type: {}".format(rtype))
    else:
        raise DTypeError(
            "Type conversion failed, lhs type: {}, rhs type: {}".format(
                ltype, rtype)
        )


# TODO(Niansong): this should be covered by cast_types, double-check before removing
def regularize_fixed_type(lhs, rhs):
    if not is_fixed_type(lhs.dtype) or not is_fixed_type(rhs.dtype):
        raise DTypeError("Should be all fixed types")
    if not lhs.dtype.frac == rhs.dtype.frac:
        raise DTypeError("Should have the same frac")
    lwidth = lhs.dtype.width
    rwidth = rhs.dtype.width
    if lwidth < rwidth:
        res_type = hcl_d.FixedType.get(rwidth, lhs.dtype.frac)
        cast = CastOp(lhs, res_type)
        return cast, rhs
    elif lwidth > rwidth:
        res_type = hcl_d.FixedType.get(lwidth, rhs.dtype.frac)
        cast = CastOp(rhs, res_type)
        return lhs, cast
    else:
        return lhs, rhs