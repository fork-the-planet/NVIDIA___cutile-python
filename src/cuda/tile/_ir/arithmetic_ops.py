# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


# This file contains implementations of arithmetic operations that work on generic
# TensorLikeTy inputs. Please avoid any cutile-specific logic outside of generate_bytecode()
# methods.


import math
import operator
from dataclasses import dataclass
from typing import Sequence, Any
from typing_extensions import override

from cuda.tile._datatype import numeric_dtype_category, DType, bool_, is_integral, is_boolean, \
    int8, get_signedness, is_float, is_arithmetic
from cuda.tile._exception import TileTypeError
from cuda.tile._numeric_semantics import RoundingMode
from cuda.tile._ir.core_ops import loosely_typed_const, strictly_typed_const
from cuda.tile._ir.ir import operand, Operation, Var, add_operation, Builder, attribute
from cuda.tile._ir.op_impl import ImplRegistry
from cuda.tile._ir.ops_utils import is_shape_broadcastable_to, promote_types, \
    get_dtype, get_default_rounding_mode, rounding_mode_to_bytecode, \
    reraise_tile_exception, check_rd_and_ftz, BINOP_REGISTRY
from cuda.tile._ir.type import TensorLikeTy, LooselyTypedScalar, Type
from cuda.tile._ir2bytecode import BytecodeContext, convert_dtype, typeid
import cuda.tile._bytecode as bc


_registry = ImplRegistry()
impl = _registry.impl


def arithmetic_impl_registry() -> ImplRegistry:
    return _registry


def binop_propagate_constant(fn: str, x: Any, y: Any, type: Type | None) -> Var:
    impl = BINOP_REGISTRY[fn].impl
    with reraise_tile_exception():
        res = impl(x, y)

    if type is None:
        return loosely_typed_const(res)
    else:
        return strictly_typed_const(res, type)


def comparison_operator_impl(registry: ImplRegistry, lhs_ty: type[Type], rhs_ty: type[Type]):
    def decorate(func):
        for name in ("eq", "ne", "lt", "le", "gt", "ge"):
            registry.impl(getattr(operator, name), fixed_args=[name],
                          overload=(lhs_ty, rhs_ty))(func)
        return func

    return decorate


@dataclass(eq=False)
class TileReshape(Operation, opcode="tile_reshape"):
    x: Var = operand()

    @override
    def generate_bytecode(self, ctx: BytecodeContext) -> bc.Value:
        x_value = ctx.get_value(self.x)
        res_type_id = ctx.typeid_of(self.result_var)
        return bc.encode_ReshapeOp(ctx.builder, res_type_id, x_value)


def reshape(x: Var[TensorLikeTy], new_shape: Sequence[int]) -> Var:
    x_ty = x.get_type()
    x_shape = x_ty.tensor_shape()
    numel = math.prod(x_shape)

    negative_one_index = None
    numel2 = 1
    for i, dim_value in enumerate(new_shape):
        if dim_value < 0:
            if dim_value < -1:
                raise TileTypeError(f"Dimension can only be -1 or non-negative, got {dim_value}")
            if negative_one_index is not None:
                raise TileTypeError(f"Only one dimension can be -1, got {new_shape}")
            negative_one_index = i
        else:
            numel2 *= dim_value

    if negative_one_index is not None:
        if numel2 == 0 or numel % numel2 != 0:
            raise TileTypeError(f"Cannot reshape {x_shape} to {new_shape}")
        new_shape = list(new_shape)
        new_shape[negative_one_index] = numel // numel2
        new_shape = tuple(new_shape)
    elif numel != numel2:
        raise TileTypeError(f"Cannot reshape {x_shape} to {new_shape}")

    if new_shape == x_shape:
        return x
    else:
        res_type = x.ctx.typing_hooks.get_tensor_like_type(x_ty.tensor_dtype(), new_shape)
        return add_operation(TileReshape, res_type, x=x)


@dataclass(eq=False)
class TileBroadcast(Operation, opcode="tile_broadcast"):
    x: Var = operand()

    @override
    def generate_bytecode(self, ctx: BytecodeContext) -> bc.Value:
        x_value = ctx.get_value(self.x)
        res_typeid = ctx.typeid_of(self.result_var)
        return bc.encode_BroadcastOp(ctx.builder, res_typeid, x_value)


def broadcast_to(x: Var[TensorLikeTy], shape: Sequence[int]) -> Var[TensorLikeTy]:
    x_ty = x.get_type()
    old_shape = x_ty.tensor_shape()

    if not is_shape_broadcastable_to(old_shape, shape):
        raise TileTypeError(f"Shape {old_shape} is not broadcastable to {tuple(shape)}")

    if len(shape) > len(old_shape):
        extra_ones = (1,) * (len(shape) - len(old_shape))
        old_shape = extra_ones + old_shape
        x = reshape(x, old_shape)

    if old_shape == shape:
        return x
    else:
        result_ty = x.ctx.typing_hooks.get_tensor_like_type(x_ty.tensor_dtype(), shape)
        return add_operation(TileBroadcast, result_ty, x=x)


@dataclass(eq=False)
class TileAsType(Operation, opcode="tile_astype"):
    x: Var = operand()

    @override
    def generate_bytecode(self, ctx: BytecodeContext) -> bc.Value:
        value = ctx.get_value(self.x)
        return convert_dtype(ctx, value, ctx.typeof(self.x), ctx.typeof(self.result_var))


def astype(x: Var[TensorLikeTy], dtype: DType) -> Var[TensorLikeTy]:
    x_ty = x.get_type()
    if x_ty.tensor_dtype() == dtype:
        return x

    if x.is_constant():
        val = numeric_dtype_category(dtype).pytype(x.get_constant())
        return strictly_typed_const(val, x.ctx.typing_hooks.get_tensor_like_type(dtype, ()))

    result_ty = x.ctx.typing_hooks.get_tensor_like_type(dtype, x_ty.tensor_shape())
    return add_operation(TileAsType, result_ty, x=x)


def promote_and_broadcast_to(x: Var, ty: TensorLikeTy) -> Var[TensorLikeTy]:
    return broadcast_to(astype(x, ty.tensor_dtype()), ty.tensor_shape())


# Does not do broadcasting or type promotion, hence the name "Raw"
@dataclass(eq=False)
class RawComparisonOperation(Operation, opcode="raw_cmp"):
    fn: str = attribute()
    lhs: Var[TensorLikeTy] = operand()
    rhs: Var[TensorLikeTy] = operand()

    @override
    def generate_bytecode(self, ctx: BytecodeContext):
        from .._ir2bytecode import encode_comparison
        lhs = ctx.get_value(self.lhs)
        rhs = ctx.get_value(self.rhs)
        dtype = self.lhs.get_type().tensor_dtype()
        result_typeid = ctx.typeid_of(self.result_var)
        return encode_comparison(ctx.builder, self.fn, lhs, rhs, dtype, result_typeid)


def compare_tensorlike_raw(fn: str,
                           x: Var[TensorLikeTy],
                           y: Var[TensorLikeTy]) -> Var[TensorLikeTy]:
    ty = x.get_type()
    assert ty == y.get_type()
    res_ty = x.ctx.typing_hooks.get_tensor_like_type(bool_, ty.tensor_shape())
    return add_operation(RawComparisonOperation, res_ty, fn=fn, lhs=x, rhs=y)


def compare_tensorlike(fn: str, x: Var[TensorLikeTy], y: Var[TensorLikeTy]) -> Var[TensorLikeTy]:
    x_ty = x.get_loose_type()
    y_ty = y.get_loose_type()

    if isinstance(x_ty, LooselyTypedScalar) and isinstance(y_ty, LooselyTypedScalar):
        return binop_propagate_constant(fn, x_ty.value, y_ty.value, None)

    common_ty = promote_types(x_ty, y_ty, Builder.get_current().ir_ctx.typing_hooks)
    x = promote_and_broadcast_to(x, common_ty)
    y = promote_and_broadcast_to(y, common_ty)

    if x.is_constant() and y.is_constant():
        res_ty = x.ctx.typing_hooks.get_tensor_like_type(bool_, common_ty.tensor_shape())
        return binop_propagate_constant(fn, x.get_constant(), y.get_constant(), res_ty)

    return compare_tensorlike_raw(fn, x, y)


@comparison_operator_impl(_registry, TensorLikeTy, TensorLikeTy)
def comparison_tensorlike_impl(fn: str, x: Var[TensorLikeTy], y: Var[TensorLikeTy]) -> Var:
    return compare_tensorlike(fn, x, y)


# Does not do broadcasting or type promotion, hence the name "Raw"
@dataclass(eq=False)
class RawBinaryBitwiseOperation(Operation, opcode="raw_binary_bitwise"):
    fn: str = attribute()
    lhs: Var = operand()
    rhs: Var = operand()

    @override
    def generate_bytecode(self, ctx: BytecodeContext):
        res_typeid = ctx.typeid_of(self.result_var)
        lhs = ctx.get_value(self.lhs)
        rhs = ctx.get_value(self.rhs)
        match self.fn:
            case "and_": return bc.encode_AndIOp(ctx.builder, res_typeid, lhs, rhs)
            case "or_": return bc.encode_OrIOp(ctx.builder, res_typeid, lhs, rhs)
            case "xor": return bc.encode_XOrIOp(ctx.builder, res_typeid, lhs, rhs)
            case _:
                raise NotImplementedError(f"Missing binary bitwise implementation for {self.fn}")


def binary_bitwise_tensorlike_raw(fn: str,
                                  x: Var[TensorLikeTy],
                                  y: Var[TensorLikeTy]) -> Var[TensorLikeTy]:
    ty = x.get_type()
    assert ty == y.get_type()
    return add_operation(RawBinaryBitwiseOperation, ty, fn=fn, lhs=x, rhs=y)


def binary_bitwise_tensorlike(fn: str,
                              x: Var[TensorLikeTy],
                              y: Var[TensorLikeTy]) -> Var[TensorLikeTy]:
    x_ty = x.get_loose_type()
    y_ty = y.get_loose_type()

    if isinstance(x_ty, LooselyTypedScalar) and isinstance(y_ty, LooselyTypedScalar):
        return binop_propagate_constant(fn, x_ty.value, y_ty.value, None)

    lhs_dtype = x_ty.tensor_dtype()
    rhs_dtype = y_ty.tensor_dtype()

    if not (is_integral(lhs_dtype) or is_boolean(lhs_dtype)) \
            or not (is_integral(rhs_dtype) or is_boolean(rhs_dtype)):
        raise TileTypeError("Bitwise operations require integers or booleans."
                            " Use an explicit cuda.tile.bitcast() for non-integer operands.")

    x_loose = isinstance(x_ty, LooselyTypedScalar)
    y_loose = isinstance(y_ty, LooselyTypedScalar)
    if x_loose == y_loose and lhs_dtype != rhs_dtype:
        msg = "Bitwise operands must have same data type, got:"
        msg += f" {lhs_dtype} and {rhs_dtype}"
        raise TileTypeError(msg)

    if {lhs_dtype, rhs_dtype} == {bool_, int8}:
        raise TileTypeError("Bitwise op does not support bool and int8")

    common_ty = promote_types(x_ty, y_ty, x.ctx.typing_hooks)
    x = promote_and_broadcast_to(x, common_ty)
    y = promote_and_broadcast_to(y, common_ty)

    if x.is_constant() and y.is_constant():
        return binop_propagate_constant(fn, x.get_constant(), y.get_constant(), common_ty)

    return binary_bitwise_tensorlike_raw(fn, x, y)


@impl(operator.and_, fixed_args=["and_"], overload=(TensorLikeTy, TensorLikeTy))
@impl(operator.or_, fixed_args=["or_"], overload=(TensorLikeTy, TensorLikeTy))
@impl(operator.xor, fixed_args=["xor"], overload=(TensorLikeTy, TensorLikeTy))
def _binary_bitwise_tensorlike_impl(fn: str,
                                    x: Var[TensorLikeTy],
                                    y: Var[TensorLikeTy]) -> Var[TensorLikeTy]:
    return binary_bitwise_tensorlike(fn, x, y)


# Does not do broadcasting or type promotion, hence the name "Raw"
@dataclass(eq=False)
class RawBitwiseShiftOperation(Operation, opcode="raw_bitwise_shift"):
    fn: str = attribute()
    lhs: Var[TensorLikeTy] = operand()
    rhs: Var[TensorLikeTy] = operand()

    @override
    def generate_bytecode(self, ctx: BytecodeContext) -> bc.Value:
        res_ty = self.result_var.get_type()
        res_type_id = typeid(ctx.type_table, res_ty)
        lhs = ctx.get_value(self.lhs)
        rhs = ctx.get_value(self.rhs)
        match self.fn:
            case "lshift":
                return bc.encode_ShLIOp(ctx.builder, res_type_id, lhs, rhs, bc.IntegerOverflow.NONE)
            case "rshift":
                return bc.encode_ShRIOp(ctx.builder, res_type_id, lhs, rhs,
                                        get_signedness(get_dtype(res_ty)))
            case _: raise NotImplementedError()


def bitwise_shift_tensorlike_raw(fn: str,
                                 x: Var[TensorLikeTy],
                                 y: Var[TensorLikeTy]) -> Var[TensorLikeTy]:
    ty = x.get_type()
    assert ty == y.get_type()
    return add_operation(RawBitwiseShiftOperation, ty, fn=fn, lhs=x, rhs=y)


def bitwise_shift_tensorlike(fn: str,
                             x: Var[TensorLikeTy],
                             y: Var[TensorLikeTy]) -> Var[TensorLikeTy]:
    x_ty = x.get_loose_type()
    y_ty = y.get_loose_type()

    if isinstance(x_ty, LooselyTypedScalar) and isinstance(y_ty, LooselyTypedScalar):
        return binop_propagate_constant(fn, x_ty.value, y_ty.value, None)

    lhs_dtype = x_ty.tensor_dtype()
    if not is_integral(lhs_dtype):
        msg = f'Bitwise shift requires an integer for left-hand side, got: {lhs_dtype}'
        raise TileTypeError(msg)

    rhs_dtype = y_ty.tensor_dtype()
    if not is_integral(rhs_dtype):
        msg = f'Bitwise shift requires an integer for right-hand side, got: {rhs_dtype}'
        raise TileTypeError(msg)

    common_ty = promote_types(x_ty, y_ty, x.ctx.typing_hooks)
    x = promote_and_broadcast_to(x, common_ty)
    y = promote_and_broadcast_to(y, common_ty)

    if x.is_constant() and y.is_constant():
        return binop_propagate_constant(fn, x.get_constant(), y.get_constant(), common_ty)

    return bitwise_shift_tensorlike_raw(fn, x, y)


@impl(operator.lshift, fixed_args=["lshift"], overload=(TensorLikeTy, TensorLikeTy))
@impl(operator.rshift, fixed_args=["rshift"], overload=(TensorLikeTy, TensorLikeTy))
def _bitwise_shift_tensorlike_impl(fn: str, x: Var[TensorLikeTy], y: Var[TensorLikeTy]):
    return bitwise_shift_tensorlike(fn, x, y)


# Does not do broadcasting or type promotion, hence the name "Raw"
@dataclass(eq=False)
class RawBinaryArithmeticOperation(Operation, opcode="raw_binary_arith"):
    fn: str = attribute()
    rounding_mode: RoundingMode | None = attribute()
    flush_to_zero: bool = attribute()
    lhs: Var[TensorLikeTy] = operand()
    rhs: Var[TensorLikeTy] = operand()

    @override
    def generate_bytecode(self, ctx: BytecodeContext) -> bc.Value:
        result_ty = self.result_var.get_type()
        dtype = get_dtype(result_ty)
        kind = "float" if is_float(dtype) else "int"
        res_typeid = typeid(ctx.type_table, result_ty)
        rm = self.rounding_mode if self.rounding_mode is not None else get_default_rounding_mode()
        rounding_mode = rounding_mode_to_bytecode[rm]
        lhs = ctx.get_value(self.lhs)
        rhs = ctx.get_value(self.rhs)

        match self.fn, kind:
            case "add", "int":
                return bc.encode_AddIOp(ctx.builder, res_typeid, lhs, rhs,
                                        overflow=bc.IntegerOverflow.NONE)
            case "add", "float":
                return bc.encode_AddFOp(ctx.builder, res_typeid, lhs, rhs,
                                        rounding_mode=rounding_mode,
                                        flush_to_zero=self.flush_to_zero)
            case "sub", "int":
                return bc.encode_SubIOp(ctx.builder, res_typeid, lhs, rhs,
                                        overflow=bc.IntegerOverflow.NONE)
            case "sub", "float":
                return bc.encode_SubFOp(ctx.builder, res_typeid, lhs, rhs,
                                        rounding_mode=rounding_mode,
                                        flush_to_zero=self.flush_to_zero)
            case "mul", "int":
                return bc.encode_MulIOp(ctx.builder, res_typeid, lhs, rhs,
                                        overflow=bc.IntegerOverflow.NONE)
            case "mul", "float":
                return bc.encode_MulFOp(ctx.builder, res_typeid, lhs, rhs,
                                        rounding_mode=rounding_mode,
                                        flush_to_zero=self.flush_to_zero)
            case "floordiv", "int":
                return bc.encode_DivIOp(ctx.builder, res_typeid, lhs, rhs,
                                        signedness=get_signedness(dtype),
                                        rounding=bc.RoundingMode.NEGATIVE_INF)
            case "floordiv", "float":
                quotient = bc.encode_DivFOp(ctx.builder, res_typeid, lhs, rhs,
                                            rounding_mode=rounding_mode,
                                            flush_to_zero=self.flush_to_zero)
                return bc.encode_FloorOp(ctx.builder, res_typeid, quotient)
            case "cdiv", "int":
                return bc.encode_DivIOp(ctx.builder, res_typeid, lhs, rhs,
                                        signedness=get_signedness(dtype),
                                        rounding=bc.RoundingMode.POSITIVE_INF)
            case "truediv", "float":
                return bc.encode_DivFOp(ctx.builder, res_typeid, lhs, rhs,
                                        rounding_mode=rounding_mode,
                                        flush_to_zero=self.flush_to_zero)
            case "pow", "float":
                return bc.encode_PowOp(ctx.builder, res_typeid, lhs, rhs)
            case "atan2", "float":
                return bc.encode_Atan2Op(ctx.builder, res_typeid, lhs, rhs)
            case "min", "int":
                return bc.encode_MinIOp(ctx.builder, res_typeid, lhs, rhs,
                                        signedness=get_signedness(dtype))
            case "min", "float":
                return bc.encode_MinFOp(ctx.builder, res_typeid, lhs, rhs,
                                        propagate_nan=False,
                                        flush_to_zero=self.flush_to_zero)
            case "max", "int":
                return bc.encode_MaxIOp(ctx.builder, res_typeid, lhs, rhs,
                                        signedness=get_signedness(dtype))
            case "max", "float":
                return bc.encode_MaxFOp(ctx.builder, res_typeid, lhs, rhs,
                                        propagate_nan=False,
                                        flush_to_zero=self.flush_to_zero)
            case "c_mod", "float":
                # C-style modulo
                return bc.encode_RemFOp(ctx.builder, res_typeid, lhs, rhs)
            case "c_mod", "int":
                # C-style modulo
                return bc.encode_RemIOp(ctx.builder, res_typeid, lhs, rhs,
                                        signedness=get_signedness(dtype))
            case _:
                raise NotImplementedError(f"Missing binary arithmetic implementation"
                                          f" for {self.fn}, {kind}")


def binary_arithmetic_tensorlike_raw(fn: str, x: Var[TensorLikeTy], y: Var[TensorLikeTy],
                                     rounding_mode: RoundingMode | None = None,
                                     flush_to_zero: bool = False) -> Var[TensorLikeTy]:
    ty = x.get_type()
    assert ty == y.get_type(), f"{ty} != {y.get_type()}"
    # FIXME: remove cutile-specific check
    check_rd_and_ftz(fn, rounding_mode, flush_to_zero, ty.tensor_dtype())
    return add_operation(RawBinaryArithmeticOperation, ty, fn=fn, lhs=x, rhs=y,
                         rounding_mode=rounding_mode, flush_to_zero=flush_to_zero)


def binary_arithmetic_tensorlike(fn: str, x: Var[TensorLikeTy], y: Var[TensorLikeTy],
                                 rounding_mode: RoundingMode | None = None,
                                 flush_to_zero: bool = False) -> Var[TensorLikeTy]:
    x_ty = x.get_loose_type()
    y_ty = y.get_loose_type()

    if not is_arithmetic(x_ty.tensor_dtype()):
        raise TileTypeError(f"Left-hand side has non-arithmetic dtype {x_ty.tensor_dtype()}")
    if not is_arithmetic(y_ty.tensor_dtype()):
        raise TileTypeError(f"Right-hard side has non-arithmetic dtype {y_ty.tensor_dtype()}")

    if isinstance(x_ty, LooselyTypedScalar) and isinstance(y_ty, LooselyTypedScalar):
        return binop_propagate_constant(fn, x_ty.value, y_ty.value, None)

    force_float = (fn == "truediv")
    common_ty = promote_types(x_ty, y_ty, x.ctx.typing_hooks, force_float=force_float)

    common_dtype = common_ty.tensor_dtype()
    if common_dtype == bool_:
        raise TileTypeError(f'Binary arithmetic op `{fn}` does not support bool, '
                            f'please cast bool to int')

    x = promote_and_broadcast_to(x, common_ty)
    y = promote_and_broadcast_to(y, common_ty)

    if x.is_constant() and y.is_constant():
        return binop_propagate_constant(fn, x.get_constant(), y.get_constant(), common_ty)

    return binary_arithmetic_tensorlike_raw(fn, x, y, rounding_mode, flush_to_zero)


@impl(operator.add, fixed_args=["add"], overload=(TensorLikeTy, TensorLikeTy))
@impl(operator.sub, fixed_args=["sub"], overload=(TensorLikeTy, TensorLikeTy))
@impl(operator.mul, fixed_args=["mul"], overload=(TensorLikeTy, TensorLikeTy))
@impl(operator.floordiv, fixed_args=["floordiv"], overload=(TensorLikeTy, TensorLikeTy))
@impl(operator.truediv, fixed_args=["truediv"], overload=(TensorLikeTy, TensorLikeTy))
@impl(operator.pow, fixed_args=["pow"], overload=(TensorLikeTy, TensorLikeTy))
@impl(min, fixed_args=["min"], overload=(TensorLikeTy, TensorLikeTy))
@impl(max, fixed_args=["max"], overload=(TensorLikeTy, TensorLikeTy))
def _binary_arithmetic_tensorlike_impl(fn: str, x: Var[TensorLikeTy], y: Var[TensorLikeTy]):
    return binary_arithmetic_tensorlike(fn, x, y)


def mod_tensorlike(x: Var[TensorLikeTy], y: Var[TensorLikeTy]) -> Var[TensorLikeTy]:
    x_ty = x.get_loose_type()
    y_ty = y.get_loose_type()
    if x_ty.tensor_dtype() == y_ty.tensor_dtype() == bool_:
        raise TileTypeError('Modulo operation does not support bool')

    if isinstance(x_ty, LooselyTypedScalar) and isinstance(y_ty, LooselyTypedScalar):
        with reraise_tile_exception():
            res = x_ty.value % y_ty.value
        return loosely_typed_const(res)

    # Usual promote & broadcast logic
    common_ty = promote_types(x_ty, y_ty, x.ctx.typing_hooks)
    x = promote_and_broadcast_to(x, common_ty)
    y = promote_and_broadcast_to(y, common_ty)

    if x.is_constant() and y.is_constant():
        with reraise_tile_exception():
            res = x.get_constant() % y.get_constant()
        return strictly_typed_const(res, common_ty)

    # TileOR rem follows the C behavior while Python's mod behavior differs.
    # So we generate the C-style mod first and then apply a correction.
    value = binary_arithmetic_tensorlike_raw("c_mod", x, y)

    # If the sign of `value` does not match the sign of `y`, apply a correction.
    zero = strictly_typed_const(0, common_ty)
    value_sign = compare_tensorlike("lt", value, zero)
    y_sign = compare_tensorlike("lt", y, zero)

    # need_fix = (value_sign ^ y_sign) & (value != 0)
    sign_mismatch = binary_bitwise_tensorlike("xor", value_sign, y_sign)
    value_not_zero = compare_tensorlike("ne", value, zero)
    need_fix = binary_bitwise_tensorlike("and_", sign_mismatch, value_not_zero)

    fixed_value = binary_arithmetic_tensorlike("add", value, y)
    return where(need_fix, fixed_value, value)


@impl(operator.mod, overload=(TensorLikeTy, TensorLikeTy))
def _mod_tensorlike_impl(x: Var[TensorLikeTy], y: Var[TensorLikeTy]) -> Var[TensorLikeTy]:
    return mod_tensorlike(x, y)


# Does not support broadcasting or type promotion
@dataclass(eq=False)
class RawWhereOperation(Operation, opcode="raw_where"):
    cond: Var = operand()
    x: Var = operand()
    y: Var = operand()

    @override
    def generate_bytecode(self, ctx: BytecodeContext) -> bc.Value:
        res_typeid = ctx.typeid_of(self.result_var)
        cond = ctx.get_value(self.cond)
        x = ctx.get_value(self.x)
        y = ctx.get_value(self.y)
        return bc.encode_SelectOp(ctx.builder, res_typeid, cond, x, y)


def where_raw(cond: Var[TensorLikeTy],
              x: Var[TensorLikeTy],
              y: Var[TensorLikeTy]) -> Var[TensorLikeTy]:
    ty = x.get_type()
    assert ty == y.get_type()
    return add_operation(RawWhereOperation, ty, cond=cond, x=x, y=y)


def where(cond: Var[TensorLikeTy],
          x: Var[TensorLikeTy],
          y: Var[TensorLikeTy]) -> Var[TensorLikeTy]:
    cond_ty = cond.get_loose_type()
    x_ty = x.get_loose_type()
    y_ty = y.get_loose_type()

    typing_hooks = cond.ctx.typing_hooks

    xy_ty = promote_types(x_ty, y_ty, typing_hooks)
    dtype = xy_ty.tensor_dtype()

    cond_like_ty = typing_hooks.get_tensor_like_type(dtype, cond_ty.tensor_shape())
    res_ty = promote_types(cond_like_ty, xy_ty, typing_hooks)

    result_shaped_bool = typing_hooks.get_tensor_like_type(bool_, res_ty.tensor_shape())

    cond = promote_and_broadcast_to(cond, result_shaped_bool)
    x = promote_and_broadcast_to(x, res_ty)
    y = promote_and_broadcast_to(y, res_ty)
    return where_raw(cond, x, y)
