# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Literal, TypeVar

import cuda.lang._datatype as datatype
from cuda.lang._execution import stub
import cuda.lang._mlir as mlir
from cuda.lang._stub._nvvm_support import (
    _IntrinsicDTypeAnnotation,
    _IntrinsicPredicateAnnotation,
)
from cuda.lang._ir.type import TileTy
from cuda.tile import TileTypeError, TileValueError
from cuda.tile._ir.op_impl import (
    require_constant_bool,
    require_constant_enum,
    require_constant_int,
)
from cuda.tile._ir.ir import Var, add_operation_variadic
from cuda.tile._ir.ops import (
    implicit_cast,
    build_tuple,
)


FuncTy = TypeVar("FuncTy", bound=Callable[..., Any])


@dataclass(frozen=True)
class ArgSpec:
    type: object
    kind: Literal["operand", "attribute"] = "operand"
    optional: bool = False
    variadic: bool = False
    unit: bool = False
    name: str = ""


@dataclass(frozen=True)
class ResultSpec:
    name: str
    type: object
    optional: bool = False
    variadic: bool = False


def is_none_constant(value: Var) -> bool:
    return value.is_constant() and value.get_constant() is None


def is_enum_type(ty) -> bool:
    return isinstance(ty, type) and issubclass(ty, Enum)


def cast_operand(spec: ArgSpec, arg: Var) -> Var:
    target_type = spec.type
    src_type = arg.get_type()
    ctx = f"Attempting to cast argument to {target_type=}"
    match target_type:
        case _IntrinsicPredicateAnnotation():
            target_type.predicate(arg)
            return arg
        case _IntrinsicDTypeAnnotation():
            return implicit_cast(arg, target_type.dtype, ctx)
        case tuple():
            for target in target_type:
                try:
                    return implicit_cast(arg, target.dtype, ctx)
                except (TileTypeError, TileValueError):
                    pass
            options = ", ".join([str(t) for t in target_type])
            raise TileTypeError(
                f"Could not cast arg of type {src_type} to any of {options}"
            )
        case _:
            raise TileTypeError("Expected a predicate, a dtype, or a tuple of dtypes")


def make_mlir_attribute(spec: ArgSpec, arg: Var) -> tuple[str, mlir.Attribute] | None:
    if spec.optional and is_none_constant(arg):
        return None

    if is_enum_type(spec.type):
        attr_cls = getattr(mlir.nvvm, spec.type.__name__ + "Attr")
        arg = require_constant_enum(arg, spec.type)
        return spec.name, attr_cls(value=arg)

    if spec.unit:
        arg = require_constant_bool(arg)
        return (spec.name, mlir.UnitAttr()) if arg else None

    dtype = spec.type.dtype
    if dtype is datatype.bool_:
        arg = require_constant_bool(arg)
        return spec.name, mlir.BoolAttr(value=arg)

    if datatype.is_integral(dtype):
        arg = require_constant_int(arg)
        ty = mlir.IntegerType.signless(dtype.bitwidth)
        attr = mlir.IntegerAttr.make(ty, int(arg))
        return spec.name, attr

    raise TileTypeError(f"Cannot convert argument into attribute: {spec}")


def get_raw_mlir_parts(
    arg_specs, has_operand_segment_sizes, args: tuple[Var, ...]
) -> tuple[tuple[Var, ...], tuple[tuple[str, mlir.Attribute], ...]]:
    operands = []
    attributes = []
    operand_segment_sizes = []
    for arg, spec in zip(args, arg_specs, strict=True):
        if spec.kind == "attribute":
            attr = make_mlir_attribute(spec, arg)
            if attr is not None:
                attributes.append(attr)

        elif spec.optional and is_none_constant(arg):
            operand_segment_sizes.append(0)

        elif spec.variadic:
            assert isinstance(arg, tuple)
            operands.extend(cast_operand(spec, item) for item in arg)
            operand_segment_sizes.append(len(arg))
        else:
            operands.append(cast_operand(spec, arg))
            operand_segment_sizes.append(1)

    if has_operand_segment_sizes:
        attributes.append(
            ("operandSegmentSizes", mlir.DenseI32ArrayAttr(operand_segment_sizes))
        )

    return tuple(operands), tuple(attributes)


def _raw_nvvm_mlir_operation_impl(stub_func, *args: Var):
    from cuda.lang._ir.ops import RawMLIROperation

    result_types = tuple(TileTy(ty.type.dtype) for ty in stub_func._results)
    operands, attrs = get_raw_mlir_parts(
        stub_func._args, stub_func._attr_sized_operand_segments, args
    )
    results = add_operation_variadic(
        RawMLIROperation,
        result_types,
        op_name=stub_func._op_name,
        operands_=operands,
        mlir_attributes=attrs,
    )
    match len(stub_func._results):
        case 0:
            return None
        case 1:
            return results[0]
        case _:
            return build_tuple(results)


_raw_nvvm_mlir_operation_impl._is_coroutine = False


def nvvm_mlir_interface_stub(
    *,
    op_name: str,
    attr_sized_operand_segments: bool = False,
    results: tuple[ResultSpec, ...] = (),
    args: tuple[ArgSpec, ...] = (),
) -> Callable[[FuncTy], FuncTy]:
    def decorate(func: FuncTy) -> FuncTy:
        func = stub(func)
        func._cutile_custom_implementation_handler = _raw_nvvm_mlir_operation_impl
        func._op_name = op_name
        func._attr_sized_operand_segments = attr_sized_operand_segments
        func._results = results
        func._args = args
        return func

    return decorate


__all__ = ("ArgSpec", "ResultSpec", "nvvm_mlir_interface_stub")
