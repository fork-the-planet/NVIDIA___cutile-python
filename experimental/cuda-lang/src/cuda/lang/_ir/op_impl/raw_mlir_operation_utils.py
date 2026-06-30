# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# Building a RawMLIROperation is tricky because one must track the operand
# segment sizes for optional or variadic operands and convert enums to
# attributes. This module simplifies the process.

from dataclasses import dataclass, replace
from typing import Any

from cuda.tile._ir.ir import add_operation_variadic
from cuda.lang._ir.ir import Var
from cuda.lang._ir.op_defs import RawMLIROperation
from cuda.lang._ir.type import Type
from cuda.lang._ir.type_checking_helpers import is_none
from cuda.lang._mlir import DenseI32ArrayAttr, UnitAttr
from cuda.lang._mlir._builtins import Attribute


@dataclass(eq=False, frozen=True, kw_only=True)
class RawMLIROperationBuilder:
    name: str
    result_types: Type | tuple[Type, ...] = ()
    operands_: tuple[Var[Any], ...] = ()
    segments_: tuple[int, ...] = ()
    attributes_: tuple[tuple[str, Attribute], ...] = ()
    add_segment_sizes_: bool = False

    @property
    def operands(self) -> tuple[Var[Any], ...]:
        return self.operands_

    @property
    def attributes(self) -> tuple[tuple[str, Attribute], ...]:
        if not self.add_segment_sizes_:
            return self.attributes_
        return (
            *self.attributes_,
            ("operandSegmentSizes", DenseI32ArrayAttr(self.segments_)),
        )

    def add_attribute(self, name: str, attribute: Attribute):
        return replace(self, attributes_=(*self.attributes_, (name, attribute)))

    def add_unit_attribute(self, name: str, value: bool = True):
        if value:
            return self.add_attribute(name, UnitAttr())
        return self

    def add_operand(self, var: Var[Any]):
        return replace(
            self,
            operands_=(*self.operands_, var),
            segments_=(*self.segments_, 1),
        )

    def add_variadic_operand(self, vars: tuple[Var[Any], ...] | list[Var[Any]]):
        vars = tuple(vars)
        return replace(
            self,
            operands_=(*self.operands_, *vars),
            segments_=(*self.segments_, len(vars)),
            add_segment_sizes_=True,
        )

    def add_optional_operand(self, var: Var[Any] | None):
        if var is None or is_none(var):
            return replace(
                self,
                segments_=(*self.segments_, 0),
                add_segment_sizes_=True,
            )
        return replace(
            self,
            operands_=(*self.operands_, var),
            segments_=(*self.segments_, 1),
            add_segment_sizes_=True,
        )

    def emit(self) -> tuple[Var[Any], ...]:
        return add_operation_variadic(
            RawMLIROperation,
            self.result_types,
            op_name=self.name,
            operands_=self.operands,
            mlir_attributes=self.attributes,
        )
