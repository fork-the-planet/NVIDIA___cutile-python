# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Any, Optional

from typing_extensions import override

import cuda.tile._bytecode as bc
from cuda.tile._ir.ir import Operation, attribute, Var, Builder
from cuda.tile._ir.type import Type, DTypeSpec, TensorLikeTy
from cuda.tile._ir.typing_support import type_of_constant_python_value, \
    loose_type_of_constant_python_value
from cuda.tile._ir2bytecode import BytecodeContext


@dataclass(eq=False)
class TypedConst(Operation, opcode="typed_const"):
    value: Any = attribute()

    @override
    def generate_bytecode(self, ctx: BytecodeContext) -> bc.Value:
        return ctx.constant(self.value, ctx.typeof(self.result_var))


def loosely_typed_const(value: Any,
                        ty: Optional[Type] = None,
                        loose_ty: Optional[Type] = None,
                        name: str | None = None) -> Var:
    builder = Builder.get_current()
    if ty is None:
        ty = type_of_constant_python_value(value, builder.ir_ctx.typing_hooks)
    assert not ty.is_aggregate(), "Use sym2var(value, constant_only=True) instead"

    # Normalize third party dtype spec objects (e.g. torch.float32 -> ct.float32)
    if isinstance(ty, DTypeSpec):
        value = ty.dtype

    ret = _strictly_typed_const_inner(builder, value, ty, name=name)
    if loose_ty is None:
        loose_ty = loose_type_of_constant_python_value(value, builder.ir_ctx.typing_hooks)
    ret.set_loose_type(loose_ty)
    return ret


def strictly_typed_const(value: Any, ty: Type, name: str | None = None) -> Var:
    return _strictly_typed_const_inner(Builder.get_current(), value, ty, name)


def _strictly_typed_const_inner(builder: Builder,
                                value: Any, ty: Type, name: str | None = None) -> Var:
    result = None if name is None else builder.ir_ctx.make_var(name, builder.loc)
    ret = builder.add_operation(TypedConst, ty, dict(value=value), result=result)
    if not isinstance(ty, TensorLikeTy) or ty.tensor_shape() == ():
        # We currently don't have a way to represent an N-dimensional tile constant
        ret.set_constant(value)
    return ret
