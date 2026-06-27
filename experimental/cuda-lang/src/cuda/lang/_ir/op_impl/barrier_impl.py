# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import cuda.lang._datatype as datatype
from cuda.lang._ir.type import ScalarTy
from cuda.lang._ir.ir import add_operation
from cuda.lang._ir.op_defs import RawNVVMIntrinsic
from cuda.tile._ir.ops import implicit_cast
from cuda.tile._ir.op_impl import (
    ImplRegistry,
    require_constant_bool,
)
from cuda.lang._ir.type_checking_helpers import (
    require_integral_scalar_type,
    require_boolean_scalar_type,
    optional_cast,
)
from cuda.lang._stub import barrier
from cuda.lang._exception import TypeCheckingError

_registry = ImplRegistry()
impl = _registry.impl


def barrier_impl_registry() -> ImplRegistry:
    return _registry


def _require_barrier_reduction_kind(op):
    if not op.is_constant():
        raise TypeCheckingError("Expected BarrierReductionKind constant")
    value = op.get_constant()
    if isinstance(value, barrier.BarrierReductionKind):
        return value
    try:
        return barrier.BarrierReductionKind(value)
    except (TypeError, ValueError):
        valid = ", ".join(kind.name for kind in barrier.BarrierReductionKind)
        raise TypeCheckingError(f"Expected BarrierReductionKind to be one of {valid}")


@impl(barrier.barrier_reduce_block)
def barrier_reduce_block_impl(
    op,
    predicate,
    number_of_threads,
    barrier_id,
    aligned,
):
    op = _require_barrier_reduction_kind(op)
    require_boolean_scalar_type(predicate)
    require_integral_scalar_type(barrier_id)
    barrier_id = implicit_cast(barrier_id, datatype.int32, "barrier id")
    number_of_threads = optional_cast(
        number_of_threads, datatype.int32, "barrier number_of_threads"
    )
    aligned = require_constant_bool(aligned)

    intrinsic = "llvm.nvvm.barrier.cta.red."
    match op:
        case barrier.BarrierReductionKind.POP_COUNT:
            intrinsic += "popc"
        case barrier.BarrierReductionKind.AND:
            intrinsic += "and"
        case barrier.BarrierReductionKind.OR:
            intrinsic += "or"
        case _:
            assert False

    if aligned:
        intrinsic += ".aligned"
    if number_of_threads is None:
        intrinsic += ".all"
        operands = (barrier_id, predicate)
    else:
        intrinsic += ".count"
        operands = (barrier_id, number_of_threads, predicate)

    result_type = (
        ScalarTy(datatype.int32)
        if op is barrier.BarrierReductionKind.POP_COUNT
        else ScalarTy(datatype.bool_)
    )
    return add_operation(
        RawNVVMIntrinsic,
        result_type,
        intrinsic=intrinsic,
        operands_=operands,
    )
