# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, cast

from cuda.tile._ir.cast_ops import implicit_cast
import cuda.lang._datatype as datatype
from cuda.lang._enums import (
    Tcgen05MMAKind,
    Tcgen05LdStShape,
    Tcgen05MMACollectorOp,
    CTAGroup,
    Tcgen05CopyMulticast,
    Tcgen05CopyShape,
    Tcgen05CopySourceFormat,
)
from cuda.lang._ir.type import PointerTy
from cuda.lang._stub import tcgen05 as tcgen05_stub
from cuda.lang._ir.ir import Var
from cuda.lang._ir.ops import (
    MemorySpace,
    ScalarTy,
    TileTypeError,
    VectorTy,
    add_operation_variadic,
    astype,
    require_scalar_type,
    strictly_typed_const,
)
from cuda.lang._ir.op_defs import RawNVVMIntrinsic
from .raw_mlir_operation_utils import RawMLIROperationBuilder
from cuda.lang._ir.enum_to_mlir import cl_enum_to_mlir_attribute
from cuda.lang._ir.type_checking_helpers import (
    is_none,
    make_type_checking_error,
    require_integral_scalar_type,
    require_mbarrier_ptr,
    require_pointer_in_memory_space,
    require_vector_type,
)
from cuda.tile._exception import TileValueError
from cuda.tile._ir.op_impl import (
    ImplRegistry,
    require_constant_bool,
    require_constant_enum,
    require_constant_int,
    require_optional_constant_enum,
)
import cuda.lang._mlir as mlir


_registry = ImplRegistry()
impl = _registry.impl


def tcgen05_impl_registry() -> ImplRegistry:
    return _registry


TCGEN05_VALID_COUNTS_BY_SHAPE = {
    Tcgen05LdStShape.SHAPE_16X64B: (1, 2, 4, 8, 16, 32, 64, 128),
    Tcgen05LdStShape.SHAPE_16X128B: (1, 2, 4, 8, 16, 32, 64),
    Tcgen05LdStShape.SHAPE_16X256B: (1, 2, 4, 8, 16, 32),
    Tcgen05LdStShape.SHAPE_32X32B: (1, 2, 4, 8, 16, 32, 64, 128),
    Tcgen05LdStShape.SHAPE_16X32BX2: (1, 2, 4, 8, 16, 32, 64, 128),
}

TCGEN05_REGISTERS_PER_COUNT = {
    Tcgen05LdStShape.SHAPE_16X64B: 1,
    Tcgen05LdStShape.SHAPE_16X128B: 2,
    Tcgen05LdStShape.SHAPE_16X256B: 4,
    Tcgen05LdStShape.SHAPE_32X32B: 1,
    Tcgen05LdStShape.SHAPE_16X32BX2: 1,
}


@impl(tcgen05_stub.tcgen05_alloc)
def tcgen05_alloc_impl(
    addr: Var,
    ncols: Var,
    cta_group: Var,
) -> None:
    require_pointer_in_memory_space(
        addr, (MemorySpace.SHARED_CLUSTER, MemorySpace.SHARED)
    )
    ncols = implicit_cast(ncols, datatype.int32, "cast num columns to int32")
    cta_group_value = cast(CTAGroup, require_constant_enum(cta_group, CTAGroup))
    intrinsic = "llvm.nvvm.tcgen05.alloc.shared." + cta_group_value.value
    add_operation_variadic(
        RawNVVMIntrinsic,
        (),
        intrinsic=intrinsic,
        operands_=(addr, ncols),
    )


@impl(tcgen05_stub.tcgen05_dealloc)
def tcgen05_dealloc_impl(
    addr: Var,
    ncols: Var,
    cta_group: Var,
) -> None:
    require_pointer_in_memory_space(addr, (MemorySpace.TENSOR,))
    ncols = implicit_cast(ncols, datatype.int32, "cast num columns to int32")
    cta_group_value = cast(CTAGroup, require_constant_enum(cta_group, CTAGroup))
    intrinsic = "llvm.nvvm.tcgen05.dealloc." + cta_group_value.value
    add_operation_variadic(
        RawNVVMIntrinsic,
        (),
        intrinsic=intrinsic,
        operands_=(addr, ncols),
    )


@impl(tcgen05_stub.tcgen05_commit)
def tcgen05_commit_impl(
    mbar: Var,
    multicast_mask: Var,
    cta_group: Var,
):
    require_mbarrier_ptr(mbar)
    operands = [mbar]
    cta_group_value = cast(CTAGroup, require_constant_enum(cta_group, CTAGroup))
    intrinsic = "llvm.nvvm.tcgen05.commit"
    if not is_none(multicast_mask):
        intrinsic += ".mc"
        mask = implicit_cast(multicast_mask, datatype.int16, "multicast mask")
        operands.append(mask)
    intrinsic += ".shared." + cta_group_value.value
    add_operation_variadic(
        RawNVVMIntrinsic,
        (),
        intrinsic=intrinsic,
        operands_=tuple(operands),
    )


@impl(tcgen05_stub.tcgen05_copy)
def tcgen05_copy_impl(
    address,
    shared_memory_descriptor,
    shape,
    cta_group,
    multicast,
    source_format,
):
    require_pointer_in_memory_space(address, (MemorySpace.TENSOR,))
    require_scalar_type(
        shared_memory_descriptor,
        lambda x: x == datatype.int64,
        "Expected shared memory descriptor to be encoded as a 64-bit integer.",
    )

    group_value = require_constant_enum(cta_group, CTAGroup)
    group_attribute = cl_enum_to_mlir_attribute(group_value)

    shape_value = require_constant_enum(shape, Tcgen05CopyShape)
    shape_attribute = cl_enum_to_mlir_attribute(shape_value)

    builder = (
        RawMLIROperationBuilder(name="nvvm.tcgen05.cp")
        .add_operand(address)
        .add_operand(shared_memory_descriptor)
        .add_attribute("group", group_attribute)
        .add_attribute("shape", shape_attribute)
    )

    if is_none(multicast):
        value = mlir.nvvm.Tcgen05CpMulticast.NONE
        multicast_attribute = mlir.nvvm.Tcgen05CpMulticastAttr(value=value)
    else:
        multicast_value = require_constant_enum(multicast, Tcgen05CopyMulticast)
        multicast_attribute = cl_enum_to_mlir_attribute(multicast_value)

    builder = builder.add_attribute("multicast", multicast_attribute)

    if not is_none(source_format):
        source_format_value = require_constant_enum(
            source_format, Tcgen05CopySourceFormat
        )
        source_format_attribute = cl_enum_to_mlir_attribute(source_format_value)
        builder = builder.add_attribute("srcFormat", source_format_attribute)

    builder.emit()


@impl(tcgen05_stub.tcgen05_store)
def tcgen05_store_impl(
    shape: Var,
    tmem_addr: Var,
    value: Var,
    unpack: Var,
    offset: Var,
):
    require_pointer_in_memory_space(tmem_addr, (MemorySpace.TENSOR,))
    shape_value = require_constant_enum(shape, Tcgen05LdStShape)
    valid_counts = TCGEN05_VALID_COUNTS_BY_SHAPE[shape_value]
    registers_per_count = TCGEN05_REGISTERS_PER_COUNT[shape_value]
    value_type = value.get_type()
    if is_none(offset):
        offset = None
    else:
        require_constant_int(offset)
        offset = astype(offset, datatype.int64)
    require_constant_bool(unpack)

    def type_error(dtype, count):
        message = (
            "Expected scalar 32-bit integer or vector of 32-bit integers "
            f"but got {count=} and {dtype=}"
        )
        raise TileTypeError(message)

    match value_type:
        case ScalarTy() as st:
            count = 1
            if st.dtype != datatype.int32:
                type_error(st.dtype, count)
        case VectorTy() as vt:
            count = vt.length
            if vt.element_dtype != datatype.int32:
                type_error(vt.element_dtype, count)
        case _:
            raise TileTypeError("Expected scalar or vector with datatype int32")

    valid_register_counts = tuple(
        valid_count * registers_per_count for valid_count in valid_counts
    )
    if count not in valid_register_counts:
        valid = ", ".join(str(count) for count in valid_register_counts)
        raise TileValueError(
            f"Expected register count for {shape_value.name} to be one of "
            f"{valid}, got {count}"
        )

    count_value = count // registers_per_count
    needs_offset = shape_value == Tcgen05LdStShape.SHAPE_16X32BX2
    has_offset = offset is not None
    if needs_offset != has_offset:
        raise TileTypeError(
            "offset parameter is only valid with shape Tcgen05LdStShape.SHAPE_16X32BX2"
        )

    operands = (
        tmem_addr,
        *([offset] if offset is not None else []),
        value,
        unpack,
    )
    intrinsic = f"llvm.nvvm.tcgen05.st.{shape_value.value}.x{count_value}"
    add_operation_variadic(
        RawNVVMIntrinsic,
        (),
        intrinsic=intrinsic,
        operands_=operands,
    )


@impl(tcgen05_stub.tcgen05_load)
def tcgen05_load_impl(
    shape: Var,
    tmem_addr: Var,
    count: Var,
    pack: Var,
    offset: Var,
) -> Var:
    require_pointer_in_memory_space(tmem_addr, (MemorySpace.TENSOR,))
    shape_value = require_constant_enum(shape, Tcgen05LdStShape)
    count_value = require_constant_int(count)
    valid_counts = TCGEN05_VALID_COUNTS_BY_SHAPE[shape_value]
    if count_value not in valid_counts:
        valid = ", ".join(str(value) for value in valid_counts)
        raise TileValueError(
            f"Expected count for {shape_value.name} to be one of {valid}, got {count_value}"
        )

    has_offset = not is_none(offset)
    uses_offset = shape_value is Tcgen05LdStShape.SHAPE_16X32BX2
    if uses_offset and not has_offset:
        raise TileTypeError("tcgen05_load with SHAPE_16X32BX2 requires offset")
    if has_offset and not uses_offset:
        raise TileTypeError("tcgen05_load offset is only valid with SHAPE_16X32BX2")

    operands = [tmem_addr]
    if has_offset:
        require_scalar_type(offset)
        operands.append(astype(offset, datatype.int64))

    if is_none(pack):
        operands.append(strictly_typed_const(False, ScalarTy(datatype.bool_)))
    else:
        require_scalar_type(
            pack,
            lambda dtype: dtype is datatype.bool_,
            f"Expected pack dtype to be {datatype.bool_}",
        )
        operands.append(pack)

    intrinsic = f"llvm.nvvm.tcgen05.ld.{shape_value.value}.x{count_value}"
    total_registers = count_value * TCGEN05_REGISTERS_PER_COUNT[shape_value]
    result_type = (
        ScalarTy(datatype.int32)
        if total_registers == 1
        else VectorTy(datatype.int32, total_registers)
    )

    [result] = add_operation_variadic(
        RawNVVMIntrinsic,
        (result_type,),
        intrinsic=intrinsic,
        operands_=tuple(operands),
    )
    return result


def _require_tcgen05_mma_matrix_a(var: Var):

    ty = var.get_type()

    def error():
        return make_type_checking_error(
            "Expected a tensor memory pointer or a shared memory descriptor "
            f"encoded as a 64 bit integer but got {ty}"
        )

    match ty:
        case PointerTy() as pt:
            info = datatype.PointerInfo(pt.pointer_dtype)
            if info.memory_space is not MemorySpace.TENSOR:
                raise error()
            return var
        case ScalarTy() as st:
            if not datatype.is_integral(st.dtype) or st.dtype.bitwidth != 64:
                raise error()
            return astype(var, datatype.int64)
        case _:
            raise error()


@impl(tcgen05_stub.tcgen05_mma)
def tcgen05_mma_impl(
    kind: Var[Any],
    cta_group: Var[Any],
    matrix_d: Var[Any],
    matrix_a: Var[Any],
    matrix_b: Var[Any],
    idesc: Var[Any],
    enable_input_d: Var[Any],
    scale_input_d: Var[Any],
    disable_output_lane: Var[Any],
    collector_op: Var[Any],
    a_shift: Var[Any],
):
    kind_value = require_constant_enum(kind, Tcgen05MMAKind)
    cta_group_value = (
        require_optional_constant_enum(cta_group, CTAGroup) or CTAGroup.CTA_1
    )
    collector_op_value = require_constant_enum(collector_op, Tcgen05MMACollectorOp)
    require_pointer_in_memory_space(matrix_d, (MemorySpace.TENSOR,))
    matrix_a = _require_tcgen05_mma_matrix_a(matrix_a)
    require_integral_scalar_type(matrix_b, bitwidth=64)
    require_integral_scalar_type(idesc)
    idesc = implicit_cast(idesc, datatype.int32, "idesc as int32")
    enable_input_d = implicit_cast(
        enable_input_d, datatype.bool_, "enable_input_d as bool_"
    )

    builder = (
        RawMLIROperationBuilder(name="nvvm.tcgen05.mma")
        .add_attribute("mmaKind", cl_enum_to_mlir_attribute(kind_value))
        .add_attribute("ctaGroup", cl_enum_to_mlir_attribute(cta_group_value))
        .add_attribute("collectorOp", cl_enum_to_mlir_attribute(collector_op_value))
        .add_operand(matrix_d)
        .add_operand(matrix_a)
        .add_operand(matrix_b)
        .add_operand(idesc)
        .add_operand(enable_input_d)
    )

    if not is_none(scale_input_d):
        value = require_constant_int(scale_input_d)
        if value < 0 or value > 15:
            raise make_type_checking_error(
                "Expected scale_input_d to be an immediate in [0, 15]"
            )
        scale_input_d = astype(scale_input_d, datatype.int32)
    builder = builder.add_optional_operand(scale_input_d)

    if not is_none(disable_output_lane):
        expected_len = 4 if cta_group_value is CTAGroup.CTA_1 else 8
        require_vector_type(disable_output_lane, expected_len)
        disable_output_lane = astype(disable_output_lane, datatype.int32)
    builder = builder.add_optional_operand(disable_output_lane)

    if not is_none(a_shift):
        value = require_constant_bool(a_shift)
        if value:
            if not isinstance(matrix_a.get_type(), PointerTy):
                raise make_type_checking_error(
                    "a_shift can only be applied if A is in tensor memory", a_shift
                )
            builder = builder.add_unit_attribute("aShift")

    builder.emit()
