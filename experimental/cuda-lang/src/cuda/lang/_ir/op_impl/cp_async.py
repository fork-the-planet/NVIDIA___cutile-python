# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from cuda.tile._ir.op_impl import ImplRegistry
from cuda.tile._ir.ops import implicit_cast
import cuda.lang._datatype as datatype
from cuda.lang._mlir import BoolAttr
from cuda.lang._enums import MemorySpace
from cuda.lang._stub import cp_async
from .raw_mlir_operation_utils import RawMLIROperationBuilder
from ..type_checking_helpers import (
    is_none,
    make_type_checking_error,
    require_boolean_scalar_type,
    require_mbarrier_ptr,
    require_none,
    require_optional,
    require_pointer_in_memory_space,
    require_uniform_int_tuple_type,
    tensor_map_descriptor_like,
)
from cuda.tile._ir.op_impl import require_constant_enum, require_optional_constant_enum
import cuda.lang._mlir.nvvm as mlir


_registry = ImplRegistry()
impl = _registry.impl


def cp_async_impl_registry() -> ImplRegistry:
    return _registry


def validate_g2s_mode(mode: cp_async.TMALoadMode, im2col_count: int) -> None:
    match mode:
        case cp_async.TMALoadMode.TILE | cp_async.TMALoadMode.TILE_GATHER4:
            if im2col_count != 0:
                raise make_type_checking_error(
                    f"{mode.name} mode does not accept im2col_offsets"
                )

        case (
            cp_async.TMALoadMode.IM2COL
            | cp_async.TMALoadMode.IM2COL_W
            | cp_async.TMALoadMode.IM2COL_W_128
        ):
            if im2col_count == 0:
                raise make_type_checking_error(
                    f"{mode.name} mode requires im2col_offsets"
                )

        case _:
            raise make_type_checking_error(f"Unsupported TMA load mode {mode}")


def optional_cast(var, dtype, context: str):
    if is_none(var):
        return None
    return implicit_cast(var, dtype, context)


@impl(cp_async.cp_async_bulk_tensor_global_to_shared)
def cp_async_bulk_tensor_global_to_shared_impl(
    src_tensor_map_descriptor,
    src_coordinates,
    dst_memory,
    mbarrier,
    im2col_offsets,
    multicast_mask,
    l2_cache_hint,
    mode,
    group,
    predicate,
):
    tensor_map = tensor_map_descriptor_like(src_tensor_map_descriptor)
    src_coordinate_vars = require_uniform_int_tuple_type(src_coordinates)
    im2col_offset_vars = require_uniform_int_tuple_type(im2col_offsets)
    require_mbarrier_ptr(mbarrier, (MemorySpace.SHARED,))
    mode = require_constant_enum(mode, cp_async.TMALoadMode)
    validate_g2s_mode(mode, len(im2col_offset_vars))
    mode = getattr(mlir.TMALoadMode, mode.name)
    dst_ty = require_pointer_in_memory_space(
        dst_memory,
        (MemorySpace.SHARED, MemorySpace.SHARED_CLUSTER),
    )
    is_cta_only = dst_ty.memory_space == MemorySpace.SHARED
    group_attr = None
    src_coordinates = tuple(
        implicit_cast(coord, datatype.int32, "TMA coordinates")
        for coord in src_coordinate_vars
    )
    im2col_offsets = tuple(
        implicit_cast(offset, datatype.int16, "TMA im2col offsets")
        for offset in im2col_offset_vars
    )
    l2_cache_hint = optional_cast(l2_cache_hint, datatype.int64, "TMA L2 cache hint")

    if is_cta_only:
        message = (
            "When the destination memory is in shared memory, the "
            "predicate, multicast mask, and group arguments are invalid."
        )
        require_none(predicate, message)
        require_none(multicast_mask, message)
        require_none(group, message)
    else:
        multicast_mask = optional_cast(
            multicast_mask, datatype.int16, "TMA multicast mask"
        )
        require_optional(predicate, require_boolean_scalar_type)
        group_value = require_optional_constant_enum(group, cp_async.CTAGroup)
        group_attr = (
            None
            if group_value is None
            else mlir.CTAGroupKindAttr(
                value=getattr(mlir.CTAGroupKind, group_value.name)
            )
        )

    builder = (
        RawMLIROperationBuilder(
            name="nvvm.cp.async.bulk.tensor.shared.cluster.global"
        )
        .add_attribute("mode", mlir.TMALoadModeAttr(value=mode))
        .add_attribute("isCTAOnly", BoolAttr(value=is_cta_only))
    )
    if not is_cta_only and group_attr is not None:
        builder = builder.add_attribute("group", group_attr)

    builder = (
        builder.add_operand(dst_memory)
        .add_operand(tensor_map)
        .add_variadic_operand(src_coordinates)
        .add_operand(mbarrier)
        .add_variadic_operand(im2col_offsets)
        .add_optional_operand(multicast_mask)
        .add_optional_operand(l2_cache_hint)
        .add_optional_operand(predicate)
    )
    builder.emit()


@impl(cp_async.cp_async_bulk_tensor_shared_to_global)
def cp_async_bulk_tensor_shared_to_global_impl(
    src_memory,
    dst_tensor_map_descriptor,
    dst_coordinates,
    l2_cache_hint,
    mode,
    predicate,
):
    require_pointer_in_memory_space(src_memory, (MemorySpace.SHARED,))
    tensor_map = tensor_map_descriptor_like(dst_tensor_map_descriptor)
    dst_coordinate_vars = require_uniform_int_tuple_type(dst_coordinates)
    mode = require_constant_enum(mode, cp_async.TMAStoreMode)
    mode = getattr(mlir.TMAStoreMode, mode.name)
    dst_coordinates = tuple(
        implicit_cast(coord, datatype.int32, "TMA coordinates")
        for coord in dst_coordinate_vars
    )
    l2_cache_hint = optional_cast(l2_cache_hint, datatype.int64, "TMA L2 cache hint")
    predicate = optional_cast(predicate, datatype.bool_, "TMA predicate")
    builder = (
        RawMLIROperationBuilder(name="nvvm.cp.async.bulk.tensor.global.shared.cta")
        .add_attribute("mode", mlir.TMAStoreModeAttr(value=mode))
        .add_operand(tensor_map)
        .add_operand(src_memory)
        .add_variadic_operand(dst_coordinates)
        .add_optional_operand(l2_cache_hint)
        .add_optional_operand(predicate)
    )
    builder.emit()
