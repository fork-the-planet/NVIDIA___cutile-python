# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from cuda.lang._enums import CachePolicy
from cuda.lang._exception import InvalidValueError, TypeCheckingError
from cuda.lang._ir.op_defs import RawNVVMIntrinsic, BitCast, InlinePTX
from ..type import (
    DTypeConstructor,
    MemorySpace,
    ScalarTy,
    PointerTy,
    VectorTy,
    type_bitwidth,
)
from cuda.lang._ir.type_checking_helpers import (
    require_constant_enum,
    require_dtype_spec,
    require_integral_scalar_type,
    require_pointer_in_memory_space,
    require_pointer_type,
    require_scalar_type,
)
from cuda.lang._stub import core_api, cache_policy
import cuda.lang._datatype as datatype
from cuda.tile._datatype import int32, opaque_pointer_dtype, pointer_dtype
from cuda.tile._ir.arithmetic_ops import astype, binary_bitwise_tensorlike
from cuda.tile._ir.core_ops import strictly_typed_const
from cuda.tile._ir.ir import Var, add_operation, add_operation_variadic
from cuda.tile._ir.op_impl import ImplRegistry, require_constant_int


_registry = ImplRegistry()
impl = _registry.impl


def core_api_impl_registry() -> ImplRegistry:
    return _registry


@impl(core_api.thread_index, fixed_args=["tid"])
@impl(core_api.thread_count, fixed_args=["ntid"])
@impl(core_api.block_index, fixed_args=["ctaid"])
@impl(core_api.block_count, fixed_args=["nctaid"])
@impl(core_api.cluster_index, fixed_args=["clusterid"])
@impl(core_api.cluster_count, fixed_args=["nclusterid"])
@impl(core_api.block_in_cluster_index, fixed_args=["cluster.ctaid"])
@impl(core_api.block_in_cluster_count, fixed_args=["cluster.nctaid"])
def read_gridlike_special_register_impl(sreg_name: str, axis: Var) -> Var:
    axis = require_constant_int(axis)
    if axis not in (0, 1, 2):
        raise TypeCheckingError(f"Axis must be 0, 1, or 2, but {axis} was given.")
    axis_name = "xyz"[axis]
    return add_operation(
        RawNVVMIntrinsic,
        ScalarTy(int32),
        intrinsic=f"llvm.nvvm.read.ptx.sreg.{sreg_name}.{axis_name}",
        operands_=()
    )


def bitcast(x: Var[ScalarTy | PointerTy | VectorTy], dtype: datatype.DType):
    x_ty = x.get_type()
    x_dtype = x_ty.tensor_dtype()
    if isinstance(dtype, VectorTy):
        # dead code for now - users have no way to construct vector dtypes
        raise TypeCheckingError("bitcast to vector is not supported")
    if datatype.bool_ in (dtype, x_dtype):
        raise TypeCheckingError("bitcast to or from bool is not supported")
    x_bitwidth = type_bitwidth(x_ty)
    if x_bitwidth != dtype.bitwidth:
        raise TypeCheckingError(
            "bitcast requires input value's type and output type to have the "
            f"same bitwidth, but input type is {x_bitwidth} bits and output "
            f"dtype has {dtype.bitwidth} bits"
        )

    # at the mlir level, we only have bitcast, inttoptr, and ptrtoint. If we
    # have a pointer, cast it to an int first then to the real dst type.
    # If we are casting *to* a pointer, first cast to int then the real dst
    # type. If both src and dst are pointer types, use a regular bitcast.
    # ir2mlir will use an address space cast.

    src_dtype, dst_dtype = x_dtype, dtype
    src_is_ptr = datatype.is_pointer_dtype(src_dtype)
    dst_is_ptr = datatype.is_pointer_dtype(dst_dtype)
    src_is_int_scalar = isinstance(x_ty, ScalarTy) and datatype.is_integral(src_dtype)
    dst_is_int_scalar = datatype.is_integral(dst_dtype)

    def direct_bitcast():
        res_ty = PointerTy(dtype) if datatype.is_pointer_dtype(dtype) else ScalarTy(dtype)
        return add_operation(BitCast, res_ty, x=x)

    def bitcast_through_int():
        intermediate_type = getattr(datatype, f'int{x_bitwidth}')
        first = bitcast(x, intermediate_type)
        return bitcast(first, dtype)

    if src_is_ptr and dst_is_ptr:
        return direct_bitcast()

    if src_is_ptr:
        if dst_is_int_scalar:
            return direct_bitcast()
        return bitcast_through_int()

    if dst_is_ptr:
        if src_is_int_scalar:
            return direct_bitcast()
        return bitcast_through_int()

    # no pointer involved: direct bitcast
    return direct_bitcast()


@impl(core_api.bitcast)
def bitcast_impl(x: Var[ScalarTy | PointerTy | VectorTy], dtype: Var[DTypeConstructor]):
    dtype = require_dtype_spec(dtype)
    return bitcast(x, dtype)


@impl(core_api.map_shared_to_cluster)
def map_shared_to_cluster_impl(pointer: Var, rank: Var):
    ptr_ty = require_pointer_type(pointer)
    rank = astype(rank, datatype.int32)
    require_pointer_in_memory_space(pointer, (MemorySpace.SHARED,))
    if ptr_ty.opaque:
        result_dtype = opaque_pointer_dtype(MemorySpace.SHARED_CLUSTER)
    else:
        result_dtype = pointer_dtype(ptr_ty.pointee_dtype, MemorySpace.SHARED_CLUSTER)
    result_ty = PointerTy(result_dtype)
    return add_operation(
        RawNVVMIntrinsic,
        result_ty,
        intrinsic="llvm.nvvm.mapa.shared.cluster",
        operands_=(pointer, rank),
    )


@impl(core_api.map_shared_to_leader_block)
def map_shared_to_leader_block(pointer: Var):
    spaces = (MemorySpace.SHARED, MemorySpace.SHARED_CLUSTER)
    pointer_type = require_pointer_in_memory_space(pointer, spaces)
    int_value = bitcast(pointer, datatype.uint32)
    mask = core_api.shared_cluster_leader_bit_mask()
    mask = strictly_typed_const(mask, ScalarTy(datatype.uint32))
    mapped = binary_bitwise_tensorlike("and_", int_value, mask)
    # TODO: should this be shared_cluster memory space?
    return bitcast(mapped, pointer_type.pointer_dtype)


@impl(core_api.setmaxregister_decrease)
def impl_setmaxregister_decrease(number_of_registers: Var[ScalarTy]):
    value = require_constant_int(number_of_registers)
    add_operation_variadic(
        InlinePTX,
        (),
        ptx_code=f"setmaxnreg.dec.sync.aligned.u32 {value};",
        read_only_operands=(),
        write_only_operands=(),
        read_write_operands=(),
    )


@impl(core_api.setmaxregister_increase)
def impl_setmaxregister_increase(number_of_registers: Var[ScalarTy]):
    value = require_constant_int(number_of_registers)
    add_operation_variadic(
        InlinePTX,
        (),
        ptx_code=f"setmaxnreg.inc.sync.aligned.u32 {value};",
        read_only_operands=(),
        write_only_operands=(),
        read_write_operands=(),
    )


@impl(cache_policy.create_range_cache_policy)
def impl_create_range_cache_policy(
    base_address,
    primary_size,
    total_size,
    primary_policy,
    secondary_policy,
):
    require_integral_scalar_type(primary_size)
    primary_size = astype(primary_size, datatype.int32)
    require_integral_scalar_type(total_size)
    total_size = astype(total_size, datatype.int32)
    require_pointer_type(base_address)
    primary_policy = require_constant_enum(primary_policy, CachePolicy)
    secondary_policy = require_constant_enum(secondary_policy, CachePolicy)
    valid = (CachePolicy.L2_EVICT_FIRST, CachePolicy.L2_EVICT_UNCHANGED)
    if secondary_policy not in valid:
        raise InvalidValueError(
            "Secondary cache policy may only be " + " or ".join(str(i) for i in valid)
        )
    code = (
        "createpolicy.range."
        + primary_policy.value
        + "."
        + secondary_policy.value
        + ".b64"
        + "  {$w0}"
        + ", [{$r0}]"
        + ", {$r1}"
        + ", {$r2};"
    )
    results = add_operation_variadic(
        InlinePTX,
        (ScalarTy(datatype.int64),),
        ptx_code=code,
        read_only_operands=(
            base_address,
            primary_size,
            total_size,
        ),
        write_only_operands=(datatype.int64,),
        read_write_operands=(),
    )
    return results[0]


@impl(cache_policy.create_fractional_cache_policy)
def impl_create_fractional_cache_policy(
    primary_policy,
    fraction,
    secondary_policy,
):
    primary_policy = require_constant_enum(primary_policy, CachePolicy)
    require_scalar_type(fraction, datatype.is_unrestricted_float)
    fraction = astype(fraction, datatype.float32)
    secondary_policy = require_constant_enum(secondary_policy, CachePolicy)
    valid = (CachePolicy.L2_EVICT_FIRST, CachePolicy.L2_EVICT_UNCHANGED)
    if secondary_policy not in valid:
        raise InvalidValueError(
            "Secondary cache policy may only be " + " or ".join(str(i) for i in valid)
        )
    code = (
        "createpolicy.fractional."
        + primary_policy.value
        + "."
        + secondary_policy.value
        + ".b64"
        + "  {$w0}"
        + ", {$r0};"
    )
    results = add_operation_variadic(
        InlinePTX,
        (ScalarTy(datatype.int64),),
        ptx_code=code,
        read_only_operands=(fraction,),
        write_only_operands=(datatype.int64,),
        read_write_operands=(),
    )
    return results[0]
