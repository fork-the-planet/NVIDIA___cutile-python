# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Literal

from .._datatype import uint32
from cuda.lang._execution import stub, function
from .bits import set_bit32, set_bits32
from .nvvm import P3, P6
from . import nvvm as _nvvm
from .._enums import (
    CTAGroup,
    SwizzleMode,
    Tcgen05MMAKind,
    Tcgen05MMABlockScaleKind,
    Tcgen05MMAScaleVectorSize,
    Tcgen05MMACollectorBBuffer,
    Tcgen05MMACollectorOp,
    Tcgen05LoadStoreShape,
    Tcgen05CopyMulticast,
    Tcgen05CopyShape,
    Tcgen05CopySourceFormat,
)
from cuda.tile import static_assert


@function()
def tcgen05_wait_load() -> None:
    _nvvm.tcgen05_wait_ld()


@function()
def tcgen05_wait_store() -> None:
    _nvvm.tcgen05_wait_st()


@function()
def tcgen05_fence_before_thread_sync() -> None:
    """
    Orders all prior async tcgen05 operations with respect to the subsequent
    tcgen05 and execution ordering operations
    """
    _nvvm.tcgen05_fence_before_thread_sync()


@function()
def tcgen05_fence_after_thread_sync() -> None:
    """
    Orders all subsequent async tcgen05 operations with respect to the prior
    tcgen05 and execution ordering operations
    """
    _nvvm.tcgen05_fence_after_thread_sync()


@function()
def tcgen05_relinquish_allocation_permit(cta_group: CTAGroup = CTAGroup.CTA_1) -> None:
    static_assert(cta_group in (CTAGroup.CTA_1, CTAGroup.CTA_2))
    if cta_group == CTAGroup.CTA_1:
        _nvvm.tcgen05_relinq_alloc_permit_cg1()
    else:
        _nvvm.tcgen05_relinq_alloc_permit_cg2()


@function()
def tcgen05_shift_down(address, cta_group: CTAGroup = CTAGroup.CTA_1) -> None:
    """
    Asynchronously shift down the rows of the matrix in the Tensor Memory for a warp.

    Args:
        address: pointer in tensor memory
        cta_group: cta group 1 or 2
    """
    static_assert(cta_group in (CTAGroup.CTA_1, CTAGroup.CTA_2))
    if cta_group == CTAGroup.CTA_1:
        _nvvm.tcgen05_shift_down_cg1(address)
    else:
        _nvvm.tcgen05_shift_down_cg2(address)


@stub
def tcgen05_allocate(
    address: P3,
    number_of_columns: int,
    *,
    cta_group: CTAGroup = CTAGroup.CTA_1,
) -> None:
    """Allocate tensor memory columns and write their address to ``address``."""
    ...


@stub
def tcgen05_deallocate(
    address: P6,
    number_of_columns: int,
    *,
    cta_group: CTAGroup = CTAGroup.CTA_1,
) -> None:
    """Deallocate tensor memory columns starting at ``address``."""
    ...


@stub
def tcgen05_tmem_offset(
    pointer: P6,
    *,
    lane_offset: int = 0,
    column_offset: int = 0,
) -> P6:
    """Offset a tensor memory pointer by lane and column coordinates.

    Args:
        pointer: Pointer in tensor memory.
        lane_offset (int): Number of tensor memory lanes to add.
        column_offset (int): Number of tensor memory columns to add.

    Returns:
        A tensor memory pointer with the same pointee type as ``pointer``.
    """
    ...


@stub
def tcgen05_commit(
    mbar: P3,
    *,
    multicast_mask: int | None = None,
    cta_group: CTAGroup = CTAGroup.CTA_1,
) -> None:
    """Commit tcgen05 tensor memory operations and arrive at ``mbar``."""
    ...


@stub
def tcgen05_load(
    shape: Tcgen05LoadStoreShape,
    tensor_memory_address: P6,
    *,
    count: int = 1,
    pack: bool | None = None,
    offset: int | None = None,
) -> Any:
    """Load registers from tensor memory using a tcgen05 load shape."""
    ...


@stub
def tcgen05_copy(
    address,
    shared_memory_descriptor,
    *,
    shape: Tcgen05CopyShape,
    cta_group: CTAGroup = CTAGroup.CTA_1,
    multicast: Tcgen05CopyMulticast | None = None,
    source_format: Tcgen05CopySourceFormat | None = None,
):
    """
    Initiates an asynchronous copy operation from shared memory to the
    location specified by ``address``.

    Args:
        address: Pointer in tensor memory allocated by tcgen05_allocate.
        shared_memory_descriptor: Shared memory descriptor encoded
            as a 64-bit integer.
        cta_group:
        shape:
        multicast:
        source_format:
    """


@stub
def tcgen05_store(
    shape: Tcgen05LoadStoreShape,
    tensor_memory_address,
    value,
    *,
    unpack: bool = False,
    offset: int | None = None,
):
    """
    Store registers to tensor memory using a tcgen05 store shape.

    Args:
        shape:
        tensor_memory_address: Pointer in tensor memory (address space 6).
        value: 32-bit signless integer or vector of 32-bit signless integer
            values of length 2/4/8/16/32/64/128
        unpack: unpack a 32-bit element in the register into two 16-bit
            elements and store them in adjacent columns.
        offset: When shape 16x32bx2 is used, base address of the first access is
            specified by ``tensor_memory_address`` and the base address of the second
            access is specified by ``tensor_memory_address + offset``, where offset is
            an immediate argument.
    """


class _Tcgen05Tf32Type(IntEnum):
    TF32 = 2


class _Tcgen05F16Type(IntEnum):
    F16 = 0
    BF16 = 1


class _Tcgen05F8F6F4Type(IntEnum):
    E4M3 = 0
    E5M2 = 1
    E2M3 = 3
    E3M2 = 4
    E2M1 = 5


class _Tcgen05I8Type(IntEnum):
    U8 = 0
    S8 = 1


class _Tcgen05Mxf4Type(IntEnum):
    E2M1 = 1


class _DType(IntEnum):
    F16 = 0
    F32 = 1
    S32 = 2


class _MaxShift(IntEnum):
    NoShift = 0
    MaxShift8 = 1
    MaxShift16 = 2
    MaxShift32 = 3


class _Mxf8f6f4ScaleFormat(IntEnum):
    UE8M0 = 1


class _Mxf4ScaleFormat(IntEnum):
    UE4M3 = 0
    UE8M0 = 1


class _Mxf4KDimension(IntEnum):
    DenseK64OrSparseK128 = 0
    DenseK96 = 1


@dataclass(frozen=True)
class Tcgen05InstructionDescriptor:
    """
    Instruction descriptor format for .kind::tf32, .kind::f16, .kind::f8f6f4 and .kind::i8
    """

    Tf32Type = _Tcgen05Tf32Type
    F16Type = _Tcgen05F16Type
    F8F6F4Type = _Tcgen05F8F6F4Type
    I8Type = _Tcgen05I8Type
    DType = _DType
    MaxShift = _MaxShift

    sparsity_selector: int = 0
    sparse: bool = False
    saturate: bool = False
    d_type: DType = DType.F16
    a_type: Tf32Type | F16Type | F8F6F4Type | I8Type = F16Type.F16
    b_type: Tf32Type | F16Type | F8F6F4Type | I8Type = F16Type.F16
    negate_a: bool = False
    negate_b: bool = False
    transpose_a: bool = False
    transpose_b: bool = False
    n: int = 0
    m: int = 0
    max_shift: MaxShift = MaxShift.NoShift

    def encode(self) -> int:
        desc = uint32(0x0000_0000)
        desc = set_bits32(desc, self.sparsity_selector, 0, 2)
        desc = set_bit32(desc, 2, self.sparse)
        desc = set_bit32(desc, 3, self.saturate)
        desc = set_bits32(desc, self.d_type, 4, 2)
        desc = set_bits32(desc, self.a_type, 7, 3)
        desc = set_bits32(desc, self.b_type, 10, 3)
        desc = set_bit32(desc, 13, self.negate_a)
        desc = set_bit32(desc, 14, self.negate_b)
        desc = set_bit32(desc, 15, self.transpose_a)
        desc = set_bit32(desc, 16, self.transpose_b)
        desc = set_bits32(desc, self.n >> 3, 17, 6)
        desc = set_bits32(desc, self.m >> 4, 24, 5)
        desc = set_bits32(desc, self.max_shift, 30, 2)
        return desc


@dataclass(frozen=True)
class Tcgen05Mxf8f6f4InstructionDescriptor:
    """Instruction descriptor format for .kind::mxf8f6f4"""

    Type = _Tcgen05F8F6F4Type
    ScaleFormat = _Mxf8f6f4ScaleFormat

    sparse: bool = False
    b_scale_id: Literal[0, 1, 2, 3] = 0
    a_type: Type = Type.E4M3
    b_type: Type = Type.E4M3
    negate_a: bool = False
    negate_b: bool = False
    transpose_a: bool = False
    transpose_b: bool = False
    n: int = 0
    scale_format: ScaleFormat = ScaleFormat.UE8M0
    m: int = 0
    a_scale_id: Literal[0, 1, 2, 3] = 0

    def encode(self) -> int:
        desc = uint32(0x0000_0000)
        desc = set_bit32(desc, 2, self.sparse)
        desc = set_bits32(desc, self.b_scale_id, 4, 2)
        desc = set_bits32(desc, self.a_type, 7, 3)
        desc = set_bits32(desc, self.b_type, 10, 3)
        desc = set_bit32(desc, 13, self.negate_a)
        desc = set_bit32(desc, 14, self.negate_b)
        desc = set_bit32(desc, 15, self.transpose_a)
        desc = set_bit32(desc, 16, self.transpose_b)
        desc = set_bits32(desc, self.n >> 3, 17, 6)
        desc = set_bit32(desc, 23, self.scale_format)
        desc = set_bits32(desc, self.m >> 7, 27, 2)
        desc = set_bits32(desc, self.a_scale_id, 29, 2)
        return desc


@dataclass(frozen=True)
class Tcgen05Mxf4InstructionDescriptor:
    """Instruction descriptor format for .kind::mxf4 and .kind::mxf4nvf4"""

    Type = _Tcgen05Mxf4Type
    ScaleFormat = _Mxf4ScaleFormat
    KDimension = _Mxf4KDimension

    sparse: bool = False
    b_scale_id: Literal[0, 2] = 0
    a_type: Type = Type.E2M1
    b_type: Type = Type.E2M1
    negate_a: bool = False
    negate_b: bool = False
    transpose_a: bool = False
    transpose_b: bool = False
    n: int = 0
    scale_format: ScaleFormat = ScaleFormat.UE8M0
    m: int = 0
    a_scale_id: Literal[0, 2] = 0
    k_dimension: KDimension = KDimension.DenseK64OrSparseK128

    def encode(self) -> int:
        desc = uint32(0x0000_0000)
        desc = set_bit32(desc, 2, self.sparse)
        desc = set_bits32(desc, self.b_scale_id, 4, 2)
        desc = set_bits32(desc, self.a_type, 7, 3)
        desc = set_bits32(desc, self.b_type, 10, 2)
        desc = set_bit32(desc, 13, self.negate_a)
        desc = set_bit32(desc, 14, self.negate_b)
        desc = set_bit32(desc, 15, self.transpose_a)
        desc = set_bit32(desc, 16, self.transpose_b)
        desc = set_bits32(desc, self.n >> 3, 17, 6)
        desc = set_bit32(desc, 23, self.scale_format)
        desc = set_bits32(desc, self.m >> 7, 27, 2)
        desc = set_bits32(desc, self.a_scale_id, 29, 2)
        desc = set_bit32(desc, 31, self.k_dimension)
        return desc


@dataclass(frozen=True)
class Tcgen05SharedMemoryDescriptor:
    class LeadingDimensionMode(IntEnum):
        ByteOffsetRelative = 0
        ByteAddressAbsolute = 1

    matrix_start_address: int
    leading_dimension_offset: int
    stride_dimension_offset: int
    base_offset: int = 0
    leading_dimension_mode: LeadingDimensionMode = (
        LeadingDimensionMode.ByteOffsetRelative
    )
    swizzle_mode: SwizzleMode = SwizzleMode.SWIZZLE_NONE

    @stub
    def encode(self) -> int: ...


@stub
def tcgen05_mma(
    kind,
    matrix_d,
    matrix_a,
    matrix_b,
    instruction_descriptor,
    *,
    accumulate,
    cta_group=CTAGroup.CTA_1,
    sparse_metadata=None,
    scale_input_d=None,
    disable_output_lane=None,
    collector_op=Tcgen05MMACollectorOp.DISCARD,
    a_shift=False,
) -> None:
    """
    Perform the 5th generation of matrix multiply and accumulate operation.

    Args:
        kind (Tcgen05MMAKind): Data type the operation should be performed in.
        matrix_d (P6): Pointer in tensor memory to the destination and optional
            accumulator matrix D.
        matrix_a (P6 | int64): Matrix A encoded as either a 64-bit shared-memory
            descriptor or a pointer in tensor memory.
        matrix_b (int64): Matrix B encoded as a 64-bit shared-memory descriptor.
        instruction_descriptor (int32 | uint32): Encoded instruction descriptor.
        accumulate (bool): Whether input matrix D is included in the result.
        cta_group (CTAGroup): Controlls whether the operation takes place in
            one block or a pair of blocks.
        sparse_metadata (P6 | None): Optional pointer in tensor memory containing
            sparsity metadata for packed sparse matrix A. ``None`` selects dense
            MMA; presence selects sparse MMA and must be compile-time known.
        scale_input_d (int | None): Optional compile-time exponent in
            ``[0, 15]`` that scales input D by ``2**-scale_input_d``.
            Supported only for ``F16`` and ``TF32`` kinds.
        disable_output_lane (vector | None): Optional vector mask selecting
            tensor-memory lanes that must not be updated.
        collector_op (Tcgen05MMACollectorOp): Collector-buffer operation for
            matrix A.
        a_shift (bool): Shifts the rows of the A matrix down by one row and
            can only be applied if A is in tensor memory
    """


@stub
def tcgen05_mma_block_scale(
    kind,
    matrix_d,
    matrix_a,
    matrix_b,
    instruction_descriptor,
    scale_a,
    scale_b,
    *,
    accumulate,
    sparse_metadata=None,
    cta_group=CTAGroup.CTA_1,
    scale_vector_size=Tcgen05MMAScaleVectorSize.DEFAULT,
    collector_op=Tcgen05MMACollectorOp.DISCARD,
) -> None:
    """
    Performs block scaled MMA operation on 5th-generation tensor cores.

    Args:
        kind (Tcgen05MMABlockScaleKind): Data type the operation should be
            performed in.
        matrix_d (P6): Pointer in tensor memory to the destination and optional
            accumulator matrix D.
        matrix_a (P6 | int64): Matrix A encoded as either a 64-bit shared-memory
            descriptor or a pointer in tensor memory.
        matrix_b (int64): Matrix B encoded as a 64-bit shared-memory descriptor.
        instruction_descriptor (int32 | uint32): Encoded instruction descriptor.
        scale_a (P6): Pointer in tensor memory to matrix A scale factors.
        scale_b (P6): Pointer in tensor memory to matrix B scale factors.
        accumulate (bool): Whether input matrix D is included in the result.
        sparse_metadata (P6 | None): Optional pointer in tensor memory containing
            sparsity metadata for packed sparse matrix A. ``None`` selects dense
            MMA; presence selects sparse MMA and must be compile-time known.
        cta_group (CTAGroup): Controlls whether the operation takes place in
            one block or a pair of blocks.
        scale_vector_size (Tcgen05MMAScaleVectorSize): Scale-vector layout.
        collector_op (Tcgen05MMACollectorOp): Collector-buffer operation for
            matrix A.
    """


@stub
def tcgen05_mma_weight_stationary(
    kind,
    matrix_d,
    matrix_a,
    matrix_b,
    instruction_descriptor,
    *,
    accumulate,
    sparse_metadata=None,
    zero_column_mask=None,
    collector_op=Tcgen05MMACollectorOp.DISCARD,
    collector_b_buffer=Tcgen05MMACollectorBBuffer.BUFFER_0,
) -> None:
    """
    Perform the 5th generation of weight stationary convolution matrix
    multiply and accumulate operation.

    Args:
        kind (Tcgen05MMAKind): Data type the operation should be performed in.
        matrix_d (P6): Pointer in tensor memory to the destination and optional
            accumulator matrix D.
        matrix_a (P6 | int64): Matrix A encoded as either a 64-bit shared-memory
            descriptor or a pointer in tensor memory.
        matrix_b (int64): Matrix B encoded as a 64-bit shared-memory descriptor.
        instruction_descriptor (int32 | uint32): Encoded instruction descriptor.
        accumulate (bool): Whether input matrix D is included in the result.
        sparse_metadata (P6 | None): Optional pointer in tensor memory containing
            sparsity metadata for packed sparse matrix A. ``None`` selects dense
            MMA; presence selects sparse MMA and must be compile-time known.
        zero_column_mask (int64 | None): Optional integral scalar containing a 64-bit
            zero-column mask descriptor for matrix B.
        collector_op (Tcgen05MMACollectorOp): Collector-buffer operation for
            matrix A.
        collector_b_buffer (Tcgen05MMACollectorBBuffer):
    """


__all__ = (
    "CTAGroup",
    "Tcgen05MMAKind",
    "Tcgen05MMABlockScaleKind",
    "Tcgen05MMAScaleVectorSize",
    "Tcgen05MMACollectorBBuffer",
    "Tcgen05MMACollectorOp",
    "Tcgen05LoadStoreShape",
    "Tcgen05CopyMulticast",
    "Tcgen05CopyShape",
    "Tcgen05CopySourceFormat",
    "Tcgen05InstructionDescriptor",
    "Tcgen05Mxf8f6f4InstructionDescriptor",
    "Tcgen05Mxf4InstructionDescriptor",
    "Tcgen05SharedMemoryDescriptor",
    "tcgen05_allocate",
    "tcgen05_deallocate",
    "tcgen05_tmem_offset",
    "tcgen05_commit",
    "tcgen05_load",
    "tcgen05_copy",
    "tcgen05_store",
    "tcgen05_mma",
    "tcgen05_mma_block_scale",
    "tcgen05_mma_weight_stationary",
    "tcgen05_wait_load",
    "tcgen05_wait_store",
    "tcgen05_fence_before_thread_sync",
    "tcgen05_fence_after_thread_sync",
    "tcgen05_shift_down",
    "tcgen05_relinquish_allocation_permit",
)
