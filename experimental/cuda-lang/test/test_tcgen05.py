# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from cuda.tile._exception import TileCompilerExecutionError
import pytest

import cuda.lang as cl
from cuda.lang._compile import KernelSignature, get_compute_capability
from cuda.lang._exception import TileTypeError, TileValueError
from test.util import make_symbolic_tensor, compile_kernel


cc = get_compute_capability()

if cc.major != 10:
    pytest.skip(reason="Blackwell only", allow_module_level=True)


@pytest.mark.parametrize(
    "mc_mask,cta_group,expect",
    [
        [
            0xAB,
            cl.CTAGroup.CTA_1,
            "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster"
            ".multicast::cluster.b64",
        ],
        [
            None,
            cl.CTAGroup.CTA_1,
            "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64",
        ],
        [
            None,
            cl.CTAGroup.CTA_2,
            "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.b64",
        ],
    ],
)
def test_commit(log_ptx, mc_mask, cta_group, expect):
    @cl.kernel
    def kernel():
        mbar = cl.shared_array(1, cl.mbarrier).get_base_pointer()
        cl.tcgen05_commit(mbar, multicast_mask=mc_mask, cta_group=cta_group)

    compile_kernel(kernel, assert_in_ptx=expect)


@pytest.mark.parametrize(
    "cta_group,expect",
    [
        [
            cl.CTAGroup.CTA_1,
            "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32",
        ],
        [
            cl.CTAGroup.CTA_2,
            "tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32",
        ],
    ],
)
def test_alloc(log_ptx, cta_group, expect):
    @cl.kernel
    def kernel():
        p3 = cl.shared_array(1, cl.uint32).get_base_pointer()
        cl.tcgen05_alloc(p3, 5, cta_group=cta_group)

    compiled = cl.compile_simt(kernel, [KernelSignature([])])
    ptx = compiled.ptx
    assert ptx is not None
    assert expect in ptx, ptx


def test_dealloc_requires_tensor_pointer():
    @cl.kernel
    def kernel():
        p3 = cl.shared_array(1, cl.uint32).get_base_pointer()
        cl.tcgen05_dealloc(p3, 5)

    with pytest.raises(
        TileTypeError,
        match="Expected pointer memory space to be MemorySpace.TENSOR "
        "but got MemorySpace.SHARED",
    ):
        cl.compile_simt(kernel, [KernelSignature([])])


@pytest.mark.parametrize(
    "cta_group,expect",
    [
        [
            cl.CTAGroup.CTA_1,
            "tcgen05.dealloc.cta_group::1.sync.aligned.b32",
        ],
        [
            cl.CTAGroup.CTA_2,
            "tcgen05.dealloc.cta_group::2.sync.aligned.b32",
        ],
    ],
)
def test_dealloc(log_ptx, cta_group, expect):
    @cl.kernel
    def kernel():
        tmem_dtype = cl.pointer_dtype(cl.int8, cl.MemorySpace.TENSOR)
        smem = cl.shared_array(1, tmem_dtype, alignment=4)
        cl.tcgen05_alloc(smem.get_base_pointer(), 128, cta_group=cta_group)
        tmem_ptr = smem[0]
        cl.tcgen05_dealloc(tmem_ptr, 128, cta_group=cta_group)

    compiled = cl.compile_simt(kernel, [KernelSignature([])])
    ptx = compiled.ptx
    assert ptx is not None
    assert expect in ptx, ptx


STORE_VALID_COUNTS_BY_SHAPE = {
    cl.Tcgen05LdStShape.SHAPE_16X64B: (1, 2, 4, 8, 16, 32, 64, 128),
    cl.Tcgen05LdStShape.SHAPE_16X128B: (1, 2, 4, 8, 16, 32, 64),
    cl.Tcgen05LdStShape.SHAPE_16X256B: (1, 2, 4, 8, 16, 32),
    cl.Tcgen05LdStShape.SHAPE_32X32B: (1, 2, 4, 8, 16, 32, 64, 128),
    cl.Tcgen05LdStShape.SHAPE_16X32BX2: (1, 2, 4, 8, 16, 32, 64, 128),
}


@pytest.mark.parametrize(
    "shape,count",
    [
        (shape, count)
        for shape, counts in STORE_VALID_COUNTS_BY_SHAPE.items()
        for count in counts
    ],
)
@pytest.mark.parametrize("unpack", (False, True))
def test_store(shape, count, unpack):
    offset = 1 if shape is cl.Tcgen05LdStShape.SHAPE_16X32BX2 else None
    registers_per_count = {
        cl.Tcgen05LdStShape.SHAPE_16X64B: 1,
        cl.Tcgen05LdStShape.SHAPE_16X128B: 2,
        cl.Tcgen05LdStShape.SHAPE_16X256B: 4,
        cl.Tcgen05LdStShape.SHAPE_32X32B: 1,
        cl.Tcgen05LdStShape.SHAPE_16X32BX2: 1,
    }[shape]
    register_count = count * registers_per_count

    def kernel():
        tmem_dtype = cl.pointer_dtype(cl.int8, cl.MemorySpace.TENSOR)
        smem = cl.shared_array(1, tmem_dtype, alignment=4)
        storage = cl.shared_array(256, cl.int32)
        v = storage.load_element(0, count=register_count)
        cl.tcgen05_alloc(smem.get_base_pointer(), 128)
        cl.tcgen05_store(shape, smem[0], v, unpack=unpack, offset=offset)
        cl.tcgen05_wait_store()
        cl.tcgen05_dealloc(smem[0], 128)

    expect = (
        f"tcgen05.st.sync.aligned.{shape.value}.x{count}"
        + (".unpack::16b" if unpack else "")
        + ".b32"
    )
    compile_kernel(kernel, assert_in_ptx=expect)


def test_store_rejects_wrong_value_dtype():
    @cl.kernel
    def kernel():
        tmem_dtype = cl.pointer_dtype(cl.int8, cl.MemorySpace.TENSOR)
        smem = cl.shared_array(1, tmem_dtype, alignment=4)
        cl.tcgen05_store(
            cl.Tcgen05LdStShape.SHAPE_16X64B,
            smem[0],
            cl.float32(0),
        )

    compile_kernel(
        kernel,
        raises=pytest.raises(TileTypeError, match="Expected scalar 32-bit integer"),
    )


def test_store_rejects_invalid_register_count():
    @cl.kernel
    def kernel():
        tmem_dtype = cl.pointer_dtype(cl.int8, cl.MemorySpace.TENSOR)
        smem = cl.shared_array(1, tmem_dtype, alignment=4)
        value = cl.Vector(cl.int32(0), cl.int32(0), cl.int32(0))

        # 16x128b requires 2 * count registers; three is invalid.
        cl.tcgen05_store(
            cl.Tcgen05LdStShape.SHAPE_16X128B,
            smem[0],
            value,
        )

    compile_kernel(
        kernel,
        raises=pytest.raises(TileValueError, match="Expected register count"),
    )


@pytest.mark.parametrize(
    "shape,offset",
    (
        (cl.Tcgen05LdStShape.SHAPE_16X32BX2, None),
        (cl.Tcgen05LdStShape.SHAPE_16X64B, 1),
    ),
)
def test_store_offset_validation(shape, offset):
    @cl.kernel
    def kernel():
        tmem_dtype = cl.pointer_dtype(cl.int8, cl.MemorySpace.TENSOR)
        smem = cl.shared_array(1, tmem_dtype, alignment=4)
        cl.tcgen05_store(shape, smem[0], cl.int32(0), offset=offset)

    compile_kernel(
        kernel,
        raises=pytest.raises(TileTypeError, match="offset"),
    )


@pytest.mark.parametrize("shape", (*tuple(cl.Tcgen05CopyShape), None))
@pytest.mark.parametrize("cta_group", (*tuple(cl.CTAGroup), None))
@pytest.mark.parametrize("multicast", (*tuple(cl.Tcgen05CopyMulticast), None, 5))
@pytest.mark.parametrize("source_format", (*tuple(cl.Tcgen05CopySourceFormat), None, 5))
def test_copy(shape, cta_group, multicast, source_format):
    allocation_group = (
        cta_group if isinstance(cta_group, cl.CTAGroup) else cl.CTAGroup.CTA_1
    )

    def kernel():
        tmem_dtype = cl.pointer_dtype(cl.int8, cl.MemorySpace.TENSOR)
        smem = cl.shared_array(1, tmem_dtype, alignment=4)
        cl.tcgen05_alloc(smem.get_base_pointer(), 128, cta_group=allocation_group)
        tmem_ptr = smem[0]
        descriptor = cl.int64(0xDEADBEEF)
        cl.tcgen05_copy(
            tmem_ptr,
            descriptor,
            cta_group=cta_group,
            shape=shape,
            multicast=multicast,
            source_format=source_format,
        )
        cl.tcgen05_dealloc(tmem_ptr, 128, cta_group=allocation_group)

    valid_multicasts = {
        cl.Tcgen05CopyShape.SHAPE_128x256b: (None,),
        cl.Tcgen05CopyShape.SHAPE_4x256b: (None,),
        cl.Tcgen05CopyShape.SHAPE_128x128b: (None,),
        cl.Tcgen05CopyShape.SHAPE_64x128b: (
            cl.Tcgen05CopyMulticast.WARPX2_02_13,
            cl.Tcgen05CopyMulticast.WARPX2_01_23,
        ),
        cl.Tcgen05CopyShape.SHAPE_32x128b: (cl.Tcgen05CopyMulticast.WARPX4,),
    }

    expect = None
    raises = None
    if cta_group not in tuple(cl.CTAGroup):
        raises = pytest.raises(TileTypeError, match="Expected CTAGroup")
    elif shape not in tuple(cl.Tcgen05CopyShape):
        raises = pytest.raises(TileTypeError, match="Expected Tcgen05CopyShape")
    elif multicast not in (*tuple(cl.Tcgen05CopyMulticast), None):
        raises = pytest.raises(TileTypeError, match="Expected Tcgen05CopyMulticast")
    elif source_format not in (*tuple(cl.Tcgen05CopySourceFormat), None):
        raises = pytest.raises(TileTypeError, match="Expected Tcgen05CopySourceFormat")
    elif multicast not in valid_multicasts[shape]:
        raises = pytest.raises(TileCompilerExecutionError)
    else:
        shape_str = shape.name.removeprefix("SHAPE_")
        group_str = "cta_group::" + str(1 if cta_group is cl.CTAGroup.CTA_1 else 2)
        multicast_str = {
            None: "",
            cl.Tcgen05CopyMulticast.WARPX2_02_13: ".warpx2::02_13",
            cl.Tcgen05CopyMulticast.WARPX2_01_23: ".warpx2::01_23",
            cl.Tcgen05CopyMulticast.WARPX4: ".warpx4",
        }[multicast]
        source_format_str = {
            None: "",
            cl.Tcgen05CopySourceFormat.B6x16_P32: ".b8x16.b6x16_p32",
            cl.Tcgen05CopySourceFormat.B4x16_P64: ".b8x16.b4x16_p64",
        }[source_format]
        expect = (
            f"tcgen05.cp.{group_str}.{shape_str}" + multicast_str + source_format_str
        )

    compile_kernel(kernel, assert_in_ptx=expect, raises=raises)


@pytest.mark.parametrize("shape", tuple(cl.Tcgen05LdStShape))
@pytest.mark.parametrize("count", (1, 2, 4, 8, 16, 32, 64, 128))
@pytest.mark.parametrize("pack", (True, False, None))
@pytest.mark.parametrize("offset", (None, 0, 1))
def test_load(log_ptx, shape, count, pack, offset):
    @cl.kernel
    def kernel():
        tmem_dtype = cl.pointer_dtype(cl.int8, cl.MemorySpace.TENSOR)
        smem = cl.shared_array(1, tmem_dtype, alignment=4)
        cl.tcgen05_alloc(smem.get_base_pointer(), 128)
        tmem_ptr = smem[0]
        cl.tcgen05_load(shape, tmem_ptr, count=count, pack=pack, offset=offset)
        cl.tcgen05_dealloc(tmem_ptr, 128)

    def do_compile():
        compiled = cl.compile_simt(kernel, [KernelSignature([])])
        ptx = compiled.ptx
        assert ptx is not None
        assert "tcgen05.ld.sync.aligned" in ptx and shape.value in ptx, ptx

    bad_args = offset is not None and shape is not cl.Tcgen05LdStShape.SHAPE_16X32BX2
    bad_args |= shape is cl.Tcgen05LdStShape.SHAPE_16X256B and count not in (
        1,
        2,
        4,
        8,
        16,
        32,
    )
    bad_args |= shape is cl.Tcgen05LdStShape.SHAPE_16X32BX2 and offset is None
    bad_args |= shape is cl.Tcgen05LdStShape.SHAPE_16X128B and count not in (
        1,
        2,
        4,
        8,
        16,
        32,
        64,
    )
    if bad_args:
        with pytest.raises((TileTypeError, TileValueError)):
            do_compile()
    else:
        do_compile()


@pytest.mark.parametrize("kind", cl.Tcgen05MMAKind)
@pytest.mark.parametrize("cta_group", cl.CTAGroup)
@pytest.mark.parametrize("collector_op", cl.Tcgen05MMACollectorOp)
def test_mma_valid_enum_combinations(kind, cta_group, collector_op):
    if kind in (
        cl.Tcgen05MMAKind.I8,
        cl.Tcgen05MMAKind.MXF8F6F4,
        cl.Tcgen05MMAKind.MXF4,
        cl.Tcgen05MMAKind.MXF4NVF4,
    ):
        pytest.xfail("needs updated mlir bindings")

    @cl.kernel
    def kernel():
        tmem_dtype = cl.pointer_dtype(cl.int8, cl.MemorySpace.TENSOR)
        tmem_smem = cl.shared_array(1, tmem_dtype, alignment=4)
        cl.tcgen05_mma(
            kind,
            cta_group,
            tmem_smem[0],
            cl.int64(0),
            cl.int64(0),
            cl.int32(0),
            False,
            collector_op=collector_op,
        )

    compiled = cl.compile_simt(kernel, [KernelSignature([])], log_ptx=True)
    ptx = compiled.ptx
    assert ptx is not None
    assert "tcgen05.mma" in ptx, ptx


@pytest.mark.parametrize("cta_group", cl.CTAGroup._member_map_.values())
@pytest.mark.parametrize("scale_input_d", (None, 0, 15))
@pytest.mark.parametrize("disable_output_lane", (False, True))
def test_mma_optional_operands(cta_group, scale_input_d, disable_output_lane):

    @cl.kernel
    def kernel():
        tmem_dtype = cl.pointer_dtype(cl.int8, cl.MemorySpace.TENSOR)
        tmem_smem = cl.shared_array(1, tmem_dtype, alignment=4)
        if disable_output_lane:
            if cta_group == cl.CTAGroup.CTA_1:
                disable_output_lane_value = cl.Vector(
                    cl.int32(0), cl.int32(0), cl.int32(0), cl.int32(0)
                )
            else:
                disable_output_lane_value = cl.Vector(
                    cl.int32(0),
                    cl.int32(0),
                    cl.int32(0),
                    cl.int32(0),
                    cl.int32(0),
                    cl.int32(0),
                    cl.int32(0),
                    cl.int32(0),
                )
        else:
            disable_output_lane_value = None

        cl.tcgen05_mma(
            cl.Tcgen05MMAKind.F16,
            cta_group,
            tmem_smem[0],
            cl.int64(0),
            cl.int64(0),
            cl.int32(0),
            False,
            collector_op=cl.Tcgen05MMACollectorOp.DISCARD,
            disable_output_lane=disable_output_lane_value,
            scale_input_d=cl.int64(scale_input_d)
            if scale_input_d is not None
            else None,
        )

    compiled = cl.compile_simt(kernel, [KernelSignature([])], log_ptx=True)
    ptx = compiled.ptx
    assert ptx is not None
    assert "tcgen05.mma" in ptx, ptx


def test_mma_matrix_a_validation():
    @cl.kernel
    def kernel():
        tmem_dtype = cl.pointer_dtype(cl.int8, cl.MemorySpace.TENSOR)
        tmem_smem = cl.shared_array(1, tmem_dtype, alignment=4)
        cl.tcgen05_mma(
            cl.Tcgen05MMAKind.F16,
            cl.CTAGroup.CTA_1,
            tmem_smem[0],
            cl.int32(0),  # wrong type!
            cl.int64(0),
            cl.int32(0),
            False,
        )

    match = (
        "Expected a tensor memory pointer or a shared memory descriptor "
        "encoded as a 64 bit integer but got int32"
    )
    with pytest.raises(TileTypeError, match=match):
        cl.compile_simt(kernel, [KernelSignature([])], log_ptx=True)


@pytest.mark.parametrize(
    "op,expect",
    (
        (cl.tcgen05_wait_load, "tcgen05.wait::ld.sync.aligned"),
        (cl.tcgen05_wait_store, "tcgen05.wait::st.sync.aligned"),
    ),
)
def test_wait(op, expect):
    def kernel():
        op()

    compile_kernel(kernel, assert_in_ptx=expect)


@pytest.mark.parametrize(
    "op,expect",
    (
        (
            cl.tcgen05_fence_before_thread_sync,
            "tcgen05.fence::before_thread_sync",
        ),
        (
            cl.tcgen05_fence_after_thread_sync,
            "tcgen05.fence::after_thread_sync",
        ),
    ),
)
def test_fence(op, expect):
    def kernel():
        op()

    compile_kernel(kernel, assert_in_ptx=expect)


@pytest.mark.parametrize(
    "group,expect",
    (
        (
            cl.CTAGroup.CTA_1,
            "tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned",
        ),
        (
            cl.CTAGroup.CTA_2,
            "tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned",
        ),
    ),
)
def test_relinquish(group, expect):
    @cl.kernel
    def kernel():
        cl.tcgen05_relinquish_allocation_permit(group)

    compiled = cl.compile_simt(kernel, [KernelSignature([])], log_ptx=True)
    assert expect in compiled.ptx, compiled.ptx


def test_relinquish_bad_group():
    @cl.kernel
    def kernel():
        cl.tcgen05_relinquish_allocation_permit(0xDEADBEEF)

    with pytest.raises(Exception):
        cl.compile_simt(kernel, [KernelSignature([])], log_ptx=True)


@pytest.mark.parametrize(
    "group,expect",
    (
        (cl.CTAGroup.CTA_1, "tcgen05.shift.cta_group::1.down"),
        (cl.CTAGroup.CTA_2, "tcgen05.shift.cta_group::2.down"),
    ),
)
def test_shift(group, expect):
    @cl.kernel
    def kernel():
        tmem_dtype = cl.pointer_dtype(cl.int8, cl.MemorySpace.TENSOR)
        tmem_smem = cl.shared_array(1, tmem_dtype, alignment=4)
        cl.tcgen05_shift_down(tmem_smem[0], group)

    compiled = cl.compile_simt(kernel, [KernelSignature([])], log_ptx=True)
    assert expect in compiled.ptx, compiled.ptx


def test_shift_bad_group():
    @cl.kernel
    def kernel():
        tmem_dtype = cl.pointer_dtype(cl.int8, cl.MemorySpace.TENSOR)
        tmem_smem = cl.shared_array(1, tmem_dtype, alignment=4)
        cl.tcgen05_shift_down(tmem_smem[0], 0xDEADBEEF)

    with pytest.raises(Exception):
        cl.compile_simt(kernel, [KernelSignature([])], log_ptx=True)


def test_shift_bad_address_space(subtests):
    with subtests.test("shared"):

        @cl.kernel
        def kernel():
            ptr = cl.shared_array(1, cl.int8).get_base_pointer()
            cl.tcgen05_shift_down(ptr, 0xDEADBEEF)

        with pytest.raises(Exception):
            cl.compile_simt(kernel, [KernelSignature([])], log_ptx=True)

    with subtests.test("local"):

        @cl.kernel
        def kernel():
            with cl.local_array(1, cl.int8) as arr:
                ptr = arr.get_base_pointer()
                cl.tcgen05_shift_down(ptr, 0xDEADBEEF)

        with pytest.raises(Exception):
            cl.compile_simt(kernel, [KernelSignature([])], log_ptx=True)

    with subtests.test("global"):

        @cl.kernel
        def kernel(arr):
            ptr = arr.get_base_pointer()
            cl.tcgen05_shift_down(ptr, 0xDEADBEEF)

        with pytest.raises(Exception):
            cl.compile_simt(
                kernel,
                [KernelSignature([make_symbolic_tensor(1, cl.int8)])],
                log_ptx=True,
            )
