# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import cuda.lang as cl
from cuda.lang._exception import TileTypeError
import cuda.lang._stub.nvvm_mlir_interfaces as nvvm
import torch
from ..util import require_hopper_or_newer


@require_hopper_or_newer()
def test_mlir_interface_enums():
    @cl.kernel
    def kernel(tensor):
        nvvm.fence_proxy_acquire(
            scope=nvvm.MemScopeKind.CTA,
            addr=tensor.get_base_pointer(),
            size=128,
            from_proxy=nvvm.ProxyKind.GENERIC,
            to_proxy=nvvm.ProxyKind.TENSORMAP,
        )

    z = torch.zeros(1, dtype=torch.int32).cuda()
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (z,))


def test_mlir_interface_results():
    @cl.kernel
    def kernel():
        permuted = nvvm.prmt(lo=12, hi=16, selector=7, mode=nvvm.PermuteMode.F4E)
        print(permuted)

    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, ())


def test_mlir_interface_error_on_non_constant_enum():
    @cl.kernel
    def kernel(cond):
        if cond:
            dyn_kind = nvvm.ProxyKind.GENERIC
        else:
            dyn_kind = nvvm.ProxyKind.TENSORMAP
        nvvm.fence_proxy(kind=dyn_kind)

    with pytest.raises(
        TileTypeError,
        match="Expected ProxyKind constant, but given value is not constant",
    ):
        cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (False,))


def test_mlir_interface_error_on_non_constant_attr():
    # cluster arrive takes a bool attr, so the value must be known at compile-time
    @cl.kernel
    def kernel(cond):
        nvvm.cluster_arrive(aligned=cond)

    with pytest.raises(
        TileTypeError,
        match="Expected a boolean constant, but given value is not constant",
    ):
        cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (False,))
