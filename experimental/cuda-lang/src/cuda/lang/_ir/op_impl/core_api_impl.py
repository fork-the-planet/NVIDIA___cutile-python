# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from cuda.lang._ir.op_defs import RawNVVMIntrinsic
from cuda.lang._ir.type import ScalarTy
from cuda.lang._stub import core_api
from cuda.lang._exception import TypeCheckingError
from cuda.tile._datatype import int32
from cuda.tile._ir.ir import Var, add_operation
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
