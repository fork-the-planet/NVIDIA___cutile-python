# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# Building a RawMLIROperation is tricky because one must track the operand
# segment sizes for optional or variadic operands and convert enums to
# attributes. This module simplifies the process.

from functools import singledispatch, partial

import cuda.lang._enums as enums
from cuda.lang._exception import TileInternalError, TileValueError
import cuda.lang._mlir as mlir


@singledispatch
def cl_enum_to_mlir_attribute(enum_value):
    raise NotImplementedError(
        f"Enum of type {type(enum_value)} does not have a registered "
        "function to build an MLIR attribute from it"
    )


def enum_to_mlir_nvvm_attribute(cl_enum_value, mlir_enum):
    mlir_attribute_class = mlir_enum.__name__ + "Attr"
    mlir_attribute = getattr(mlir.nvvm, mlir_attribute_class, None)
    if mlir_attribute is None:
        raise TileInternalError(
            f"Expected mlir module to have class {mlir_attribute_class} "
            "but it could not be found"
        )
    mlir_enum_value = getattr(mlir_enum, cl_enum_value.name, None)
    if mlir_enum_value is None:
        raise TileInternalError(
            f"Expected enum {type(cl_enum_value)} to have corresponding "
            "enum in mlir bindings but it could not be found"
        )
    return mlir_attribute(value=mlir_enum_value)


def invalid_enum_member(enum_value, value_map):
    valid = ", ".join(str(value) for value in value_map)
    return TileValueError(f"Expected one of {valid}, got {enum_value}")


cl_enum_to_mlir_attribute.register(
    enums.CTAGroup,
    partial(enum_to_mlir_nvvm_attribute, mlir_enum=mlir.nvvm.CTAGroupKind),
)
cl_enum_to_mlir_attribute.register(
    enums.MemoryOrder,
    partial(enum_to_mlir_nvvm_attribute, mlir_enum=mlir.nvvm.MemOrderKind),
)
cl_enum_to_mlir_attribute.register(
    enums.TMALoadMode,
    partial(enum_to_mlir_nvvm_attribute, mlir_enum=mlir.nvvm.TMALoadMode),
)
cl_enum_to_mlir_attribute.register(
    enums.TMAStoreMode,
    partial(enum_to_mlir_nvvm_attribute, mlir_enum=mlir.nvvm.TMAStoreMode),
)
cl_enum_to_mlir_attribute.register(
    enums.Tcgen05CopyMulticast,
    partial(enum_to_mlir_nvvm_attribute, mlir_enum=mlir.nvvm.Tcgen05CpMulticast),
)
cl_enum_to_mlir_attribute.register(
    enums.Tcgen05CopyShape,
    partial(enum_to_mlir_nvvm_attribute, mlir_enum=mlir.nvvm.Tcgen05CpShape),
)
cl_enum_to_mlir_attribute.register(
    enums.Tcgen05CopySourceFormat,
    partial(enum_to_mlir_nvvm_attribute, mlir_enum=mlir.nvvm.Tcgen05CpSrcFormat),
)
cl_enum_to_mlir_attribute.register(
    enums.Tcgen05MMAKind,
    partial(enum_to_mlir_nvvm_attribute, mlir_enum=mlir.nvvm.Tcgen05MMAKind),
)
cl_enum_to_mlir_attribute.register(
    enums.Tcgen05MMACollectorOp,
    partial(enum_to_mlir_nvvm_attribute, mlir_enum=mlir.nvvm.Tcgen05MMACollectorOp),
)
cl_enum_to_mlir_attribute.register(
    enums.Tcgen05LdStShape,
    partial(enum_to_mlir_nvvm_attribute, mlir_enum=mlir.nvvm.Tcgen05LdStShape),
)
cl_enum_to_mlir_attribute.register(
    enums.Tcgen05WaitKind,
    partial(enum_to_mlir_nvvm_attribute, mlir_enum=mlir.nvvm.Tcgen05WaitKind),
)


@cl_enum_to_mlir_attribute.register(enums.MemorySpace)
def memory_space_to_mlir_attribute(enum_value):
    value_map = {
        enums.MemorySpace.SHARED: mlir.nvvm.SharedSpace.shared_cta,
        enums.MemorySpace.SHARED_CLUSTER: mlir.nvvm.SharedSpace.shared_cluster,
    }
    if mlir_enum_value := value_map.get(enum_value):
        return mlir.nvvm.SharedSpaceAttr(value=mlir_enum_value)
    raise invalid_enum_member(enum_value, value_map)


@cl_enum_to_mlir_attribute.register(enums.MemoryScope)
def memory_scope_to_mlir_attribute(enum_value):
    value_map = {
        enums.MemoryScope.BLOCK: mlir.nvvm.MemScopeKind.CTA,
        enums.MemoryScope.CLUSTER: mlir.nvvm.MemScopeKind.CLUSTER,
        enums.MemoryScope.DEVICE: mlir.nvvm.MemScopeKind.GPU,
        enums.MemoryScope.SYS: mlir.nvvm.MemScopeKind.SYS,
    }
    if mlir_enum_value := value_map.get(enum_value):
        return mlir.nvvm.MemScopeKindAttr(value=mlir_enum_value)
    raise invalid_enum_member(enum_value, value_map)


@cl_enum_to_mlir_attribute.register(enums.FenceProxyKind)
def fence_proxy_kind_to_mlir_attribute(enum_value):
    value_map = {
        enums.FenceProxyKind.ALIAS: mlir.nvvm.ProxyKind.alias,
        enums.FenceProxyKind.ASYNC: mlir.nvvm.ProxyKind.async_,
        enums.FenceProxyKind.ASYNC_GLOBAL: mlir.nvvm.ProxyKind.async_global,
        enums.FenceProxyKind.ASYNC_SHARED: mlir.nvvm.ProxyKind.async_shared,
        enums.FenceProxyKind.TENSORMAP: mlir.nvvm.ProxyKind.TENSORMAP,
        enums.FenceProxyKind.GENERIC: mlir.nvvm.ProxyKind.GENERIC,
    }
    if mlir_enum_value := value_map.get(enum_value):
        return mlir.nvvm.ProxyKindAttr(value=mlir_enum_value)
    raise invalid_enum_member(enum_value, value_map)


__all__ = ("cl_enum_to_mlir_attribute",)
