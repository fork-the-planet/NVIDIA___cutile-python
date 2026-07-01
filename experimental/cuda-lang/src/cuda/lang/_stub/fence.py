# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from typing import Literal

from cuda.lang._execution import function
from .._enums import FenceProxyKind, MemoryOrder, MemoryScope, MemorySpace
from . import nvvm_mlir_interfaces as _mlir


@function()
def fence_sc_cluster() -> None:
    _mlir.fence_sc_cluster()


@function()
def fence_mbarrier_initialize() -> None:
    _mlir.fence_mbarrier_init()


@function()
def fence_sync_restrict(
    order: Literal[MemoryOrder.ACQUIRE, MemoryOrder.RELEASE],
) -> None:
    """
    Uni-directional proxy fence operation with sync_restrict.

    Args:
        order: MemoryOrder.ACQUIRE or MemoryOrder.RELEASE
    """
    _mlir.fence_sync_restrict(order=order)


@function()
def fence_proxy(
    kind: FenceProxyKind,
    *,
    space: MemorySpace | None = None,
) -> None:
    """
    Fence operation with proxy to establish an ordering between memory accesses
    that may happen through different proxies.

    Args:
        kind (FenceProxyKind):
        space (MemorySpace):
    """
    _mlir.fence_proxy(kind=kind, space=space)


@function()
def fence_proxy_acquire(
    address,
    size: int,
    *,
    scope: MemoryScope,
    from_proxy: FenceProxyKind = FenceProxyKind.GENERIC,
    to_proxy: FenceProxyKind = FenceProxyKind.TENSORMAP,
) -> None:
    """
    Uni-directional proxy fence operation with acquire semantics.

    Args:
        address (pointer):
        size (int):
        scope (MemoryScope):
        from_proxy (FenceProxyKind):
        to_proxy (FenceProxyKind):
    """
    _mlir.fence_proxy_acquire(
        addr=address,
        size=size,
        scope=scope,
        from_proxy=from_proxy,
        to_proxy=to_proxy,
    )


@function()
def fence_proxy_release(
    *,
    scope: MemoryScope,
    from_proxy: FenceProxyKind = FenceProxyKind.GENERIC,
    to_proxy: FenceProxyKind = FenceProxyKind.TENSORMAP,
) -> None:
    """
    Uni-directional proxy fence operation with release semantics.

    Args:
        scope (MemoryScope):
        from_proxy (FenceProxyKind):
        to_proxy (FenceProxyKind):
    """
    _mlir.fence_proxy_release(
        scope=scope,
        from_proxy=from_proxy,
        to_proxy=to_proxy,
    )


@function()
def fence_proxy_sync_restrict(
    order: Literal[MemoryOrder.ACQUIRE, MemoryOrder.RELEASE],
    *,
    from_proxy: FenceProxyKind = FenceProxyKind.GENERIC,
    to_proxy: FenceProxyKind = FenceProxyKind.ASYNC,
) -> None:
    """
    Uni-directional proxy fence operation with sync_restrict.

    Args:
        order: MemoryOrder.ACQUIRE or MemoryOrder.RELEASE
        from_proxy (FenceProxyKind):
        to_proxy (FenceProxyKind):
    """
    _mlir.fence_proxy_sync_restrict(
        order=order,
        from_proxy=from_proxy,
        to_proxy=to_proxy,
    )


__all__ = (
    "FenceProxyKind",
    "fence_sync_restrict",
    "fence_sc_cluster",
    "fence_mbarrier_initialize",
    "fence_proxy_sync_restrict",
    "fence_proxy",
    "fence_proxy_acquire",
    "fence_proxy_release",
)
