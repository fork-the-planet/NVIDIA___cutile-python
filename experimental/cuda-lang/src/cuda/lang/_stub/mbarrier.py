# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from typing import Literal

from cuda.lang._enums import MbarrierScope
from cuda.lang._execution import stub
from cuda.lang._datatype import uint64, bool_
from cuda.tile._memory_model import MemoryOrder


ArriveOrdering = Literal[MemoryOrder.RELAXED, MemoryOrder.RELEASE]
WaitOrdering = Literal[MemoryOrder.RELAXED, MemoryOrder.ACQUIRE]


@stub
def mbarrier_init(mbar, participants: int) -> None:
    ...


@stub
def mbarrier_invalidate(mbar) -> None:
    ...


@stub
def mbarrier_arrive(
    mbar,
    count: int = 1,
    *,
    drop: bool = False,
    scope: MbarrierScope = MbarrierScope.BLOCK,
    ordering: ArriveOrdering = MemoryOrder.RELEASE,
) -> "uint64 | None":
    """Arrive at ``mbar``. When the mbarrier resides in ``MemorySpace.SHARED``,
    an opaque 64-bit value capturing the phase of the mbarrier object _prior_
    to this arrive operation is returned. ``drop=True`` drops a participant
    from the barrier.
    """


@stub
def mbarrier_arrive_expect_tx(
    mbar,
    bytes: int,
    *,
    drop: bool = False,
    scope: MbarrierScope = MbarrierScope.BLOCK,
    ordering: ArriveOrdering = MemoryOrder.RELEASE,
) -> "uint64 | None":
    ...


@stub
def mbarrier_expect_tx(
    mbar,
    bytes: int,
    *,
    scope: MbarrierScope = MbarrierScope.BLOCK,
) -> None:
    ...


@stub
def mbarrier_complete_tx(
    mbar,
    bytes: int,
    *,
    scope: MbarrierScope = MbarrierScope.BLOCK,
) -> None:
    ...


@stub
def mbarrier_test_wait(
    mbar,
    state,
    *,
    scope: MbarrierScope = MbarrierScope.BLOCK,
    ordering: WaitOrdering = MemoryOrder.ACQUIRE,
) -> "bool_":
    """Non-blocking test whether ``mbar`` has completed."""


@stub
def mbarrier_test_wait_parity(
    mbar,
    parity: int,
    *,
    scope: MbarrierScope = MbarrierScope.BLOCK,
    ordering: WaitOrdering = MemoryOrder.ACQUIRE,
) -> "bool_":
    """Phase-parity variant of ``mbarrier_test_wait``.
    ``parity`` is the 0/1 integer parity of the phase to test for.
    """


@stub
def mbarrier_try_wait(
    mbar,
    state,
    *,
    time_hint: int | None = None,
    scope: MbarrierScope = MbarrierScope.BLOCK,
    ordering: WaitOrdering = MemoryOrder.ACQUIRE,
) -> "bool_":
    """Bounded-wait test whether ``mbar`` has completed."""


@stub
def mbarrier_try_wait_parity(
    mbar,
    parity: int,
    *,
    time_hint: int | None = None,
    scope: MbarrierScope = MbarrierScope.BLOCK,
    ordering: WaitOrdering = MemoryOrder.ACQUIRE,
) -> "bool_":
    """Phase-parity variant of ``mbarrier_try_wait``.
    ``parity`` is the 0/1 integer parity of the phase to test for.
    """


__all__ = (
    "mbarrier_init",
    "mbarrier_invalidate",
    "mbarrier_arrive",
    "mbarrier_arrive_expect_tx",
    "mbarrier_expect_tx",
    "mbarrier_complete_tx",
    "mbarrier_test_wait",
    "mbarrier_test_wait_parity",
    "mbarrier_try_wait",
    "mbarrier_try_wait_parity",
)
