# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import cuda.lang as cl
from cuda.lang._exception import InvalidValueError, TypeCheckingError
import pytest
from test.util import compile_kernel


VALID_SECONDARY_POLICIES = (
    cl.CachePolicy.L2_EVICT_FIRST,
    cl.CachePolicy.L2_EVICT_UNCHANGED,
)
BAD_SECONDARY_POLICY_RAISES = pytest.raises(
    InvalidValueError,
    match=(
        "Secondary cache policy may only be "
        "CachePolicy.L2_EVICT_FIRST or CachePolicy.L2_EVICT_UNCHANGED"
    ),
)


@pytest.mark.parametrize("primary", (*tuple(cl.CachePolicy), False, 1))
@pytest.mark.parametrize("fraction", (0.0, 1.0, 2.0, -1.0, 1, True))
@pytest.mark.parametrize("secondary", (*tuple(cl.CachePolicy), False, 1))
def test_fractional_cache_policy(primary, fraction, secondary):
    def kernel():
        cl.create_fractional_cache_policy(
            primary,
            fraction=fraction,
            secondary_policy=secondary,
        )

    if (
        not isinstance(secondary, cl.CachePolicy)
        or not isinstance(primary, cl.CachePolicy)
        or not isinstance(fraction, float)
    ):
        compile_kernel(
            kernel,
            raises=pytest.raises(TypeCheckingError),
        )
    elif secondary not in VALID_SECONDARY_POLICIES:
        compile_kernel(
            kernel,
            raises=BAD_SECONDARY_POLICY_RAISES,
        )
    else:
        compile_kernel(
            kernel,
            assert_in_ptx=f"createpolicy.fractional.{primary.value}.{secondary.value}",
        )


@pytest.mark.parametrize("primary", (*tuple(cl.CachePolicy), False, 1))
@pytest.mark.parametrize("secondary", (*tuple(cl.CachePolicy), False, 1))
@pytest.mark.parametrize("base_size", (10, 10.0, True))
@pytest.mark.parametrize("total_size", (20, 10.0, True))
def test_range_cache_policy(primary, secondary, base_size, total_size):
    def kernel():
        base = cl.shared_array(1, cl.int32).get_base_pointer()
        cl.create_range_cache_policy(
            base,
            base_size,
            total_size,
            primary,
            secondary_policy=secondary,
        )

    if (
        not isinstance(secondary, cl.CachePolicy)
        or not isinstance(primary, cl.CachePolicy)
        or not isinstance(base_size, int)
        or not isinstance(total_size, int)
        or isinstance(base_size, bool)
        or isinstance(total_size, bool)
    ):
        compile_kernel(
            kernel,
            raises=pytest.raises(TypeCheckingError),
        )
    elif secondary not in VALID_SECONDARY_POLICIES:
        compile_kernel(
            kernel,
            raises=BAD_SECONDARY_POLICY_RAISES,
        )
    else:
        compile_kernel(
            kernel,
            assert_in_ptx=f"createpolicy.range.{primary.value}.{secondary.value}",
        )
