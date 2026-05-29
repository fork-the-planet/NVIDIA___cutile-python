# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import cuda.lang as cl
import torch


@pytest.mark.parametrize(
    "dtype",
    [torch.int32, torch.int64, torch.float32, torch.float64],
)
def test_print(dtype):

    @cl.kernel
    def kernel(A):
        print(A[0])

    A = torch.tensor([5], dtype=dtype).cuda()
    cl.launch(
        torch.cuda.current_stream(),
        (1,),
        (1,),
        kernel,
        (A,),
    )
