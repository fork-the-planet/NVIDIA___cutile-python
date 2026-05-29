# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import torch.cuda

import cuda.lang as cl
from cuda.tile import static_assert


def test_dtype_of():
    @cl.kernel
    def kern(x: cl.Array):
        ptr = x.get_base_pointer()
        static_assert(cl.dtype_of(ptr) == cl.pointer_dtype(cl.int32))
        ptr_dtype = cl.dtype_of(ptr)
        static_assert(ptr_dtype == cl.pointer_dtype(cl.int32))

        val = ptr.load()
        static_assert(cl.dtype_of(val) == cl.int32)
        val_dtype = cl.dtype_of(val)
        static_assert(val_dtype == cl.int32)

        static_assert(cl.dtype_of(123) == cl.int32)
        int_dtype = cl.dtype_of(123)
        static_assert(int_dtype == cl.int32)

        static_assert(cl.dtype_of(1.5) == cl.float32)
        float_dtype = cl.dtype_of(1.5)
        static_assert(float_dtype == cl.float32)

    x = torch.zeros(10, dtype=torch.int32, device="cuda")
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kern, (x,))
