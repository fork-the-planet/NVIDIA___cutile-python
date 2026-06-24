# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import inspect

import cuda.lang as cl
import torch


def test_load_store_scalar_index():
    @cl.kernel
    def kernel(A, out):
        A.store_element(2, 7)
        A.store_element(0, 3)
        out[0] = A.load_element(0)
        out[1] = A.load_element(2)

    A = torch.zeros(4, dtype=torch.int32).cuda()
    out = torch.zeros(2, dtype=torch.int32, device="cuda")
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (A, out))
    assert A.cpu().tolist() == [3, 0, 7, 0]
    assert out.cpu().tolist() == [3, 7]


def test_load_store_tuple_index():
    @cl.kernel
    def kernel(A, out):
        A.store_element((0, 2), 7)
        A.store_element((1, 0), 3)
        out[0] = A.load_element((0, 2))
        out[1] = A.load_element((1, 0))

    A = torch.zeros(3, 3, dtype=torch.int32).cuda()
    out = torch.zeros(2, dtype=torch.int32, device="cuda")
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (A, out))
    assert A.cpu().tolist() == [[0, 0, 7], [3, 0, 0], [0, 0, 0]]
    assert out.cpu().tolist() == [7, 3]


def test_load_store_element_vector():
    @cl.kernel
    def kernel(inp, out):
        v = inp.load_element(0, count=4, alignment=16)
        out.store_element(0, v, alignment=16)

    inp = torch.tensor([1, 2, 3, 4], dtype=torch.int32).cuda()
    out = torch.zeros(4, dtype=torch.int32, device="cuda")
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (inp, out))
    assert out.cpu().tolist() == [1, 2, 3, 4]


def _keyword_only_params(fn):
    return {
        name: param.default
        for name, param in inspect.signature(fn).parameters.items()
        if param.kind is inspect.Parameter.KEYWORD_ONLY
    }


def test_element_accessors_forward_pointer_kwargs():
    assert _keyword_only_params(cl.Array.load_element) == _keyword_only_params(cl.Pointer.load)
    assert _keyword_only_params(cl.Array.store_element) == _keyword_only_params(cl.Pointer.store)
