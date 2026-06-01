# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import torch
import cuda.tile as ct
from cuda.tile._cext import dev_features_enabled
from cuda.tile._compile import _get_max_supported_bytecode_version
from functools import cache
import tempfile


@cache
def get_tileiras_version():
    return _get_max_supported_bytecode_version(tempfile.gettempdir(),
                                               allow_dev=dev_features_enabled())


def swizzle_2d_from_bid(M, N, tm, tn, GROUP_SIZE_M, bid):
    # Get the global IDs of a given block in a 1D grid.
    num_bid_m = ct.cdiv(M, tm)
    num_bid_n = ct.cdiv(N, tn)
    num_bid_in_group = GROUP_SIZE_M * num_bid_n
    group_id = bid // num_bid_in_group
    first_bid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_bid_m - first_bid_m, GROUP_SIZE_M)
    bid_m = first_bid_m + (bid % group_size_m)
    bid_n = (bid % num_bid_in_group) // group_size_m
    return bid_m, bid_n


def swizzle_2d(M, N, tm, tn, GROUP_SIZE_M):
    # Get the global IDs of the current block in a 1D grid.
    bid = ct.bid(0)
    return swizzle_2d_from_bid(M, N, tm, tn, GROUP_SIZE_M, bid)


def block_quantize(x: torch.Tensor, block_size: int, dtype: torch.dtype = torch.float8_e4m3fn):

    """
    Args:
        x (torch.Tensor):       input tensor.
        block_size (int):       size of block.
        dtype (torch.dtype):    the torch datatype to which the tensor will be converted.

    Returns:
        tuple[torch.Tensor, torch.Tensor, int]: A tuple containing:
            - x (torch.Tensor):         in dtype.
            - scale (torch.Tensor):     in torch.float8_e8m0fnu.

    Raises:
        ValueError: If the requested block size is larger than the inner most
                    dimension of the input tensor.
    """

    dtype_max = torch.finfo(torch.float8_e4m3fn).max

    if x.shape[-1] < block_size:
        raise ValueError(f'The requested block size {block_size} is larger than the inner most '
                         f'dimension of the input tensor {x.shape[-1]}')

    if x.shape[-1] % block_size != 0:
        print('[WARNING] block size is not a multiple of the inner most '
              'dimension of the input tensor')
        print('          padding the input tensor...')
        pad_len = (block_size - (x.shape[-1] % block_size)) % block_size
        x = torch.nn.functional.pad(x, (0, pad_len), value=0)

    x_block = x.reshape(*x.shape[:-1], x.shape[-1] // block_size, block_size)

    scale = torch.max(x_block.abs(), dim=-1, keepdims=True)[0]
    scale = torch.clamp(scale / dtype_max, min=1e-12)
    scale = torch.pow(2.0, torch.ceil(torch.log2(scale)))

    x_q = x_block / scale

    x_q = x_q.to(dtype).reshape(x.shape)
    scale = scale.to(torch.float8_e8m0fnu).squeeze(-1)

    return x_q, scale


def swizzle_32_4_4(scale):
    '''
    Prepare the original scale tensor to align with the expected tmem layout.
    With the innermost dimensions being (m1=32, m2=4, k1=4), and the outer dimensions
    being (m0=(M // m1 * m2), k0=(K_s // k1)).

    Reference: PTX ISA tcgen05.mma scale factor layout.
    https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-scale-factor-a-layout-1x
    '''
    m1, m2, k1 = 32, 4, 4

    M, K_s = scale.shape
    m0 = M // (m1 * m2)
    k0 = K_s // k1
    scale = scale.reshape(m0, m2, m1, k0, k1).permute(0, 3, 2, 1, 4).contiguous()
    return scale.reshape(m0, k0, 32, 16)


def unswizzle_32_4_4(tile_swizzled_scale):
    '''
    Kernel-side inverse of ``swizzle_32_4_4``: take a tile loaded
    from the host swizzled scale tensor and recover the ``(M, K_s)``
    view that ``ct.mma_scaled`` expects.
    '''
    m1, m2, k1 = 32, 4, 4
    m0, k0, _, _ = tile_swizzled_scale.shape

    return (tile_swizzled_scale.reshape((m0, k0, m1, m2, k1))
                               .permute((0, 3, 2, 1, 4))
                               .reshape((m0 * m2 * m1, k0 * k1)))
