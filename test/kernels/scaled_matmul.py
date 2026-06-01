# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import cuda.tile as ct

from kernels.kernel_utils import swizzle_2d, unswizzle_32_4_4

ConstInt = ct.Constant[int]


@ct.kernel(num_ctas=ct.ByTarget(sm_100=2))
def block_scaled_matmul_kernel(
                    A, A_scale, B, B_scale, C,
                    tm: ConstInt,         # Tile size along M dimension (rows of C)
                    tn: ConstInt,         # Tile size along N dimension (columns of C)
                    tk: ConstInt,        # Tile size along K dimension (inner product dimension)
                    scaling_block_size: ConstInt):

    """
    cuTile kernel for block-scaled matrix multiplication.

    Computes C = (A * A_scale) @ (B * B_scale), accumulating in float32.
    Each TileBlock computes one tm x tn output tile. The K dimension is processed
    in chunks of tk, with tks scale values per K tile.

    If packed swizzle scales are passed, they get unswizzled into logical 2D scale tiles,
    then passed to ct.mma_scaled.

    Args:
        A:              Input matrix A (M x K).
        A_scale:        2D scale of (M, K // scaling_block_size) or swizzle scale of
                        (M // 32 // 4, K // scaling_block_size // 4, 32, 4, 4) reshaped
                        into (M // 32 // 4, K // scaling_block_size // 4, 32, 16).
        B:              Input matrix B (K x N).
        B_scale:        2D scale of (M, K // scaling_block_size) or swizzled scale of
                        (M // 32 // 4, K // scaling_block_size // 4, 32, 4, 4) reshaped
                        into (M // 32 // 4, K // scaling_block_size // 4, 32, 16).
        C:              Output matrix C (M x N).
        tm (ConstInt):  The height of the output tile computed by this block.
                        Corresponds to rows of A and C.
        tn (ConstInt):  The width of the output tile computed by this block.
                        Corresponds to columns of B and C.
        tk (ConstInt):  The depth of the inner loop (K-dimension) tile size.
                        Corresponds to columns of A and rows of B.
        scaling_block_size (ConstInt): the scaling block size.
    """
    GROUP_SIZE_M = 8
    M = A.shape[0]
    N = B.shape[1]
    bidx, bidy = swizzle_2d(M, N, tm, tn, GROUP_SIZE_M)

    tks = tk // scaling_block_size

    # Calculate the total number of tiles along the K-dimension that need to be processed.
    # `ct.num_tiles(A, axis=1, shape=(tm, tk))` means:
    #   "View A as an MxK tensor tiled by (tm, tk), and return the number of tiles along
    #    axis 1 (the K dimension)."
    # We pass shape=(tm, tk) to describe the 2D tiling, only `tk` matters for axis=1.
    num_tiles_k = ct.num_tiles(A, axis=1, shape=(tm, tk))

    # Initialize an accumulator for the current output tile (tm x tn).
    # It's common practice to use `float32` for accumulation even with `float16` inputs
    # to maintain higher precision during the sum-reduction of the matrix multiplication.
    accumulator = ct.full((tm, tn), 0, dtype=ct.float32)
    zero_pad = ct.PaddingMode.ZERO

    # K-dimension loop: Iterate over the K-dimension in chunks of 'tk'.
    # In each iteration, a `tm` x `tk` tile from A and a `tk` x `tn` tile from B
    # are loaded, multiplied, and accumulated.
    for k in range(num_tiles_k):
        # Load tile from matrix A.
        # The `index=(bidx, k_tile_idx)` specifies which (M-tile, K-tile) to load
        # from global memory A. `shape=(tm, tk)` defines the size of this tile.
        a = ct.load(A, index=(bidx, k), shape=(tm, tk), padding_mode=zero_pad)

        if len(A_scale.shape) == 2:
            # 2D scale path. A_scale is already stored in logical shape (M, K_s).
            a_scale = ct.load(A_scale, index=(bidx, k), shape=(tm, tks), padding_mode=zero_pad)
        else:
            # Load the packed scale tile, unswizzle it, and reshape to
            # the logical ct.mma_scaled shape (tm, tks).

            # unswizzle
            a_scale_swizzled = ct.load(A_scale, index=(bidx, k, 0, 0),
                                       shape=(tm // scaling_block_size // 4, tks // 4, 32, 16),
                                       padding_mode=zero_pad)
            a_scale = unswizzle_32_4_4(a_scale_swizzled)

        # Load tile from matrix B.
        # The `index=(k_tile_idx, bidy)` specifies which (K-tile, N-tile) to load
        # from global memory B. `shape=(tk, tn)` defines the size of this tile.
        b = ct.load(B, index=(k, bidy), shape=(tk, tn), padding_mode=zero_pad)

        if len(B_scale.shape) == 2:
            b_scale = ct.load(B_scale, index=(k, bidy), shape=(tks, tn), padding_mode=zero_pad)
        else:
            # B scales are stored N-major. Unswizzle it, reshape it to
            # (tn, tks), then transpose it to the logical ct.mma_scaled shape (tks, tn).

            # unswizzle
            b_scale_swizzled = ct.load(B_scale, index=(bidy, k, 0, 0),
                                       shape=(tn // scaling_block_size // 4, tks // 4, 32, 16),
                                       padding_mode=zero_pad)
            b_scale = unswizzle_32_4_4(b_scale_swizzled).permute((1, 0))

        # Perform Scaled Matrix Multiplication for the current tiles.
        # `ct.mma_scaled` computes the product of the two loaded tiles
        # and scales and accumulates the result.
        accumulator = ct.mma_scaled(a, a_scale, b, b_scale, accumulator)

    # Store the computed tile to the global memory of the output matrix C.
    # The `(bidx, bidy)` directly corresponds to the tile's position in the 2D output matrix.
    ct.store(C, index=(bidx, bidy), tile=accumulator)
