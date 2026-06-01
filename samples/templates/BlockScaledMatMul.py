# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import cuda.tile as ct
import torch
import sys
from cuda.tile._cext import get_compute_capability
from cuda.tile._bytecode.version import BytecodeVersion


from test.kernels.kernel_utils import block_quantize, swizzle_32_4_4, get_tileiras_version
from test.kernels.scaled_matmul import block_scaled_matmul_kernel


def cutile_block_scaled_matmul(A: torch.Tensor, A_scale: torch.Tensor,
                               B: torch.Tensor, B_scale: torch.Tensor) -> torch.Tensor:

    """
    Performs block-scaled matrix multiplication using a cuTile kernel.

    This wrapper function handles input validation, determines appropriate
    tile sizes based on data type, calculates the necessary grid dimensions,
    and launches the `block_scaled_matmul_kernel`.

    Args:
        A (torch.Tensor):       The first input matrix (M x K). Must be on a CUDA device.
        B (torch.Tensor):       The second input matrix (K x N). Must be on a CUDA device
                                and have its K dimension match A's K dimension.
        A_scale (torch.Tensor): Either 2D scale with shape (M, K // scaling_block_size) or
                                swizzled scale of (M // scaling_block_size // 4,
                                K // scaling_block_size // 4, 32, 16).
        B_scale (torch.Tensor): Either 2D scale with shape (M, K // scaling_block_size) or
                                swizzled scale of (M // scaling_block_size // 4,
                                K // scaling_block_size // 4, 32, 16).

    Returns:
        torch.Tensor: The resulting matrix C (M x N) on the CUDA device.

    Raises:
        ValueError: If matrices are incompatible (K dimensions don't match),
                    or if they are not on a CUDA device.
    """
    # --- Input Validation ---
    if A.shape[1] != B.shape[0]:
        raise ValueError(f"Incompatible matrices: K dimension of A ({A.shape[1]}) "
                         f"must match K dimension of B ({B.shape[0]})")
    if A.device != B.device or A.device != A_scale.device or A.device != B_scale.device:
        raise ValueError("Input tensors must be on the same device.")
    if not A.is_cuda or not A_scale.is_cuda or not B.is_cuda or not B_scale.is_cuda:
        raise ValueError("Input tensors must be on a CUDA device.")
    # Note: cuTile handles dtype compatibility within the kernel,
    # but inputs should generally match.

    tm, tn, tk, scaling_block_size = 256, 256, 128, 32

    # --- Get Matrix Dimensions ---
    m, _ = A.shape
    _, n = B.shape

    # --- Calculate Grid Dimensions for Kernel Launch (1D Grid) ---
    # The grid defines how many CUDA blocks (CTAs) will be launched.
    # Each block computes one (tm x tn) output tile of matrix C.
    # `ct.cdiv(total_dim, tile_dim)` ensures enough blocks are launched to cover
    # the entire matrix, even if dimensions are not perfect multiples of tile sizes.
    grid_x = ct.cdiv(m, tm)  # Number of blocks needed along the M dimension (rows of C)
    grid_y = ct.cdiv(n, tn)  # Number of blocks needed along the N dimension (columns of C)
    grid_size = grid_x * grid_y

    grid = (grid_size, 1, 1)

    # --- Create Output Tensor C ---
    # The output tensor `C` is initialized with the correct dimensions (M x N),
    # on the same device, and with the same data type as the input matrices.
    C = torch.empty((m, n), device=A.device, dtype=torch.float32)

    # --- Launch the cuTile Kernel ---
    # The `block_scaled_matmul_kernel` is launched with the calculated grid dimensions.
    # `tm`, `tn`, and `tk` are passed as Constant integers to the kernel.
    kernel = block_scaled_matmul_kernel
    ct.launch(torch.cuda.current_stream(), grid, kernel, (
        A, A_scale, B, B_scale, C, tm, tn, tk, scaling_block_size))

    return C


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--correctness-check",
        action="store_true",
        help="Check the correctness of the results",
    )
    args = parser.parse_args()

    if get_compute_capability()[0] < 10:
        print("Skipped test: NOT Running cuTile Block Scaled Matrix Multiplication Examples "
              "Blackwell or newer required.")
        sys.exit(0)

    if get_tileiras_version() < BytecodeVersion.V_13_3:
        print("Skipped test: NOT Running cuTile Block Scaled Matrix Multiplication Examples "
              "tileiras versiom 13.3 required.")
        sys.exit(0)

    # --- Running cuTile Block Scaled Matrix Multiplication Examples ---
    print("--- Running cuTile Block Scaled Matrix Multiplication Examples ---")

    # Define common matrix dimensions for the examples
    M_dim = 512
    N_dim = 512
    K_dim = 768

    scaling_block_size = 32
    KS_dim = K_dim // scaling_block_size

    print("\n--- Test Case: Block Scaled Matrix Multiplication with M = 512, N = 512, "
          "K = 768, Scaling Block Size = 32 ---")

    A = torch.rand((M_dim, K_dim), device='cuda')
    B = torch.rand((N_dim, K_dim), device='cuda')

    A, A_scale = block_quantize(A, scaling_block_size)
    B, B_scale = block_quantize(B, scaling_block_size)

    B = B.T
    B_scale = B_scale.T

    k = A.shape[-1]
    ks = k // scaling_block_size

    A_s_swizzled = swizzle_32_4_4(A_scale)
    B_s_swizzled = swizzle_32_4_4(B_scale.T.contiguous())

    print(f"Input A shape: {A.shape}, dtype: {A.dtype}")
    print(f"Input B shape: {B.shape}, dtype: {B.dtype}")

    atol, rtol = 1e-4, 1e-3

    # Perform matrix multiplication using the cuTile wrapper function.
    C_cutile = cutile_block_scaled_matmul(A, A_scale, B, B_scale)
    C_cutile_swizzled = cutile_block_scaled_matmul(A, A_s_swizzled, B, B_s_swizzled)
    print(f"cuTile Output C shape: {C_cutile.shape}, dtype: {C_cutile.dtype}")

    if args.correctness_check:
        ref_A_scale = torch.repeat_interleave(A_scale, scaling_block_size, dim=1).to(torch.float32)
        ref_B_scale = torch.repeat_interleave(B_scale, scaling_block_size, dim=0).to(torch.float32)
        ref = (A.to(torch.float32) * ref_A_scale) @ (B.to(torch.float32) * ref_B_scale)

        torch.testing.assert_close(C_cutile, ref, atol=atol, rtol=rtol)
        torch.testing.assert_close(C_cutile_swizzled, ref, atol=atol, rtol=rtol)
        print("Correctness check passed")
    else:
        print("Correctness check disabled")

    print("\n--- cuTile block scaled matrix multiplication example completed. ---")
