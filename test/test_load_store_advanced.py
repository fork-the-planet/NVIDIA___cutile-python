# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import cuda.tile as ct
from cuda.tile._bytecode.version import BytecodeVersion
from cuda.tile._exception import TileTypeError
from util import assert_equal
from conftest import requires_tileiras

pytestmark = requires_tileiras(BytecodeVersion.V_13_3)

# ===========================================================================================
# ct.load_advanced / ct.store_advanced: basic load/store
# ===========================================================================================


@ct.kernel
def load_store_advanced_rows(x, y, ROWS: ct.Constant[int], COLS: ct.Constant[int]):
    indices = ct.arange(ROWS, dtype=ct.int32)
    tile = ct.load_advanced(x, (indices, ct.Slice(0, COLS)))
    ct.store_advanced(y, (indices, ct.Slice(0, COLS)), tile)


def test_store_basic():
    rows, cols = 8, 4
    x = torch.arange(rows * cols, device='cuda', dtype=torch.int32).reshape(rows, cols)
    y = torch.zeros(rows, cols, device='cuda', dtype=torch.int32)
    ct.launch(torch.cuda.current_stream(), (1,), load_store_advanced_rows, (x, y, rows, cols))
    assert_equal(x, y)


# ===========================================================================================
# ct.load_advanced/store_advanced: non-contiguous row indices (actual gather/scatter)
# ===========================================================================================


@ct.kernel
def gather_even_rows(x, y, ROWS: ct.Constant[int], COLS: ct.Constant[int]):
    indices = ct.arange(ROWS, dtype=ct.int32) * 2  # [0, 2, 4, 6]
    tile = ct.load_advanced(x, (indices, ct.Slice(0, COLS)))
    ct.store(y, (0, 0), tile)


def test_gather_non_contiguous():
    rows, cols = 8, 4
    x = torch.arange(rows * cols, device='cuda', dtype=torch.int32).reshape(rows, cols)
    y_rows, y_cols = rows // 2, cols // 2
    y = torch.zeros(y_rows, y_cols, device='cuda', dtype=torch.int32)
    ct.launch(torch.cuda.current_stream(), (1,), gather_even_rows, (x, y, y_rows, y_cols))
    expected = x[::2, :y_cols]
    assert_equal(expected, y)


@ct.kernel
def scatter_even_rows(y, ROWS: ct.Constant[int], COLS: ct.Constant[int], col_start):
    indices = ct.arange(ROWS, dtype=ct.int32) * 2  # [0, 2, 4, 6]
    tile = ct.full((ROWS, COLS), 99, dtype=y.dtype)
    ct.store_advanced(y, (indices, ct.Slice(col_start, COLS)), tile)


def test_scatter_non_contiguous():
    y_rows, y_cols = 8, 4
    y = torch.zeros(y_rows, y_cols, device='cuda', dtype=torch.int32)
    ct.launch(torch.cuda.current_stream(), (1,), scatter_even_rows, (y, y_rows, y_cols, 0))
    expected = torch.zeros(y_rows, y_cols, device='cuda', dtype=torch.int32)
    expected[::2] = 99
    assert_equal(expected, y)


# ===========================================================================================
# ct.load_advanced: ct.Slice with dynamic start
# ===========================================================================================


@ct.kernel
def load_advanced_dynamic_col(x, y, ROWS: ct.Constant[int], COLS: ct.Constant[int],
                              col_start):
    indices = ct.arange(ROWS, dtype=ct.int32)
    tile = ct.load_advanced(x, (indices, ct.Slice(col_start, COLS)))
    ct.store(y, (0, 0), tile)


@pytest.mark.parametrize("col_start", [0, 1, 2])
def test_load_dynamic_col_start(col_start):
    rows, cols = 8, 4
    y_cols = cols // 2
    x = torch.arange(rows * cols, device='cuda', dtype=torch.int32).reshape(rows, cols)
    y = torch.zeros(rows, y_cols, device='cuda', dtype=torch.int32)
    ct.launch(torch.cuda.current_stream(), (1,),
              load_advanced_dynamic_col, (x, y, rows, y_cols, col_start))
    assert_equal(x[:, col_start:col_start + y_cols], y)


# ===========================================================================================
# ct.load_advanced: ct.Slice with constant start
# ===========================================================================================


@ct.kernel
def load_advanced_const_col_start(x, y, ROWS: ct.Constant[int], COLS: ct.Constant[int]):
    indices = ct.arange(ROWS, dtype=ct.int32)
    tile = ct.load_advanced(x, (indices, ct.Slice(2, COLS)))
    ct.store(y, (0, 0), tile)


def test_load_constant_col_start():
    rows, x_cols, tile_cols = 8, 8, 4
    x = torch.arange(rows * x_cols, device='cuda', dtype=torch.int32).reshape(rows, x_cols)
    y = torch.zeros(rows, tile_cols, device='cuda', dtype=torch.int32)
    ct.launch(torch.cuda.current_stream(), (1,),
              load_advanced_const_col_start, (x, y, rows, tile_cols))
    assert_equal(x[:, 2:2 + tile_cols], y)


# ===========================================================================================
# ct.load_advanced: out-of-order sparse indices gather rows in specified order.
# ===========================================================================================


def test_load_out_of_order_sparse():
    @ct.kernel
    def kernel(x, y, ROWS: ct.Constant[int], COLS: ct.Constant[int]):
        i = ct.arange(ROWS, dtype=ct.int32)
        # indices [7, 4, 2, 3]: i=0→7, i=1→4, i≥2→i
        indices = ct.where(i == 0, ct.full((ROWS,), 7, dtype=ct.int32),
                           ct.where(i == 1, ct.full((ROWS,), 4, dtype=ct.int32), i))
        tile = ct.load_advanced(x, (indices, ct.Slice(0, COLS)))
        ct.store(y, (0, 0), tile)

    x = torch.arange(32, device='cuda', dtype=torch.int32).reshape(8, 4)
    y = torch.zeros(4, 4, device='cuda', dtype=torch.int32)
    ct.launch(torch.cuda.current_stream(), (1,), kernel, (x, y, 4, 4))
    expected = x[[7, 4, 2, 3], :]
    assert_equal(y, expected)


# ===========================================================================================
# ct.load_advanced: OOB
# ===========================================================================================


def test_load_zero_padding():
    @ct.kernel
    def load_advanced_zero_padding(x, y, ROWS: ct.Constant[int], COLS: ct.Constant[int],
                                   col_start):
        indices = ct.arange(ROWS, dtype=ct.int32)
        tile = ct.load_advanced(x, (indices, ct.Slice(col_start, COLS)),
                                padding_mode=ct.PaddingMode.ZERO)
        ct.store(y, (0, 0), tile)
    rows, cols = 4, 8
    y_cols = cols // 2
    x = torch.arange(rows * cols, device='cuda', dtype=torch.int32).reshape(rows, cols) + 1
    y = torch.full((rows, y_cols), -1, device='cuda', dtype=torch.int32)
    col_start = 6
    ct.launch(torch.cuda.current_stream(), (1,),
              load_advanced_zero_padding, (x, y, rows, y_cols, col_start))
    expected = torch.zeros(rows, y_cols, device='cuda', dtype=torch.int32)
    expected[:, :cols - col_start] = x[:, col_start:]
    assert_equal(expected, y)


def test_load_sparse_partial_oob_zero_padding():
    """Sparse-dim partial OOB indices are zero-padded when padding_mode=ZERO."""
    @ct.kernel
    def kernel(x, y, ROWS: ct.Constant[int], COLS: ct.Constant[int]):
        # indices [6, 7, 8, 9]: 6 and 7 are in-bounds, 8 and 9 are OOB for an 8-row array
        indices = ct.arange(ROWS, dtype=ct.int32) + 6
        tile = ct.load_advanced(x, (indices, ct.Slice(0, COLS)),
                                padding_mode=ct.PaddingMode.ZERO)
        ct.store(y, (0, 0), tile)

    x = torch.arange(32, device='cuda', dtype=torch.int32).reshape(8, 4)
    y = torch.full((4, 4), -1, device='cuda', dtype=torch.int32)
    ct.launch(torch.cuda.current_stream(), (1,), kernel, (x, y, 4, 4))
    expected = torch.zeros(4, 4, device='cuda', dtype=torch.int32)
    expected[:2] = x[6:8]
    assert_equal(y, expected)


def test_load_repeated_sparse_correct():
    """Repeated in-bounds sparse indices are defined: each repeated index loads the same row."""
    @ct.kernel
    def kernel(x, y, ROWS: ct.Constant[int], COLS: ct.Constant[int]):
        i = ct.arange(ROWS, dtype=ct.int32)
        # indices = [0, 0, 4, 6]: first two repeat row 0, last two are distinct
        indices = ct.where(i < 2, ct.zeros((ROWS,), dtype=ct.int32), i * 2)
        tile = ct.load_advanced(x, (indices, ct.Slice(0, COLS)))
        ct.store(y, (0, 0), tile)

    x = torch.arange(32, device='cuda', dtype=torch.int32).reshape(8, 4)
    y = torch.zeros(4, 4, device='cuda', dtype=torch.int32)
    ct.launch(torch.cuda.current_stream(), (1,), kernel, (x, y, 4, 4))
    expected = x[[0, 0, 4, 6], :]
    assert_equal(expected, y)


# ===========================================================================================
# ct.store_advanced semantics
# ===========================================================================================


def test_store_repeated_sparse_ub():
    """Verify that repeated sparse indices on store does not affect non-repeated indices."""
    @ct.kernel
    def kernel(y, ROWS: ct.Constant[int], COLS: ct.Constant[int]):
        i = ct.arange(ROWS, dtype=ct.int32)
        # indices = [0, 0, 4, 6]: first two repeat row 0 (UB), last two are distinct
        indices = ct.where(i < 2, ct.zeros((ROWS,), dtype=ct.int32), i * 2)
        tile = ct.full((ROWS, COLS), 99, dtype=y.dtype)
        ct.store_advanced(y, (indices, ct.Slice(0, COLS)), tile)

    y = torch.zeros(8, 4, device='cuda', dtype=torch.int32)
    ct.launch(torch.cuda.current_stream(), (1,), kernel, (y, 4, 4))
    torch.cuda.synchronize()
    # row 0 is UB (written by indices[0] and indices[1]) — no assertion on it
    # rows 4 and 6 have distinct indices and must be correctly written
    assert_equal(y[4], torch.full((4,), 99, device='cuda', dtype=torch.int32))
    assert_equal(y[6], torch.full((4,), 99, device='cuda', dtype=torch.int32))


def test_store_dense_oob_ignored():
    """Dense-dim elements extending past the array boundary are silently ignored."""
    rows, array_cols = 8, 4
    tile_rows, tile_cols = 4, 4
    col_start = 2  # slice [2, 6) but array only has cols [0, 4) → cols 4-5 are OOB
    y = torch.zeros(rows, array_cols, device='cuda', dtype=torch.int32)
    ct.launch(torch.cuda.current_stream(), (1,), scatter_even_rows,
              (y, tile_rows, tile_cols, col_start))
    expected = torch.zeros(rows, array_cols, device='cuda', dtype=torch.int32)
    expected[::2, col_start:] = 99  # only in-bounds cols [2, 4) on even rows
    assert_equal(expected, y)


# ===========================================================================================
# Error cases
# ===========================================================================================


def test_error_2d_tile_as_sparse():
    @ct.kernel
    def kernel(x):
        indices = ct.zeros((4, 4), dtype=ct.int32)
        ct.load_advanced(x, (indices, ct.Slice(0, 4)))

    x = torch.zeros(8, 8, device='cuda', dtype=torch.int32)
    with pytest.raises(TileTypeError, match="1D"):
        ct.launch(torch.cuda.current_stream(), (1,), kernel, (x,))


def test_error_no_sparse_dim_load():
    @ct.kernel
    def kernel(x, y, col_start):
        result = ct.load_advanced(x, (ct.Slice(0, 4), ct.Slice(col_start, 4)))
        ct.store(y, (0, 0), result)

    x = torch.arange(64, device='cuda', dtype=torch.int32).reshape(8, 8)
    y = torch.zeros(4, 4, device='cuda', dtype=torch.int32)
    with pytest.raises(TileTypeError, match="exactly one index must be a 1D integer Tile"):
        ct.launch(torch.cuda.current_stream(), (1,), kernel, (x, y, 0))


def test_error_no_sparse_dim_store():
    @ct.kernel
    def kernel(y):
        tile = ct.full((4, 4), 99, dtype=y.dtype)
        ct.store_advanced(y, (ct.Slice(2, 4), ct.Slice(1, 4)), tile)

    y = torch.zeros(8, 8, device='cuda', dtype=torch.int32)
    with pytest.raises(TileTypeError, match="exactly one index must be a 1D integer Tile"):
        ct.launch(torch.cuda.current_stream(), (1,), kernel, (y,))


def test_error_multiple_sparse_dims_load():
    @ct.kernel
    def kernel(x, y):
        r = ct.arange(4, dtype=ct.int32) * 2
        c = ct.arange(4, dtype=ct.int32) + 1
        result = ct.load_advanced(x, (r, c))
        ct.store(y, (0,), result)

    x = torch.arange(64, device='cuda', dtype=torch.int32).reshape(8, 8)
    y = torch.zeros(4, device='cuda', dtype=torch.int32)
    with pytest.raises(TileTypeError, match="exactly one index must be a 1D integer Tile"):
        ct.launch(torch.cuda.current_stream(), (1,), kernel, (x, y))


def test_error_multiple_sparse_dims_store():
    @ct.kernel
    def kernel(y):
        r = ct.arange(4, dtype=ct.int32) * 2
        c = ct.arange(4, dtype=ct.int32) + 1
        tile = ct.full((4,), 99, dtype=y.dtype)
        ct.store_advanced(y, (r, c), tile)

    y = torch.zeros(8, 8, device='cuda', dtype=torch.int32)
    with pytest.raises(TileTypeError, match="exactly one index must be a 1D integer Tile"):
        ct.launch(torch.cuda.current_stream(), (1,), kernel, (y,))


def test_error_wrong_index_rank():
    @ct.kernel
    def kernel(x):
        indices = ct.arange(4, dtype=ct.int32)
        ct.load_advanced(x, (indices, ct.Slice(0, 4), ct.Slice(0, 4)))

    x = torch.zeros(8, 8, device='cuda', dtype=torch.int32)
    with pytest.raises(TileTypeError, match="does not match array rank"):
        ct.launch(torch.cuda.current_stream(), (1,), kernel, (x,))


def test_error_non_power_of_2_slice_length():
    @ct.kernel
    def kernel(x):
        indices = ct.arange(4, dtype=ct.int32)
        ct.load_advanced(x, (indices, ct.Slice(0, 3)))

    x = torch.zeros(8, 8, device='cuda', dtype=torch.int32)
    with pytest.raises(TileTypeError, match="power of two"):
        ct.launch(torch.cuda.current_stream(), (1,), kernel, (x,))
