<!--- SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!--- SPDX-License-Identifier: Apache-2.0 -->

- New `ct.load_advanced(array, indices)` and `ct.store_advanced(array, indices, tile)` for gathering/scattering along one dimension from/to a 2D or higher-rank array. A 1D integer `Tile` selects the sparse dimension; `ct.Slice(start, length)` selects a contiguous range along each dense dimension.
