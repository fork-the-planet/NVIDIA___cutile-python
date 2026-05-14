<!--- SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!--- SPDX-License-Identifier: Apache-2.0 -->

- New `TiledView.atomic_store_add`, `TiledView.atomic_store_max`,
  `TiledView.atomic_store_min`, `TiledView.atomic_store_and`,
  `TiledView.atomic_store_or`, and `TiledView.atomic_store_xor` methods for
  performing element-wise atomic read-modify-write operations on a tiled view
  at a given tile index.
