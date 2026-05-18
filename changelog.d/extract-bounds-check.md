<!--- SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!--- SPDX-License-Identifier: Apache-2.0 -->

- `ct.extract` now raises a compile-time `TileTypeError` when a constant index is out of bounds for the tile grid. Dynamic indices are unaffected.
