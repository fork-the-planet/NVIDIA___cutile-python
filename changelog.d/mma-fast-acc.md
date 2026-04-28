<!--- SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!--- SPDX-License-Identifier: Apache-2.0 -->

- Add `use_fast_acc` keyword argument to `ct.mma()` to enable fast accumulation
  mode for fp8 inputs (`float8_e4m3fn`, `float8_e5m2`) on Hopper GPUs.
