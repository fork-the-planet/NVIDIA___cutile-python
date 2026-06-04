.. SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
..
.. SPDX-License-Identifier: Apache-2.0

(Experimental) cuda.lang
========================

|cuda lang| brings low-level CUDA programming to Python. It exposes the familiar
CUDA C++ execution model while adding Pythonic APIs for modern hardware features
such as Tensor Memory Accelerator (TMA) and Tensor Cores.

|cuda lang| and |cuTile| share unified language design and compiler infrastructure
so applications can combine explicit SIMT control with higher-level tile programming.

Getting Started
---------------

The following kernel launches one CUDA block with 4 threads to normalize a 1D
array. Each thread vector-loads 4 values from global memory, atomically adds its
partial sum into one shared-memory value through a pointer, then divides each
loaded value by the block sum and stores the normalized values back to global
memory:

.. testcode::
   :template: setup_only.py

   @cl.kernel
   def normalize_kernel(values, normalized):
       total = cl.shared_array(shape=(1,), dtype=cl.float32)
       tx = cl.thread_idx(0)
       offset = tx * 4

       if tx == 0:
           total[0] = 0.0
       cl.syncthreads()

       loaded = values.get_element_pointer(offset).load(count=4, alignment=16)
       partial_sum = loaded[0] + loaded[1] + loaded[2] + loaded[3]
       cl.atomic_add(total.get_element_pointer(0), partial_sum)
       cl.syncthreads()

       loaded = loaded / total[0]
       normalized[offset + 0] = loaded[0]
       normalized[offset + 1] = loaded[1]
       normalized[offset + 2] = loaded[2]
       normalized[offset + 3] = loaded[3]

   values = torch.tensor(
       [
           1.0, 1.0, 2.0, 4.0,
           1.0, 1.0, 2.0, 4.0,
           1.0, 1.0, 2.0, 4.0,
           1.0, 1.0, 2.0, 4.0,
       ],
       dtype=torch.float32,
       device="cuda",
   )
   normalized = torch.empty_like(values)
   cl.launch(stream,
             (1,),  # grid size
             (4,),  # block size
             normalize_kernel, (values, normalized))
   print(normalized.cpu().tolist())

.. testoutput::

   [0.03125, 0.03125, 0.0625, 0.125, 0.03125, 0.03125, 0.0625, 0.125, 0.03125, 0.03125, 0.0625, 0.125, 0.03125, 0.03125, 0.0625, 0.125]


.. toctree::
   :maxdepth: 2
   :hidden:

   data
   operations
   private
