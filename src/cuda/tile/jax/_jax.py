# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import io
import inspect
import struct
import hashlib
from functools import partial
from typing import Any, Sequence
from dataclasses import dataclass
from collections import defaultdict

import cuda.tile as ct
import cuda.tile._cext as cext
from cuda.tile._annotated_function import LeafAnnotationNode
from cuda.tile._datatype import DType
from cuda.tile.compilation._signature import (
    ArrayConstraint, ConstantConstraint, ScalarConstraint, ParameterConstraint)
from cuda.tile.compilation import (KernelSignature, CallingConvention, export_kernel)
import cuda.tile._compile as ct_compile

try:
    import jax
    import jax.extend
    import numpy as np
    from jax.interpreters import mlir
    from jax._src.lib.mlir import ir
    HAS_JAX = True
except ImportError:
    HAS_JAX = False


# =============================== API ===============================


@dataclass
class OutputPlaceholder:
    """Represents an output buffer passed to cutile_call."""

    shape: tuple[int, ...]
    dtype: Any


@dataclass
class InputOutput:
    """Wraps an input buffer to alias an output buffer to be returned by cutile_call."""

    array: "jax.Array"


def cutile_call(grid: tuple[int, ...],
                kernel,
                args: tuple[Any, ...]):
    """Launch a cuTile kernel from a JAX-traced graph.

    Args:
        grid: Tuple of up to 3 grid dimensions to execute
            the |kernel| over. Padded with 1s on the right.
        kernel: The |kernel| to execute.
        args: Positional arguments to pass to the kernel.
            Each entry must match the kernel's corresponding parameter:

            - :class:`jax.Array`: read-only input buffer.
            - :class:`OutputPlaceholder`: output buffer with the given
              ``shape`` and ``dtype``; allocated by JAX and returned from
              this call.
            - :class:`InputOutput`: input buffer aliased to an output slot,
              enabling in-place updates.
            - ``bool``, ``int``, or ``float``: A scalar argument to the kernel.
              Because JAX treats scalar as 0D Array, to pass a scalar argument,
              it must be a static argument of the JAX function using
              ``static_argnums`` or ``static_argnames``.

    Returns:
        For a kernel with one output, the output array. For multiple
        outputs (multiple ``OutputPlaceholder`` / ``InputOutput`` args), a
        tuple of arrays in declaration order.

    Notes:
        1. Array passed to cutile_call will use default XLA row-major order. Customizing
        layout will be supported in the future.

    Example:

    .. testcode::
        :template: jax_setup.py

        @ct.kernel
        def scale(x, y, c: ct.Constant, TILE_SIZE: ct.Constant):
            bid = ct.bid(0)
            tx = ct.load(x, bid, TILE_SIZE) * c
            ct.store(y, bid, tx)


        @ct.kernel
        def inplace(x, TILE_SIZE: ct.Constant):
            bid = ct.bid(0)
            tx = ct.load(x, bid, TILE_SIZE)
            ct.store(x, bid, tx / 2)


        @ct.kernel
        def sincos(x, y1, y2, TILE_SIZE: ct.Constant):
            bid = ct.bid(0)
            tx = ct.load(x, bid, TILE_SIZE)
            ct.store(y1, bid, ct.sin(tx))
            ct.store(y2, bid, ct.cos(tx))



        @jax.jit(static_argnums=[1, 2])
        def graph(x, c, tile_size):
            grid = (ct.cdiv(x.shape[0], tile_size),)
            ph = OutputPlaceholder(x.shape, x.dtype)

            y = cutile_call(grid, scale, (x, ph, c, tile_size))

            # inplace update
            y = cutile_call(grid, inplace, (InputOutput(y), tile_size))

            # multiple outputs
            ysin, ycos = cutile_call(grid, sincos, (y, ph, ph, tile_size))

            return ysin + ycos

        x = jnp.arange(10, dtype=jnp.float32)
        y = graph(x, jnp.pi, 4)
        print(y)

    .. testoutput::

        [ 1.  1. -1. -1.  1.  1. -1. -1.  1.  1.]
    """
    outputs: list[jax.ShapeDtypeStruct] = []
    constants: list[bool | int | float] = []
    scalars: list[bool | int | float] = []
    input_arrays: list[jax.Array] = []
    alias_group: list[str | None] = []
    roles: list[str] = []
    grid = grid + (1,) * (3 - len(grid))
    alias_map = defaultdict(list)

    ann = kernel._annotated_function
    annotations = ann.parameter_annotations
    kernel_name = ann.pyfunc.__name__
    params = list(ann.pysig.parameters)
    if len(args) != len(params):
        raise TypeError(
            f"{kernel_name} expects {len(params)} arguments, got {len(args)}"
        )

    # The JAX/FFI integration is one-arg-one-role and treats every annotation as a flat
    # leaf (it reads `.constant`/`.int64_index`/`.int64_scalar` directly). Tuple parameters
    # produce non-leaf annotation nodes, which it cannot flatten into buffers/constraints.
    for i, ann_node in enumerate(annotations):
        if not isinstance(ann_node, LeafAnnotationNode):
            raise NotImplementedError(
                f"{kernel_name}: argument {i} ('{params[i]}') is a tuple parameter; "
                f"tuple parameters are not supported via the JAX/FFI integration"
            )

    for (i, (x, ann_node)) in enumerate(zip(args, annotations)):
        if isinstance(x, OutputPlaceholder):
            roles.append('o')
            outputs.append(jax.ShapeDtypeStruct(x.shape, x.dtype))
        elif isinstance(x, jax.Array):
            roles.append('i')
            alias_map[id(x)].append(len(input_arrays))
            input_arrays.append(x)
        elif isinstance(x, InputOutput):
            roles.append('io')
            alias_map[id(x.array)].append(len(input_arrays))
            input_arrays.append(x.array)
            outputs.append(jax.ShapeDtypeStruct(x.array.shape, x.array.dtype))
        elif isinstance(x, (bool, int, float)):
            if ann_node.constant:
                roles.append('c')
                constants.append(x)
            else:
                roles.append('s')
                scalars.append(x)
        else:
            raise TypeError(f"Unexpected type for argument[{i}]: {type(x)}")

    _check_roles(kernel_name, params, annotations, roles)

    for i, x in enumerate(input_arrays):
        aliases = alias_map[id(x)]
        alias_group.append(f"G{aliases[0]}" if len(aliases) > 1 else None)

    @partial(jax.jit, inline=True)
    def wrapper(*input_args):
        out = cutile_call_ffi_p.bind(
            *input_args,
            kernel=kernel,
            grid=grid,
            output_shape_dtypes=tuple(outputs),
            constants=tuple(constants),
            scalars=tuple(scalars),
            roles=tuple(roles),
            alias_group=tuple(alias_group),
        )
        if len(outputs) == 1:
            return out[0]
        else:
            return out

    return wrapper(*input_arrays)


def _check_roles(kernel_name: str,
                 params: Sequence[inspect.Parameter],
                 annotations: Sequence[Any],
                 roles: Sequence[str]):
    for i, (role, ann_node, pname) in enumerate(zip(roles, annotations, params)):
        if ann_node.constant and role != 'c':
            raise TypeError(
                f"{kernel_name}: argument {i} ('{pname}') is annotated ct.Constant; "
                f"expected a static scalar argument"
            )

# =========================== FFI registration ===========================


def register_ffi():
    call_type_id = cext.xla_ffi_get_call_type_id()
    call_type_info = cext.xla_ffi_get_call_type_info()
    jax.ffi.register_ffi_type(
        "cutile_launch",
        {"type_id": call_type_id, "type_info": call_type_info},
        platform='CUDA')

    call_handler = cext.xla_ffi_get_call_handler()
    jax.ffi.register_ffi_target(
        "cutile_launch",
        {"instantiate": call_handler, "execute": call_handler},
        platform='CUDA')


# =========================== Primitive ===========================
cutile_call_ffi_p = None


def register_primitive():
    global cutile_call_ffi_p

    cutile_call_ffi_p = jax.extend.core.Primitive("cutile_call_ffi")
    cutile_call_ffi_p.multiple_results = True

    mlir.register_lowering(
        cutile_call_ffi_p,
        _cutile_call_ffi_p_lower,
        platform="cuda",
    )

    @cutile_call_ffi_p.def_abstract_eval
    def _cutile_call_ffi_abstract(*input_args, kernel, grid, output_shape_dtypes,
                                  constants, scalars, roles, alias_group):
        return [jax.core.ShapedArray(x.shape, x.dtype) for x in output_shape_dtypes]


def pack_scalar(value: bool | int | float,
                want_int64: bool = False) -> tuple[DType, int]:
    if isinstance(value, bool):
        return (ct.int32, 1 if value else 0)
    if isinstance(value, int):
        if want_int64:
            return (ct.int64, value & 0xFFFFFFFFFFFFFFFF)
        if not (-(1 << 31) <= value < (1 << 31)):
            raise OverflowError(
                f"Runtime int scalar {value} doesn't fit in int32; annotate "
                f"the kernel parameter with ct.ScalarInt64.")
        return (ct.int32, value & 0xFFFFFFFF)
    if isinstance(value, float):
        bits = int.from_bytes(struct.pack("<f", value), "little", signed=False)
        return (ct.float32, bits)
    raise TypeError(
        f"Unsupported runtime scalar type: {type(value).__name__}")


def _cutile_call_ffi_p_lower(
    ctx: "mlir.LoweringRuleContext",
    *input_args: "ir.Value",
    kernel: ct.kernel,
    grid: tuple[int, int, int],
    output_shape_dtypes: tuple["jax.ShapeDtypeStruct", ...],
    constants: tuple[bool | int | float, ...],
    scalars: tuple[bool | int | float, ...],
    roles: tuple[str, ...],
    alias_group: tuple[str | None, ...]
):
    num_inputs = len(input_args)
    num_outputs = len(output_shape_dtypes)

    constraints: list[ParameterConstraint] = []
    # launch argument id to buffer id, input buffers followed by output_buffers
    buffer_ids: list[int] = []
    # Parallel to `buffer_ids`: index width (32 or 64) the kernel was
    # compiled with for each buffer. Used at execute-time to refuse
    # passing oversize shape/stride to an i32-indexed kernel.
    index_bitwidths: list[int] = []
    input_output_aliases: dict[int, int] = {}

    ann_func = kernel._annotated_function
    annotations = ann_func.parameter_annotations
    scalar_packed: list[int] = []

    ni, no, nc, ns = 0, 0, 0, 0
    for pos, role in enumerate(roles):
        is_i64_index = annotations[pos].int64_index
        idx_dtype = ct.int64 if is_i64_index else ct.int32
        if role == 'i':
            buffer_ids.append(ni)
            index_bitwidths.append(64 if is_i64_index else 32)
            constraints.append(_array_constraint(ctx.avals_in[ni], idx_dtype, alias_group[ni]))
            ni += 1
        elif role == 'o':
            buffer_ids.append(no + num_inputs)
            index_bitwidths.append(64 if is_i64_index else 32)
            constraints.append(_array_constraint(ctx.avals_out[no], idx_dtype, None))
            no += 1
        elif role == 'io':
            buffer_ids.append(ni)
            index_bitwidths.append(64 if is_i64_index else 32)
            constraints.append(_array_constraint(ctx.avals_in[ni], idx_dtype, alias_group[ni]))
            input_output_aliases[ni] = no
            ni += 1
            no += 1
        elif role == 's':
            buffer_ids.append(ns + num_inputs + num_outputs)
            index_bitwidths.append(0)  # unused for scalar slot
            dtype, packed = pack_scalar(scalars[ns],
                                        want_int64=annotations[pos].int64_scalar)
            constraints.append(ScalarConstraint(dtype))
            scalar_packed.append(packed)
            ns += 1
        elif role == 'c':
            constraints.append(ConstantConstraint(constants[nc]))
            nc += 1

    symbol, cubin_bytes, cubin_id = compile_kernel_cached(kernel, constraints)

    # cubin_code and function_name are attributes on the launch op itself.
    # MLIR bytecode uniques identical attributes, so a graph that launches
    # the same cubin from N sites still serializes its bytes only once. The
    # FFI handler load-or-finds the kernel under a process-wide registry
    # keyed by cubin_id at INSTANTIATE.
    return jax.ffi.ffi_lowering(
        "cutile_launch",
        operand_output_aliases=input_output_aliases or None,
        api_version=4,
    )(
        ctx, *input_args,
        buffer_ids=np.asarray(buffer_ids, dtype=np.int32),
        scalar_packed=np.asarray(scalar_packed, dtype=np.uint64),
        cubin_code=cubin_bytes,
        cubin_id=np.frombuffer(cubin_id, dtype=np.uint8),
        function_name=symbol,
        index_bitwidths=np.asarray(index_bitwidths, dtype=np.int32),
        num_inputs=np.int32(num_inputs),
        num_outputs=np.int32(num_outputs),
        grid_x=np.int32(grid[0]),
        grid_y=np.int32(grid[1]),
        grid_z=np.int32(grid[2]),
    )


# =========================== Compilation ===========================
_COMPILE_CACHE = {}


def compile_kernel(kernel: ct.kernel,
                   constraints: Sequence[ParameterConstraint]
                   ) -> tuple[str, bytes]:
    pyfunc = kernel._annotated_function.pyfunc
    signature = KernelSignature(
        constraints, CallingConvention.cutile_python_v1(),
    ).with_mangled_symbol(pyfunc.__name__)
    function_name = signature.symbol
    buf = io.BytesIO()
    export_kernel(
        kernel, signatures=[signature], output_file=buf,
        gpu_code=ct_compile.get_sm_arch(), output_format="cubin",
    )
    cubin_code = buf.getvalue()
    return function_name, cubin_code


def _constraint_cache_key(c: ParameterConstraint) -> Any:
    if isinstance(c, ConstantConstraint):
        return ("const", type(c.value).__name__, c.value)
    return c


def compile_kernel_cached(kernel: ct.kernel,
                          constraints: Sequence[ParameterConstraint]
                          ) -> tuple[str, bytes, bytes]:
    """Returns (function_name, cubin_code, cubin_id). cubin_id is the
    32-byte SHA-256 digest of cubin_code, used as the registry key on the
    C++ side."""
    constraints = tuple(constraints)
    key = (id(kernel), tuple(_constraint_cache_key(c) for c in constraints))
    hit = _COMPILE_CACHE.get(key)
    if hit is not None:
        return hit
    function_name, cubin_code = compile_kernel(kernel, constraints)
    cubin_id = hashlib.sha256(cubin_code).digest()
    out = (function_name, cubin_code, cubin_id)
    _COMPILE_CACHE[key] = out
    return out


_XLA_BASE_ADDR_ALIGN = 256
_DIVISOR_16 = 16
_BYTE_BITWIDTH = 8


def _array_constraint(aval: "jax.core.ShapedArray",
                      index_dtype: DType,
                      alias_group: str | None) -> ArrayConstraint:
    from cuda.tile._ir.typing_support import to_dtype
    for d in aval.shape:
        if not isinstance(d, int):
            raise NotImplementedError(
                f"cutile_call does not yet support shape-polymorphic dims (got {d!r} "
                f"of type {type(d).__name__} in shape {aval.shape})"
            )
    dtype = to_dtype(aval.dtype)
    shape = aval.shape
    ndim = len(shape)
    bits = dtype.bitwidth

    # Row-major contiguous strides in elements (XLA default layout).
    strides = [1] * ndim
    for i in range(ndim - 2, -1, -1):
        strides[i] = strides[i + 1] * shape[i + 1]

    if index_dtype is ct.int32:
        i32_max = (1 << 31) - 1
        for i, (d, s) in enumerate(zip(shape, strides)):
            if d > i32_max or s > i32_max:
                raise TypeError(
                    f"Array shape={shape} dim {i} (size={d}, stride={s}) "
                    f"exceeds the int32 index range; annotate the parameter "
                    f"with ct.IndexedWithInt64 to use 64-bit shape/stride."
                )

    stride_constant = []
    stride_divisible_by = []
    shape_divisible_by = []
    div16_bits = _DIVISOR_16 * _BYTE_BITWIDTH
    stride_divisor = div16_bits // bits if div16_bits % bits == 0 else 1

    for i in range(ndim):
        s, d = strides[i], shape[i]
        stride_constant.append(1 if s == 1 else None)
        shape_divisible_by.append(_DIVISOR_16 if d % _DIVISOR_16 == 0 else 1)
        stride_divisible_by.append(stride_divisor if s % stride_divisor == 0 else 1)

    return ArrayConstraint(
        dtype=dtype,
        ndim=ndim,
        index_dtype=index_dtype,
        stride_lower_bound_incl=0,
        alias_groups=() if alias_group is None else (alias_group,),
        may_alias_internally=False,
        stride_constant=tuple(stride_constant),
        stride_divisible_by=tuple(stride_divisible_by),
        shape_divisible_by=tuple(shape_divisible_by),
        base_addr_divisible_by=_XLA_BASE_ADDR_ALIGN,
    )
