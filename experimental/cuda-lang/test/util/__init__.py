# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import pytest

import cuda.lang as cl
from cuda.lang._compile import get_compute_capability
from cuda.lang._logging import get_log_flags
from cuda.lang.compilation import KernelSignature

from .filecheck_utils import filecheck, get_source
from .ir_utils import (
    get_ir,
    make_symbolic_scalar,
    make_symbolic_tensor,
    compile_for_arguments,
)


def require_blackwell_or_newer():
    return pytest.mark.skipif(
        get_compute_capability() < (10, 0),
        reason="feature requires Blackwell or newer",
    )


def require_blackwell_cc100():
    cc = get_compute_capability()
    return pytest.mark.skipif(
        cc.major != 10,
        reason="feature requires Blackwell with compute capability 100",
    )


def require_hopper_or_newer():
    return pytest.mark.skipif(
        get_compute_capability() < (9, 0),
        reason="feature requires Hopper or newer",
    )


@pytest.fixture
def log_ptx():
    log_flags = get_log_flags()
    old_log_ptx = log_flags.log_ptx
    log_flags.log_ptx = True
    try:
        yield
    finally:
        log_flags.log_ptx = old_log_ptx


@pytest.fixture
def no_log_ptx():
    log_flags = get_log_flags()
    old_log_ptx = log_flags.log_ptx
    log_flags.log_ptx = False
    try:
        yield
    finally:
        log_flags.log_ptx = old_log_ptx


def compile_kernel(
    kernel,
    signature=KernelSignature([]),
    assert_in_ptx=None,
    assert_not_in_ptx=None,
    assert_in_mlir=None,
    assert_not_in_mlir=None,
    filecheck_ptx=None,
    raises=None,
):
    if raises is not None:
        assert assert_in_ptx is None
        assert assert_in_mlir is None
        assert filecheck_ptx is None
        with raises:
            cl.compile_simt(kernel, [signature], log_ptx=True)
        return

    compiled = cl.compile_simt(kernel, [signature], log_ptx=True)
    assert compiled.ptx

    def tuple_or_str_check(check, scrutinee, predicate=lambda x, y: x in y):
        match check:
            case None:
                pass
            case str():
                assert predicate(check, scrutinee), (
                    f"{predicate=} failed with\n{check=}\n{scrutinee}"
                )
            case tuple() | list():
                for single_check in check:
                    assert predicate(single_check, scrutinee), (
                        f"{predicate=} failed with\n{single_check=}\n{scrutinee}"
                    )
            case _:
                assert False, "expected assert_in_ptx to be str or iterable of str"

    tuple_or_str_check(assert_in_ptx, compiled.ptx)
    tuple_or_str_check(assert_not_in_ptx, compiled.ptx, lambda x, y: x not in y)
    tuple_or_str_check(assert_in_mlir, compiled.mlir)
    tuple_or_str_check(assert_not_in_mlir, compiled.mlir, lambda x, y: x not in y)

    if filecheck_ptx is not None:
        assert isinstance(filecheck_ptx, str)
        filecheck(compiled.ptx, filecheck_ptx)


__all__ = (
    "filecheck",
    "get_source",
    "get_ir",
    "make_symbolic_scalar",
    "make_symbolic_tensor",
    "compile_for_arguments",
    "log_ptx",
    "require_hopper_or_newer",
    "require_blackwell_or_newer",
)
