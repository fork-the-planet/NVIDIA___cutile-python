# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import threading
from contextlib import contextmanager
from typing import TYPE_CHECKING

from ._exception import TileStaticEvalError, TypeCheckingError

if TYPE_CHECKING:
    from ._ir.hir import StaticEvalKind


class DispatchMode:
    @staticmethod
    def get_current() -> "DispatchMode":
        return _current_mode.mode

    @contextmanager
    def as_current(self):
        old_mode = _current_mode.mode
        _current_mode.mode = self
        try:
            yield self
        finally:
            _current_mode.mode = old_mode

    def call_tile_function_from_host(self, func, args, kwargs):
        raise NotImplementedError()


class NormalMode(DispatchMode):
    def call_tile_function_from_host(self, func, args, kwargs):
        raise RuntimeError("Tile functions can only be called from tile code.")


class StaticEvalMode(DispatchMode):
    def __init__(self, kind: "StaticEvalKind"):
        self._kind = kind

    def call_tile_function_from_host(self, func, args, kwargs):
        from cuda.tile import static_eval, static_assert, static_iter
        if func in (static_eval, static_assert, static_iter):
            what = f"{func.__name__}() cannot be used"
        else:
            func_name = getattr(func, "__name__", "")
            if len(func_name) > 0:
                func_name = func_name + ": "
            what = f"{func_name}Tile functions cannot be called"

        where = self._kind._value_
        raise TileStaticEvalError(f"{what} inside {where}.")


class MetafunctionMode(DispatchMode):
    def call_tile_function_from_host(self, func, args, kwargs):
        from cuda.tile._ir.ir import Builder
        from cuda.tile._ir.core_ops import sym2var
        from cuda.tile._cext import run_coroutine
        from cuda.tile._passes.hir2ir import call
        from cuda.tile._ir.type import var2sym

        if Builder.get_current().is_terminated:
            raise TypeCheckingError(
                    "Current block has already been terminated with a break/continue;"
                    " cannot add any more operations to it.")

        func = sym2var(func)
        args = tuple(sym2var(x) for x in args)
        kwargs = {k: sym2var(v) for k, v in kwargs.items()}
        res = run_coroutine(call(func, args, kwargs))
        return var2sym(res)


class _CurrentModeTL(threading.local):
    mode: DispatchMode = NormalMode()


_current_mode = _CurrentModeTL()
