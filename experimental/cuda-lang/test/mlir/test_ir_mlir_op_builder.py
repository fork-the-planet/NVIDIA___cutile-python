# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
import struct
from typing import Any, cast

from cuda.lang._ir.ir import Var
from cuda.lang._ir.op_impl.raw_mlir_operation_utils import RawMLIROperationBuilder
import cuda.lang._mlir as mlir
from cuda.lang._mlir import DenseI32ArrayAttr, nvvm


@dataclass
class FakeVar:
    value: int | None

    def is_constant(self):
        return self.value is None

    def get_constant(self):
        assert self.value is None
        return None


def test_mlir_operation_builder():
    def fake_var(value: int | None) -> Var[Any]:
        return cast(Var[Any], FakeVar(value))

    b = RawMLIROperationBuilder(name="foo.bar")

    b = b.add_attribute("baz", mlir.UnitAttr())
    b = b.add_attribute(
        "fib", nvvm.Tcgen05MMACollectorOpAttr(value=nvvm.Tcgen05MMACollectorOp.DISCARD)
    )

    b = b.add_operand(fake_var(0))
    assert set(dict(b.attributes)) == {"baz", "fib"}

    b = b.add_variadic_operand([fake_var(1), fake_var(2)])
    b = b.add_variadic_operand([])

    b = b.add_optional_operand(fake_var(3))
    b = b.add_optional_operand(None)
    b = b.add_optional_operand(fake_var(None))
    b = b.add_unit_attribute("not_present", False)

    assert b.operands == (
        fake_var(0),
        fake_var(1),
        fake_var(2),
        fake_var(3),
    )

    attributes = dict(b.attributes)
    assert set(attributes) == {"baz", "fib", "operandSegmentSizes"}
    assert isinstance(attributes["baz"], mlir.UnitAttr)
    assert attributes["fib"] == nvvm.Tcgen05MMACollectorOpAttr(
        value=nvvm.Tcgen05MMACollectorOp.DISCARD
    )
    segments = attributes["operandSegmentSizes"]
    assert isinstance(segments, DenseI32ArrayAttr)
    assert segments.size == 6
    assert struct.unpack("=" + "i" * segments.size, segments.rawData) == (1, 2, 0, 1, 0, 0)
