- Fixed a bug where strictly typed numeric constants
  (e.g., `ct.uint32(-3)`, `ct.int8(300)`, `ct.float16(0.2)`)
  were not truncated/rounded according to their dtype.
  Out-of-range integer values are now wrapped modulo the dtype's range,
  and float values are rounded/clamped according to the dtype's precision and range.
