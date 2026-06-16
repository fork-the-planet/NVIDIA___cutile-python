- Fixed a bug where strictly typed integer constants (e.g., `ct.uint32(-3)`, `ct.int8(300)`)
  were not truncated to their declared dtype's bit width.
  Out-of-range values are now wrapped modulo the dtype's range.