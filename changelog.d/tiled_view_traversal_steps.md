- Add support for `tiled_view` with `traversal_steps`.
  When the traversal step is not equal to the tile shape, the tile space is
  adjusted accordingly, enabling iteration patterns such as gapped or overlapping
  tile access.
