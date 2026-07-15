- Added a ``check_bounds`` option to ``ct.load()``, ``ct.store()`` and the
  ``TiledView.load()`` / ``TiledView.store()`` methods. It defaults to ``True``.
  when set to ``False``, it declares all elements in the tile is guaranteed to
  stay within the array bounds and skips the out-of-bounds check to speed up the
  load/store. Setting it to ``False`` requires tileiras 13.4+.
