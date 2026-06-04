# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS      ?= -W
VENV            ?=
BINDIR           = $(if $(VENV),$(VENV)/bin/,)
SPHINXBUILD     ?= $(BINDIR)sphinx-build
SPHINXAUTOBUILD ?= $(BINDIR)sphinx-autobuild
PYTHON          ?= $(BINDIR)python
SOURCEDIR        = source
BUILDDIR         = build
PROJECTDIR       = $(CURDIR)/..
AUTOBUILD_OPTS   = --watch $(PROJECTDIR)/src/cuda/lang --host 0.0.0.0 --port 8000

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile clean

clean:
	rm -rf $(SOURCEDIR)/generated
	rm -rf $(BUILDDIR)

serve: html
	$(PYTHON) -m http.server -d "$(BUILDDIR)/html"

watch:
	$(SPHINXAUTOBUILD) $(AUTOBUILD_OPTS) "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option. $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O) --show-traceback
