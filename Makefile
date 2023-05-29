BAZEL_BIN_DIR := extra-deps/tensorflow-haskell/third_party/tensorflow/bazel-bin

.PHONY: all help setup submodules tflitebuild tf-lib-links cabal.project

all: help

help:
	@echo "This Makefile defines the following targets:"
	@echo "  'setup': all of the following:"
	@echo "    'submodules': Sets up the Git submodules"
	@echo "    'tflitebuild': Builds tensorflow lite inside build/"
	@echo "    'tfbuild': Builds full tensorflow with bazel"
	@echo "    'tf-lib-links': Additional .so symlinks in bazel bin dir"
	@echo "    'cabal.project': Creates cabal.project from cabal.project.in (with envsubst)"

setup: submodules tflitebuild tfbuild cabal.project

submodules:
	@if git status --porcelain | grep extra-deps >/dev/null; then \
		echo "'git status' detects changes in the extra-deps directory, refusing to init submodules"; \
		exit 1; \
		fi
	git submodule update --init --recursive

tflitebuild:
	mkdir -p build
	cd build && cmake ../extra-deps/tensorflow-haskell/third_party/tensorflow/tensorflow/lite -DBUILD_SHARED_LIBS=1
	cd build && cmake --build . -j

tfbuild: bazel511
	cd extra-deps/tensorflow-haskell/third_party/tensorflow && ../../../../bazel511 build //tensorflow/tools/pip_package:build_pip_package
	cd extra-deps/tensorflow-haskell/third_party/tensorflow && ../../../../run-with-PATH-dir.sh python=python3 -- bazel-bin/tensorflow/tools/pip_package/build_pip_package ../../../../accelerate-tensorflow-lite/tf-python-venv
	virtualenv accelerate-tensorflow-lite/tf-python-venv
	accelerate-tensorflow-lite/tf-python-venv/bin/pip3 install accelerate-tensorflow-lite/tf-python-venv/tensorflow-2.10.1-*.whl

bazel511:
	curl -L https://github.com/bazelbuild/bazel/releases/download/5.1.1/bazel-5.1.1-linux-x86_64 >$@
	chmod +x $@

tf-lib-links: $(BAZEL_BIN_DIR)/tensorflow/libtensorflow.so $(BAZEL_BIN_DIR)/tensorflow/libtensorflow.so.2 $(BAZEL_BIN_DIR)/tensorflow/libtensorflow.so.2.10

$(BAZEL_BIN_DIR)/tensorflow/libtensorflow.so:
	ln -vs $@.2.10.1 $@
$(BAZEL_BIN_DIR)/tensorflow/libtensorflow.so.2:
	ln -vs $@.10.1 $@
$(BAZEL_BIN_DIR)/tensorflow/libtensorflow.so.2.10:
	ln -vs $@.1 $@

cabal.project: cabal.project.in
	envsubst '$$PWD' <$< >$@
