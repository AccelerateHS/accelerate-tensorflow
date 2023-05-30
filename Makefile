BAZEL_BIN_DIR := extra-deps/tensorflow-haskell/third_party/tensorflow/bazel-bin

TF_LIB_SO_BASE := $(BAZEL_BIN_DIR)/tensorflow/libtensorflow.so
TF_LIB_LINKS := $(TF_LIB_SO_BASE) $(TF_LIB_SO_BASE).2 $(TF_LIB_SO_BASE).2.10

.PHONY: all help setup submodules tflitebuild tf-lib-links libedgetpu libedgetpu-lib-links cabal.project

all: help

help:
	@echo "This Makefile defines the following targets:"
	@echo "  'setup': all of the following:"
	@echo "    'submodules': Sets up the Git submodules"
	@echo "    'tflitebuild': Builds tensorflow lite inside build/"
	@echo "    'tfbuild': Builds full tensorflow with bazel"
	@echo "    'tf-lib-links': Additional .so symlinks in bazel bin dir"
	@echo "    'libedgetpu': Build libedgetpu.so"
	@echo "    'libedgetpu-lib-links': Additional .so symlinks in libedgetpu bin dir"
	@echo "    'cabal.project': Creates cabal.project from cabal.project.in (with envsubst)"
	@echo "Note: DO NOT USE -j with this Makefile. Parallelism is exploited already, and you can only break things."

# Don't list these targets as dependencies here so that things don't go _quite_ as horribly wrong when someone misguidedly uses -j with this Makefile
setup:
	$(MAKE) submodules
	$(MAKE) tflitebuild tfbuild
	$(MAKE) tf-lib-links
	$(MAKE) libedgetpu
	$(MAKE) libedgetpu-lib-links
	$(MAKE) cabal.project

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

tf-lib-links: $(TF_LIB_LINKS)

$(TF_LIB_LINKS):
	ln -s libtensorflow.so.2.10.1 $@

libedgetpu:
	cd extra-deps/libedgetpu && \
		env TFROOT=$(PWD)/extra-deps/tensorflow-haskell/third_party/tensorflow \
		make -f makefile_build/Makefile \
			CXXFLAGS=-I$(PWD)/build/flatbuffers/include \
			LDFLAGS='-L$(PWD)/build/_deps/flatbuffers-build -L$(PWD)/build/_deps/abseil-cpp-build/absl/flags' \
			FLATC=$(PWD)/extra-deps/tensorflow-haskell/third_party/tensorflow/bazel-bin/external/flatbuffers/flatc \
			libedgetpu-throttled \
			-j$(shell nproc)

libedgetpu-lib-links: extra-deps/libedgetpu/out/throttled/k8/libedgetpu.so

extra-deps/libedgetpu/out/throttled/k8/libedgetpu.so:
	ln -s libedgetpu.so.1.0 $@

cabal.project: cabal.project.in
	envsubst '$$PWD' <$< >$@
