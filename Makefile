.PHONY: all help setup submodules tfbuild libtf-dl/lib/libtensorflow.so cabal.project

all: help

help:
	@echo "This Makefile defines the following targets:"
	@echo "  'setup': all of the following:"
	@echo "    'submodules': Sets up the Git submodules"
	@echo "    'tfbuild': Builds tensorflow inside build/"
	@echo "    'libtf-dl/lib/libtensorflow.so': Downloads libtensorflow.so"
	@echo "    'cabal.project': Creates cabal.project from cabal.project.in (with envsubst)"

setup: submodules tfbuild libtf-dl/lib/libtensorflow.so cabal.project

submodules:
	@if git status --porcelain | grep extra-deps >/dev/null; then \
		echo "'git status' detects changes in the extra-deps directory, refusing to init submodules"; \
		exit 1; \
		fi
	git submodule update --init --recursive

tfbuild:
	mkdir -p build
	cd build && cmake ../extra-deps/tensorflow-haskell/third_party/tensorflow/tensorflow/lite -DBUILD_SHARED_LIBS=1
	cd build && cmake --build . -j

libtf-dl/lib/libtensorflow.so:
	mkdir libtf-dl
	curl https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-linux-x86_64-2.10.1.tar.gz | tar xz -C libtf-dl

cabal.project: cabal.project.in
	envsubst '$$PWD' <$< >$@
