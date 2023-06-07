<div align="center">
<img width="450" src="https://github.com/AccelerateHS/accelerate/raw/master/images/accelerate-logo-text-v.png?raw=true" alt="henlo, my name is Theia"/>

# TensorFlow backend for the Accelerate array language

[![CI-Linux](https://github.com/tmcdonell/accelerate-tensorflow/workflows/ci-linux/badge.svg)](https://github.com/tmcdonell/accelerate-tensorflow/actions?query=workflow%3Aci-linux)
[![CI-MacOS](https://github.com/tmcdonell/accelerate-tensorflow/workflows/ci-macos/badge.svg)](https://github.com/tmcdonell/accelerate-tensorflow/actions?query=workflow%3Aci-macos)
[![CI-Windows](https://github.com/tmcdonell/accelerate-tensorflow/workflows/ci-windows/badge.svg)](https://github.com/tmcdonell/accelerate-tensorflow/actions?query=workflow%3Aci-windows)
<br>
[![Stackage LTS](https://stackage.org/package/accelerate-tensorflow/badge/lts)](https://stackage.org/lts/package/accelerate-tensorflow)
[![Stackage Nightly](https://stackage.org/package/accelerate-tensorflow/badge/nightly)](https://stackage.org/nightly/package/accelerate-tensorflow)
[![Hackage](https://img.shields.io/hackage/v/accelerate-tensorflow.svg)](https://hackage.haskell.org/package/accelerate-tensorflow)

</div>

This package compiles Accelerate code to a [TensorFlow](https://www.tensorflow.org) graph. For details on
Accelerate, refer to the [main repository](https://github.com/AccelerateHS/accelerate).

Contributions and bug reports are welcome!<br>
Please feel free to contact me through [GitHub](https://github.com/AccelerateHS/accelerate) or [gitter.im](https://gitter.im/AccelerateHS/Lobby).

## Dependencies

Many things are built from source in this repository, either because we need to apply some patches or because we need somewhat different transitive dependency versions than the upstream pre-compiled download has.
Still, a few things need to be installed on the system.

1. From the [Coral apt repository](https://coral.ai/software/#debian-packages), get the `edgetpu_compiler` package.
   (The other edgetpu libraries are either not needed or compiled from source here.)

2. **TODO: is protoc required?**
   To build the required TensorFlow and TensorFlow-haskell packages, you need to have protoc installed.
   If you do not have it installed, follow the directions on [this webpage](https://google.github.io/proto-lens/installing-protoc.html).
   (The Ubuntu package name is `protobuf-compiler`.)


## Compiling using Cabal

```sh
make setup  # see 'make help' for what this does; note, can take hours as this also builds a full copy of Tensorflow

ENV="$PWD/extra-deps/tensorflow-haskell/third_party/tensorflow/bazel-bin/tensorflow:$PWD/extra-deps/libedgetpu/out/throttled/k8:$PWD/build:$(echo "$PWD/build/_deps/abseil-cpp-build/absl/"{flags,hash,container,strings} | sed 's/ /:/g')"

env LD_LIBRARY_PATH="$ENV" cabal build all

# To run tests:
env LD_LIBRARY_PATH="$ENV" cabal run nofib-tensorflow-lite
# Without cabal:
env LD_LIBRARY_PATH="$ENV" accelerate_tensorflow_lite_datadir="$PWD/accelerate-tensorflow-lite" "$(cabal list-bin nofib-tensorflow-lite)"
```

This uses the Tensorflow submodule already contained within the tensorflow-haskell submodule.
Currently, this is Tensorflow 2.10.1.
