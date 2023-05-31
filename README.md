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

TFLite is compiled from the submodule and TF itself is not because TFLite can be compiled using CMake, and TF itself seems to need compilation with Bazel, which is annoying to install and use. Hence we download the required .so from the upstream release for TF itself.

## Installing the edgetpu library

TODO: Make sure everything in this section is correct; at the moment, the list
of what to install might be incomplete.
TODO: non-debian Linux instructions.
Follow the instructions from [Coral](https://coral.ai/software/#debian-packages) to get access to their debian packages through apt(-get). Then, install the following libraries:
 - libedgetpu-dev (TODO: check necessity, probably required)
 - edgetpu\_compiler
 - libedgetpu1-std (recommended unless the higher frequency is required)

## TODO check if this is still required

### Installing protoc

To build the required TensorFlow and TensorFlow-haskell packages, you need to
have protoc installed. If you do not have it installed, follow the directions on
[this webpage](https://google.github.io/proto-lens/installing-protoc.html).

### Installing other dependencies

Other dependencies have to be installed manually before running `stack build`.
Among these are cpuinfo, farmhash. (TODO: find out what exactly is on this
list.) These exist in the Ubuntu package management system and can be installed
through apt:
```bash
sudo apt install libcpuinfo-dev libfarmhash-dev
```

