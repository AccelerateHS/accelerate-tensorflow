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

## Here be dragons

To build a (local) copy of tensorflow-lite:

```sh
mkdir build
cd build
cmake ../extra-deps/tensorflow/tensorflow/lite
cmake --build . -j
```

## Installing the TensorFlow C-libraries

The TensorFlow C-bindings are required to build this project. In order to
install them, follow the instructions provided by
[TensorFlow](https://www.tensorflow.org/install/lang_c). Make sure to install
the TensorFlow 2.3.0
[CPU](https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-linux-x86_64-2.3.0.tar.gz)
or
[GPU](https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-gpu-linux-x86_64-2.3.0.tar.gz)
bindings, and not the latest version, since this is the version required by the
TensorFlow-Haskell dependency. As such, we recommend not installing in the
default location (/usr/local on Linux or MacOS systems), but to a different
location. To make sure the build succeeds, you need to tell stack where to find
these files, using the `extra-lib-dirs` and `extra-include-dirs` fields. Make
sure to set the `LIBRARY_PATH` and `LD_LIBRARY_PATH` as described in the
installation instructions as well.

## Installing protoc

To build the required TensorFlow and TensorFlow-haskell packages, you need to
have protoc installed. If you do not have it installed, follow the directions on
[this webpage](https://google.github.io/proto-lens/installing-protoc.html).

## Installing other dependencies

Other dependencies have to be installed manually before running `stack build`.
Among these are cpuinfo, farmhash. (TODO: find out what exactly is on this
list.) These exist in the Ubuntu package management system and can be installed
through apt:
```bash
sudo apt install libcpuinfo-dev libfarmhash-dev
```

