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
# The following 3 commands can also be done in one go using `make setup`:
make submodules     # git submodule update --init --recursive
make tfbuild        # build Tensorflow inside newly-created build/ directory
make cabal.project  # rewrites $PWD in cabal.project.in to make cabal.project

cabal build all
```

This uses the Tensorflow submodule already contained within the tensorflow-haskell submodule.
Currently, this is Tensorflow 2.10.1.

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

