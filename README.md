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

# EXPERIMENTAL: BE AWARE

This repository contains experimental code.
Neither the plain TensorFlow CPU backend nor the TFLite backend support all Accelerate primitives, and there are multiple compatibility as well as correctness issues with Coral's TPU SDK and (probably) our usage of it.

The packages here are **NOT** yet suitable for actual use.

## Packages

This repository contains two packages:
- `accelerate-tensorflow` compiles
  [Accelerate](https://github.com/AccelerateHS/accelerate) code to a
  [TensorFlow](https://www.tensorflow.org) graph. It allows running that graph
  on the default TensorFlow backend (CPU).
- `accelerate-tensorflow-lite` extends `accelerate-tensorflow` and allows
  running the graph on a TPU using [Coral (from Google)](https://coral.ai/)'s
  `libedgetpu` and the accompanying `edgetpu_compiler`. This works by
  converting the graph to a TensorFlow Lite (TFLite) model, hence the name of
  the package.

Contributions and bug reports are welcome.

## Dependencies

Many things are built from source in this repository, either because we need to apply some patches or because we need somewhat different transitive dependency versions than the upstream pre-compiled download has.
Still, a few things need to be installed on the system.

1. From the [Coral apt repository](https://coral.ai/software/#debian-packages), get the `edgetpu_compiler` package.
   (The other edgetpu libraries are either not needed or compiled from source here.)

2. Furthermore, install the following packages:
   - GHC 8.10.7 and its dependencies (see the ghcup instructions)
   - All dependencies needed by GHC
   - The protobuf compiler `protoc` (`protobuf-compiler` on Ubuntu, or see [here](https://google.github.io/proto-lens/installing-protoc.html)).
   - `curl`, `gawk`, `libusb-1.0-0-dev` (on Ubuntu)
   - Python 3.10 (not 3.11!), including `pip`. Furthermore ensure that `python3` is available in PATH under the name `python`, to satisfy TensorFlow's build process.

A sequence of commands starting from a (virtual) machine running fresh, "minimal" Ubuntu Server 22.04.3 and ending up with running the test suite is listed in [ubuntu-build-instructions.txt](ubuntu-build-instructions.txt), but note that this excludes two things:
1. The installation of the Coral `edgetpu_compiler`;
2. The setup of the udev rules: see the next section.


## Udev rules

In order to be able to access the TPU hardware using a non-root user, you will need to set up udev rules.
For the Coral USB Accelerator, the following rules work: (note that both rules are necessary)
```
$ cat /etc/udev/rules.d/99-edgetpu-accelerator.rules
SUBSYSTEM=="usb",ATTRS{idVendor}=="1a6e",GROUP="edgetpu"
SUBSYSTEM=="usb",ATTRS{idVendor}=="18d1",GROUP="edgetpu"
```
To get access to the hardware, one then needs to add your Linux user to the `edgetpu` group.
(This is the `GROUP=` field in the udev rules; you can name this group differently if you want.)
Do not forget to re-login after updating your Linux user.

Note that while this may also work for other Coral EdgeTPU hardware, you may need to consult the vendor of your specific hardware for details on the device IDs.

A symptom that these udev rules are not set up correctly is getting `ERROR: Failed to retrieve TPU context.`; when `strace`ing the executable, one then sees that you're getting "Permission denied" errors when accessing the USB device.


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

A shorthand for the `env LD_LIBRARY_PATH="..."` prefix is [`./in-env.sh`](in-env.sh).

The version of TensorFlow being used is that in the submodule contained within the `tensorflow-haskell` submodule.
This is TensorFlow version 2.10.1.
Both repositories have some patches applied at the time of writing, and are hence forks of upstream.


## Usage example

For an example of how to use both backends defined in this repository, see [accelerate-tensorflow-lite/test/examples/Main.hs](accelerate-tensorflow-lite/test/examples/Main.hs).


## Further documentation

Some further documentation on the implementation of the backends can be found in the [doc](doc) directory.
