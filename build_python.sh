#!/usr/bin/env bash
set -euo pipefail

rootdir="$PWD"

mkdir -p build-python
mkdir -p python-3.10-prefix

curl -L 'https://www.python.org/ftp/python/3.10.0/Python-3.10.0.tar.xz' >build-python/Python-3.10.0.tar.xz
tar -C build-python -xf build-python/Python-3.10.0.tar.xz

cd build-python/Python-3.10.0
./configure --prefix="$rootdir/python-3.10-prefix" --enable-optimizations
make -j32
make install

ln -vs "$rootdir/python-3.10-prefix/bin/python3" "$rootdir/python-3.10-prefix/bin/python"
