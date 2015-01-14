#!/bin/bash
git submodule update --init --recursive
pushd .
cd lib/world
./waf configure && ./waf
popd
