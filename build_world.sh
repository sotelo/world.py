#!/bin/bash
git submodule update
pushd .
cd lib/world
./waf configure && ./waf
popd
