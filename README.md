[![Build Status](https://travis-ci.org/jimsotelo/world.py.svg?branch=master)](https://travis-ci.org/jimsotelo/world.py)

This code currently only works in python 2.7!
First, run
`bash build_world.sh`

This should download and compile the WORLD code in lib/world.
Next, we need to build the cython extension using
`python setup.py develop`

To run test code the current directory *must* be on LD_LIBRARY_PATH!
`export LD_LIBRARY_PATH += .` 

Note the . which stands for current directory - this command was run in the same folder as test.py
