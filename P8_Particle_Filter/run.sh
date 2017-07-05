#!/bin/bash
# Script to run particle filter!
#
# Written by Tiffany Huang, 12/14/2016
#
rm -rf build
mkdir build && cd build
cmake .. && make

# Run particle filter
# cd ./build
./particle_filter
