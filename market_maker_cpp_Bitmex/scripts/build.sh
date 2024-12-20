#!/bin/bash
set -e

# Create build directory
mkdir -p build
cd build

# Configure CMake
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_TESTING=ON \
    -DUSE_CUDA=ON \
    -DUSE_TVM=ON

# Build
cmake --build . -j$(nproc)

# Install
cmake --install .

echo "Build completed successfully!" 