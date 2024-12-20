#!/bin/bash
set -e

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Install system dependencies
if command_exists apt-get; then
    # Debian/Ubuntu
    sudo apt-get update
    sudo apt-get install -y \
        build-essential \
        cmake \
        git \
        python3-dev \
        python3-pip \
        libtorch-dev \
        libprotobuf-dev \
        protobuf-compiler \
        nlohmann-json3-dev \
        libssl-dev \
        libcurl4-openssl-dev

elif command_exists yum; then
    # RHEL/CentOS
    sudo yum groupinstall -y "Development Tools"
    sudo yum install -y \
        cmake3 \
        python3-devel \
        openssl-devel \
        libcurl-devel

elif command_exists brew; then
    # macOS
    brew install \
        cmake \
        protobuf \
        nlohmann-json \
        libtorch \
        openssl \
        curl
fi

# Install Python dependencies
pip3 install --user \
    torch \
    numpy \
    pandas \
    matplotlib \
    seaborn

# Clone and build TVM
if [ ! -d "tvm" ]; then
    git clone --recursive https://github.com/apache/tvm tvm
    cd tvm
    mkdir -p build
    cp cmake/config.cmake build
    cd build
    cmake ..
    make -j$(nproc)
    cd ../..
fi

echo "All dependencies installed successfully!" 