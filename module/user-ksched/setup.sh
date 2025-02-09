#!/bin/bash
sudo apt-get update
sudo apt-get install -y make \
    gcc \
    cmake \
    pkg-config \
    libnl-3-dev \
    libnl-route-3-dev \
    libnuma-dev \
    uuid-dev \
    libssl-dev \
    libaio-dev \
    libcunit1-dev \
    libclang-dev \
    libncurses-dev \
    meson \
    python3-pyelftools \
    g++ \
    build-essential \
    libnuma-dev \
    python3 \
    python3-pip \
    python3-pyelftools \
    pkg-config \
    meson \
    ninja-build \
    libaio-dev \
    libcunit1-dev \
    uuid-dev \
    libjson-c-dev \
    libssl-dev \
    libncurses5-dev \
    libncursesw5-dev

# Rollback and patch Caladan
cd ../caladan/
git reset --hard 14a57f0f405cdbf54f897436002ee472ede2ca40
git apply ../user-ksched/caladan.patch

# Build and insert ksched module
make submodules
export LIBRARY_PATH=$LIBRARY_PATH:$(dirname $(gcc -print-libgcc-file-name))
make clean && make
pushd ksched
make clean && make
popd
sudo ./scripts/setup_machine.sh && lsmod | grep ksched
