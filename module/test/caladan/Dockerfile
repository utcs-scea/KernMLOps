# This Dockerfile acts more as a guide for building and benchmarking the Caladan environment
# We recommend following these steps in a bare-metal environment

FROM ubuntu:20.04

# Instructions for installing Caladan
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    build-essential \
    make \
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
    libstdc++-10-dev \
    git \
    curl \
    sudo \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /caladan
# Use '.' to clone into current directory
# Applying the patch sets up ksched for shmem benchmarking
RUN git clone https://github.com/shenango/caladan.git . && \
    git reset --hard 14a57f0f405cdbf54f897436002ee472ede2ca40 && \
    git apply ~/KernMLOps/caladan.patch && \
    make submodules

# Build Caladan
ENV LIBRARY_PATH=$LIBRARY_PATH:/usr/lib/gcc/x86_64-linux-gnu/10
RUN make clean && make && \
    cd ksched && make clean && make && cd ..

# Setup the module
RUN ./scripts/setup_machine.sh && lsmod | grep ksched

# Run and measure the Caladan benchmark
RUN cd ~/KernMLOps/module/test/caladan/ && g++ -Wall -O3 -o ksched_user ksched_user.cpp
RUN cd ../.. && ./scripts/take_measure.sh

RUN ./scripts/graph_measure.sh | ./scripts/graph.py -t "Performance Comparison" -x "Data Size" -y "Time (ns)" -g 5 -o test.png

######

# Install Rust
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y && \
    . "$HOME/.cargo/env" && \
    rustup default nightly

# Copy configuration files
COPY server.config /caladan/
COPY client.config /caladan/
COPY ksched_user.c /caladan/

# Build synthetic application
RUN cd apps/synthetic && \
    cargo clean && \
    cargo update && \
    cargo build --release
