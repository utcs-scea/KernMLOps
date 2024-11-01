#!/bin/bash
if [ -d "data/db" ]; then
    echo "Directory /data/db already exists."
    exit 0
fi
mkdir -p data/db

YCSB_BENCHMARK_NAME="ycsb"
BENCHMARK_DIR_NAME="kernmlops-benchmark"

BENCHMARK_DIR="${BENCHMARK_DIR:-$HOME/$BENCHMARK_DIR_NAME}"
YCSB_BENCHMARK_DIR="$BENCHMARK_DIR/$YCSB_BENCHMARK_NAME"

# Start MongoDB server without sysctl
mongod --dbpath /data/db --fork --logpath /var/log/mongodb.log

# Run YCSB load and run mongodb commands. It is expected for the benchmark to fail here, it will still load for actual use.
"$YCSB_BENCHMARK_DIR/ycsb-0.17.0/bin/ycsb" load mongodb -s -P "$YCSB_BENCHMARK_DIR/ycsb-0.17.0/workloads/workloada" \
    -p recordcount=1000000 \
    -p mongodb.url=mongodb://localhost:27017/ycsb \
    -p mongodb.writeConcern=acknowledged
