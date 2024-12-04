#!/bin/bash
# Define variables first
YCSB_BENCHMARK_NAME="ycsb"
BENCHMARK_DIR_NAME="kernmlops-benchmark"
BENCHMARK_DIR="${BENCHMARK_DIR:-$HOME/$BENCHMARK_DIR_NAME}"
YCSB_BENCHMARK_DIR="$BENCHMARK_DIR/$YCSB_BENCHMARK_NAME"

# Then use them
if [ -d "$YCSB_BENCHMARK_DIR/mongo_db" ]; then
    echo "Directory $YCSB_BENCHMARK_DIR/mongo_db already exists."
    exit 0
fi
mkdir -p "$YCSB_BENCHMARK_DIR/mongo_db"

# Start MongoDB server
sudo mongod --dbpath "$YCSB_BENCHMARK_DIR/mongo_db" --bind_ip_all --fork --logpath /var/log/mongodb.log

# Load workload
"$YCSB_BENCHMARK_DIR/ycsb-0.17.0/bin/ycsb" load mongodb -s \
    -P "$YCSB_BENCHMARK_DIR/ycsb-0.17.0/workloads/workloada" \
    -p mongodb.url=mongodb://128.83.122.76:27017/ycsb
