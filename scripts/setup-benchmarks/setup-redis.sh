#!/bin/bash

# Exit on any error
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "${SCRIPT_DIR}/common.sh"

echo "Setting up Redis benchmark..."

# Install Redis server if not already installed
if ! command -v redis-server &> /dev/null; then
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if [ -f /etc/debian_version ]; then
            sudo apt-get update
            sudo apt-get install -y redis-server
        elif [ -f /etc/redhat-release ]; then
            sudo dnf install -y redis
        fi
    else
        echo "Unsupported operating system"
        exit 1
    fi
fi

# Create Redis data directory
REDIS_DATA_DIR="${BENCHMARK_DIR}/redis"
mkdir -p "${REDIS_DATA_DIR}"

# Create Redis configuration
cat > "${REDIS_DATA_DIR}/redis.conf" << EOF
port 6379
dir ${REDIS_DATA_DIR}
maxmemory 82gb
maxmemory-policy allkeys-lru
EOF

echo "Redis benchmark setup complete"