#!/bin/bash

set -e

# Check if binary exists
if [ ! -f "target/release/train_tokenizer" ]; then
    echo "Binary not found. Building first..."
    ./build.sh
fi

echo "Starting tokenizer training..."
echo "Using $(nproc) CPU cores"
echo ""

# Run with timing
time ./target/release/train_tokenizer

echo ""
echo "✓ Training complete!"

