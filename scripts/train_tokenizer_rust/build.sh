#!/bin/bash

set -e

echo "Building high-performance tokenizer trainer..."
echo "This may take a few minutes on first build (compiling dependencies)"
echo ""

# Check if Rust is installed
if ! command -v cargo &> /dev/null; then
    echo "ERROR: Rust is not installed!"
    echo "Install it with: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
    exit 1
fi

# Build in release mode with optimizations
cargo build --release

echo ""
echo "✓ Build complete!"
echo ""
echo "Binary location: target/release/train_tokenizer"
echo "To run: cargo run --release"
echo "Or directly: ./target/release/train_tokenizer"
echo ""
echo "Performance tip: The program will use all available CPU cores by default."
echo "To limit threads: RAYON_NUM_THREADS=64 cargo run --release"

