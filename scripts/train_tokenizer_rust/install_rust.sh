#!/bin/bash

set -e

echo "Installing Rust toolchain..."
echo ""

# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

# Source cargo environment
source $HOME/.cargo/env

echo ""
echo "✓ Rust installed successfully!"
echo ""
echo "Rust version:"
rustc --version
cargo --version
echo ""
echo "You can now build the tokenizer trainer with: ./build.sh"

