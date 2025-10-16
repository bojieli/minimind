# High-Performance BPE Tokenizer Trainer

A blazingly fast BPE tokenizer trainer written in Rust, designed to fully utilize multi-core systems (tested on 128-core servers).

## Performance Advantages & Limitations

### What's Fast (Multi-threaded):
- ✅ **Parallel JSON parsing** - uses all CPU cores with rayon
- ✅ **Parallel file I/O** - large buffers (8MB) for sequential reads
- ✅ **Parallel pre-tokenization** - ByteLevel encoding uses multiple threads
- ✅ **Zero GC pauses** - predictable performance
- ✅ **No Python overhead** - native Rust, no binding layer

### What's Slow (Single-threaded):
- ⚠️ **BPE merge learning** - inherently sequential algorithm
  - The tokenizers library trains BPE merges on a single thread
  - This is the main bottleneck for large vocabularies
  - Each merge iteration depends on the previous one

### Expected Speedup:
- **JSON parsing**: 10-100x faster than Python (multi-core)
- **BPE training**: 2-5x faster than Python (native code, no GIL, but still sequential)
- **Overall**: 5-20x faster depending on dataset size and vocab size

The larger the vocabulary, the more time is spent in the sequential BPE merge phase.

## Prerequisites

Install Rust if you haven't already:
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
```

## Building

```bash
cd train_tokenizer_rust
cargo build --release
```

The `--release` flag is crucial for maximum performance. It enables:
- Level 3 optimizations
- Link-time optimization (LTO)
- Single codegen unit for better optimization

## Usage

### Basic usage (using default paths):
```bash
cargo run --release
```

### Custom paths via command line:
```bash
# Custom data path and output directory
cargo run --release -- --data-path /path/to/data.jsonl --output-dir /path/to/output

# Custom vocabulary size
cargo run --release -- --vocab-size 8000

# Train on more data (e.g., 5GB instead of default 1GB)
cargo run --release -- --max-data-gb 5.0

# Limit number of threads
cargo run --release -- --threads 64

# All options combined
cargo run --release -- \
  --data-path ../../dataset/pretrain_data.jsonl \
  --output-dir ../../model \
  --vocab-size 6400 \
  --max-data-gb 1.0 \
  --threads 64
```

### Using the compiled binary:
```bash
# After building, use the binary directly
./target/release/train_tokenizer --help

# Run with custom options
./target/release/train_tokenizer \
  --data-path /path/to/data.jsonl \
  --output-dir /path/to/output \
  --vocab-size 8000
```

### Available options:
- `-d, --data-path <PATH>`: Input JSONL file path (default: `../../dataset/pretrain_data.jsonl`)
- `-o, --output-dir <DIR>`: Output directory (default: `../../model`)
- `-v, --vocab-size <SIZE>`: Target vocabulary size (default: `6400`)
- `-m, --max-data-gb <GB>`: Maximum data to load in GB (default: `1.0`)
- `-t, --threads <NUM>`: Number of threads to use (default: all available cores)
- `-h, --help`: Print help information
- `-V, --version`: Print version information

## Input Format

The program expects JSONL files where each line contains:
```json
{"text": "Your training text here..."}
```

**Note on Data Size**: By default, the program loads only the first **1GB** of JSONL data. This is intentional:
- ✅ 1GB is typically sufficient for high-quality tokenizers
- ✅ Dramatically faster training (BPE merge phase is sequential)
- ✅ Lower memory usage
- ✅ You can always increase with `--max-data-gb` if needed

Research shows diminishing returns beyond 1-5GB of training data for tokenizers.

## Output Files

The trained tokenizer will be saved to the output directory:
- `tokenizer.json` - Complete tokenizer configuration
- `vocab.json` - Vocabulary mappings
- `merges.txt` - BPE merge operations
- `tokenizer_config.json` - HuggingFace compatibility config

## Performance Tuning

### Thread Control
Thread count mainly affects JSON parsing and pre-tokenization phases:
```bash
# Limit threads (useful for shared servers)
cargo run --release -- --threads 64
```

Note: The BPE merge phase is single-threaded regardless of thread count.

### Memory Usage
The program uses an 8MB buffer for file I/O. For systems with limited memory, you can reduce this in `src/main.rs`:
```rust
let reader = BufReader::with_capacity(2 * 1024 * 1024, file); // 2MB buffer
```

### Optimizing Training Speed

**For faster training:**
1. **Use less data** (default is 1GB, which is usually sufficient)
   ```bash
   cargo run --release -- --max-data-gb 0.5  # Use only 500MB
   ```

2. **Reduce vocabulary size** - Fewer merges = faster training
   ```bash
   cargo run --release -- --vocab-size 3200  # Half the merges
   ```

3. **Balance quality vs speed**:
   - 100MB-500MB: Very fast, good for initial experiments
   - 1GB (default): Fast, good quality for most use cases
   - 5GB+: Slower, diminishing returns on quality

4. **Trade-off**: The sequential BPE merge phase time is roughly:
   - Linear with vocabulary size (more merges = more iterations)
   - Logarithmic with dataset size (more text = slower frequency updates)
   - **Recommendation**: Start with 1GB and smaller vocab, increase if needed

**What WON'T help:**
- Adding more CPU cores (BPE merging is sequential)
- Adding more RAM (not memory-bound)
- Using faster storage (after JSON parsing completes)

## Compatibility

The output is 100% compatible with HuggingFace Transformers:
```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("../model/")
```

## Special Tokens

The tokenizer includes three special tokens with fixed IDs:
- `<|endoftext|>` (ID: 0) - Padding/unknown token
- `<|im_start|>` (ID: 1) - Begin of sequence
- `<|im_end|>` (ID: 2) - End of sequence

## Future Improvements

The main performance bottleneck is the sequential BPE merge algorithm. Potential improvements:

1. **Parallel BPE variants** - Research implementations exist (e.g., FastBPE) but sacrifice some quality
2. **Approximate BPE** - Use sampling or approximations for faster training
3. **WordPiece/Unigram** - Alternative algorithms with better parallelization potential
4. **Pre-computed frequencies** - Cache and reuse frequency tables across training runs

For now, this implementation prioritizes:
- ✅ Correctness (100% compatible with HuggingFace)
- ✅ Fast I/O and preprocessing (multi-threaded)
- ✅ Reasonable BPE training speed (native Rust, no Python overhead)

If you need faster training, consider reducing vocabulary size or using a smaller training corpus.
