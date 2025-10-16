#!/bin/bash
# One-command setup script for MiniMind training data preparation
# This script downloads datasets and prepares both pretrain and SFT data

set -e  # Exit on error

echo "========================================================"
echo "MiniMind Training Data Setup"
echo "========================================================"
echo ""

# Configuration
DATASET_DIR="/root/minimind/dataset"
PRETRAIN_SIZE_GB=15
MAX_TOKENS_512=512

# Create dataset directory
mkdir -p "$DATASET_DIR"
cd /root/minimind

echo "Step 1: Checking dependencies..."
/root/miniconda3/bin/python -c "import pandas, pyarrow, transformers" 2>/dev/null || {
    echo "Installing required packages..."
    /root/miniconda3/bin/pip install pandas pyarrow transformers -q
}
echo "✓ Dependencies ready"
echo ""

# Check if datasets need to be downloaded
echo "Step 2: Checking datasets..."

if [ ! -d "$DATASET_DIR/fineweb_edu_chinese_4_5/4_5" ]; then
    echo "Downloading Fineweb-Edu-Chinese-4_5 dataset..."
    echo "(This will take a while, ~70GB download)"
    huggingface-cli download opencsg/Fineweb-Edu-Chinese-V2.1 \
      --repo-type dataset \
      --include "4_5/*" \
      --local-dir "$DATASET_DIR/fineweb_edu_chinese_4_5"
    echo "✓ Fineweb dataset downloaded"
else
    echo "✓ Fineweb dataset already exists"
fi

if [ ! -d "$DATASET_DIR/smoltalk-chinese/data" ]; then
    echo "Downloading smoltalk-chinese dataset..."
    huggingface-cli download opencsg/smoltalk-chinese \
      --repo-type dataset \
      --local-dir "$DATASET_DIR/smoltalk-chinese"
    echo "✓ smoltalk dataset downloaded"
else
    echo "✓ smoltalk dataset already exists"
fi
echo ""

# Prepare pretrain data with chunking
echo "Step 3: Preparing pretrain data with chunking (${PRETRAIN_SIZE_GB}GB, max ${MAX_TOKENS_512} tokens)..."
if [ -f "$DATASET_DIR/pretrain_data.jsonl" ]; then
    echo "pretrain_data.jsonl already exists. Skipping..."
    echo "To regenerate with chunking, delete the file first:"
    echo "  rm $DATASET_DIR/pretrain_data.jsonl"
else
    /root/miniconda3/bin/python scripts/prepare_pretrain_data.py \
      --input_dir "$DATASET_DIR/fineweb_edu_chinese_4_5/4_5" \
      --output_file "$DATASET_DIR/pretrain_data.jsonl" \
      --max_gb $PRETRAIN_SIZE_GB \
      --min_length 10 \
      --max_seq_len $MAX_TOKENS_512 \
      --min_chunk_length 50 \
      --tokenizer_path model/
    echo "✓ Pretrain data ready (with chunking for max utilization)"
fi
echo ""

# Prepare SFT data for different sequence lengths
echo "Step 4: Preparing SFT data..."

echo "  - Preparing SFT data for 512 tokens..."
if [ -f "$DATASET_DIR/sft_data_512.jsonl" ]; then
    echo "    sft_data_512.jsonl already exists. Skipping..."
else
    /root/miniconda3/bin/python scripts/prepare_sft_data.py \
      --input_dir "$DATASET_DIR/smoltalk-chinese/data" \
      --output_file "$DATASET_DIR/sft_data_512.jsonl" \
      --tokenizer_path model/ \
      --min_tokens 50 \
      --max_tokens $MAX_TOKENS_512
    echo "    ✓ SFT 512 ready"
fi
echo ""

# Summary
echo "========================================================"
echo "SETUP COMPLETE!"
echo "========================================================"
echo ""
echo "Data files created:"
ls -lh "$DATASET_DIR"/*.jsonl 2>/dev/null || echo "No JSONL files found"
echo ""
echo "========================================================"
echo "Next Steps - Training Commands"
echo "========================================================"
echo ""
echo "1. Train Tokenizer (Optional, already provided):"
echo "   Rust version (faster):"
echo "   cd scripts/train_tokenizer_rust && bash run.sh"
echo "   Python version:"
echo "   cd scripts && python train_tokenizer.py"
echo ""
echo "2. Pretrain (Learn Knowledge):"
echo "   Single GPU:"
echo "   cd trainer && python train_pretrain.py --data_path ../dataset/pretrain_data.jsonl --use_wandb"
echo ""
echo "   8 GPUs (DDP):"
echo "   cd trainer && torchrun --nproc_per_node=8 train_pretrain.py --data_path ../dataset/pretrain_data.jsonl --use_wandb"
echo ""
echo "3. Supervised Fine-Tuning (Learn Dialogue):"
echo "   Single GPU:"
echo "   cd trainer && python train_full_sft.py --data_path ../dataset/sft_data_512.jsonl --use_wandb"
echo ""
echo "   8 GPUs (DDP):"
echo "   cd trainer && torchrun --nproc_per_node=8 train_full_sft.py --data_path ../dataset/sft_data_512.jsonl --use_wandb"
echo ""
echo "4. DPO Training (Optional - Preference Alignment):"
echo "   Single GPU:"
echo "   cd trainer && python train_dpo.py --data_path ../dataset/dpo.jsonl --use_wandb"
echo ""
echo "   8 GPUs (DDP):"
echo "   cd trainer && torchrun --nproc_per_node=8 train_dpo.py --data_path ../dataset/dpo.jsonl --use_wandb"
echo ""
echo "========================================================"
echo "See scripts/QUICKSTART.md for detailed documentation."

