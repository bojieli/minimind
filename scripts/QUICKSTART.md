# Quick Start: Data Preparation on New Server

## TL;DR - One Command Setup

```bash
cd /root/minimind
bash scripts/setup_training_data.sh
```

This will:
1. Download Fineweb-Edu-Chinese (pretrain) and smoltalk-chinese (SFT)
2. Prepare 15GB pretrain data
3. Prepare SFT data for 512 and 1024 token lengths

---

## Manual Setup (Step by Step)

### 1. Download Datasets

```bash
# Fineweb for pretraining (~70GB)
huggingface-cli download opencsg/Fineweb-Edu-Chinese-V2.1 \
  --repo-type dataset \
  --include "4_5/*" \
  --local-dir /root/minimind/dataset/fineweb_edu_chinese_4_5

# smoltalk for SFT
huggingface-cli download opencsg/smoltalk-chinese \
  --repo-type dataset \
  --local-dir /root/minimind/dataset/smoltalk-chinese
```

### 2. Prepare Pretrain Data (15GB limit)

```bash
cd /root/minimind
python scripts/prepare_pretrain_data.py \
  --max_gb 15 \
  --output_file dataset/pretrain_data.jsonl
```

### 3. Prepare SFT Data

```bash
# For 512 tokens max_length
python scripts/prepare_sft_data.py \
  --max_tokens 512 \
  --output_file dataset/sft_data_512.jsonl

# For 1024 tokens max_length  
python scripts/prepare_sft_data.py \
  --max_tokens 1024 \
  --output_file dataset/sft_data_1024.jsonl
```

---

## Training

### Pretrain
```bash
cd /root/minimind/trainer
python train_pretrain.py \
  --data_path ../dataset/pretrain_data.jsonl \
  --max_seq_len 512 \
  --epochs 2
```

### Fine-tune
```bash
python train_full_sft.py \
  --data_path ../dataset/sft_data_512.jsonl \
  --max_seq_len 512 \
  --epochs 2
```

---

## Scripts Created

| Script | Purpose |
|--------|---------|
| `prepare_pretrain_data.py` | Convert Fineweb parquet to pretrain JSONL (with size limit) |
| `prepare_sft_data.py` | Filter smoltalk by token count for SFT |
| `setup_training_data.sh` | All-in-one automated setup |
| `DATA_PREPARATION.md` | Comprehensive documentation |

---

## File Outputs

After running setup:
- `dataset/pretrain_data.jsonl` (~15GB) - For pretraining
- `dataset/sft_data_512.jsonl` - For 512-token SFT
- `dataset/sft_data_1024.jsonl` - For 1024-token SFT

---

## Customization

### Different Pretrain Size
```bash
python scripts/prepare_pretrain_data.py --max_gb 5   # 5GB
python scripts/prepare_pretrain_data.py --max_gb 30  # 30GB
```

### Different Token Lengths
```bash
python scripts/prepare_sft_data.py --min_tokens 30 --max_tokens 256   # Short
python scripts/prepare_sft_data.py --min_tokens 100 --max_tokens 2048 # Long
```

---

## Troubleshooting

**Problem**: `huggingface-cli: command not found`
```bash
pip install -U huggingface_hub
```

**Problem**: Dataset download too slow
```bash
# Use mirror or download manually from Hugging Face website
```

**Problem**: Out of disk space
```bash
# Reduce pretrain size
python scripts/prepare_pretrain_data.py --max_gb 5
```

---

For detailed documentation, see `DATA_PREPARATION.md`.

