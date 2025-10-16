#!/root/miniconda3/bin/python
"""
Prepare SFT (Supervised Fine-Tuning) data from smoltalk-chinese dataset.
Filters samples by token count to match the target training sequence length.
"""

import os
import sys
import json
import glob
from pathlib import Path

try:
    import pandas as pd
    import pyarrow.parquet as pq
    HAS_PARQUET = True
except ImportError:
    HAS_PARQUET = False
    print("Warning: pandas/pyarrow not available")

try:
    from transformers import AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("Error: transformers not available. Install with: pip install transformers")
    sys.exit(1)


def count_tokens(text, tokenizer):
    """Count tokens in text."""
    if not text or not isinstance(text, str):
        return 0
    tokens = tokenizer.encode(text, add_special_tokens=False)
    return len(tokens)


def format_conversation(sample):
    """Format sample into conversation string."""
    # Try different formats
    if 'conversations' in sample:
        conversations = sample['conversations']
        text_parts = []
        for turn in conversations:
            role = turn.get('from', turn.get('role', ''))
            content = turn.get('value', turn.get('content', ''))
            if role and content:
                text_parts.append(f"{role}: {content}")
        return '\n'.join(text_parts)
    
    elif 'messages' in sample:
        messages = sample['messages']
        text_parts = []
        for msg in messages:
            role = msg.get('role', '')
            content = msg.get('content', '')
            if role and content:
                text_parts.append(f"{role}: {content}")
        return '\n'.join(text_parts)
    
    elif 'instruction' in sample and 'output' in sample:
        instruction = sample.get('instruction', '')
        input_text = sample.get('input', '')
        output = sample.get('output', '')
        
        if input_text:
            return f"instruction: {instruction}\ninput: {input_text}\noutput: {output}"
        else:
            return f"instruction: {instruction}\noutput: {output}"
    
    elif 'text' in sample:
        return sample['text']
    
    else:
        # Concatenate all string values
        text_parts = []
        for key, value in sample.items():
            if isinstance(value, str) and value.strip():
                text_parts.append(value)
        return '\n'.join(text_parts)


def prepare_sft_from_smoltalk(input_dir, output_file, tokenizer_path, 
                               min_tokens=50, max_tokens=512):
    """
    Prepare SFT data from smoltalk-chinese parquet files.
    
    Args:
        input_dir: Directory with parquet files
        output_file: Output JSONL file
        tokenizer_path: Path to tokenizer
        min_tokens: Minimum token count
        max_tokens: Maximum token count
    """
    if not HAS_PARQUET:
        print("Error: pandas/pyarrow required. Install with: pip install pandas pyarrow")
        sys.exit(1)
    
    print(f"Loading tokenizer from {tokenizer_path}...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    print(f"✓ Tokenizer loaded (vocab size: {tokenizer.vocab_size})")
    
    # Find parquet files
    parquet_files = glob.glob(os.path.join(input_dir, "*.parquet"))
    if not parquet_files:
        print(f"✗ No parquet files found in {input_dir}")
        return
    
    print(f"\nPreparing SFT Data")
    print(f"="*60)
    print(f"Input dir: {input_dir}")
    print(f"Output: {output_file}")
    print(f"Token range: {min_tokens}-{max_tokens}")
    print(f"Found {len(parquet_files)} parquet files")
    print(f"="*60)
    
    total_samples = 0
    kept_samples = 0
    token_stats = []
    
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for i, parquet_file in enumerate(sorted(parquet_files), 1):
            file_name = os.path.basename(parquet_file)
            print(f"[{i}/{len(parquet_files)}] {file_name}...", end=" ", flush=True)
            
            try:
                df = pd.read_parquet(parquet_file)
                file_kept = 0
                
                for _, row in df.iterrows():
                    total_samples += 1
                    sample = row.to_dict()
                    
                    # Format conversation
                    text = format_conversation(sample)
                    
                    # Count tokens
                    num_tokens = count_tokens(text, tokenizer)
                    token_stats.append(num_tokens)
                    
                    # Filter by token count
                    if min_tokens <= num_tokens <= max_tokens:
                        # Convert sample to JSON-serializable format
                        output_sample = {}
                        for key, value in sample.items():
                            # Convert numpy types to native Python types
                            if hasattr(value, 'tolist'):  # numpy array
                                output_sample[key] = value.tolist()
                            elif hasattr(value, 'item'):  # numpy scalar
                                output_sample[key] = value.item()
                            else:
                                output_sample[key] = value
                        
                        output_sample['_token_count'] = int(num_tokens)
                        output_sample['_source_file'] = file_name
                        
                        outfile.write(json.dumps(output_sample, ensure_ascii=False) + '\n')
                        kept_samples += 1
                        file_kept += 1
                
                print(f"kept {file_kept}/{len(df)} ({file_kept/len(df)*100:.1f}%)")
                
            except Exception as e:
                print(f"✗ Error: {e}")
                continue
    
    # Summary
    token_stats.sort()
    print(f"\n{'='*60}")
    print(f"SFT DATA SUMMARY")
    print(f"{'='*60}")
    print(f"Total samples:     {total_samples:,}")
    print(f"Samples kept:      {kept_samples:,}")
    print(f"Keep ratio:        {kept_samples/total_samples*100:.2f}%")
    
    if token_stats:
        print(f"\nToken statistics:")
        print(f"  Min:      {min(token_stats)}")
        print(f"  Max:      {max(token_stats)}")
        print(f"  Mean:     {sum(token_stats)/len(token_stats):.1f}")
        print(f"  Median:   {token_stats[len(token_stats)//2]}")
    
    file_size_mb = os.path.getsize(output_file) / (1024**2)
    print(f"\nOutput: {output_file}")
    print(f"Size:   {file_size_mb:.2f}MB")
    print(f"\n✓ SFT data ready!")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare SFT data from smoltalk-chinese")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="/root/minimind/dataset/smoltalk-chinese/data",
        help="Input directory with parquet files"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="/root/minimind/dataset/sft_data_512.jsonl",
        help="Output JSONL file path"
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="/root/minimind/model",
        help="Path to MiniMind tokenizer"
    )
    parser.add_argument(
        "--min_tokens",
        type=int,
        default=50,
        help="Minimum token count"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=512,
        help="Maximum token count"
    )
    
    args = parser.parse_args()
    
    # Check inputs
    if not os.path.exists(args.input_dir):
        print(f"✗ Error: Input directory not found: {args.input_dir}")
        print(f"\nTo download smoltalk-chinese:")
        print(f"  huggingface-cli download opencsg/smoltalk-chinese \\")
        print(f"    --repo-type dataset \\")
        print(f"    --local-dir /root/minimind/dataset/smoltalk-chinese")
        sys.exit(1)
    
    if not os.path.exists(args.tokenizer_path):
        print(f"✗ Error: Tokenizer not found: {args.tokenizer_path}")
        sys.exit(1)
    
    # Create output directory
    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    prepare_sft_from_smoltalk(
        input_dir=args.input_dir,
        output_file=args.output_file,
        tokenizer_path=args.tokenizer_path,
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens
    )


if __name__ == "__main__":
    main()

