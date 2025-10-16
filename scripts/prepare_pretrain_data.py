#!/root/miniconda3/bin/python
"""
Prepare pretrain data from various sources.
Converts parquet/other formats to MiniMind's JSONL format.
Includes size limiting to avoid overly large datasets.
"""

import os
import sys
import json
import glob
from pathlib import Path

try:
    import pandas as pd
    import pyarrow.parquet as pq
except ImportError as e:
    print(f"Error: Required libraries not found: {e}")
    print("Please install: pip install pandas pyarrow")
    sys.exit(1)


def convert_fineweb_to_pretrain(input_dir, output_file, max_gb=15, min_length=10):
    """
    Convert Fineweb-Edu-Chinese parquet files to MiniMind pretrain format.
    Stops when reaching target size.
    
    Args:
        input_dir: Directory containing parquet files
        output_file: Output JSONL file path
        max_gb: Maximum output file size in GB
        min_length: Minimum text length (characters) to keep
    """
    parquet_files = sorted(glob.glob(os.path.join(input_dir, "*.parquet")))
    
    if not parquet_files:
        print(f"✗ No parquet files found in {input_dir}")
        return
    
    max_bytes = max_gb * 1024 * 1024 * 1024
    current_bytes = 0
    
    print(f"Converting Fineweb to Pretrain Format")
    print(f"="*60)
    print(f"Input dir: {input_dir}")
    print(f"Output: {output_file}")
    print(f"Max size: {max_gb}GB")
    print(f"Min text length: {min_length} chars")
    print(f"Found {len(parquet_files)} parquet files")
    print(f"="*60)
    
    total_samples = 0
    kept_samples = 0
    
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for i, parquet_file in enumerate(parquet_files, 1):
            if current_bytes >= max_bytes:
                print(f"\n✓ Reached target size of {max_gb}GB, stopping.")
                break
            
            file_name = os.path.basename(parquet_file)
            print(f"[{i}/{len(parquet_files)}] Processing {file_name}...", end=" ", flush=True)
            
            try:
                df = pd.read_parquet(parquet_file)
                file_kept = 0
                
                for _, row in df.iterrows():
                    if current_bytes >= max_bytes:
                        break
                    
                    total_samples += 1
                    text = row.get('text', '')
                    
                    # Filter by length
                    if not text or len(text) < min_length:
                        continue
                    
                    # Write in MiniMind pretrain format
                    output_record = {"text": text}
                    line = json.dumps(output_record, ensure_ascii=False) + '\n'
                    line_bytes = len(line.encode('utf-8'))
                    
                    outfile.write(line)
                    current_bytes += line_bytes
                    kept_samples += 1
                    file_kept += 1
                
                current_gb = current_bytes / (1024**3)
                print(f"kept {file_kept}/{len(df)} ({current_gb:.2f}GB so far)")
                
            except Exception as e:
                print(f"✗ Error: {e}")
                continue
    
    # Summary
    final_gb = current_bytes / (1024**3)
    print(f"\n{'='*60}")
    print(f"CONVERSION SUMMARY")
    print(f"{'='*60}")
    print(f"Total samples processed: {total_samples:,}")
    print(f"Samples kept:            {kept_samples:,}")
    print(f"Output size:             {final_gb:.2f}GB")
    print(f"Output file:             {output_file}")
    print(f"\n✓ Pretrain data ready!")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare pretrain data from Fineweb")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="/root/minimind/dataset/fineweb_edu_chinese_4_5/4_5",
        help="Input directory with parquet files"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="/root/minimind/dataset/pretrain_data.jsonl",
        help="Output JSONL file path"
    )
    parser.add_argument(
        "--max_gb",
        type=float,
        default=15,
        help="Maximum output file size in GB"
    )
    parser.add_argument(
        "--min_length",
        type=int,
        default=10,
        help="Minimum text length in characters"
    )
    
    args = parser.parse_args()
    
    # Check input exists
    if not os.path.exists(args.input_dir):
        print(f"✗ Error: Input directory not found: {args.input_dir}")
        print(f"\nTo download Fineweb-Edu-Chinese-4_5 dataset:")
        print(f"  huggingface-cli download opencsg/Fineweb-Edu-Chinese-V2.1 \\")
        print(f"    --repo-type dataset \\")
        print(f"    --include \"4_5/*\" \\")
        print(f"    --local-dir {args.input_dir.rsplit('/', 1)[0]}")
        sys.exit(1)
    
    # Create output directory
    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    convert_fineweb_to_pretrain(
        input_dir=args.input_dir,
        output_file=args.output_file,
        max_gb=args.max_gb,
        min_length=args.min_length
    )


if __name__ == "__main__":
    main()

