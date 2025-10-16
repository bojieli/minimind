#!/root/miniconda3/bin/python
"""
Prepare pretrain data from various sources with parallel chunking.
Converts parquet/other formats to MiniMind's JSONL format.
Automatically chunks long texts to maximize data utilization.
"""

import os
import sys
import json
import glob
import warnings
import multiprocessing as mp
from pathlib import Path
from functools import partial

try:
    import pandas as pd
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
except ImportError as e:
    print(f"Error: Required libraries not found: {e}")
    print("Please install: pip install pandas pyarrow transformers")
    sys.exit(1)


def chunk_text(text, tokenizer, max_length=512, min_chunk_length=50):
    """
    将长文本分块为多个小段
    
    Args:
        text: 输入文本
        tokenizer: 分词器
        max_length: 最大tokens长度
        min_chunk_length: 最小chunk长度
    
    Returns:
        List of text chunks
    """
    if not text or len(text) < min_chunk_length:
        return []
    
    # 关闭警告
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tokens = tokenizer(str(text), add_special_tokens=False, truncation=False).input_ids
    
    chunks = []
    chunk_size = max_length - 1  # 留一个位置给padding/special token
    
    for i in range(0, len(tokens), chunk_size):
        chunk_tokens = tokens[i:i + chunk_size]
        if len(chunk_tokens) >= min_chunk_length:
            chunk_text = tokenizer.decode(chunk_tokens)
            chunks.append(chunk_text)
    
    return chunks


def process_parquet_file(args):
    """
    并行处理函数：处理单个parquet文件
    每个worker处理一个完整的parquet文件，返回所有chunks
    """
    parquet_file, tokenizer_path, max_length, min_chunk_length, min_text_length = args
    
    import logging
    logging.getLogger("transformers").setLevel(logging.ERROR)
    
    # 每个进程加载自己的tokenizer
    if not hasattr(process_parquet_file, 'tokenizer'):
        process_parquet_file.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    tokenizer = process_parquet_file.tokenizer
    
    try:
        df = pd.read_parquet(parquet_file)
        all_chunks = []
        texts_processed = 0
        
        for _, row in df.iterrows():
            text = row.get('text', '')
            if text and len(text) >= min_text_length:
                texts_processed += 1
                chunks = chunk_text(text, tokenizer, max_length, min_chunk_length)
                all_chunks.extend(chunks)
        
        return {
            'file': os.path.basename(parquet_file),
            'chunks': all_chunks,
            'texts_processed': texts_processed,
            'chunks_generated': len(all_chunks)
        }
    except Exception as e:
        return {
            'file': os.path.basename(parquet_file),
            'error': str(e),
            'chunks': [],
            'texts_processed': 0,
            'chunks_generated': 0
        }


def convert_fineweb_to_pretrain_chunked(
    input_dir, 
    output_file, 
    max_gb=15, 
    min_length=10,
    max_seq_len=512,
    min_chunk_length=50,
    num_workers=None,
    tokenizer_path=None
):
    """
    Convert Fineweb-Edu-Chinese parquet files to MiniMind pretrain format with chunking.
    
    Args:
        input_dir: Directory containing parquet files
        output_file: Output JSONL file path
        max_gb: Maximum output file size in GB
        min_length: Minimum text length (characters) to keep before chunking
        max_seq_len: Maximum sequence length for chunking (tokens)
        min_chunk_length: Minimum chunk length (tokens)
        num_workers: Number of parallel workers (default: CPU count)
        tokenizer_path: Path to tokenizer (default: ../model/)
    """
    # Auto-detect workers
    if num_workers is None:
        num_workers = min(mp.cpu_count(), 128)
    
    # Default tokenizer path
    if tokenizer_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        tokenizer_path = os.path.join(script_dir, '../model/')
    
    parquet_files = sorted(glob.glob(os.path.join(input_dir, "*.parquet")))
    
    if not parquet_files:
        print(f"✗ No parquet files found in {input_dir}")
        return
    
    max_bytes = max_gb * 1024 * 1024 * 1024
    current_bytes = 0
    
    print(f"Converting Fineweb to Pretrain Format (with Chunking)")
    print(f"="*60)
    print(f"Input dir: {input_dir}")
    print(f"Output: {output_file}")
    print(f"Max size: {max_gb}GB")
    print(f"Min text length: {min_length} chars")
    print(f"Max seq length: {max_seq_len} tokens")
    print(f"Min chunk length: {min_chunk_length} tokens")
    print(f"Parallel workers: {num_workers}")
    print(f"Found {len(parquet_files)} parquet files")
    print(f"="*60)
    
    total_texts = 0
    total_chunks = 0
    files_processed = 0
    
    # Prepare arguments for parallel processing
    worker_args = [
        (pf, tokenizer_path, max_seq_len, min_chunk_length, min_length)
        for pf in parquet_files
    ]
    
    print(f"\nProcessing {len(parquet_files)} parquet files with {num_workers} parallel workers...")
    print("(Each worker processes a complete parquet file)")
    
    with open(output_file, 'w', encoding='utf-8') as outfile:
        # Process files in batches of num_workers
        with mp.Pool(processes=num_workers) as pool:
            # Use imap for streaming results
            for result in pool.imap(process_parquet_file, worker_args):
                if current_bytes >= max_bytes:
                    print(f"\n✓ Reached target size of {max_gb}GB, stopping.")
                    break
                
                files_processed += 1
                
                # Handle errors
                if 'error' in result:
                    print(f"\r[{files_processed}/{len(parquet_files)}] {result['file']}: ✗ Error: {result['error']}")
                    continue
                
                # Write chunks to file
                file_chunks_written = 0
                for chunk_text in result['chunks']:
                    if current_bytes >= max_bytes:
                        break
                    
                    output_record = {"text": chunk_text}
                    line = json.dumps(output_record, ensure_ascii=False) + '\n'
                    line_bytes = len(line.encode('utf-8'))
                    
                    outfile.write(line)
                    current_bytes += line_bytes
                    file_chunks_written += 1
                
                total_texts += result['texts_processed']
                total_chunks += file_chunks_written
                
                # Progress update
                current_gb = current_bytes / (1024**3)
                expansion = file_chunks_written / result['texts_processed'] if result['texts_processed'] > 0 else 0
                print(f"\r[{files_processed}/{len(parquet_files)}] {result['file']}: "
                      f"{file_chunks_written:,} chunks from {result['texts_processed']:,} texts "
                      f"({expansion:.1f}x) | Total: {total_chunks:,} chunks, {current_gb:.2f}GB", 
                      end='', flush=True)
                
                if files_processed % 10 == 0:
                    print()  # New line every 10 files
    
    # Summary
    final_gb = current_bytes / (1024**3)
    print(f"\n\n{'='*60}")
    print(f"CONVERSION SUMMARY")
    print(f"{'='*60}")
    print(f"Files processed: {files_processed:,}")
    print(f"Texts processed: {total_texts:,}")
    print(f"Chunks generated: {total_chunks:,}")
    print(f"Expansion rate: {total_chunks/total_texts if total_texts > 0 else 0:.2f}x")
    print(f"Output size: {final_gb:.2f}GB")
    print(f"Output file: {output_file}")
    print(f"\n✓ Pretrain data ready with parallel chunking!")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare pretrain data from Fineweb with chunking")
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
        help="Minimum text length in characters before chunking"
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=512,
        help="Maximum sequence length for chunks (tokens)"
    )
    parser.add_argument(
        "--min_chunk_length",
        type=int,
        default=50,
        help="Minimum chunk length (tokens)"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: auto-detect)"
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=None,
        help="Path to tokenizer directory (default: ../model/)"
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
    
    convert_fineweb_to_pretrain_chunked(
        input_dir=args.input_dir,
        output_file=args.output_file,
        max_gb=args.max_gb,
        min_length=args.min_length,
        max_seq_len=args.max_seq_len,
        min_chunk_length=args.min_chunk_length,
        num_workers=args.num_workers,
        tokenizer_path=args.tokenizer_path
    )


if __name__ == "__main__":
    main()
