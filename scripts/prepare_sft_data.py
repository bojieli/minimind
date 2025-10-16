#!/root/miniconda3/bin/python
"""
Prepare SFT (Supervised Fine-Tuning) data from smoltalk-chinese dataset with parallel processing.
Extracts conversation rounds (user-assistant) before reaching token limit.
If first round exceeds limit, discards the entire sample.
"""

import os
import sys
import json
import glob
import warnings
import multiprocessing as mp
from queue import Empty
from pathlib import Path

try:
    import pandas as pd
    import pyarrow.parquet as pq
    HAS_PARQUET = True
except ImportError:
    HAS_PARQUET = False
    print("Error: pandas/pyarrow required. Install with: pip install pandas pyarrow")
    sys.exit(1)

try:
    from transformers import AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("Error: transformers required. Install with: pip install transformers")
    sys.exit(1)


def extract_conversation_rounds(conversations, tokenizer, max_tokens=512):
    """
    提取对话轮次，直到达到token限制
    如果第一轮就超过限制，返回None表示丢弃整个样本
    
    Args:
        conversations: List of conversation turns
        tokenizer: Tokenizer instance
        max_tokens: Maximum tokens allowed
    
    Returns:
        List of valid conversation turns, or None if first round exceeds limit
    """
    if not conversations or len(conversations) < 2:
        return None
    
    valid_rounds = []
    current_tokens = 0
    
    i = 0
    while i < len(conversations) - 1:
        # Ensure we have a user-assistant pair
        if i + 1 >= len(conversations):
            break
        
        user_turn = conversations[i]
        assistant_turn = conversations[i + 1]
        
        # Verify roles
        user_role = user_turn.get('from', user_turn.get('role', ''))
        assistant_role = assistant_turn.get('from', assistant_turn.get('role', ''))
        
        if user_role not in ['user', 'human'] or assistant_role not in ['assistant', 'gpt', 'model']:
            i += 1
            continue
        
        # Get content
        user_content = user_turn.get('value', user_turn.get('content', ''))
        assistant_content = assistant_turn.get('value', assistant_turn.get('content', ''))
        
        if not user_content or not assistant_content:
            i += 2
            continue
        
        # Count tokens for this round
        round_text = f"user: {user_content}\nassistant: {assistant_content}"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            round_tokens = len(tokenizer.encode(round_text, add_special_tokens=False))
        
        # If this is the first round and it exceeds limit, discard entire sample
        if len(valid_rounds) == 0 and current_tokens + round_tokens > max_tokens:
            return None
        
        # If adding this round would exceed limit, stop
        if current_tokens + round_tokens > max_tokens:
            break
        
        # Add this round
        valid_rounds.extend([user_turn, assistant_turn])
        current_tokens += round_tokens
        i += 2
    
    # Must have at least one complete round
    return valid_rounds if len(valid_rounds) >= 2 else None


def process_sample_worker(args):
    """
    Worker function to process a single sample
    每个worker处理一个样本，返回处理结果
    """
    sample, tokenizer_path, max_tokens, source_file = args
    
    import logging
    logging.getLogger("transformers").setLevel(logging.ERROR)
    
    # Load tokenizer once per worker
    if not hasattr(process_sample_worker, 'tokenizer'):
        process_sample_worker.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    tokenizer = process_sample_worker.tokenizer
    
    try:
        # Extract conversations
        conversations = sample.get('conversations', [])
        if not conversations:
            return {'status': 'skip', 'reason': 'no_conversations'}
        
        # Extract valid rounds
        valid_rounds = extract_conversation_rounds(conversations, tokenizer, max_tokens)
        
        if valid_rounds is None:
            return {'status': 'skip', 'reason': 'first_round_too_long'}
        
        # Create output sample
        output_sample = {
            'conversations': valid_rounds
        }
        
        # Add metadata
        for key in ['system_prompt_key', 'magpie_model', 'difficulty', 'score', 'classify']:
            if key in sample:
                output_sample[key] = sample[key]
        
        # Calculate token count
        text_parts = []
        for turn in valid_rounds:
            role = turn.get('from', turn.get('role', ''))
            content = turn.get('value', turn.get('content', ''))
            text_parts.append(f"{role}: {content}")
        full_text = '\n'.join(text_parts)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            token_count = len(tokenizer.encode(full_text, add_special_tokens=False))
        
        output_sample['n_turn'] = len(valid_rounds) / 2
        output_sample['_token_count'] = token_count
        output_sample['_source_file'] = source_file
        
        return {'status': 'success', 'sample': output_sample}
        
    except Exception as e:
        return {'status': 'error', 'reason': str(e)}


def prepare_sft_parallel(input_dir, output_file, tokenizer_path,
                         min_tokens=50, max_tokens=512, num_workers=None):
    """
    Prepare SFT data with parallel processing using queue pattern
    
    Args:
        input_dir: Directory with parquet files
        output_file: Output JSONL file
        tokenizer_path: Path to tokenizer
        min_tokens: Minimum token count (not used in round extraction)
        max_tokens: Maximum token count per sample
        num_workers: Number of parallel workers (default: auto-detect)
    """
    # Auto-detect workers
    if num_workers is None:
        num_workers = min(mp.cpu_count(), 128)
    
    print(f"Preparing SFT Data with Parallel Processing")
    print(f"="*60)
    print(f"Input dir: {input_dir}")
    print(f"Output: {output_file}")
    print(f"Max tokens: {max_tokens}")
    print(f"Parallel workers: {num_workers}")
    print(f"="*60)
    
    # Find parquet files
    parquet_files = glob.glob(os.path.join(input_dir, "*.parquet"))
    if not parquet_files:
        print(f"✗ No parquet files found in {input_dir}")
        return
    
    print(f"Found {len(parquet_files)} parquet files")
    print(f"\nProcessing samples with {num_workers} parallel workers...")
    
    total_samples = 0
    kept_samples = 0
    skipped_first_round = 0
    skipped_no_conv = 0
    errors = 0
    
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for file_idx, parquet_file in enumerate(sorted(parquet_files), 1):
            file_name = os.path.basename(parquet_file)
            print(f"\n[{file_idx}/{len(parquet_files)}] Processing {file_name}...")
            
            try:
                df = pd.read_parquet(parquet_file)
                print(f"  Loaded {len(df):,} samples")
                
                # Prepare arguments for parallel processing
                worker_args = []
                for _, row in df.iterrows():
                    sample = row.to_dict()
                    # Convert numpy types to native Python types
                    sample_cleaned = {}
                    for key, value in sample.items():
                        if hasattr(value, 'tolist'):  # numpy array
                            sample_cleaned[key] = value.tolist()
                        elif hasattr(value, 'item'):  # numpy scalar
                            sample_cleaned[key] = value.item()
                        else:
                            sample_cleaned[key] = value
                    
                    worker_args.append((sample_cleaned, tokenizer_path, max_tokens, file_name))
                
                # Process in parallel
                file_kept = 0
                file_skipped_first = 0
                file_skipped_no_conv = 0
                file_errors = 0
                
                with mp.Pool(processes=num_workers) as pool:
                    # Process in batches for progress tracking
                    batch_size = 1000
                    for batch_start in range(0, len(worker_args), batch_size):
                        batch_end = min(batch_start + batch_size, len(worker_args))
                        batch = worker_args[batch_start:batch_end]
                        
                        results = pool.map(process_sample_worker, batch)
                        
                        for result in results:
                            total_samples += 1
                            
                            if result['status'] == 'success':
                                outfile.write(json.dumps(result['sample'], ensure_ascii=False) + '\n')
                                kept_samples += 1
                                file_kept += 1
                            elif result['status'] == 'skip':
                                if result['reason'] == 'first_round_too_long':
                                    skipped_first_round += 1
                                    file_skipped_first += 1
                                elif result['reason'] == 'no_conversations':
                                    skipped_no_conv += 1
                                    file_skipped_no_conv += 1
                            elif result['status'] == 'error':
                                errors += 1
                                file_errors += 1
                        
                        # Progress update
                        if batch_end % 10000 == 0 or batch_end == len(worker_args):
                            print(f"  Progress: {batch_end:,}/{len(worker_args):,} "
                                  f"(kept: {file_kept:,}, skipped: {file_skipped_first + file_skipped_no_conv:,})",
                                  end='\r', flush=True)
                
                print(f"\n  ✓ File complete: kept {file_kept:,}/{len(df):,} ({file_kept/len(df)*100:.1f}%)")
                print(f"    - Skipped (first round too long): {file_skipped_first:,}")
                print(f"    - Skipped (no conversations): {file_skipped_no_conv:,}")
                if file_errors > 0:
                    print(f"    - Errors: {file_errors:,}")
                
            except Exception as e:
                print(f"  ✗ Error loading file: {e}")
                continue
    
    # Summary
    print(f"\n\n{'='*60}")
    print(f"SFT DATA PREPARATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total samples processed:        {total_samples:,}")
    print(f"Samples kept:                   {kept_samples:,}")
    print(f"Keep ratio:                     {kept_samples/total_samples*100:.2f}%")
    print(f"\nSkipped reasons:")
    print(f"  First round too long:         {skipped_first_round:,}")
    print(f"  No conversations:             {skipped_no_conv:,}")
    if errors > 0:
        print(f"  Errors:                       {errors:,}")
    
    file_size_mb = os.path.getsize(output_file) / (1024**2)
    print(f"\nOutput file: {output_file}")
    print(f"Output size: {file_size_mb:.2f}MB")
    print(f"\n✓ SFT data ready with parallel processing!")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare SFT data from smoltalk-chinese (parallel)")
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
        help="Minimum token count (not used in round extraction)"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=512,
        help="Maximum token count per sample"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: auto-detect, max 128)"
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
    
    prepare_sft_parallel(
        input_dir=args.input_dir,
        output_file=args.output_file,
        tokenizer_path=args.tokenizer_path,
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
        num_workers=args.num_workers
    )


if __name__ == "__main__":
    main()
