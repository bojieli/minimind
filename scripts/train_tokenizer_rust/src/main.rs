use anyhow::{Context, Result};
use clap::Parser;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use tokenizers::models::bpe::{BpeTrainerBuilder, BPE};
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::{AddedToken, DecoderWrapper, Model, NormalizerWrapper, PostProcessorWrapper, PreTokenizerWrapper, TokenizerBuilder};

/// High-performance BPE tokenizer trainer for JSONL datasets
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to input JSONL file (each line: {"text": "..."})
    #[arg(short, long, default_value = "../../dataset/pretrain_data.jsonl")]
    data_path: String,

    /// Output directory for trained tokenizer files
    #[arg(short, long, default_value = "../../model")]
    output_dir: String,

    /// Target vocabulary size
    #[arg(short, long, default_value_t = 6400)]
    vocab_size: usize,

    /// Number of threads to use (default: all available cores)
    #[arg(short, long)]
    threads: Option<usize>,

    /// Maximum data size to load in GB (default: 1GB)
    #[arg(short = 'm', long, default_value_t = 1.0)]
    max_data_gb: f64,
}

#[derive(Debug, Deserialize)]
struct JsonlRecord {
    text: String,
}

#[derive(Debug, Serialize)]
struct TokenizerConfig {
    add_bos_token: bool,
    add_eos_token: bool,
    add_prefix_space: bool,
    added_tokens_decoder: serde_json::Value,
    additional_special_tokens: Vec<String>,
    bos_token: String,
    clean_up_tokenization_spaces: bool,
    eos_token: String,
    legacy: bool,
    model_max_length: usize,
    pad_token: String,
    sp_model_kwargs: serde_json::Value,
    spaces_between_special_tokens: bool,
    tokenizer_class: String,
    unk_token: String,
    chat_template: String,
}

/// Read texts from JSONL file in parallel chunks for better I/O performance
/// Stops reading after reaching max_bytes to limit memory usage and training time
fn read_texts_from_jsonl<P: AsRef<Path>>(path: P, max_bytes: usize) -> Result<Vec<String>> {
    let file = File::open(path.as_ref())
        .with_context(|| format!("Failed to open file: {:?}", path.as_ref()))?;
    
    let reader = BufReader::with_capacity(8 * 1024 * 1024, file); // 8MB buffer
    
    // Read lines until we hit the size limit (fast sequential I/O)
    let mut lines = Vec::new();
    let mut total_bytes = 0usize;
    
    for line in reader.lines() {
        let line = line.context("Failed to read line from file")?;
        let line_bytes = line.len();
        
        // Check if adding this line would exceed the limit
        if total_bytes + line_bytes > max_bytes && !lines.is_empty() {
            println!("Reached size limit of {:.2} GB, stopping read", max_bytes as f64 / 1_073_741_824.0);
            break;
        }
        
        total_bytes += line_bytes;
        lines.push(line);
    }
    
    println!("Read {} lines ({:.2} GB), parsing JSON in parallel...", 
             lines.len(), 
             total_bytes as f64 / 1_073_741_824.0);
    
    // Parse JSON in parallel using all available cores
    let texts: Vec<String> = lines
        .par_iter()
        .map(|line| {
            serde_json::from_str::<JsonlRecord>(line)
                .map(|record| record.text)
                .unwrap_or_else(|_| String::new())
        })
        .filter(|text| !text.is_empty())
        .collect();
    
    let total_text_bytes: usize = texts.iter().map(|s| s.len()).sum();
    println!("Parsed {} valid text records ({:.2} GB of text)", 
             texts.len(),
             total_text_bytes as f64 / 1_073_741_824.0);
    
    Ok(texts)
}

fn train_tokenizer(
    data_path: &str,
    output_dir: &str,
    vocab_size: usize,
    max_data_gb: f64,
) -> Result<()> {
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║          BPE Tokenizer Training (Rust Implementation)         ║");
    println!("╚════════════════════════════════════════════════════════════════╝");
    println!();
    println!("Configuration:");
    println!("  • Data path:    {}", data_path);
    println!("  • Output dir:   {}", output_dir);
    println!("  • Vocab size:   {}", vocab_size);
    println!("  • Max data:     {:.2} GB", max_data_gb);
    println!("  • Threads:      {}", rayon::current_num_threads());
    println!();
    
    // Convert GB to bytes
    let max_bytes = (max_data_gb * 1_073_741_824.0) as usize;
    
    // Read texts from JSONL file
    println!("Phase 1/3: Reading and parsing JSONL (multi-threaded)");
    println!("─────────────────────────────────────────────────────────────");
    let texts = read_texts_from_jsonl(data_path, max_bytes)?;
    
    if texts.is_empty() {
        anyhow::bail!("No valid text data found in input file");
    }
    
    println!("Loaded {} texts for training", texts.len());
    
    // Define special tokens (order matters - they'll be assigned IDs 0, 1, 2)
    let special_tokens = vec![
        AddedToken::from("<|endoftext|>", true),
        AddedToken::from("<|im_start|>", true),
        AddedToken::from("<|im_end|>", true),
    ];
    
    // Create BPE trainer
    let mut trainer = BpeTrainerBuilder::new()
        .vocab_size(vocab_size)
        .show_progress(true)
        .special_tokens(special_tokens.clone())
        .initial_alphabet(ByteLevel::alphabet())
        .build();
    
    // Initialize tokenizer with BPE model
    let mut tokenizer = TokenizerBuilder::new()
        .with_model(BPE::default())
        .with_normalizer(None::<NormalizerWrapper>)
        .with_pre_tokenizer(Some(PreTokenizerWrapper::ByteLevel(
            ByteLevel::new(false, false, false) // add_prefix_space=false
        )))
        .with_post_processor(None::<PostProcessorWrapper>)
        .with_decoder(Some(DecoderWrapper::ByteLevel(ByteLevel::default())))
        .build()
        .map_err(|e| anyhow::anyhow!("Failed to build tokenizer: {}", e))?;
    
    // Train the tokenizer
    println!("Training tokenizer on {} texts...", texts.len());
    
    // Write texts to temporary files for training (tokenizers API requires files)
    let temp_dir = std::env::temp_dir();
    let temp_files: Vec<String> = texts
        .par_chunks(10000)
        .enumerate()
        .map(|(idx, chunk)| {
            let temp_path = temp_dir.join(format!("train_chunk_{}.txt", idx));
            let mut file = std::fs::File::create(&temp_path)
                .expect("Failed to create temp file");
            for text in chunk {
                writeln!(file, "{}", text).expect("Failed to write to temp file");
            }
            temp_path.to_string_lossy().to_string()
        })
        .collect();
    
    tokenizer
        .train_from_files(&mut trainer, temp_files.clone())
        .map_err(|e| anyhow::anyhow!("Failed to train tokenizer: {}", e))?;
    
    // Clean up temp files
    for temp_file in temp_files {
        let _ = std::fs::remove_file(temp_file);
    }
    
    // Verify special token IDs
    let endoftext_id = tokenizer.token_to_id("<|endoftext|>")
        .context("Failed to find <|endoftext|> token")?;
    let im_start_id = tokenizer.token_to_id("<|im_start|>")
        .context("Failed to find <|im_start|> token")?;
    let im_end_id = tokenizer.token_to_id("<|im_end|>")
        .context("Failed to find <|im_end|> token")?;
    
    println!("Special token IDs:");
    println!("  <|endoftext|>: {}", endoftext_id);
    println!("  <|im_start|>: {}", im_start_id);
    println!("  <|im_end|>: {}", im_end_id);
    
    assert_eq!(endoftext_id, 0, "Expected <|endoftext|> to have ID 0");
    assert_eq!(im_start_id, 1, "Expected <|im_start|> to have ID 1");
    assert_eq!(im_end_id, 2, "Expected <|im_end|> to have ID 2");
    
    // Create output directory
    std::fs::create_dir_all(output_dir)
        .with_context(|| format!("Failed to create output directory: {}", output_dir))?;
    
    // Save tokenizer
    let tokenizer_path = format!("{}/tokenizer.json", output_dir);
    tokenizer
        .save(&tokenizer_path, false)
        .map_err(|e| anyhow::anyhow!("Failed to save tokenizer to {}: {}", tokenizer_path, e))?;
    
    println!("Tokenizer saved to: {}", tokenizer_path);
    
    // Save vocab and merges files (for compatibility)
    let vocab_path = format!("{}/vocab.json", output_dir);
    let merges_path = format!("{}/merges.txt", output_dir);
    tokenizer.get_model().save(Path::new(output_dir), Some(""))
        .map_err(|e| anyhow::anyhow!("Failed to save vocab and merges: {}", e))?;
    
    println!("Vocab saved to: {}", vocab_path);
    println!("Merges saved to: {}", merges_path);
    
    // Create tokenizer config file
    let config = create_tokenizer_config();
    let config_path = format!("{}/tokenizer_config.json", output_dir);
    let config_file = File::create(&config_path)
        .with_context(|| format!("Failed to create config file: {}", config_path))?;
    serde_json::to_writer_pretty(config_file, &config)
        .context("Failed to write config file")?;
    
    println!("Config saved to: {}", config_path);
    println!("Tokenizer training completed successfully!");
    
    Ok(())
}

fn create_tokenizer_config() -> TokenizerConfig {
    let added_tokens_decoder = serde_json::json!({
        "0": {
            "content": "<|endoftext|>",
            "lstrip": false,
            "normalized": false,
            "rstrip": false,
            "single_word": false,
            "special": true
        },
        "1": {
            "content": "<|im_start|>",
            "lstrip": false,
            "normalized": false,
            "rstrip": false,
            "single_word": false,
            "special": true
        },
        "2": {
            "content": "<|im_end|>",
            "lstrip": false,
            "normalized": false,
            "rstrip": false,
            "single_word": false,
            "special": true
        }
    });
    
    let chat_template = "{% if messages[0]['role'] == 'system' %}{% set system_message = messages[0]['content'] %}{{ '<|im_start|>system\\n' + system_message + '<|im_end|>\\n' }}{% else %}{{ '<|im_start|>system\\nYou are a helpful assistant<|im_end|>\\n' }}{% endif %}{% for message in messages %}{% set content = message['content'] %}{% if message['role'] == 'user' %}{{ '<|im_start|>user\\n' + content + '<|im_end|>\\n<|im_start|>assistant\\n' }}{% elif message['role'] == 'assistant' %}{{ content + '<|im_end|>' + '\\n' }}{% endif %}{% endfor %}".to_string();
    
    TokenizerConfig {
        add_bos_token: false,
        add_eos_token: false,
        add_prefix_space: false,
        added_tokens_decoder,
        additional_special_tokens: vec![],
        bos_token: "<|im_start|>".to_string(),
        clean_up_tokenization_spaces: false,
        eos_token: "<|im_end|>".to_string(),
        legacy: true,
        model_max_length: 32768,
        pad_token: "<|endoftext|>".to_string(),
        sp_model_kwargs: serde_json::json!({}),
        spaces_between_special_tokens: false,
        tokenizer_class: "PreTrainedTokenizerFast".to_string(),
        unk_token: "<|endoftext|>".to_string(),
        chat_template,
    }
}

fn main() -> Result<()> {
    // Parse command-line arguments
    let args = Args::parse();
    
    // Configure rayon thread pool if specified
    if let Some(threads) = args.threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()
            .context("Failed to configure thread pool")?;
    }
    
    let num_threads = rayon::current_num_threads();
    println!("Using {} threads for parallel processing", num_threads);
    println!();
    
    // Train tokenizer with provided arguments
    train_tokenizer(&args.data_path, &args.output_dir, args.vocab_size, args.max_data_gb)?;
    
    Ok(())
}

