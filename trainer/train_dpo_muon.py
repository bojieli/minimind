import os
import sys
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time
import math
import warnings
import torch
import torch.nn.functional as F
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer, AutoModelForCausalLM
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from dataset.lm_dataset import DPODataset
from .muon import Muon, DistMuon

warnings.filterwarnings('ignore')


def Logger(content):
    if not ddp or dist.get_rank() == 0:
        print(content)


def get_lr(current_step, total_steps, lr):
    return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))


def logits_to_probs(logits, labels):
    log_probs = F.log_softmax(logits, dim=2)
    probs = torch.gather(log_probs, dim=2, index=labels.unsqueeze(2)).squeeze(-1)
    return probs


def dpo_loss(ref_probs, probs, mask, beta):
    seq_lengths = mask.sum(dim=1, keepdim=True)
    ref_probs = (ref_probs * mask).sum(dim=1) / seq_lengths.squeeze()
    probs = (probs * mask).sum(dim=1) / seq_lengths.squeeze()

    batch_size = ref_probs.shape[0]
    chosen_ref_probs = ref_probs[:batch_size // 2]
    reject_ref_probs = ref_probs[batch_size // 2:]
    chosen_probs = probs[:batch_size // 2]
    reject_probs = probs[batch_size // 2:]

    pi_logratios = chosen_probs - reject_probs
    ref_logratios = chosen_ref_probs - reject_ref_probs
    logits = pi_logratios - ref_logratios
    loss = -F.logsigmoid(beta * logits)
    return loss.mean()


def train_epoch(epoch, wandb):
    start_time = time.time()
    for step, batch in enumerate(train_loader):
        x_chosen = batch['x_chosen'].to(args.device)
        x_rejected = batch['x_rejected'].to(args.device)
        y_chosen = batch['y_chosen'].to(args.device)
        y_rejected = batch['y_rejected'].to(args.device)
        mask_chosen = batch['mask_chosen'].to(args.device)
        mask_rejected = batch['mask_rejected'].to(args.device)
        x = torch.cat([x_chosen, x_rejected], dim=0)
        y = torch.cat([y_chosen, y_rejected], dim=0)
        mask = torch.cat([mask_chosen, mask_rejected], dim=0)

        # Update learning rates for both optimizers
        lr_muon = get_lr(epoch * iter_per_epoch + step, args.epochs * iter_per_epoch, args.learning_rate_muon)
        lr_adamw = get_lr(epoch * iter_per_epoch + step, args.epochs * iter_per_epoch, args.learning_rate)
        
        for param_group in optimizer_muon.param_groups:
            param_group['lr'] = lr_muon
        for param_group in optimizer_adamw.param_groups:
            param_group['lr'] = lr_adamw

        with ctx:
            with torch.no_grad():
                ref_outputs = ref_model(x)
                ref_logits = ref_outputs.logits
            ref_probs = logits_to_probs(ref_logits, y)
            ref_probs = ref_probs * mask
            outputs = model(x)
            logits = outputs.logits
            probs = logits_to_probs(logits, y)
            probs = probs * mask
            loss = dpo_loss(ref_probs, probs, mask, beta=0.1)
            loss = loss / args.accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % args.accumulation_steps == 0:
            # Update Muon parameters (2D params)
            scaler.unscale_(optimizer_muon)
            torch.nn.utils.clip_grad_norm_(muon_params, args.grad_clip)
            scaler.step(optimizer_muon)
            
            # Update AdamW parameters (1D params, embeddings, norms)
            scaler.unscale_(optimizer_adamw)
            torch.nn.utils.clip_grad_norm_(adamw_params, args.grad_clip)
            scaler.step(optimizer_adamw)
            
            scaler.update()

            optimizer_muon.zero_grad(set_to_none=True)
            optimizer_adamw.zero_grad(set_to_none=True)

        if step % args.log_interval == 0:
            spend_time = time.time() - start_time
            Logger(
                'Epoch:[{}/{}]({}/{}) loss:{:.3f} lr_muon:{:.8f} lr_adamw:{:.8f} epoch_Time:{}min:'.format(
                    epoch + 1,
                    args.epochs,
                    step,
                    iter_per_epoch,
                    loss.item() * args.accumulation_steps,
                    optimizer_muon.param_groups[-1]['lr'],
                    optimizer_adamw.param_groups[-1]['lr'],
                    spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60))

            if (wandb is not None) and (not ddp or dist.get_rank() == 0):
                wandb.log({"loss": loss.item() * args.accumulation_steps,
                           "lr_muon": optimizer_muon.param_groups[-1]['lr'],
                           "lr_adamw": optimizer_adamw.param_groups[-1]['lr'],
                           "epoch_Time": spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60})

        if (step + 1) % args.save_interval == 0 and (not ddp or dist.get_rank() == 0):
            model.eval()
            moe_path = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/dpo_muon_{lm_config.hidden_size}{moe_path}.pth'
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            state_dict = {k: v.half() for k, v in state_dict.items()}
            torch.save(state_dict, ckp)
            model.train()


def init_model(lm_config):
    tokenizer = AutoTokenizer.from_pretrained('../model')

    # Load the reference model (frozen)
    ref_model = MiniMindForCausalLM(lm_config)
    moe_path = '_moe' if lm_config.use_moe else ''
    ckp_ref = f'{args.save_dir}/full_sft_{lm_config.hidden_size}{moe_path}.pth'
    state_dict = torch.load(ckp_ref, map_location=args.device)
    ref_model.load_state_dict(state_dict)
    ref_model = ref_model.to(args.device)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    # Load the trainable model
    model = MiniMindForCausalLM(lm_config)
    model.load_state_dict(state_dict)
    model = model.to(args.device)

    Logger(f'LLM Total Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f}M')
    
    # Separate parameters for Muon and AdamW
    muon_params = []
    adamw_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        # Muon: 2D parameters (attention and FFN layers)
        # AdamW: 1D/0D parameters (embeddings, norms, lm_head)
        if param.ndim == 2 and 'embed' not in name and 'norm' not in name and 'lm_head' not in name:
            muon_params.append(param)
        else:
            adamw_params.append(param)
    
    Logger(f'✨ Muon optimizer params: {sum(p.numel() for p in muon_params) / 1e6:.3f}M ({len(muon_params)} tensors)')
    Logger(f'✨ AdamW optimizer params: {sum(p.numel() for p in adamw_params) / 1e6:.3f}M ({len(adamw_params)} tensors)')
    
    return model, ref_model, tokenizer, muon_params, adamw_params


def init_distributed_mode():
    if not ddp: return
    global ddp_local_rank, DEVICE

    dist.init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    DEVICE = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(DEVICE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind DPO with Muon Optimizer")
    parser.add_argument("--out_dir", type=str, default="../out")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-8, help="AdamW LR (very low for DPO)")
    parser.add_argument("--learning_rate_muon", type=float, default=0.001, help="Muon LR for DPO (very low)")
    parser.add_argument("--momentum", type=float, default=0.85, help="Momentum for Muon (lower for DPO)")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-DPO-Muon")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--ddp", action="store_true")
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_iters", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--hidden_size', default=768, type=int)
    parser.add_argument('--num_hidden_layers', default=16, type=int)
    parser.add_argument('--max_seq_len', default=512, type=int)
    parser.add_argument('--use_moe', default=False, type=bool)
    parser.add_argument("--data_path", type=str, default="../dataset/dpo.jsonl")
    args = parser.parse_args()

    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=args.use_moe)
    args.save_dir = os.path.join(args.out_dir)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    tokens_per_iter = args.batch_size * args.max_seq_len
    device_type = "cuda" if "cuda" in args.device else "cpu"

    args.wandb_run_name = f"MiniMind-DPO-Muon-Epoch{args.epochs}-BS{args.batch_size}"

    ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast()
    ddp = int(os.environ.get("RANK", -1)) != -1
    ddp_local_rank, DEVICE = 0, "cuda:0"
    base_seed = 1337
    torch.manual_seed(base_seed)
    torch.cuda.manual_seed(base_seed)

    if ddp:
        init_distributed_mode()
        args.device = torch.device(DEVICE)
        rank = dist.get_rank()
        torch.manual_seed(base_seed + rank)
        torch.cuda.manual_seed(base_seed + rank)

    if args.use_wandb and (not ddp or ddp_local_rank == 0):
        import wandb
        wandb.init(project=args.wandb_project, name=args.wandb_run_name)
    else:
        wandb = None

    model, ref_model, tokenizer, muon_params, adamw_params = init_model(lm_config)

    train_ds = DPODataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if ddp else None
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        pin_memory=True,
        drop_last=False,
        shuffle=(not ddp),
        num_workers=args.num_workers,
        sampler=train_sampler
    )

    Logger(f'Total dataset size: {len(train_ds):,} samples')
    if ddp:
        world_size = dist.get_world_size()
        Logger(f'DDP mode: {world_size} GPUs, ~{len(train_ds)//world_size:,} samples per GPU')
    Logger(f'Steps per epoch: {len(train_loader):,} (batch_size={args.batch_size})')

    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype in ['float16', 'bfloat16']))
    
    # Create dual optimizers with DPO-appropriate parameters (very conservative)
    if ddp and len(muon_params) > 0:
        optimizer_muon = DistMuon(muon_params, lr=args.learning_rate_muon, momentum=args.momentum, nesterov=True)
    else:
        optimizer_muon = Muon(muon_params, lr=args.learning_rate_muon, momentum=args.momentum, nesterov=True)
    
    optimizer_adamw = optim.AdamW(adamw_params, lr=args.learning_rate)
    
    Logger(f'✅ Using Muon optimizer for 2D parameters (lr={args.learning_rate_muon}, momentum={args.momentum})')
    Logger(f'✅ Using AdamW optimizer for 1D/0D parameters (lr={args.learning_rate})')
    Logger(f'⚠️  Note: Muon for DPO is highly experimental. Monitor training closely.')

    if ddp:
        model._ddp_params_and_buffers_to_ignore = {"pos_cis"}
        model = DistributedDataParallel(model, device_ids=[ddp_local_rank])

    iter_per_epoch = len(train_loader)
    for epoch in range(args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        train_epoch(epoch, wandb)

