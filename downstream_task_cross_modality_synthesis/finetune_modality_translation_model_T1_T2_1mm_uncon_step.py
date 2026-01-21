# finetune code script
import torch
import torch.nn as nn
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from collections import OrderedDict
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os
import json
from models import DiT_models, LabelEmbedder
from model_utils import UKBDataset_modality_translation_T1_T2_1mm_finetune_uncon,PatchEmbed_1d
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from diffusion import create_diffusion

#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag

def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/finetune_log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


#################################################################################
#                             Define Finetune Model                             #
#################################################################################

class DiT_UKB_finetune(nn.Module):
    """
    Diffusion model with disease finetune module.
    """
    def __init__(self, dit_model, image_part_num, finetune_mode='full', finetune_block=None):
        super().__init__()
        
        self.dit_model = dit_model
        self.hidden_size = self.dit_model.y1_embedder.hidden_size
        self.dit_model.x_embedder = PatchEmbed_1d(input_size=self.dit_model.x_embedder.input_size, 
                                  patch_size=self.dit_model.x_embedder.patch_size, padding_size=self.dit_model.x_embedder.padding_size,
                                  in_chans=2, embed_dim=self.hidden_size)

        self.image_part_index_embedder = LabelEmbedder(image_part_num, self.hidden_size, self.dit_model.class_dropout_prob)
        nn.init.zeros_(self.image_part_index_embedder.embedding_table.weight)

        self.finetune_mode = finetune_mode
        self.finetune_block = finetune_block if finetune_block is not None else []

        self.set_finetune_mode()

    def set_finetune_mode(self):
        """
        Sets the mode of fine-tuning (full, new_module, part).
        """
        if self.finetune_mode == 'full':
            requires_grad(self.dit_model, True)  # Unfreeze all model parameters
            requires_grad(self.dit_model.x_embedder, True)  # Unfreeze disease embedder
        elif self.finetune_mode == 'new_module':
            requires_grad(self.dit_model, False)  # Freeze the base model
            requires_grad(self.dit_model.x_embedder, True)
        elif self.finetune_mode == 'part':
            requires_grad(self.dit_model, False)  # Freeze the base model
            requires_grad(self.dit_model.x_embedder, True)  # Unfreeze disease embedder
            # Unfreeze specific transformer blocks
            for block_num in self.finetune_block:
                if block_num < len(self.dit_model.blocks):
                    requires_grad(self.dit_model.blocks[block_num], True)

    def forward(self, x, t, y1, y2, y3, y4,source_x):
        x_cmb = torch.cat([x,source_x],dim=1)
        x = self.dit_model.x_embedder(x_cmb) + self.dit_model.pos_embed
        t = self.dit_model.t_embedder(t)

        drop_mask = torch.rand(x.shape[0], device=x.device) < self.dit_model.class_dropout_prob

        y1 = self.dit_model.y1_embedder(y1, self.training, drop_mask)  # (N, D)  age
        y2 = self.dit_model.y2_embedder(y2, self.training, drop_mask)  # (N, D)  sex
        y3 = self.dit_model.y3_embedder(y3, self.training, drop_mask)  # (N, D)  modality
        y4 = self.image_part_index_embedder(y4, self.training, drop_mask)    # (N, D)  image_part_index

        c = t + y1 + y2 + y3 + y4

        for block in self.dit_model.blocks:
            x = block(x, c)
        x = self.dit_model.final_layer(x, c)
        x = self.dit_model.unpatchify(x)
        return x

    def forward_with_cfg(self, x, t, y1, y2, y3, y4,source_x, cfg_scale):
        """
        Forward pass of DiT with classifier-free guidance.
        """
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y1, y2, y3, y4, source_x)
        eps, rest = model_out[:, :self.dit_model.in_channels], model_out[:, self.dit_model.in_channels:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Finetune a new DiT model from a checkpoint.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    
    # Setup DDP:
    dist.init_process_group("nccl")
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")
    
    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.model.replace("/", "-")  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}-finetune_mode-{args.finetune_mode}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger = create_logger(None)

    with open(args.data_info, 'r') as file:
        data_info = json.load(file)
        image_data_info = data_info['Image Data']
        condition_info = data_info['Condition Data']

    dit_model = DiT_models[args.model](condition_info=condition_info)

    if args.ckpt_path:
        checkpoint_path = f"{args.ckpt_path}/checkpoints/{args.ckpt_epoch:07d}.pt"
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
            dit_model.load_state_dict(checkpoint['model'])
            logger.info(f"Finetune model from checkpoint {checkpoint_path} (epoch {args.ckpt_epoch})")
        else:
            logger.warning(f"Checkpoint {checkpoint_path} not found. Starting from scratch.")
            
    finetune_mode = args.finetune_mode
    finetune_block = args.finetune_block

    model = DiT_UKB_finetune(dit_model,args.image_part_num,finetune_mode,finetune_block)
    ema = deepcopy(model).to(device)
    requires_grad(ema, False)

    if "ema" in checkpoint:
        checkpoint_ema = checkpoint["ema"]
        if 'x_embedder.proj.weight' in checkpoint_ema:
            del checkpoint_ema['x_embedder.proj.weight']  # delete x_embedder's part
        ema.dit_model.load_state_dict(checkpoint_ema,strict=False)
        logger.info("Loaded EMA weights from checkpoint.")
    else:
        logger.warning("No EMA weights found in checkpoint.")

    model = DDP(model.to(device), device_ids=[rank])
    diffusion = create_diffusion(timestep_respacing="")
    
    logger.info(f"Finetune DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, weight_decay=0)

    # Setup data (same as before):
    transform = True
    dataset = UKBDataset_modality_translation_T1_T2_1mm_finetune_uncon(args.data_path, args.data_file, transform=transform)
    
    sampler = DistributedSampler(dataset, num_replicas=dist.get_world_size(), rank=rank, shuffle=True, seed=args.global_seed)
    loader = DataLoader(dataset, batch_size=int(args.global_batch_size // dist.get_world_size()), shuffle=False, sampler=sampler, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    logger.info(f"Dataset contains {len(dataset):,} images ({args.data_path})")

    # Prepare models for training:
    model.train()
    ema.eval()  # EMA model should always be in eval mode

    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(0, args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch+1}...")
        for data in loader:
            source_x = data['T1_image'].to(device)
            y1 = data['age'].to(device)
            y2 = data['sex'].to(device)
            y3 = data['mod'].to(device)
            y4 = data['image_part'].to(device)
            target_x = data['T2_image'].to(device)

            t = torch.randint(0, diffusion.num_timesteps, (target_x.shape[0],), device=device)
            model_kwargs = dict(y1=y1,y2=y2,y3=y3,y4=y4, source_x=source_x)
            loss_dict = diffusion.training_losses(model, target_x, t, model_kwargs)
            loss = loss_dict["loss"].mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            update_ema(ema, model.module)

            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Save DiT checkpoint:
            if rank == 0 and (train_steps+1)%10000 ==0:
                checkpoint = {
                    "model": model.module.state_dict(),
                    "ema": ema.state_dict(),
                    "opt": opt.state_dict(),
                    "args": args
                }
                checkpoint_path = f"{checkpoint_dir}/step_{train_steps+1:07d}.pt"
                torch.save(checkpoint, checkpoint_path)
                logger.info(f"Saved checkpoint to {checkpoint_path}")
        dist.barrier()

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str) #dataset_ukb_1mm
    parser.add_argument("--data_file", type=str) 
    parser.add_argument("--data_info", type=str) # data_info.json
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--results_dir", type=str)
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-UKB-L/256")
    parser.add_argument("--global-batch-size", type=int, default=64)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--log-every", type=int, default=20)

    # Arguments for finetune 
    parser.add_argument("--ckpt_epoch", type=int, default=200, help="Epoch number to checkpoint")
    parser.add_argument("--ckpt_path", type=str, help="Path to the checkpoint directory")
    parser.add_argument("--finetune_mode", type=str, choices=['full','part','new_module'],default='full')
    parser.add_argument("--finetune_block", type=list, default=[20,21,22,23])
    parser.add_argument("--image_part_num", type=int, default=8)

    args = parser.parse_args()
    main(args)


# torchrun --nnodes=1 --nproc_per_node=2 finetune_modality_translation_model_T1_T2_1mm_uncon_step.py