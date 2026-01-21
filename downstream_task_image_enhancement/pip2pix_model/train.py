import os
import json
import argparse
from glob import glob
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from model import Pix2PixModel
from utils import UKBDataset_enhancement_finetune
import logging
###############################################################################
#                               Helper utils                                  #
###############################################################################

def cleanup():
    dist.destroy_process_group()

def is_main_process():
    return dist.get_rank() == 0

def save_checkpoint(state: dict, path: str):
    torch.save(state, path)

###############################################################################
#                                Main                                         #
###############################################################################

def main(args):
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dist.init_process_group("nccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank % torch.cuda.device_count())
    device = torch.device("cuda", rank % torch.cuda.device_count())

    seed = args.seed + rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # ------------------------------ dirs ------------------------------------
    if is_main_process():
        os.makedirs(args.results_dir, exist_ok=True)
        exp_idx = len(glob(os.path.join(args.results_dir, "exp_*")))
        exp_dir = os.path.join(args.results_dir, f"exp_{exp_idx:03d}")
        ckpt_dir = os.path.join(exp_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)

        # set up logger
        logging.basicConfig(
            level=logging.INFO,
            format='[%(asctime)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(os.path.join(exp_dir, 'train.log')),
            ],
        )
        logger = logging.getLogger(__name__)
        logger.info("Experiment: %s", exp_dir)
    else:
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
        exp_dir = ""
        ckpt_dir = ""

    obj_list = [exp_dir, ckpt_dir]
    dist.broadcast_object_list(obj_list, src=0)
    exp_dir, ckpt_dir = obj_list

    # ------------------------------ data ------------------------------------
    with open(args.data_info, "r") as f:
        data_info = json.load(f)["Image Data"]

    dataset = UKBDataset_enhancement_finetune(args.data_path, args.cor_data_path,args.data_file, data_info, transform=True)
    sampler = DistributedSampler(dataset, world_size, rank, shuffle=True, seed=args.seed)
    loader = DataLoader(dataset,batch_size=int(args.global_batch_size // world_size),sampler=sampler,shuffle=False,num_workers=args.num_workers,pin_memory=True,drop_last=True)

    # ------------------------------ model -----------------------------------
    class Opt:
        def __init__(self, gpu_id):
            self.gpu_ids = [gpu_id]
            self.isTrain = True
            self.input_nc = 1
            self.output_nc = 1
            self.ngf = 64
            self.num_down =5
            self.netG = "unet_256"
            self.netD = "basic"
            self.n_layers_D = 3
            self.ndf = 64
            self.norm = "batch"
            self.init_type = "normal"
            self.init_gain = 0.02
            self.no_dropout = False
            self.gan_mode = "lsgan"
            self.lr = args.lr
            self.beta1 = 0.5
            self.lambda_L1 = 100.0
            self.direction = "AtoB"

    opt = Opt(device.index)
    pix2pix = Pix2PixModel(opt)
    model = DDP(pix2pix, device_ids=[device.index])
    optimizer_G, optimizer_D = model.module.optimizer_G, model.module.optimizer_D

    # ------------------------------ train -----------------------------------
    global_step = 0
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        logger.info("Begin epoch %s", epoch)
        for batch in loader:
            model.module.set_input(batch)
            model.module.optimize_parameters()
            global_step += 1

            if global_step % args.log_every == 0 and is_main_process():
                lg = model.module.loss_G.item()
                ld = model.module.loss_D.item()
                logger.info(f"[E{epoch+1} S{global_step}] G={lg:.4f} D={ld:.4f}")


            if global_step % args.ckpt_every == 0 and is_main_process():
                save_path = os.path.join(ckpt_dir, f"ckpt_{global_step:07d}.pt")
                save_checkpoint({
                    "step": global_step,
                    "model": model.module.state_dict(),
                    "opt_G": optimizer_G.state_dict(),
                    "opt_D": optimizer_D.state_dict(),
                    "args": vars(args)
                }, save_path)
                logger.info(f"Saved checkpoint to {save_path}")
        dist.barrier()

    if is_main_process():
        print("Training finished.")
    cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default='/corrupt_more_3d_dataset/clean')
    parser.add_argument("--cor_data_path", type=str, default='/corrupt_more_3d_dataset/corrupt')
    parser.add_argument("--data_file", type=str, default='../labels/image_quality_enhancement_training_data_2000.csv')
    parser.add_argument("--data_info", type=str, default='../labels/data_info.json')
    parser.add_argument("--results_dir", type=str, default="Save_path/pix2pix_model")
    parser.add_argument("--epochs", type=int, default=1630)
    parser.add_argument("--global_batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=32)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--ckpt-every", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()
    main(args)



# torchrun --nnodes=1 --nproc_per_node=2 train.py