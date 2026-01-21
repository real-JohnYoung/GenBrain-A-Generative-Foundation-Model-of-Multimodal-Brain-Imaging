import torch
import torch.nn as nn
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from diffusion import create_diffusion
import os
import argparse
from models import DiT_models,LabelEmbedder
from torch.utils.data import DataLoader
from model_utils import UKBDataset_T1_super_resolution_eval,PatchEmbed_1d
import json
import numpy as np

def find_model(model_name):
    """
    Finds a pre-trained DiT model, downloading it if necessary. Alternatively, loads a model from a local path.
    """
    if model_name in pretrained_models:  # Find/download our pre-trained DiT checkpoints
        return download_model(model_name)
    else:  # Load a custom DiT checkpoint:
        assert os.path.isfile(model_name), f'Could not find DiT checkpoint at {model_name}'
        checkpoint = torch.load(model_name, map_location=lambda storage, loc: storage)
        if "model" in checkpoint:       # supports checkpoints from train.py
            checkpoint = checkpoint["model"]
        return checkpoint

def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag

#################################################################################
#                             Define Finetune Model                             #
#################################################################################

class DiT_UKB_finetune(nn.Module):
    """
    Diffusion model with disease finetune module.
    """
    def __init__(self, dit_model, image_part_num, finetune_mode='full', finetune_block=None ):
        super().__init__()
        
        self.dit_model = dit_model
        self.hidden_size = self.dit_model.y1_embedder.hidden_size
        self.dit_model.x_embedder = PatchEmbed_1d(input_size=self.dit_model.x_embedder.input_size, patch_size=self.dit_model.x_embedder.patch_size, 
                                                  padding_size=self.dit_model.x_embedder.padding_size,in_chans=2, embed_dim=self.hidden_size)

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

    def forward(self, x, t, y1, y2, y3, y4,lr_x):
        """
        Forward pass of DiT with disease fine-tuning module.
        """
        x_cmb = torch.cat([x,lr_x],dim=1)
        x = self.dit_model.x_embedder(x_cmb) + self.dit_model.pos_embed
        t = self.dit_model.t_embedder(t)

        drop_mask = torch.rand(x.shape[0], device=x.device) < self.dit_model.class_dropout_prob
        y1 = self.dit_model.y1_embedder(y1, self.training, drop_mask)        # (N, D)  age
        y2 = self.dit_model.y2_embedder(y2, self.training, drop_mask)        # (N, D)  sex
        y3 = self.dit_model.y3_embedder(y3, self.training, drop_mask)        # (N, D)  modality
        y4 = self.image_part_index_embedder(y4, self.training, drop_mask)    # (N, D)  image_part_index

        c = t + y1 + y2 + y3 + y4

        for block in self.dit_model.blocks:
            x = block(x, c)
        x = self.dit_model.final_layer(x, c)
        x = self.dit_model.unpatchify(x)
        return x

    def forward_with_cfg(self, x, t, y1, y2, y3, y4, lr_x, cfg_scale):
        """
        Forward pass of DiT with classifier-free guidance.
        """
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y1, y2, y3, y4, lr_x)
        eps, rest = model_out[:, :self.dit_model.in_channels], model_out[:, self.dit_model.in_channels:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(args.data_info, 'r') as file:
        data_info = json.load(file)
        image_data_info = data_info['Image Data']
        condition_info = data_info['Condition Data']

    dit_model = DiT_models[args.model](condition_info=condition_info).to(device)

    finetune_mode = args.finetune_mode
    finetune_block = args.finetune_block

    model = DiT_UKB_finetune(dit_model,args.image_part_num,finetune_mode,finetune_block)
    
    ckpt_path = args.ckpt
    checkpoint = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
    if "model" in checkpoint:  # supports checkpoints from train.py
        checkpoint = checkpoint["model"]

    model.load_state_dict(checkpoint)
    model.to(device)

    model.eval()  # important!
    model.class_dropout_prob = -1
    diffusion = create_diffusion(str(args.num_sampling_steps))

    average_num = args.average_num

    batch_size = args.batch_size
    save_dir = os.path.join(args.save_dir,f"step_{args.ckpt_step}")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    dataset = UKBDataset_T1_super_resolution_eval(args.data_path, args.data_file, transform=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=args.num_workers)


    print("Begin sampling...")
    for i,data in enumerate(loader):
        for sample_num in range(average_num):

            eid_batch  = data['eid'].cpu().numpy()
            y1  = data['age'].to(device)
            y2  = data['sex'].to(device)
            y3  = data['mod'].to(device)
            y4  = data['image_part'].to(device)
            lr_x  = data['T1_image_itp'].to(device)

            real_sub_num = len(eid_batch)

            z  = torch.randn(real_sub_num, 1, args.image_size, device=device)

            z = torch.cat([z, z], 0)  # Uncomment if intentionally doubling z

            real_batch_size = real_sub_num 

            y1_null = torch.tensor([0] * real_batch_size, dtype=torch.int32, device=device)
            y1 = torch.cat([y1, y1_null], 0)

            y2_null = torch.tensor([2] * real_batch_size, dtype=torch.int32, device=device)
            y2 = torch.cat([y2, y2_null], 0)

            y3_null = torch.tensor([args.modality_num] * real_batch_size, dtype=torch.int32, device=device)
            y3 = torch.cat([y3, y3_null], 0)

            y4_null = torch.tensor([args.image_part_num] * real_batch_size, dtype=torch.int32, device=device)
            y4 = torch.cat([y4, y4_null], 0)

            lr_x_null = torch.zeros_like(lr_x,dtype=torch.float32, device=device)
            lr_x = torch.cat([lr_x, lr_x_null], 0)

            # Assert tensor shapes
            assert y1.shape == y2.shape == y3.shape, "Condition tensors must have the same shape"
 
            model_kwargs = dict(y1=y1, y2=y2, y3=y3, y4=y4, lr_x =lr_x ,cfg_scale=args.cfg_scale)

            with torch.no_grad():
                # Perform sampling with the diffusion model
                if args.use_ddim:
                    samples = diffusion.ddim_sample_loop(
                        model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
                    )
                else:
                    samples = diffusion.p_sample_loop(
                        model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
                    )

            samples, _ = samples.chunk(2, dim=0)

            save_batch_data(eid_batch, data['image_part'].numpy(), args.image_part_num, samples.cpu(), save_dir, sample_num)

            print(f"Batch {i} saved successfully, sample num {sample_num}, saved successfully.")

        if args.get_average_samples ==1:
            get_average_samples(eid_batch,average_num, save_dir)
    
    print(f"All generated images and metadata saved in {save_dir}")


def get_average_samples(eid_batch, average_num, save_dir):
    sub_num = len(eid_batch)

    for i in range(sub_num):
        modality_name = 'T1w_1mm'
        modality_data = []
        for sample_num in range(average_num):
            data_path = os.path.join(save_dir,  str(eid_batch[i]), f"{modality_name}_{sample_num}.npy")
            data = np.load(data_path)
            modality_data.append(data)

        modality_data = np.stack(modality_data)
        modality_data = np.mean(modality_data,axis=0)
        save_path = os.path.join(save_dir,str(eid_batch[i]), f"{modality_name}_average_{average_num}.npy")
        np.save(save_path, modality_data)


def save_batch_data(eid_batch, image_part_batch, image_part_num, data, save_dir, sample_num):
    sub_num = len(eid_batch)
    for i in range(sub_num):
        x = data[i]
        modality_name = 'T1w_1mm'
        os.makedirs(os.path.join(save_dir, str(eid_batch[i])),exist_ok=True)
        np.save(os.path.join(save_dir, str(eid_batch[i]), f"{modality_name}_image_part_{image_part_batch[i]}_{sample_num}.npy"), x)
    
    eid_list = list(set(eid_batch))
    # merge files
    
    image_part = {
        'part_0': {'start': 0, 'end': 228452},
        'part_1': {'start': 228377, 'end': 456829},
        'part_2': {'start': 456754, 'end': 685206},
        'part_3': {'start': 685131, 'end': 913583},
        'part_4': {'start': 913508, 'end': 1141960},
        'part_5': {'start': 1141885, 'end': 1370337},
        'part_6': {'start': 1370262, 'end': 1598714},
        'part_7': {'start': 1598642, 'end': 1827094}
    }

    overlap_mask = np.load('./labels/sr_overlap_mask.npy')
    merged_image =np.zeros(1827095)
    for eid in eid_list:
        for i in range(image_part_num):
            start = image_part[f'part_{i}']['start']
            end = image_part[f'part_{i}']['end']
            part_image = np.load(os.path.join(save_dir, str(eid), f"{modality_name}_image_part_{i}_{sample_num}.npy")).reshape(-1)
            merged_image[start:end+1] += part_image
        merged_image /= overlap_mask
        np.save(os.path.join(save_dir, str(eid), f"{modality_name}_{sample_num}.npy"), merged_image)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-UKB-L/256")
    parser.add_argument("--cfg-scale", type=float, default=1.2)
    parser.add_argument("--num-sampling-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)       
    parser.add_argument("--image-size", type=int, default=228453)
    parser.add_argument("--ckpt", type=str)
    parser.add_argument("--ckpt_step", type=int, default= 50000)
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--data_info", type=str)
    parser.add_argument("--data_file", type=str)
    parser.add_argument("--batch_size", type=int, default=8, help="Number of samples per batch")
    parser.add_argument("--modality_num", type=int, default=34, help="Number of modalities per sample")
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--average_num", type=int, default=5, help="Average number per sample")
    parser.add_argument("--get_average_samples", type=int, default=1) # need 1, not need 0
    parser.add_argument("--use_ddim", default=True)

    # Arguments for finetune model
    parser.add_argument("--finetune_mode", type=str, choices=['full','part','new_module'],default='full')
    parser.add_argument("--finetune_block", type=list, default=[20,21,22,23])
    parser.add_argument("--image_part_num", type=int, default=8)

    args = parser.parse_args()
    main(args)
