import torch
import torch.nn as nn
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import os
import sys,os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from diffusion import create_diffusion
import argparse
import pandas as pd
from models import DiT_models
import json
import numpy as np
from model_utils import PatchEmbed_1d

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
    def __init__(self, dit_model, finetune_mode='full', finetune_block=None):
        super().__init__()
        
        self.dit_model = dit_model
        self.hidden_size = self.dit_model.y1_embedder.hidden_size
        self.dit_model.x_embedder = PatchEmbed_1d(input_size=self.dit_model.x_embedder.input_size, 
                                  patch_size=self.dit_model.x_embedder.patch_size, padding_size=self.dit_model.x_embedder.padding_size,
                                  in_chans=2, embed_dim=self.hidden_size)

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


    def forward(self, x, t, y1, y2, y3, cor_x):
        """
        Forward pass of DiT with disease fine-tuning module.
        """
        x_cmb = torch.cat([x,cor_x],dim=1)
        x = self.dit_model.x_embedder(x_cmb) + self.dit_model.pos_embed
        t = self.dit_model.t_embedder(t)

        drop_mask = torch.rand(x.shape[0], device=x.device) < self.dit_model.class_dropout_prob

        y1 = self.dit_model.y1_embedder(y1, self.training, drop_mask)  # (N, D)  age
        y2 = self.dit_model.y2_embedder(y2, self.training, drop_mask)  # (N, D)  sex
        y3 = self.dit_model.y3_embedder(y3, self.training, drop_mask)  # (N, D)  modality
        
        c = t + y1 + y2 + y3

        for block in self.dit_model.blocks:
            x = block(x, c)
        x = self.dit_model.final_layer(x, c)
        x = self.dit_model.unpatchify(x)
        return x

    def forward_with_cfg(self, x, t, y1, y2, y3, cor_x, cfg_scale):
        """
        Forward pass of DiT with classifier-free guidance.
        """
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y1, y2, y3, cor_x)
        eps, rest = model_out[:, :self.dit_model.in_channels], model_out[:, self.dit_model.in_channels:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)



def load_corrupt_images(cor_data_path,eid_batch,modality_batch,device):
    sub_num = len(eid_batch)
    cor_images = []

    for i in range(sub_num):
        modality = modality_batch[i]
        if modality == 11:
            modality_name = 'T1_brain_nonlinear_2mm'
        elif modality == 13:
            modality_name = 'T2_FLAIR_brain_to_MNI'
        cor_image_path = os.path.join(cor_data_path,str(eid_batch[i]),modality_name+'.npy')
        cor_image = np.load(cor_image_path)
        cor_image = (cor_image - cor_image.mean())/cor_image.std() # attention !!!
        cor_image = torch.tensor(cor_image, dtype=torch.float32,device=device).unsqueeze(0)
        cor_images.append(cor_image)
    
    cor_images = torch.cat(cor_images, dim = 0).unsqueeze(1)
    return cor_images


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

    model = DiT_UKB_finetune(dit_model,finetune_mode,finetune_block)
    
    ckpt_path = args.ckpt
    checkpoint = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
    if "model" in checkpoint:  # supports checkpoints from train.py
        checkpoint = checkpoint["model"]

    model.load_state_dict(checkpoint)
    model.to(device)


    model.eval()  # important!
    model.dit_model.class_dropout_prob = -1
    diffusion = create_diffusion(str(args.num_sampling_steps))

    sample_data = pd.read_csv(args.sample_file)
    n = len(sample_data)
    average_num = args.average_num


    eid_labels = sample_data['eid'].astype(int).tolist()
    mod_labels = sample_data['modality'].astype(int).tolist()

    batch_size = args.batch_size
    save_dir = os.path.join(args.save_dir,f"step_{args.ckpt_step}")
    cor_data_path = args.cor_data_path

    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    print("Begin sampling...")
    for i in range( 0,n, batch_size):  # (0, n, batch_size)

        for sample_num in range(average_num):

            eid_batch = eid_labels[i:i + batch_size]
            modality_batch = mod_labels[i:i+batch_size]

            real_sub_num = len(eid_batch)

            z = torch.randn(real_sub_num, 1, args.image_size, device=device)
            y1 = torch.tensor([0] * real_sub_num, dtype=torch.int32, device=device)
            y2 = torch.tensor([2] * real_sub_num, dtype=torch.int32, device=device)
            y3 = torch.tensor(modality_batch, dtype=torch.int32, device=device)       # modality 
            cor_x = load_corrupt_images(cor_data_path,eid_batch,modality_batch,device)

            z = torch.cat([z, z], 0)  # Uncomment if intentionally doubling z

            y1_null = torch.tensor([0] * real_sub_num, dtype=torch.int32, device=device)
            y1 = torch.cat([y1, y1_null], 0)

            y2_null = torch.tensor([2] * real_sub_num, dtype=torch.int32, device=device)
            y2 = torch.cat([y2, y2_null], 0)

            y3_null = torch.tensor([args.modality_num] * real_sub_num, dtype=torch.int32, device=device)
            y3 = torch.cat([y3, y3_null], 0)

            cor_x_null = torch.zeros_like(cor_x,dtype=torch.int32, device=device)

            cor_x = torch.cat([cor_x,cor_x_null],0)

            assert y1.shape == y2.shape == y3.shape, "Condition tensors must have the same shape"

            model_kwargs = dict(y1=y1, y2=y2, y3=y3, cor_x=cor_x, cfg_scale=args.cfg_scale)

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

            save_batch_data(eid_batch, modality_batch,samples.cpu(), save_dir, image_data_info,sample_num)

            print(f"Batch {i // batch_size + 1} saved successfully, sample num {sample_num}, saved successfully.")

        if args.need_average:
            get_average_samples(eid_batch,modality_batch,average_num, save_dir)


    print(f"All generated images and metadata saved in {save_dir}")


def get_average_samples(eid_batch, modality_batch, average_num, save_dir):
    sub_num = len(eid_batch)

    for i in range(sub_num):
        modality = modality_batch[i]
        
        if modality == 11:
            modality_name = 'T1_brain_nonlinear_2mm'
        elif modality == 13:
            modality_name = 'T2_FLAIR_brain_to_MNI'
        
        modality_data = []

        for sample_num in range(average_num):
            data_path = os.path.join(save_dir,f"enhanced_{str(eid_batch[i])}",f"{modality_name}_{sample_num}.npy")
            data = np.load(data_path)
            modality_data.append(data)

        modality_data = np.stack(modality_data)
        modality_data = np.mean(modality_data,axis=0)
        save_path = os.path.join(save_dir,f"enhanced_{str(eid_batch[i])}",f"{modality_name}_average_{average_num}.npy")
        np.save(save_path, modality_data)


def save_batch_data(eid_batch,modality_batch, data, save_dir, image_data_info, sample_num):
    sub_num = len(eid_batch)
    for i in range(sub_num):
        addr = os.path.join(save_dir, f"enhanced_{str(eid_batch[i])}")
        os.makedirs(addr, exist_ok=True)
        # Correct slicing with colon
        x = data[i]
        index = modality_batch[i]
        modality = image_data_info[str(index)]['modality']
        mean = image_data_info[str(index)]['mean']
        std = image_data_info[str(index)]['std']
        x = (x * std + mean).numpy()
        # Provide the array to save
        np.save(os.path.join(addr, f"{modality}_{sample_num}.npy"), x)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-UKB-L/256")
    parser.add_argument("--cor_data_path", type=str, default="Path_to_Corrupted_Image_Dataset")
    parser.add_argument("--cfg-scale", type=float, default=1.2)
    parser.add_argument("--num-sampling-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=228453)
    parser.add_argument("--ckpt", type=str, default="Path_to_Train_from_Scratch/checkpoints/step_0050000.pt")
    parser.add_argument("--ckpt_step", type=int, default=50000)
    parser.add_argument("--save_dir", type=str, default="Save_Directory/avg_evaluation")
    parser.add_argument("--data_info", type=str, default="../labels/data_info.json")
    parser.add_argument("--sample_file", type=str, default="./labels/image_quality_enhancement_testing_data.csv")
    parser.add_argument("--batch_size", type=int, default=5, help="Number of samples per batch")
    parser.add_argument("--modality_num", type=int, default=34, help="Number of modalities per sample")
    parser.add_argument("--average_num", type=int, default=5, help="Average number per sample")
    parser.add_argument("--need_average", type=int, default=1) # need_average: Yes 1, NO 0
    parser.add_argument("--use_ddim", default=True)

    # Argument for finetune model
    parser.add_argument("--finetune_mode", type=str, choices=['full','part','new_module','eval'],default='full')
    parser.add_argument("--finetune_block", type=list, default=[20,21,22,23])

    args = parser.parse_args()
    main(args)