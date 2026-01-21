import torch
import torch.nn as nn
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from diffusion import create_diffusion
import os
import argparse
import pandas as pd
from models import DiT_models,LabelEmbedder
import json
import numpy as np
 
def find_model(model_name):
    assert os.path.isfile(model_name), f'Could not find DiT checkpoint at {model_name}'
    checkpoint = torch.load(model_name, map_location=lambda storage, loc: storage)
    if "model" in checkpoint:  # supports checkpoints from train.py
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
    def __init__(self, dit_model, disease_num, finetune_mode='full', finetune_block=None):
        super().__init__()
        
        self.dit_model = dit_model
        self.hidden_size = self.dit_model.y1_embedder.hidden_size
        self.disease_embedder = LabelEmbedder(num_classes=disease_num, hidden_size=self.hidden_size, dropout_prob=self.dit_model.class_dropout_prob)
        nn.init.zeros_(self.disease_embedder.embedding_table.weight)
        self.finetune_mode = finetune_mode
        self.finetune_block = finetune_block if finetune_block is not None else []

        self.set_finetune_mode()

    def set_finetune_mode(self):
        """
        Sets the mode of fine-tuning (full, new_module, part).
        """
        if self.finetune_mode == 'full':
            requires_grad(self.dit_model, True)  # Unfreeze all model parameters
            requires_grad(self.disease_embedder, True)  # Unfreeze disease embedder
        elif self.finetune_mode == 'new_module':
            requires_grad(self.dit_model, False)  # Freeze the base model
            requires_grad(self.disease_embedder, True)  # Unfreeze only the disease embedder
        elif self.finetune_mode == 'part':
            requires_grad(self.dit_model, False)  # Freeze the base model
            requires_grad(self.disease_embedder, True)  # Unfreeze disease embedder
            # Unfreeze specific transformer blocks
            for block_num in self.finetune_block:
                if block_num < len(self.dit_model.blocks):
                    requires_grad(self.dit_model.blocks[block_num], True)

    def forward(self, x, t, y1, y2, y3, y4):
        """
        Forward pass of DiT with disease fine-tuning module.
        """
        x = self.dit_model.x_embedder(x) + self.dit_model.pos_embed
        t = self.dit_model.t_embedder(t)

        drop_mask = torch.rand(x.shape[0], device=x.device) < self.dit_model.class_dropout_prob

        y1 = self.dit_model.y1_embedder(y1, self.training, drop_mask)  # (N, D)  age
        y2 = self.dit_model.y2_embedder(y2, self.training, drop_mask)  # (N, D)  sex
        y3 = self.dit_model.y3_embedder(y3, self.training, drop_mask)  # (N, D)  modality
        y4 = self.disease_embedder(y4, self.training, drop_mask)   # (N, D)  disease
        
        c = t + y1 + y2 + y3 + y4

        for block in self.dit_model.blocks:
            x = block(x, c)
        x = self.dit_model.final_layer(x, c)
        x = self.dit_model.unpatchify(x)
        return x

    def forward_with_cfg(self, x, t, y1, y2, y3, y4, cfg_scale):
        """
        Forward pass of DiT with classifier-free guidance.
        """
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y1, y2, y3, y4)
        eps, rest = model_out[:, :self.dit_model.in_channels], model_out[:, self.dit_model.in_channels:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


class ExtendedLabelEmbedder(LabelEmbedder):

    def __init__(
        self,
        old_embedder: LabelEmbedder,
        n_new_modalities: int,
        init_std: float = 0.02
    ):
        self.original_cfg_id = old_embedder.num_classes

        super().__init__(
            num_classes=old_embedder.num_classes + n_new_modalities,
            hidden_size=old_embedder.embedding_table.embedding_dim,
            dropout_prob=old_embedder.dropout_prob
        )

        with torch.no_grad():
            old_w = old_embedder.embedding_table.weight
            new_w = self.embedding_table.weight
            new_w[: old_w.size(0)] = old_w
            start = old_w.size(0)
            end   = start + n_new_modalities
            nn.init.normal_(new_w[start:end],mean=0.0,std=init_std)

    def token_drop(self, labels, force_drop_ids=None):
        
        if force_drop_ids is None:
            drop_mask = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_mask = force_drop_ids == 1

        # labels[drop_mask] = original cfg id
        return torch.where(drop_mask, self.original_cfg_id, labels)

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings



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

    disease_num = args.disease_num
    finetune_mode = args.finetune_mode
    finetune_block = args.finetune_block

    model = DiT_UKB_finetune(dit_model,disease_num,finetune_mode,finetune_block)
    model.dit_model.y3_embedder = ExtendedLabelEmbedder(model.dit_model.y3_embedder,n_new_modalities=1) # tian subcortex : 35~50

    ckpt_path = args.ckpt
    checkpoint = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
    if "model" in checkpoint:  # supports checkpoints from train.py
        checkpoint = checkpoint["model"]

    model.load_state_dict(checkpoint)
    
    model.to(device)

    model.eval()  # important!
    model.class_dropout_prob = -1
    diffusion = create_diffusion(str(args.num_sampling_steps))

    sample_data = pd.read_csv(args.sample_file)
    n = len(sample_data)
    average_num = args.average_num


    id_labels = sample_data['FILE_ID'].astype(str).tolist()
    age_labels = sample_data['age'].astype(float).tolist()
    sex_labels = sample_data['sex(F0M1)'].astype(int).tolist()
    dis_labels = sample_data['label'].astype(int).tolist()
    

    batch_size = args.batch_size
    save_dir = os.path.join(args.save_dir,f"step_{args.ckpt_step}")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    subcortex_id = args.subcortex_id

    print("Begin sampling...")
    for i in range(0, n, batch_size):  # (0, n, batch_size)

        for sample_num in range(0,average_num):
            print(f"Begin average num {sample_num}")
            id_batch = id_labels[i:i + batch_size]
            age_batch = age_labels[i:i + batch_size]
            sex_batch = sex_labels[i:i + batch_size]
            dis_batch = dis_labels[i:i + batch_size]
            

            real_sub_num = len(id_batch)
            mod_batch = [35] * real_sub_num

            z = torch.randn(real_sub_num, 1, args.image_size, device=device)
            y1 = torch.tensor(age_batch, dtype=torch.float32, device=device)
            y2 = torch.tensor(sex_batch, dtype=torch.int32, device=device)
            y3 = torch.tensor(mod_batch, dtype=torch.int32, device=device)
            y4 = torch.tensor(dis_batch,  dtype=torch.int32, device=device)
            
            # if sample_num ==0:
            #     continue
            z = torch.cat([z, z], 0)  # Uncomment if intentionally doubling z

            real_batch_size = real_sub_num 

            y1_null = torch.tensor([0] * real_batch_size, dtype=torch.int32, device=device)
            y1 = torch.cat([y1, y1_null], 0)

            y2_null = torch.tensor([2] * real_batch_size, dtype=torch.int32, device=device)
            y2 = torch.cat([y2, y2_null], 0)

            y3_null = torch.tensor([args.modality_num] * real_batch_size, dtype=torch.int32, device=device)
            y3 = torch.cat([y3, y3_null], 0)

            y4_null = torch.tensor([args.disease_num] * real_batch_size, dtype=torch.int32, device=device)
            y4 = torch.cat([y4, y4_null], 0)

            # Assert tensor shapes
            assert y1.shape == y2.shape == y3.shape, "Condition tensors must have the same shape"

            model_kwargs = dict(y1=y1, y2=y2, y3=y3, y4=y4,cfg_scale=args.cfg_scale)

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

            save_batch_data(id_batch, mod_batch,subcortex_id,samples.cpu(), save_dir, image_data_info,sample_num)

            print(f"Batch {i // batch_size + 1} saved successfully, sample num {sample_num}, saved successfully.")
        
        if args.get_average_samples ==1:
            get_average_samples(id_batch,average_num, subcortex_id, save_dir)
    
    
    print(f"All generated images and metadata saved in {save_dir}")


def get_average_samples(eid_batch, average_num, subcortex_id, save_dir):
    sub_num = len(eid_batch)
    for i in range(sub_num):
        modality_data = []
        for sample_num in range(average_num):
            data_path = os.path.join(save_dir,  eid_batch[i] +f"_tian{subcortex_id}_{sample_num}.npy")
            data = np.load(data_path)
            modality_data.append(data)
            modality_data = np.stack(modality_data)
            modality_data = np.mean(modality_data,axis=0)
            save_path = os.path.join(save_dir,eid_batch[i]+f"_tian{subcortex_id}_average_{average_num}.npy")
            np.save(save_path, modality_data)


def save_batch_data(eid_batch,modality_batch, subcortex_id, data, save_dir, image_data_info, sample_num):
    sub_num = len(eid_batch)

    with open('./labels/tian_subcortex_mean_std.json', 'r', encoding='utf-8') as f:
        data_mean_std = json.load(f)

    for i in range(sub_num):
        x = data[i]
        mod_info = data_mean_std[f'Tian_subcortex_{subcortex_id}']
        data_mean = mod_info['mean']
        data_std = mod_info['std']    
        x = (x * data_std + data_mean).numpy()
        # Provide the array to save
        np.save(os.path.join(save_dir, eid_batch[i] + f"_Tian_Subcortex_seedFC_{subcortex_id}.npy"), x)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-UKB-L/256")
    parser.add_argument("--cfg-scale", type=float, default=1.2)
    parser.add_argument("--num-sampling-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)       
    parser.add_argument("--image-size", type=int, default=228453)
    parser.add_argument("--ckpt", type=str, default=None)#step_0010000.pt
    parser.add_argument("--ckpt_step", type=int, default=10000)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--data_info", type=str, default=None)# data_info.json
    parser.add_argument("--sample_file", type=str, default=None)
    parser.add_argument("--subcortex_id", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=5, help="Number of samples per batch")
    parser.add_argument("--modality_num", type=int, default=34, help="Number of modalities per sample")
    parser.add_argument("--average_num", type=int, default=1, help="Average number per sample")
    parser.add_argument("--get_average_samples", type=int, default=0) # need 1, not need 0
    parser.add_argument("--use_ddim", default=True)

    # Arguments for finetune model
    parser.add_argument("--finetune_mode", type=str, choices=['full','part','new_module'],default='full')
    parser.add_argument("--finetune_block", type=list, default=[20,21,22,23])
    parser.add_argument("--disease_num", type=int, default=2)

    args = parser.parse_args()
    main(args)
