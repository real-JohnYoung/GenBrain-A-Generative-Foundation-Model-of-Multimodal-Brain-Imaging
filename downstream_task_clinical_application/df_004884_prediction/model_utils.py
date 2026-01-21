import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import nibabel as nib
import os
import random

mask = nib.load('MNI152_T1_2mm_brain.nii.gz')
mask_data = mask.get_fdata()
nonzero = mask_data.nonzero()


class ADC3DMixedDataset(Dataset):

    def __init__(self, real_image_dir, real_csv, gen_image_dir, gen_csv, ratio=0.5,random_state=42):
        self.real_df = pd.read_csv(real_csv) if real_csv and os.path.exists(real_csv) else pd.DataFrame(columns=['participant_id', 'wab_aq'])
        self.gen_df = (pd.read_csv(gen_csv).sample(frac=1, random_state=random_state).reset_index(drop=True)if gen_csv and os.path.exists(gen_csv)else pd.DataFrame(columns=['participant_id', 'wab_aq']))
        self.real_image_dir = real_image_dir
        self.gen_image_dir = gen_image_dir

        self.n_real = len(self.real_df)
        self.n_gen  = len(self.gen_df)
        self.ratio = ratio

        self.only_gen = (self.n_real == 0 and self.n_gen > 0)

        if not self.only_gen:
            self.n_gen_sample = int(self.n_real * ratio) if ratio > 0 else 0
        else:
            self.n_gen_sample = self.n_gen
        
        print(f"Using {self.n_gen_sample} generated samples.") 

    def __len__(self):
        if self.only_gen:
            return self.n_gen
        else:
            return self.n_real + self.n_gen_sample

    def __getitem__(self, idx):
        if self.only_gen:
            row = self.gen_df.iloc[idx]
            image_path = os.path.join(self.gen_image_dir, 'gen_' + row['participant_id'], f"{row['participant_id']}_all_FA_0.npy")
        else:
            if idx < self.n_real:
                row = self.real_df.iloc[idx]
                image_path = os.path.join(self.real_image_dir, f"{row['participant_id']}_all_FA.npy")
            else:
                gen_idx = idx -self.n_real
                row = self.gen_df.iloc[gen_idx]
                image_path = os.path.join(self.gen_image_dir, 'gen_' + row['participant_id'], f"{row['participant_id']}_all_FA_0.npy")
        
        image_1d = np.load(image_path).reshape(-1)
        image_1d= np.clip(image_1d,0,1)
        image_1d = (image_1d-0.162)/0.152
        image = mask_data.copy()
        image[nonzero] = image_1d
        image = image[9:81, 10:100, 1:77]
        image = np.expand_dims(image, axis=0)  # (1, D, H, W)
        label = float(row['wab_aq'])
        return torch.tensor(image, dtype=torch.float32), torch.tensor(label, dtype=torch.float32).unsqueeze(0)
