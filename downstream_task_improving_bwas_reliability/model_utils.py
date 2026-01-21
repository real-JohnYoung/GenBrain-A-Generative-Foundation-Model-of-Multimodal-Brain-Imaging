# This file contains modulues used for Dit_1d
import torch
import torch.nn as nn
import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torch.nn.functional as F
import json

class PatchEmbed_1d(nn.Module):
    """ 1D Image to Patch Embedding
    """
    def __init__(self, input_size, patch_size=256, padding_size=155, in_chans=1, embed_dim=768, norm_layer=None):
        super().__init__()

        self.input_size = input_size
        self.patch_size = patch_size
        self.num_patches = (self.input_size + padding_size) // self.patch_size
        self.padding_size = padding_size

        assert self.num_patches * self.patch_size == self.input_size + padding_size, \
            f"Hidden size error: input size extend ({self.input_size + padding_size}) must be divisible by hidden size ({self.patch_size})"

        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = F.pad(x, (0, self.padding_size))
        x = self.proj(x)
        x = x.transpose(1, 2)  
        x = self.norm(x)
        return x


# This dataset is used for downstream task [multisite scz finetune for vbm modality].
class UKBDataset_multisite_scz_vbm_finetune(Dataset):
    def __init__(self, data_path, data_label_file,image_data_info,transform=None):

        self.data_path = data_path
        self.data_label_file = pd.read_csv(data_label_file)

        self.transform = transform
        self.data_info = image_data_info
        
    def __len__(self):
        return len(self.data_label_file)
    
    def __getitem__(self, idx):
        # Get the row for the current index
        row = self.data_label_file.iloc[idx]
        sub_id= row['file_name']
        modality = 15        # for VBM modality
        
        age = row['age']
        sex = row['Sex(M1F0)']
        disease = row['label']

        image_info = self.data_info[str(modality)]
        modality_name = 'VBM_2mm'

        image_path = os.path.join(self.data_path, sub_id+".npy")
        
        image = np.load(image_path)
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)

        if self.transform: 
            # smoothed VBM
            image = torch.clamp(image, 0, 3)  
            image = (image-0.41)/0.27
        
        # Convert to tensor
        data = {}
        age = torch.tensor(age, dtype=torch.float32)
        sex = torch.tensor(sex, dtype=torch.int32)
        modality = torch.tensor(modality,dtype=torch.int32)
        disease = torch.tensor(disease,dtype=torch.int32)

        data ={ 'image':image, 
                'age': age,'sex': sex,'mod':modality,
                'dis': disease
               }
        return data

# This dataset is used for downstream task [istbi depression finetune for vbm modality].
class UKBDataset_istbi_depression_vbm_finetune(Dataset):
    def __init__(self, data_path, data_label_file,image_data_info,transform=None):

        self.data_path = data_path
        self.data_label_file = pd.read_csv(data_label_file)

        self.transform = transform
        self.data_info = image_data_info
        
    def __len__(self):
        return len(self.data_label_file)
    
    def __getitem__(self, idx):
        # Get the row for the current index

        row = self.data_label_file.iloc[idx]
        sub_id= row['ID']
        modality = 15        # for VBM modality
        
        age = row['Age']
        sex = row['Sex']
        disease = row['label']

        image_info = self.data_info[str(modality)]

        image_path = os.path.join(self.data_path, sub_id+".npy")
        
        image = np.load(image_path)
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)

        if self.transform: 
            # smoothed VBM
            image = torch.clamp(image, 0, 3)  
            image = (image-0.34)/0.18
        
        # Convert to tensor
        data = {}
        age = torch.tensor(age, dtype=torch.float32)
        sex = torch.tensor(sex, dtype=torch.int32)
        modality = torch.tensor(modality,dtype=torch.int32)
        disease = torch.tensor(disease,dtype=torch.int32)

        data ={ 'image':image, 
                'age': age,'sex': sex,'mod':modality,
                'dis': disease
               }
        return data






class UKBDataset_abide_autism_tian_subcortex_finetune_separately(Dataset):
    def __init__(self, data_path, data_label_file,image_data_info,subcortex_id,transform=None):

        self.data_path = data_path
        self.data_label_file = pd.read_csv(data_label_file)
        self.transform = transform
        self.data_info = image_data_info
        self.subcortex_id = subcortex_id

        with open('。/labels/tian_subcortex_mean_std.json', 'r', encoding='utf-8') as f:
            self.data_mean_std = json.load(f)
        
        
    def __len__(self):
        return len(self.data_label_file)
    
    def __getitem__(self, idx):
        # Get the row for the current index

        row = self.data_label_file.iloc[idx]
        sub_id= row['FILE_ID']

        age = row['age']
        sex = row['sex(F0M1)']
        disease = row['label']


        image_path = os.path.join(self.data_path, f"abide{row['abide']}_tian" ,sub_id+f'_Tian_Subcortex_seedFC_{self.subcortex_id}.npy')
        image = np.load(image_path)
        image = np.nan_to_num(image)

        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)

        if self.transform: 
            mod_info = self.data_mean_std[f'Tian_subcortex_{self.subcortex_id}']
            data_mean = mod_info['mean']
            data_std = mod_info['std']
            image = torch.clamp(image, -1, 1)
            image = (image-data_mean)/data_std
        
        # Convert to tensor
        data = {}
        age = torch.tensor(age, dtype=torch.float32)
        sex = torch.tensor(sex, dtype=torch.int32)
        modality = torch.tensor(35,dtype=torch.int32)
        disease = torch.tensor(disease,dtype=torch.int32)

        data ={ 'image':image, 
                'age': age,'sex': sex,'mod':modality,
                'dis': disease
               }
        return data




