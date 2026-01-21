# This file contains modulues used for Dit_1d
import torch
import torch.nn as nn
import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import nibabel as nib
import torch.nn.functional as F
import csv

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



class UKBDataset_enhancement_finetune(Dataset):
    def __init__(self, data_path, cor_data_path, data_file, image_data_info,transform=None):

        # Read the CSV file to get labels
        self.data_path = data_path
        self.cor_data_path = cor_data_path
        self.data_file = pd.read_csv(data_file)
        self.transform = transform
        self.data_info = image_data_info

    def __len__(self):
        return len(self.data_file)
    
    def __getitem__(self, idx):
        """
        Finetune Label:
        - cor_image (torch.Tensor): The corrupted image.  # [corrupt image]
        """
        # Get the row for the current index
        row = self.data_file.iloc[idx]
        eid = row['eid'].astype(int)
        modality = row['modality'].astype(int)
        
        age = 0  # unkown  condition
        sex = 2  # unknown condition           

        image_info = self.data_info[str(modality)]
        modality_name = image_info['modality']

        # Construct the .npy file path
        image_path = os.path.join(self.data_path, str(eid), f"{modality_name}.npy")
        image = np.load(image_path)
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)

        cor_image_path = os.path.join(self.cor_data_path, str(eid), f"{modality_name}.npy")
        cor_image = np.load(cor_image_path)
        cor_image = torch.tensor(cor_image, dtype=torch.float32).unsqueeze(0)

        if self.transform:
            data_min = image_info['min']
            data_max = image_info['max']
            image = torch.clamp(image, data_min, data_max)
            if modality_name in ["T1_brain_nonlinear_2mm",'SWI_2mm','T2_FLAIR_brain_to_MNI']:
                mean = image.mean()
                std = image.std()
                cor_mean = cor_image.mean()
                cor_std = cor_image.std()
            else: 
                mean = image_info['mean']
                std = image_info['std']

            image = (image-mean)/std
            cor_image = (cor_image-cor_mean)/cor_std
        
        # Convert to tensor
        data = {}
        age = torch.tensor(age, dtype=torch.float32)
        sex = torch.tensor(sex, dtype=torch.int32)
        modality = torch.tensor(modality,dtype=torch.int32)
        data ={ 'image':image, 'cor_image':cor_image, 
                'age': age,'sex': sex,'mod':modality
               }
        return data 



class UKBDataset_enhancement_finetune_s(Dataset):
    def __init__(self, data_path, cor_data_path, data_file, image_data_info,transform=None):

        # Read the CSV file to get labels
        self.data_path = data_path
        self.cor_data_path = cor_data_path
        self.data_file = pd.read_csv(data_file)
        self.transform = transform
        self.data_info = image_data_info

    def __len__(self):
        return len(self.data_file)
    
    def __getitem__(self, idx):
        """
        Finetune Label:
        - cor_image (torch.Tensor): The corrupted image.  # [corrupt image]
        """
        # Get the row for the current index
        row = self.data_file.iloc[idx]
        eid = row['eid'].astype(int)
        file_index= row['file_index'].astype(int)
        modality = row['modality'].astype(int)
        
        age = 0  # unkown  condition
        sex = 2  # unknown condition           

        image_info = self.data_info[str(modality)]
        modality_name = image_info['modality']

        # Construct the .npy file path
        image_path = os.path.join(self.data_path, str(eid), f"{modality_name}.npy")
        image = np.load(image_path)
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)

        cor_image_path = os.path.join(self.cor_data_path, str(eid), f"{modality_name}_{file_index}.npy")
        cor_image = np.load(cor_image_path)
        cor_image = torch.tensor(cor_image, dtype=torch.float32).unsqueeze(0)

        if self.transform:# for T1, FLAIR, SWI
            data_min = image_info['min']
            data_max = image_info['max']
            image = torch.clamp(image, data_min, data_max)
            cor_image = torch.clamp(cor_image, data_min, data_max)
            image = (image-image.mean())/image.std()
            cor_image = (cor_image-cor_image.mean())/cor_image.std()


        # Convert to tensor
        data = {}
        age = torch.tensor(age, dtype=torch.float32)
        sex = torch.tensor(sex, dtype=torch.int32)
        modality = torch.tensor(modality,dtype=torch.int32)
        data ={ 'image':image, 'cor_image':cor_image, 
                'age': age,'sex': sex,'mod':modality
               }
        return data 