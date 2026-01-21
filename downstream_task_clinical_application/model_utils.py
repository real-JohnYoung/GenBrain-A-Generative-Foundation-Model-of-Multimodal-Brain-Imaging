# This file contains modulues used for Dit_1d
import torch
import torch.nn as nn
import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torch.nn.functional as F

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


# This dataset is used for downstream task [ for ADC modality].
class UKBDataset_clinical_4889_finetune(Dataset):
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
        sub_id= row['participant_id']
        modality = 35                            # for ADC modality 35 new modality embedder
        
        age = row['age']
        sex = row['sex']
        if sex == 'F':
            sex = 0
        elif sex == 'M':
            sex =1
        score = row['nihss']

        image_path = os.path.join(self.data_path, sub_id+f'_ADC_brain_MNI.npy')
        image = np.load(image_path)
        image = np.nan_to_num(image)

        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)

        if self.transform: 
            image = torch.clamp(image, 0, 36712)
            #image = (image-7656)/12397
            image = (image -image.mean())/image.std()
        
        # Convert to tensor
        data = {}
        age = torch.tensor(age, dtype=torch.float32)
        sex = torch.tensor(sex, dtype=torch.int32)
        modality = torch.tensor(modality,dtype=torch.int32)
        score = torch.tensor(score,dtype=torch.float32)

        data ={ 'image':image, 
                'age': age,'sex': sex,'mod':modality,
                'score': score
               }
        return data

# This dataset is used for downstream task [ for FA modality].
class UKBDataset_clinical_4884_finetune(Dataset):
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
        sub_id= row['participant_id']
        modality = 0
        age = row['age_at_stroke']
        sex = row['sex']
        if sex == 'F':
            sex = 0
        elif sex == 'M':
            sex =1
        score = row['wab_aq']  # WAB (Western Aphasia Battery-Revised) Aphasia Quotient

        image_path = os.path.join(self.data_path, sub_id+f'_all_FA.npy')
        image = np.load(image_path)
        image = np.nan_to_num(image)

        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)

        if self.transform: 
            image = torch.clamp(image, 0, 1)
            image = (image-0.162)/0.152
        
        # Convert to tensor
        data = {}
        age = torch.tensor(age, dtype=torch.float32)
        sex = torch.tensor(sex, dtype=torch.int32)
        modality = torch.tensor(modality,dtype=torch.int32)
        score = torch.tensor(score,dtype=torch.float32)

        data ={ 'image':image, 
                'age': age,'sex': sex,'mod':modality,
                'score': score
               }
        return data