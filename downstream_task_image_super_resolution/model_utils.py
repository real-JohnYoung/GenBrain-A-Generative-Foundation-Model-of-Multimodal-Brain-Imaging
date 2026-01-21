# This file contains modulues used for Dit_1d
import os
import torch
import torch.nn as nn
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


class UKBDataset_T1_super_resolution(Dataset):
    def __init__(self, data_path, data_file,transform=None):

        # Read the CSV file to get labels
        self.data_path = data_path
        self.data_file = pd.read_csv(data_file)
        self.transform = transform
        self.image_part = {
                'part_0': {'start': 0, 'end': 228452},
                'part_1': {'start': 228377, 'end': 456829},
                'part_2': {'start': 456754, 'end': 685206},
                'part_3': {'start': 685131, 'end': 913583},
                'part_4': {'start': 913508, 'end': 1141960},
                'part_5': {'start': 1141885, 'end': 1370337},
                'part_6': {'start': 1370262, 'end': 1598714},
                'part_7': {'start': 1598642, 'end': 1827094}
            }


    def __len__(self):
        return len(self.data_file)
    
    def __getitem__(self, idx):

        # Get the row for the current index
        row = self.data_file.iloc[idx]
        eid = row['eid'].astype(int)
        part_index = row['part'].astype(int)
        start = self.image_part[f'part_{part_index}']['start']
        end   = self.image_part[f'part_{part_index}']['end']

        modality = 11    # target modality
        
        age = 0
        sex = 2  

        # T1 image: 1mm T1 image
        # T1 image itp : 2mm T1 image interpolate to 1mm T1 image

        T1_image_path = os.path.join(self.data_path, str(eid), f"T1_brain_to_MNI.npy")
        T1_image = np.load(T1_image_path)
        T1_image = torch.tensor(T1_image, dtype=torch.float32)

        T1_image_itp_path = os.path.join(self.data_path, str(eid), f"T1_brain_to_MNI_itp.npy")
        T1_image_itp = np.load(T1_image_itp_path)
        T1_image_itp = torch.tensor(T1_image_itp, dtype=torch.float32)

        if self.transform:

            T1_image = torch.clamp(T1_image, 0, 2e3)
            T1_image_itp = torch.clamp(T1_image_itp, 0, 2e3)

            T1_image = (T1_image - T1_image.mean())/T1_image.std()
            T1_image_itp = (T1_image_itp - T1_image_itp.mean())/T1_image_itp.std()

            T1_image = T1_image[start:end+1].unsqueeze(0)
            T1_image_itp = T1_image_itp[start:end+1].unsqueeze(0)
        
        # Convert to tensor
        data = {}
        age = torch.tensor(age, dtype=torch.float32)
        sex = torch.tensor(sex, dtype=torch.int32)
        modality = torch.tensor(modality,dtype=torch.int32)
        part_index = torch.tensor(part_index,dtype=torch.int32)

        data ={ 'T1_image':T1_image, 'T1_image_itp':T1_image_itp, 
                'age': age,'sex': sex,'mod':modality,'image_part':part_index
               }

        return data 



class UKBDataset_T1_super_resolution_eval(Dataset):
    def __init__(self, data_path, data_file,transform=None):

        # Read the CSV file to get labels
        self.data_path = data_path
        self.data_file = pd.read_csv(data_file)
        self.transform = transform
        self.image_part = {
                'part_0': {'start': 0, 'end': 228452},
                'part_1': {'start': 228377, 'end': 456829},
                'part_2': {'start': 456754, 'end': 685206},
                'part_3': {'start': 685131, 'end': 913583},
                'part_4': {'start': 913508, 'end': 1141960},
                'part_5': {'start': 1141885, 'end': 1370337},
                'part_6': {'start': 1370262, 'end': 1598714},
                'part_7': {'start': 1598642, 'end': 1827094}
            }


    def __len__(self):
        return len(self.data_file)
    
    def __getitem__(self, idx):

        # Get the row for the current index
        row = self.data_file.iloc[idx]
        eid = row['eid'].astype(int)
        part_index = row['part'].astype(int)
        start = self.image_part[f'part_{part_index}']['start']
        end   = self.image_part[f'part_{part_index}']['end']

        modality = 11    # target modality
        
        age = 0
        sex = 2  

        # T1 image: 1mm T1 image
        # T1 image itp : 2mm T1 image interpolate to 1mm T1 image

        T1_image_itp_path = os.path.join(self.data_path, str(eid), f"T1_brain_to_MNI_itp.npy")
        T1_image_itp = np.load(T1_image_itp_path)
        image_mean = T1_image_itp.mean()
        image_std = T1_image_itp.std()
        T1_image_itp = T1_image_itp[start:end+1]
        T1_image_itp = torch.tensor(T1_image_itp, dtype=torch.float32).unsqueeze(0)

        if self.transform:
            T1_image_itp = torch.clamp(T1_image_itp, 0, 2e3)
            T1_image_itp = (T1_image_itp - image_mean)/image_std
        
        # Convert to tensor
        data = {}
        age = torch.tensor(age, dtype=torch.float32)
        sex = torch.tensor(sex, dtype=torch.int32)
        modality = torch.tensor(modality,dtype=torch.int32)
        part_index = torch.tensor(part_index,dtype=torch.int32)

        data ={ 'eid':eid, 'T1_image_itp':T1_image_itp, 
                'age': age,'sex': sex,'mod':modality,'image_part':part_index
               }

        return data 



class UKBDataset_T1_super_resolution_syn(Dataset):
    def __init__(self, data_path, data_file,transform=None):

        # Read the CSV file to get labels
        self.data_path = data_path
        self.data_file = pd.read_csv(data_file)
        self.transform = transform
        self.image_part = {
                'part_0': {'start': 0, 'end': 228452},
                'part_1': {'start': 228377, 'end': 456829},
                'part_2': {'start': 456754, 'end': 685206},
                'part_3': {'start': 685131, 'end': 913583},
                'part_4': {'start': 913508, 'end': 1141960},
                'part_5': {'start': 1141885, 'end': 1370337},
                'part_6': {'start': 1370262, 'end': 1598714},
                'part_7': {'start': 1598642, 'end': 1827094}
            }


    def __len__(self):
        return len(self.data_file)
    
    def __getitem__(self, idx):

        # Get the row for the current index
        row = self.data_file.iloc[idx]
        eid = row['eid'].astype(int)
        part_index = row['part'].astype(int)
        start = self.image_part[f'part_{part_index}']['start']
        end   = self.image_part[f'part_{part_index}']['end']

        modality = 11    # target modality
        
        age = 0
        sex = 2  

        # T1 image: 1mm T1 image
        # T1 image itp : 2mm T1 image interpolate to 1mm T1 image

        T1_image_itp_path = os.path.join(self.data_path, str(eid), f"T1_brain_to_MNI_itp.npy")
        T1_image_itp = np.load(T1_image_itp_path)
        image_mean = T1_image_itp.mean()
        image_std = T1_image_itp.std()
        T1_image_itp = T1_image_itp[start:end+1]
        T1_image_itp = torch.tensor(T1_image_itp, dtype=torch.float32).unsqueeze(0)

        if self.transform:
            T1_image_itp = (T1_image_itp - image_mean)/image_std
        
        # Convert to tensor
        data = {}
        age = torch.tensor(age, dtype=torch.float32)
        sex = torch.tensor(sex, dtype=torch.int32)
        modality = torch.tensor(modality,dtype=torch.int32)
        part_index = torch.tensor(part_index,dtype=torch.int32)

        data ={ 'eid':eid, 'T1_image_itp':T1_image_itp, 
                'age': age,'sex': sex,'mod':modality,'image_part':part_index
               }

        return data 