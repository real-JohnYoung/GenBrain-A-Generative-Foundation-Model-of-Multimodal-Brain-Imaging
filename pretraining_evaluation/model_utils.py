# This file contains modulues used for Dit_1d
import torch
import torch.nn as nn
import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import nibabel as nib
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


class UKBDataset(Dataset):
    def __init__(self, data_path, data_file, label_file, image_data_info,transform=None):
        """
        Initializes the dataset with paths to label CSV file and data directory.

        Args:
        - data_path (str): Path to the directory containing .npy files.
        - data_file (str): Path to the CSV file containing data info.
        - label_file (str): Path to the CSV file containing labels.
        - image_data_info (dict)
        - transforms
        """
        # Read the CSV file to get labels
        self.data_path = data_path
        self.data_file = pd.read_csv(data_file)
        self.label_file = pd.read_csv(label_file)
        self.transform = transform
        self.data_info = image_data_info
        
        # Convert labels to integer type
        self.label_file['31-0.0'] = self.label_file['31-0.0'].fillna(2)
        self.label_file['21003-2.0'] = self.label_file['21003-2.0'].fillna(64.275)

        self.label_file['31-0.0'] = self.label_file['31-0.0'].astype(int)
        self.label_file['21003-2.0'] = self.label_file['21003-2.0'].astype(float)

    def __len__(self):
        """
        Returns the number of samples in the dataset.
        """
        return len(self.data_file)
    
    def __getitem__(self, idx):
        """
        Returns a sample from the dataset.

        Args:
        - idx (int): Index of the sample to retrieve.

        Returns: 
        A data dictionary, which contains:

        Imaging Data:
        - image: The 1D image data as a tensor.

        Meta Data:
        - age (torch.Tensor): The age label as a tensor.
        - sex (torch.Tensor): The sex label as a tensor.
        - mod (torch.Tensor): The modality "mod" as a tensor.
        """
        # Get the row for the current index
        row = self.data_file.iloc[idx]
        eid = row['eid'].astype(int)
        modality = row['modality'].astype(int)
        
        condition = self.label_file.query(f'eid == {eid}')
        age = condition['21003-2.0']
        sex = condition['31-0.0']                 

        image_info = self.data_info[str(modality)]
        modality_name = image_info['modality']

        # Construct the .npy file path
        image_path = os.path.join(self.data_path, str(eid), f"{modality_name}.npy")
        image = np.load(image_path)
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)

        if self.transform:
            data_min = image_info['min']
            data_max = image_info['max']
            image = torch.clamp(image, data_min, data_max)
            if modality_name in ["T1_brain_nonlinear_2mm",'SWI_2mm','T2_FLAIR_brain_to_MNI']:
                mean = image.mean()
                std = image.std()
            else: 
                mean = image_info['mean']
                std = image_info['std']
            
            image = (image-mean)/std
        
        # Convert to tensor
        data = {}
        age = torch.tensor(age.to_numpy(), dtype=torch.float32)
        sex = torch.tensor(sex.to_numpy(), dtype=torch.int32)
        modality = torch.tensor(modality,dtype=torch.int32)
        data ={ 'image':image, 
                'age': age,'sex': sex,'mod':modality
               }
        return data 

