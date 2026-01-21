# This file contains modulues used for Dit_1d
import torch
import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torch.nn.functional as F

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
            mni_mask = np.load('../mni_96.npy')
            data_mean = image[0][mni_mask!=0].mean()
            data_std  = image[0][mni_mask!=0].std()
            cor_data_mean = cor_image[0][mni_mask!=0].mean()
            cor_data_std  = cor_image[0][mni_mask!=0].std()
        
            image = torch.clamp(image, data_min, data_max)
            cor_image = torch.clamp(cor_image, data_min, data_max)

            image = (image-data_mean)/data_std
            cor_image = (cor_image-cor_data_mean)/cor_data_std


        # Convert to tensor
        data = {}
        data ={ 'B':image, 'A':cor_image}
        return data 