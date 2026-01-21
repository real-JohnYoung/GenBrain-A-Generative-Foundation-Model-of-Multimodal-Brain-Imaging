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




class UKBDataset_modality_translation_T1_T2_finetune_uncon(Dataset):
    def __init__(self, data_path, data_file, label_file,transform=None):

        # Read the CSV file to get labels
        self.data_path = data_path
        self.data_file = pd.read_csv(data_file)
        self.label_file = pd.read_csv(label_file)
        self.transform = transform
        
        # Convert labels to integer type
        self.label_file['31-0.0'] = self.label_file['31-0.0'].fillna(2)
        self.label_file['21003-2.0'] = self.label_file['21003-2.0'].fillna(64.275)

        self.label_file['31-0.0'] = self.label_file['31-0.0'].astype(int)
        self.label_file['21003-2.0'] = self.label_file['21003-2.0'].astype(float)

    def __len__(self):
        return len(self.data_file)
    
    def __getitem__(self, idx):

        # Get the row for the current index
        row = self.data_file.iloc[idx]
        eid = row['eid'].astype(int)
        modality = 13   # target modality
        
        age = 0
        sex = 2  

        # Construct the .npy file path
        T1_image_path = os.path.join(self.data_path, str(eid), f"T1_brain_nonlinear_2mm.npy")
        T1_image = np.load(T1_image_path)
        T1_image = torch.tensor(T1_image, dtype=torch.float32).unsqueeze(0)

        T2_image_path = os.path.join(self.data_path, str(eid), f"T2_FLAIR_brain_to_MNI.npy")
        T2_image = np.load(T2_image_path)
        T2_image = torch.tensor(T2_image, dtype=torch.float32).unsqueeze(0)

        if self.transform:

            T1_image = torch.clamp(T1_image, 0, 2e3)
            T2_image = torch.clamp(T2_image, 0, 1e3)

            T1_image = (T1_image-T1_image.mean())/T1_image.std()
            T2_image = (T2_image-T2_image.mean())/T2_image.std()
        
        # Convert to tensor
        data = {}
        age = torch.tensor(age, dtype=torch.float32)
        sex = torch.tensor(sex, dtype=torch.int32)
        modality = torch.tensor(modality,dtype=torch.int32)

        data ={ 'T1_image':T1_image, 'T2_image':T2_image, 
                'age': age,'sex': sex,'mod':modality
               }

        return data 




class UKBDataset_modality_translation_T2_T1_finetune_uncon(Dataset):
    def __init__(self, data_path, data_file, label_file,transform=None):

        # Read the CSV file to get labels
        self.data_path = data_path
        self.data_file = pd.read_csv(data_file)
        self.label_file = pd.read_csv(label_file)
        self.transform = transform
        
        # Convert labels to integer type
        self.label_file['31-0.0'] = self.label_file['31-0.0'].fillna(2)
        self.label_file['21003-2.0'] = self.label_file['21003-2.0'].fillna(64.275)

        self.label_file['31-0.0'] = self.label_file['31-0.0'].astype(int)
        self.label_file['21003-2.0'] = self.label_file['21003-2.0'].astype(float)

    def __len__(self):
        return len(self.data_file)
    
    def __getitem__(self, idx):

        # Get the row for the current index
        row = self.data_file.iloc[idx]
        eid = row['eid'].astype(int)
        modality = 11   # target modality
        
        age = 0
        sex = 2

        # Construct the .npy file path
        T1_image_path = os.path.join(self.data_path, str(eid), f"T1_brain_nonlinear_2mm.npy")
        T1_image = np.load(T1_image_path)
        T1_image = torch.tensor(T1_image, dtype=torch.float32).unsqueeze(0)

        T2_image_path = os.path.join(self.data_path, str(eid), f"T2_FLAIR_brain_to_MNI.npy")
        T2_image = np.load(T2_image_path)
        T2_image = torch.tensor(T2_image, dtype=torch.float32).unsqueeze(0)

        if self.transform:

            T1_image = torch.clamp(T1_image, 0, 2e3)
            T2_image = torch.clamp(T2_image, 0, 1e3)

            T1_image = (T1_image-T1_image.mean())/T1_image.std()
            T2_image = (T2_image-T2_image.mean())/T2_image.std()
        
        # Convert to tensor
        data = {}
        age = torch.tensor(age, dtype=torch.float32)
        sex = torch.tensor(sex, dtype=torch.int32)
        modality = torch.tensor(modality,dtype=torch.int32)

        data ={ 'T2_image':T2_image, 'T1_image':T1_image, 
                'age': age,'sex': sex,'mod':modality
               }
        return data 



class UKBDataset_modality_translation_DU15_zstat_finetune_uncon(Dataset):
    def __init__(self, data_path, data_file,image_data_info,transform=None):

        # Read the CSV file to get labels
        self.data_path = data_path
        self.data_file = pd.read_csv(data_file)
        self.data_info = image_data_info
        self.transform = transform
        
    def __len__(self):
        return len(self.data_file)
    
    def __getitem__(self, idx):

        # Get the row for the current index
        row = self.data_file.iloc[idx]
        eid = row['eid'].astype(int)
        zstat_modality = row['modality_y'].astype(int)   # target modality
        
        zstat_image_info = self.data_info[str(zstat_modality)]
        zstat_modality_name = zstat_image_info['modality']

        age = 0
        sex = 2

        # Construct the .npy file path
        zstat_image_path = os.path.join(self.data_path, str(eid), f"{zstat_modality_name}.npy")
        zstat_image = np.load(zstat_image_path)
        zstat_image = torch.tensor(zstat_image, dtype=torch.float32).unsqueeze(0)

        if self.transform:
            data_min = zstat_image_info['min']
            data_max = zstat_image_info['max']
            data_mean = zstat_image_info['mean']
            data_std = zstat_image_info['std']

            zstat_image = torch.clamp(zstat_image, data_min, data_max)
            zstat_image = (zstat_image-data_mean)/data_std

        # load DU15 image  modality:[19,33]
        DU15_image = []
        for i in range(19,34):
            du_i_image_info = self.data_info[str(i)]
            du_i_modality_name = du_i_image_info['modality']
            du_i_image_path = os.path.join(self.data_path, str(eid), f"{du_i_modality_name}.npy")
            du_i_image= np.load(du_i_image_path)
            du_i_image = torch.tensor(du_i_image, dtype=torch.float32)
            
            if self.transform:
                data_min =  du_i_image_info['min']
                data_max =  du_i_image_info['max']
                data_mean = du_i_image_info['mean']
                data_std =  du_i_image_info['std']

                du_i_image = torch.clamp(du_i_image, data_min, data_max)
                du_i_image = (du_i_image-data_mean)/data_std

            DU15_image.append(du_i_image)
        DU15_image = np.array(DU15_image)
        DU15_image = torch.tensor(DU15_image)


        # Convert to tensor
        data = {}
        age = torch.tensor(age, dtype=torch.float32)
        sex = torch.tensor(sex, dtype=torch.int32)
        zstat_modality = torch.tensor(zstat_modality,dtype=torch.int32)

        data ={ 'DU15_image':DU15_image, 'zstat_image':zstat_image, 
                'age': age,'sex': sex,'mod':zstat_modality
               }
        return data 




class UKBDataset_modality_translation_DU15_zstat_evaluation_uncon(Dataset):
    def __init__(self, data_path, data_file,image_data_info,transform=None):

        # Read the CSV file to get labels
        self.data_path = data_path
        self.data_file = pd.read_csv(data_file)
        self.data_info = image_data_info
        self.transform = transform
        

    def __len__(self):
        return len(self.data_file)
    
    def __getitem__(self, idx):

        # Get the row for the current index
        row = self.data_file.iloc[idx]
        eid = row['eid'].astype(int)
        zstat_modality = row['modality_y'].astype(int)   # target modality
        
        zstat_image_info = self.data_info[str(zstat_modality)]
        zstat_modality_name = zstat_image_info['modality']

        age = 0
        sex = 2

        # Construct the .npy file path
        zstat_image_path = os.path.join(self.data_path, str(eid), f"{zstat_modality_name}.npy")
        zstat_image = np.load(zstat_image_path)
        zstat_image = torch.tensor(zstat_image, dtype=torch.float32).unsqueeze(0)

        if self.transform:
            data_min = zstat_image_info['min']
            data_max = zstat_image_info['max']
            data_mean = zstat_image_info['mean']
            data_std = zstat_image_info['std']

            zstat_image = torch.clamp(zstat_image, data_min, data_max)
            zstat_image = (zstat_image-data_mean)/data_std

        # load DU15 image  modality:[19,33]
        DU15_image = []
        for i in range(19,34):
            du_i_image_info = self.data_info[str(i)]
            du_i_modality_name = du_i_image_info['modality']
            du_i_image_path = os.path.join(self.data_path, str(eid), f"{du_i_modality_name}.npy")
            du_i_image= np.load(du_i_image_path)
            du_i_image = torch.tensor(du_i_image, dtype=torch.float32)
            
            if self.transform:
                data_min =  du_i_image_info['min']
                data_max =  du_i_image_info['max']
                data_mean = du_i_image_info['mean']
                data_std =  du_i_image_info['std']

                du_i_image = torch.clamp(du_i_image, data_min, data_max)
                du_i_image = (du_i_image-data_mean)/data_std

            DU15_image.append(du_i_image)
        DU15_image = np.array(DU15_image)
        DU15_image = torch.tensor(DU15_image)


        # Convert to tensor
        data = {}
        age = torch.tensor(age, dtype=torch.float32)
        sex = torch.tensor(sex, dtype=torch.int32)
        zstat_modality = torch.tensor(zstat_modality,dtype=torch.int32)

        data ={ 'eid':eid, 'DU15_image':DU15_image, 'zstat_image':zstat_image, 
                'age': age,'sex': sex,'mod':zstat_modality
               }
        return data 

class UKBDataset_modality_translation_DWI_DU15_finetune_uncon(Dataset):
    def __init__(self, data_path, data_file,image_data_info,transform=None):

        # Read the CSV file to get labels
        self.data_path = data_path
        self.data_file = pd.read_csv(data_file)
        self.data_info = image_data_info
        self.transform = transform
        
    def __len__(self):
        return len(self.data_file)
    
    def __getitem__(self, idx):

        # Get the row for the current index
        row = self.data_file.iloc[idx]
        eid = row['eid'].astype(int)
        du_modality = row['modality_y'].astype(int)   # target modality
        
        du_image_info = self.data_info[str(du_modality)]
        du_modality_name = du_image_info['modality']

        age = 0
        sex = 2

        # Construct the .npy file path
        du_image_path = os.path.join(self.data_path, str(eid), f"{du_modality_name}.npy")
        du_image = np.load(du_image_path)
        du_image = torch.tensor(du_image, dtype=torch.float32).unsqueeze(0)

        if self.transform:
            data_min = du_image_info['min']
            data_max = du_image_info['max']
            data_mean = du_image_info['mean']
            data_std = du_image_info['std']

            du_image = torch.clamp(du_image, data_min, data_max)
            du_image = (du_image-data_mean)/data_std

        # load DWI image  modality:[0,8]
        DWI_image = []
        for i in range(0,9):
            dwi_i_image_info = self.data_info[str(i)]
            dwi_i_modality_name = dwi_i_image_info['modality']
            dwi_i_image_path = os.path.join(self.data_path, str(eid), f"{dwi_i_modality_name}.npy")
            dwi_i_image= np.load(dwi_i_image_path)
            dwi_i_image = torch.tensor(dwi_i_image, dtype=torch.float32)
            
            if self.transform:
                data_min =  dwi_i_image_info['min']
                data_max =  dwi_i_image_info['max']
                data_mean = dwi_i_image_info['mean']
                data_std =  dwi_i_image_info['std']

                dwi_i_image = torch.clamp(dwi_i_image, data_min, data_max)
                dwi_i_image = (dwi_i_image-data_mean)/data_std

            DWI_image.append(dwi_i_image)
        DWI_image = np.array(DWI_image)
        DWI_image = torch.tensor(DWI_image)


        # Convert to tensor
        data = {}
        age = torch.tensor(age, dtype=torch.float32)
        sex = torch.tensor(sex, dtype=torch.int32)
        du_modality = torch.tensor(du_modality,dtype=torch.int32)

        data ={ 'DWI_image':DWI_image, 'du_image':du_image, 
                'age': age,'sex': sex,'mod':du_modality
               }
        return data 




class UKBDataset_modality_translation_DWI_DU15_evaluation_uncon(Dataset):
    def __init__(self, data_path, data_file,image_data_info,transform=None):

        # Read the CSV file to get labels
        self.data_path = data_path
        self.data_file = pd.read_csv(data_file)
        self.data_info = image_data_info
        self.transform = transform
        
    def __len__(self):
        return len(self.data_file)
    
    def __getitem__(self, idx):

        # Get the row for the current index
        row = self.data_file.iloc[idx]
        eid = row['eid'].astype(int)
        du_modality = row['modality_y'].astype(int)   # target modality
        
        du_image_info = self.data_info[str(du_modality)]
        du_modality_name = du_image_info['modality']

        age = 0
        sex = 2

        # Construct the .npy file path
        du_image_path = os.path.join(self.data_path, str(eid), f"{du_modality_name}.npy")
        du_image = np.load(du_image_path)
        du_image = torch.tensor(du_image, dtype=torch.float32).unsqueeze(0)

        if self.transform:
            data_min = du_image_info['min']
            data_max = du_image_info['max']
            data_mean = du_image_info['mean']
            data_std = du_image_info['std']

            du_image = torch.clamp(du_image, data_min, data_max)
            du_image = (du_image-data_mean)/data_std

        # load DWI image  modality:[0,8]
        DWI_image = []
        for i in range(0,9):
            dwi_i_image_info = self.data_info[str(i)]
            dwi_i_modality_name = dwi_i_image_info['modality']
            dwi_i_image_path = os.path.join(self.data_path, str(eid), f"{dwi_i_modality_name}.npy")
            dwi_i_image= np.load(dwi_i_image_path)
            dwi_i_image = torch.tensor(dwi_i_image, dtype=torch.float32)
            
            if self.transform:
                data_min =  dwi_i_image_info['min']
                data_max =  dwi_i_image_info['max']
                data_mean = dwi_i_image_info['mean']
                data_std =  dwi_i_image_info['std']

                dwi_i_image = torch.clamp(dwi_i_image, data_min, data_max)
                dwi_i_image = (dwi_i_image-data_mean)/data_std

            DWI_image.append(dwi_i_image)
        DWI_image = np.array(DWI_image)
        DWI_image = torch.tensor(DWI_image)


        # Convert to tensor
        data = {}
        age = torch.tensor(age, dtype=torch.float32)
        sex = torch.tensor(sex, dtype=torch.int32)
        du_modality = torch.tensor(du_modality,dtype=torch.int32)

        data ={ 'eid':eid, 'DWI_image':DWI_image, 'du_image':du_image, 
                'age': age,'sex': sex,'mod':du_modality
               }
        return data 




class UKBDataset_modality_translation_T1_T2_1mm_finetune_uncon(Dataset):
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
        
        modality = 13   # target modality FLAIR
        age = 0
        sex = 2

        # Construct the .npy file path
        T1_image_path = os.path.join(self.data_path, str(eid), f"T1_brain_to_MNI.npy")
        T1_image = np.load(T1_image_path)
        T1_image = torch.tensor(T1_image, dtype=torch.float32)

        T2_image_path = os.path.join(self.data_path, str(eid), f"T2_FLAIR_brain_to_MNI.npy")
        T2_image = np.load(T2_image_path)
        T2_image = torch.tensor(T2_image, dtype=torch.float32)

        if self.transform:

            T1_image = torch.clamp(T1_image, 0, 2e3)
            T2_image = torch.clamp(T2_image, 0, 1e3)

            T1_image = (T1_image-T1_image.mean())/T1_image.std()
            T2_image = (T2_image-T2_image.mean())/T2_image.std()
        
            T1_image = T1_image[start:end+1].unsqueeze(0)
            T2_image = T2_image[start:end+1].unsqueeze(0)

        # Convert to tensor
        data = {}
        age = torch.tensor(age, dtype=torch.float32)
        sex = torch.tensor(sex, dtype=torch.int32)
        modality = torch.tensor(modality,dtype=torch.int32)

        data ={ 'T1_image':T1_image, 'T2_image':T2_image, 
                'age': age,'sex': sex,'mod':modality,
                'image_part':part_index
               }

        return data 


class UKBDataset_modality_translation_T2_T1_1mm_finetune_uncon(Dataset):
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
        
        modality = 11   # target modality T1
        age = 0
        sex = 2

        # Construct the .npy file path
        T1_image_path = os.path.join(self.data_path, str(eid), f"T1_brain_to_MNI.npy")
        T1_image = np.load(T1_image_path)
        T1_image = torch.tensor(T1_image, dtype=torch.float32)

        T2_image_path = os.path.join(self.data_path, str(eid), f"T2_FLAIR_brain_to_MNI.npy")
        T2_image = np.load(T2_image_path)
        T2_image = torch.tensor(T2_image, dtype=torch.float32)

        if self.transform:

            T1_image = torch.clamp(T1_image, 0, 2e3)
            T2_image = torch.clamp(T2_image, 0, 1e3)

            T1_image = (T1_image-T1_image.mean())/T1_image.std()
            T2_image = (T2_image-T2_image.mean())/T2_image.std()
        
            T1_image = T1_image[start:end+1].unsqueeze(0)
            T2_image = T2_image[start:end+1].unsqueeze(0)

        # Convert to tensor
        data = {}
        age = torch.tensor(age, dtype=torch.float32)
        sex = torch.tensor(sex, dtype=torch.int32)
        modality = torch.tensor(modality,dtype=torch.int32)

        data ={ 'T1_image':T1_image, 'T2_image':T2_image, 
                'age': age,'sex': sex,'mod':modality,
                'image_part':part_index
               }

        return data 