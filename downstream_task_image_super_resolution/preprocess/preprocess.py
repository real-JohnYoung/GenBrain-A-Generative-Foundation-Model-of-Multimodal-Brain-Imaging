import os
from tqdm import tqdm
import torch
import nibabel as nib
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd

root_dir = 'Path_to/ukb_multimodal'
data_file_path = "T1_image_super_resolution_test_data.csv"
save_dir = 'processed_1mm_1d'


mask_2mm_path = 'MNI152_T1_2mm_brain.nii.gz'
mask_2mm_img = nib.load(mask_2mm_path)
mask_2mm_data = mask_2mm_img.get_fdata()
mask_2mm_nonzero = (mask_2mm_data != 0)



mask_1mm_path = 'MNI152_T1_1mm_Brain.nii.gz'
mask_1mm_img = nib.load(mask_1mm_path)
mask_1mm_data = mask_1mm_img.get_fdata()
mask_1mm_nonzero = (mask_1mm_data != 0)


data_file = pd.read_csv(data_file_path)
eids = data_file['eid'].values


def downsample(data_2mm_1d, target_size=(182, 218, 182), mode='nearest'):
    try:
        image_x = mask_2mm_data.copy()
        image_x[mask_2mm_nonzero] = data_2mm_1d
        image_x = torch.FloatTensor(image_x).unsqueeze(0).unsqueeze(0)
        image_x[torch.isnan(image_x)] = 0.0
        image_x[torch.isinf(image_x)] = 0.0
        image_x = torch.nn.functional.interpolate(image_x, target_size, mode=mode)[0, 0, :, :, :]
        data_1mm_1d = image_x.numpy()[mask_1mm_nonzero]
        return data_1mm_1d
    except Exception as e:
        print(f"error: {e}")
        return None



def process_file(eid):

    data_2mm_path = os.path.join(root_dir, str(eid), 'T1_brain_nonlinear_2mm.npy')
    data_2mm_1d = np.load(data_2mm_path)
    downsample(data_2mm_1d)

    data_1mm = downsample(data_2mm_1d)
    
    if data_1mm is not None:
        save_folder = os.path.join(save_dir, str(eid))
        os.makedirs(save_folder, exist_ok=True)
        save_file = os.path.join(save_folder, 'T1_brain_to_MNI_itp.npy')
        np.save(save_file, data_1mm)
    else:
        print(f"{data_path} fail。")



with ProcessPoolExecutor(max_workers=16) as executor:
    futures = [executor.submit(process_file, eid) for eid in eids]
    for future in tqdm(as_completed(futures), total=len(futures)):
        try:
            future.result()
        except Exception as e:
            print(f"error: {e}")