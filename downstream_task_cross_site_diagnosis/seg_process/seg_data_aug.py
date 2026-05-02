import os
import numpy as np
import pandas as pd
import nibabel as nib
from tqdm import tqdm
import subprocess

data_step = 20000

data_file = "Path_To/adni_nc_ad_mci/data_aug_labels/train_data_augmented_files.csv"
origin_1d_dir = f"Path_To/ADNI_T1/finetune_adni_nc_mci_ad_T1/000-DiT-UKB-L-256-finetune_mode-full/data_aug/step_{data_step}"
origin_3d_save_dir = f"Path_To/ADNI_T1/finetune_adni_nc_mci_ad_T1/000-DiT-UKB-L-256-finetune_mode-full/data_aug_seg/data_3d/step_{data_step}"
seg_3d_save_dir = f"Path_To/ADNI_T1/finetune_adni_nc_mci_ad_T1/000-DiT-UKB-L-256-finetune_mode-full/data_aug_seg/seg_3d/step_{data_step}"
seg_csv_save_dir = f"Path_To/ADNI_T1/finetune_adni_nc_mci_ad_T1/000-DiT-UKB-L-256-finetune_mode-full/data_aug_seg/seg_csv/step_{data_step}"
inference_path = "Path_To/freesurfer-freesurfer-dev-mri_WMHsynthseg/WMHSynthSeg/inference.py"
os.makedirs(origin_3d_save_dir,exist_ok=True)
os.makedirs(seg_3d_save_dir,exist_ok=True)
os.makedirs(seg_csv_save_dir,exist_ok=True)


mask_path = "Path_To/preprocess/MNI152_T1_2mm_brain.nii.gz"
mask_img = nib.load(mask_path)

def save_1d_to_3d(data_dir,save_dir,id,mask):
    
    data_path = os.path.join(data_dir, id+"_T1_0.npy")
    data_1d = np.load(data_path).reshape(-1)
    data_1d = np.clip(data_1d,-3,3)
    mask_data = mask.get_fdata()
    mask_nonzero = (mask_data != 0)
    data_3d = mask_data.copy()
    data_3d[~mask_nonzero] = -3
    data_3d[mask_nonzero] = data_1d
    data_3d = nib.Nifti1Image(data_3d, affine=mask.affine, header=mask.header)
    save_path = os.path.join(save_dir, id +".nii.gz")
    nib.save(data_3d, save_path)



def run_inference(inference_path, id,origin_3d_save_dir, seg_3d_save_dir, seg_csv_save_dir, device="cuda"):

    input_path = os.path.join(origin_3d_save_dir,id + ".nii.gz")
    output_path = os.path.join(seg_3d_save_dir,id +"_seg.nii.gz")
    csv_vols_path = os.path.join(seg_csv_save_dir,id+"_seg.csv")
    command = [
                "python", inference_path,
                "--i", input_path,
                "--o", output_path,
                "--csv_vols", csv_vols_path,
                "--device", device,
                "--threads", str(4)
                ]
    try:
        result = subprocess.run(command, check=True, text=True, capture_output=True)
        print("Inference completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error during inference execution: {e}")


data_df = pd.read_csv(data_file)
id_list = data_df['ID'].to_list()

for id in tqdm(id_list):
    save_1d_to_3d(origin_1d_dir, origin_3d_save_dir, id, mask_img)
    run_inference(inference_path, id, origin_3d_save_dir, seg_3d_save_dir, seg_csv_save_dir)
