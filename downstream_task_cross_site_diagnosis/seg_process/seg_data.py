import os
import numpy as np
import pandas as pd
import nibabel as nib
from tqdm import tqdm
import subprocess


data_file = "/Path_To/downstream_task_data_augmentation/adni_nc_ad_mci/labels/adni_info.csv"

seg_3d_save_dir = f"Path_To/ADNI_T1/seg_3d"
seg_csv_save_dir = f"Path_To/ADNI_T1/seg_csv"
inference_path = "/Path_To/freesurfer-freesurfer-dev-mri_WMHsynthseg/WMHSynthSeg/inference.py"
os.makedirs(seg_3d_save_dir,exist_ok=True)
os.makedirs(seg_csv_save_dir,exist_ok=True)



def run_inference(inference_path, sub_info, seg_3d_save_dir, seg_csv_save_dir, device="cuda"):

    input_path = sub_info['Path']
    file_name = sub_info['ID']+'_'+str(sub_info['Age'])+'_'+str(sub_info['Sex'])+'_'+str(sub_info['Diagnosis'])
    output_path = os.path.join(seg_3d_save_dir,file_name +"_seg.nii.gz")
    csv_vols_path = os.path.join(seg_csv_save_dir,file_name+"_seg.csv")

    command = [
                "python", inference_path,
                "--i", input_path,
                "--o", output_path,
                "--csv_vols", csv_vols_path,
                "--device", device
                ]
    try:
        result = subprocess.run(command, check=True, text=True, capture_output=True)
        print("Inference completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error during inference execution: {e}")


data_df = pd.read_csv(data_file)

n = len(data_df)

for i in tqdm(range(n//2,n)):
    sub_info = data_df.iloc[i]
    run_inference(inference_path, sub_info, seg_3d_save_dir, seg_csv_save_dir)
