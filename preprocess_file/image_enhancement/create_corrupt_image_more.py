import os 
import logging
import numpy as np
import pandas as pd
import nibabel as nib
from tqdm import tqdm
import scipy.ndimage as ndimage
from motion_artifact_simulation import add_motion_artifacts
from noise_simulation import add_gaussian_noise_more, add_rician_noise_more

# create corrupt MRI image 

sub_file_path = '../image_quality_enhancement/image_quality_enhancement_training_data_2000.csv'
data_root_1 = 'UKB_FILE_for_T1w'        # nii.gz file
data_root_2 = 'UKB_FILE_for_FLAIR'      # 1d file
save_root = '/corrupted_data_for_3d'    # 3d
save_1d_root = '/corrupted_data_for_1d' # 1d
atlas_path = '../MNI152_T1_2mm_brain.nii.gz'
log_file = './process_log/testing_log_file.txt'
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

sub_file = pd.read_csv(sub_file_path)
data_num = len(sub_file)

atlas = nib.load(atlas_path)
atlas_data = atlas.get_fdata()
nonzero = (atlas_data != 0)

for i in tqdm(range(data_num)):
    x = sub_file.iloc[i]
    eid = x['eid']
    modality = x['modality']
    
    if modality == 11:
        modality_name = 'T1_brain_nonlinear_2mm'
        data_path = os.path.join(data_root_1, str(eid), modality_name + '.nii.gz')
        image = nib.load(data_path)
        image_data = image.get_fdata()

    elif modality == 13:
        modality_name = 'T2_FLAIR_brain_to_MNI'
        data_path = os.path.join(data_root_2, str(eid), modality_name + '_2mm' + '.npy')
        image_data = np.load(data_path)
    

    save_dir = os.path.join(save_root, str(eid))
    os.makedirs(save_dir, exist_ok=True)

    save_1d_dir = os.path.join(save_1d_root, str(eid))
    os.makedirs(save_1d_dir, exist_ok=True)

    num_movements = np.random.randint(0, 4)           # 0~3 movements
    gaussian_noise_level = np.random.randint(0, 11)   # 10 noise levels
    rician_noise_level = np.random.randint(0, 11)     # 10 noise levels


    image_data 
    if num_movements>0:
        image_data = add_motion_artifacts(image_data, num_movements, rotation_range=(-30, 30), translation_range=(-5, 5), rotation_lam=10, translation_lam=3, phase_direction='X')
    if gaussian_noise_level > 0:
        image_data = add_gaussian_noise_more(image_data, gaussian_noise_level)
    if rician_noise_level > 0:
        image_data = add_rician_noise_more(image_data, rician_noise_level)

    log_text = f'sub {eid} {modality_name} motion artifact num_movements: {num_movements}, gaussian noise level: {gaussian_noise_level}, rician noise level: {rician_noise_level}.'

    corrupt_image = nib.Nifti1Image(image_data, atlas.affine)

    # save 3d data
    save_path = os.path.join(save_dir, modality_name + '.nii.gz')
    nib.save(corrupt_image, save_path)

    # save 1d data
    save_1d_path = os.path.join(save_1d_dir, modality_name + '.npy')
    corrupt_image_1d = corrupt_image.get_fdata()[nonzero]
    np.save(save_1d_path, corrupt_image_1d)

    logging.info(log_text)
