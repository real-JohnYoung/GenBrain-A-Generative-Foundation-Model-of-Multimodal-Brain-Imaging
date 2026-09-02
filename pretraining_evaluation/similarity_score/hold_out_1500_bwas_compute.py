import pandas as pd
import os
import numpy as np
from tqdm import tqdm


def BWAS_correlation(fMRI_2D_1, fMRI_2D_2):
    fMRI_2D_1 = (fMRI_2D_1 - fMRI_2D_1.mean(axis=0)) / fMRI_2D_1.std(axis=0)
    fMRI_2D_2 = (fMRI_2D_2 - fMRI_2D_2.mean(axis=0)) / fMRI_2D_2.std(axis=0)
    r = np.dot(np.transpose(fMRI_2D_1), fMRI_2D_2) / fMRI_2D_1.shape[0]
    return r

modalities =['dti_FA_2mm', 'dti_L1_2mm', 'dti_L2_2mm', 'dti_L3_2mm', 'dti_MD_2mm',
              'dti_MO_2mm','NODDI_ICVF_2mm','NODDI_ISOVF_2mm','NODDI_OD_2mm','QSM_2mm','SWI_2mm',
              'T1_brain_nonlinear_2mm','T1_warp_Jac_2mm','T2_FLAIR_brain_to_MNI','T2star_2mm', 'VBM_2mm',
              'zstat1s','zstat2s','zstat5s','DU15_1','DU15_2','DU15_3','DU15_4','DU15_5',
              'DU15_6','DU15_7','DU15_8','DU15_9','DU15_10','DU15_11','DU15_12',
              'DU15_13','DU15_14','DU15_15']


relative_modaloties =  ['SWI_2mm','T1_brain_nonlinear_2mm','T2_FLAIR_brain_to_MNI']



save_dir = "xxx/pretrain_bwas_analysis/hold_out_1500_pattern"
data_dir = "Path_to_UKB_Multimodal_Brain_Image_Dataset"
hold_out_df = pd.read_csv("xxx/test_data_label.csv") # hold out 1500 subjects
hold_out_age_labels = hold_out_df['21003-2.0'].values
hold_out_sex_labels = hold_out_df['31-0.0'].values
hold_out_eids = hold_out_df['eid'].to_list()



for modality in modalities:
    print(f"Begin {modality}")
    hold_out_data = []
    for eid in tqdm(hold_out_eids):
        data = np.load(os.path.join(data_dir,str(eid),f'{modality}.npy'))
        hold_out_data.append(data)
    hold_out_data = np.array(hold_out_data)
    
    if modality in relative_modaloties:
        hold_out_data_mean = hold_out_data.mean(axis=1, keepdims=True)
        hold_out_data_std = hold_out_data.std(axis=1, keepdims=True)
        hold_out_data = (hold_out_data -hold_out_data_mean) / hold_out_data_std
        

    hold_out_age_coef = BWAS_correlation(hold_out_data,hold_out_age_labels)
    hold_out_sex_coef = BWAS_correlation(hold_out_data,hold_out_sex_labels)
        
    hold_out_age_coef= np.array(hold_out_age_coef)
    hold_out_sex_coef= np.array(hold_out_sex_coef)
    
    np.save(os.path.join(save_dir,f'hold_out_{modality}_age.npy'),hold_out_age_coef)
    np.save(os.path.join(save_dir,f'hold_out_{modality}_sex.npy'),hold_out_sex_coef)
