import os
import numpy as np
import pandas as pd

def compute_similarity(bwas_dir, save_dir, index):

    modalities = ['dti_FA_2mm', 'dti_L1_2mm', 'dti_L2_2mm', 'dti_L3_2mm', 'dti_MD_2mm',
                  'dti_MO_2mm', 'NODDI_ICVF_2mm', 'NODDI_ISOVF_2mm', 'NODDI_OD_2mm', 'QSM_2mm', 'SWI_2mm',
                  'T1_brain_nonlinear_2mm', 'T1_warp_Jac_2mm', 'T2_FLAIR_brain_to_MNI', 'T2star_2mm', 'VBM_2mm',
                  'zstat1s', 'zstat2s', 'zstat5s', 'DU15_1', 'DU15_2', 'DU15_3', 'DU15_4', 'DU15_5',
                  'DU15_6', 'DU15_7', 'DU15_8', 'DU15_9', 'DU15_10', 'DU15_11', 'DU15_12',
                  'DU15_13', 'DU15_14', 'DU15_15']
    results = []

    n_train = 1500
    n_gen = 100 

    for modality in modalities:

        # age
        r_train_age = np.load(os.path.join('xxx/pretrain_bwas_analysis/hold_out_1500_pattern', f'hold_out_{modality}_age.npy'))
        t_train_age = r_train_age * np.sqrt((n_train - 2) / (1 - r_train_age ** 2))

        r_gen_age = np.load(os.path.join(bwas_dir, f'gen_{modality}_age.npy'))
        t_gen_age = r_gen_age * np.sqrt((n_gen - 2) / (1 - r_gen_age ** 2))

        age_similarity = cosine_similarity(t_train_age.flatten(), t_gen_age.flatten())

        # sex
        r_train_sex = np.load(os.path.join('xxx/pretrain_bwas_analysis/hold_out_1500_pattern', f'hold_out_{modality}_sex.npy'))
        t_train_sex = r_train_sex * np.sqrt((n_train - 2) / (1 - r_train_sex ** 2))

        r_gen_sex = np.load(os.path.join(bwas_dir, f'gen_{modality}_sex.npy'))
        t_gen_sex = r_gen_sex * np.sqrt((n_gen - 2) / (1 - r_gen_sex ** 2))

        sex_similarity = cosine_similarity(t_train_sex.flatten(), t_gen_sex.flatten())

        print(modality, age_similarity, sex_similarity)
        results.append({'modality': modality, 'age_similarity': age_similarity, 'sex_similarity': sex_similarity})

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(save_dir, f'similarities_{index}.csv'), index=False)
    print("similarities.csv is saved...")


def cosine_similarity(vec1, vec2):
    vec1 = np.nan_to_num(vec1, nan=0.0)
    vec2 = np.nan_to_num(vec2, nan=0.0)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    cosine_sim = np.dot(vec1, vec2) / (norm1 * norm2)
    return cosine_sim


save_dir = 'xxx/pretrain_bwas_analysis/pretrain_bwas_eavluation_1500_results/gen'
gen_root = 'xxx/evaluations_model'
for i in range(5):
    bwas_dir = os.path.join(gen_root,f'epoch_200_{i}')
    compute_similarity(bwas_dir, save_dir, index=i)

