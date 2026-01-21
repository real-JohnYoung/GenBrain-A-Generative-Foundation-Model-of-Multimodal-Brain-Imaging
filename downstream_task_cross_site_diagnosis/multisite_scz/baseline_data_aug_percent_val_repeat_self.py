# baseline_seg: Baseline method is based on LightGBM method. LightGBM: https://github.com/microsoft/LightGBM.
import os
import numpy as np
import pandas as pd
from utils import perform_param_search
from sklearn.utils import shuffle

def main():

    train_site_list = ['fBIRN','MCIC','ds000030','SH_JZ1', 'SH_JZ2','SH_drug1', 'NUSDAST', 'COBRE','chengdu','zhengzhou'] 
    test_site_list  = ['fBIRN','MCIC','ds000030','SH_JZ1', 'SH_JZ2','SH_drug1', 'NUSDAST', 'COBRE','chengdu','zhengzhou']

    for i in range(0,31):
        data_aug_percent = i / 10

        for train_site in train_site_list:
            for test_site in test_site_list:

                train_file_path = f'Path_To/multisite_scz_one_site/labels/{train_site}/{train_site}_train_data.csv'
                aug_file_path = f'Path_To/multisite_scz_one_site/labels/{train_site}/seg_data_aug_data.csv'

                if train_site == test_site :
                    test_file_path = f'Path_To/multisite_scz_one_site/labels/{test_site}/{test_site}_test_data.csv'
                else:
                    test_file_path = f'Path_To/multisite_scz_one_site/labels/{test_site}/{test_site}_all_data.csv'

                val_file_path = f'Path_To/multisite_scz_one_site/labels/{train_site}/{train_site}_val_data.csv'

                train_file = pd.read_csv(train_file_path)
                val_file = pd.read_csv(val_file_path)
                test_file = pd.read_csv(test_file_path)

                train_num = len(train_file)
                aug_file  = pd.read_csv(aug_file_path)[:train_num*3]
                data_aug_num = int(data_aug_percent * train_num)
                aug_file = aug_file.sample(frac=data_aug_num/len(aug_file),random_state=42).reset_index(drop=True)
                train_labels = train_file['label'].to_list()
                val_labels = val_file['label'].to_list()
                test_labels = test_file['label'].to_list()
                aug_labels = aug_file['label'].to_list()

                gpu_device_id = 0
                train_data = np.array(train_file.iloc[:, 7:])

                mean = train_data.mean(axis = 0)
                std = train_data.std(axis = 0)
                train_data = (train_data-mean)/std

                test_data = np.array(test_file.iloc[:, 7:])
                test_data = (test_data-mean)/std

                val_data = np.array(val_file.iloc[:, 7:])
                val_data = (val_data-mean)/std

                aug_data = np.array(aug_file.iloc[:, 6:])
                aug_data = (aug_data-mean)/std


                train_data_aug = np.concatenate((train_data,aug_data),axis=0)
                train_labels_aug = train_labels + aug_labels
                train_data_aug = np.array(train_data_aug)
                train_labels_aug = np.array(train_labels_aug)

                train_data_aug, train_labels_aug = shuffle(train_data_aug, train_labels_aug, random_state=42)

                param_grid = {  
                                'n_estimators': [25, 50, 100, 200, 300],
                                'max_depth': np.linspace(5, 30, 6).astype('int32').tolist(),
                                'num_leaves': np.linspace(5, 30, 6).astype('int32').tolist(),
                                'subsample': np.linspace(0.6, 1, 9).tolist(),
                                'learning_rate': [0.1, 0.05, 0.01, 0.001],
                                'colsample_bytree': np.linspace(0.6, 1, 9).tolist(),
                                'objective': ['binary'],
                                'metric': ['binary_logloss'],
                                'boosting_type': ['gbdt'],
                                'seed':[42]
                            }
                
                output_root = f'Path_To/multisite_scz_one_site/results/data_aug_repeat_test_all/data_aug_{round(data_aug_percent*100)}_percent_train_on_{train_site}_test_on_{test_site}'
                os.makedirs(output_root,exist_ok=True)
                output_dir = output_root + f'/param_search'

                results = perform_param_search(train_data_aug, train_labels_aug, val_data, val_labels, test_data, test_labels, param_grid, output_dir,gpu_device_id)
                print(f"Parameter search completed. Results saved to {output_dir}")

if __name__ == "__main__":
    main()