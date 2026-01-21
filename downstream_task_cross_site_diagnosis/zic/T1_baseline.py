# baseline_seg: Baseline method is based on LightGBM method. LightGBM: https://github.com/microsoft/LightGBM.
import os
import numpy as np
import pandas as pd
from utils import perform_param_search

def main():
    train_file_path = 'Path_To/liuyuan_ad_nc_mci_ad/labels/T1_train_data.csv'
    val_file_path = 'Path_To/liuyuan_ad_nc_mci_ad/labels/T1_val_data.csv'
    test_file_path = 'Path_To/liuyuan_ad_nc_mci_ad/labels/T1_test_data.csv'


    train_file = pd.read_csv(train_file_path)
    val_file = pd.read_csv(val_file_path)
    test_file = pd.read_csv(test_file_path)

    train_labels = train_file['label'].to_list()
    val_labels = val_file['label'].to_list()
    test_labels = test_file['label'].to_list()

    gpu_device_id = 0

    train_data = np.array(train_file.iloc[:, 8:])
    mean = train_data.mean(axis = 0)
    std = train_data.std(axis = 0)
    train_data = (train_data-mean)/std

    test_data = np.array(test_file.iloc[:, 8:])
    test_data = (test_data-mean)/std

    val_data = np.array(val_file.iloc[:, 8:])
    val_data = (val_data-mean)/std

    param_grid = {  
                    'n_estimators': [25, 50, 100, 200, 300],
                    'max_depth': np.linspace(5, 30, 6).astype('int32').tolist(),
                    'num_leaves': np.linspace(5, 30, 6).astype('int32').tolist(),
                    'subsample': np.linspace(0.6, 1, 9).tolist(),
                    'learning_rate': [0.1, 0.05, 0.01, 0.001],
                    'colsample_bytree': np.linspace(0.6, 1, 9).tolist(),
                    'boosting_type': ['gbdt'],
                    'seed':[42]
                }
    
    output_root = 'Path_To/liuyuan_ad_nc_mci_ad/T1_results/baseline'
    os.makedirs(output_root,exist_ok=True)
    output_dir = output_root + f'/param_search'

    results = perform_param_search(train_data, train_labels, val_data, val_labels, test_data, test_labels, param_grid, output_dir,gpu_device_id)
    print(f"Parameter search completed. Results saved to {output_dir}")

if __name__ == "__main__":
    main()