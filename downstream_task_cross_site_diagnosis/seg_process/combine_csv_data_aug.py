import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import subprocess



seg_csv_save_dir = "Path_To/ADNI_T1/seg_csv"
train_data_file = "Path_To/adni_nc_ad_mci/labels/adni_info.csv"

train_df = pd.read_csv(train_data_file)


n = len(train_df)

train_seg_df_list =[]
for i  in tqdm(range(n)):
    sub_info = train_df.iloc[i]
    file_name = sub_info['ID']+'_'+str(sub_info['Age'])+'_'+str(sub_info['Sex'])+'_'+str(sub_info['Diagnosis'])
    csv_vols_path = os.path.join(seg_csv_save_dir,file_name+"_seg.csv")
    train_seg_df = pd.read_csv(csv_vols_path)
    train_seg_df_list.append(train_seg_df)


train_seg_merged_df = pd.concat(train_seg_df_list, ignore_index=True)
train_seg_merged_df.to_csv("Path_To/adni_nc_ad_mci/labels/seg_results.csv", index=False)


