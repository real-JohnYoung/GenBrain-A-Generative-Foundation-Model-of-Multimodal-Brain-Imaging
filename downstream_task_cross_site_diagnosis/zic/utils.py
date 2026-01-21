import os
import random
import pandas as pd
import numpy as np
from itertools import product
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve, auc,roc_auc_score
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay
import torch
from lightgbm import LGBMClassifier



def select_params_combo(my_dict, nb_items, my_seed):
    combo_list = [dict(zip(my_dict.keys(), v)) for v in product(*my_dict.values())]
    random.seed(my_seed)
    return random.sample(combo_list, nb_items)


def perform_param_search(train_data, train_labels, val_data, val_labels, test_data, test_labels, param_grid, output_dir,gpu_device_id=0):
    results = []
    
    candidate_params_lst = select_params_combo(param_grid, 20, my_seed=42)
    
    for idx, params in tqdm(enumerate(candidate_params_lst), total=len(candidate_params_lst), desc="Hyperparameter Search"):
        print(f"Training with parameters: {params}")
        result = train_and_evaluate_lightgbm(params, train_data, train_labels, val_data, val_labels, test_data, test_labels, output_dir, gpu_device_id,idx)
        results.append(result)
        result_df = pd.DataFrame([{
            'params': result['params'],
            'val_accuracy': result['val_accuracy'],
            'val_precision': result['val_precision'],
            'val_recall': result['val_recall'],
            'val_f1': result['val_f1'],
            'val_auroc':result['val_auroc'],
            'test_accuracy': result['test_accuracy'],
            'test_precision': result['test_precision'],
            'test_recall': result['test_recall'],
            'test_f1': result['test_f1'],
            'test_auroc':result['test_auroc']
        }])

        result_df.to_csv(os.path.join(output_dir, 'param_search_results.csv'), mode='a', header=not os.path.exists(os.path.join(output_dir, 'param_search_results.csv')), index=False)

    return results

def save_evaluation_results(pred_labels, preds, true_labels, output_dir,idx):
    os.makedirs(os.path.join(output_dir,str(idx)), exist_ok=True)

    accuracy = accuracy_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels,average='macro')
    recall = recall_score(true_labels, pred_labels,average='macro')
    f1 = f1_score(true_labels, pred_labels,average='macro')
    auroc = roc_auc_score(true_labels, preds,average='macro', multi_class='ovr')
    with open(os.path.join(output_dir, str(idx),'evaluation_metrics.txt'), 'w') as f:
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")
        f.write(f"Auroc Score: {auroc:.4f}\n")
    

def train_and_evaluate_lightgbm(params, train_data, train_labels, val_data, val_labels, test_data, test_labels, output_dir,gpu_device_id=0,idx=0):
    model = LGBMClassifier(
        objective='multiclass',
        num_class=3,           
        metric='multi_logloss',
        n_jobs=1, 
        device='gpu' if torch.cuda.is_available() else 'cpu', 
        gpu_platform_id=0, 
        gpu_device_id=gpu_device_id,  
        **params
    )

    model.fit(train_data, train_labels, eval_set=[(val_data, val_labels)])

    val_preds = model.predict_proba(val_data)

    val_pred_labels = np.argmax(val_preds, axis=1)
    val_accuracy = accuracy_score(val_labels, val_pred_labels)
    val_precision = precision_score(val_labels, val_pred_labels,average='macro')
    val_recall = recall_score(val_labels, val_pred_labels,average='macro')
    val_f1 = f1_score(val_labels, val_pred_labels,average='macro')
    val_auroc = roc_auc_score(val_labels, val_preds,average='macro', multi_class='ovr')

    save_evaluation_results(val_pred_labels, val_preds, val_labels, output_dir,idx)


    test_preds = model.predict_proba(test_data)
    np.save(os.path.join(output_dir, str(idx),'test_results.npy'), test_preds)
    test_pred_labels = np.argmax(test_preds, axis=1)
    test_accuracy = accuracy_score(test_labels, test_pred_labels)
    test_precision = precision_score(test_labels, test_pred_labels,average='macro')
    test_recall = recall_score(test_labels, test_pred_labels,average='macro')
    test_f1 = f1_score(test_labels, test_pred_labels,average='macro')
    test_auroc = roc_auc_score(test_labels, test_preds,average='macro', multi_class='ovr')

    with open(os.path.join(output_dir, str(idx),'test_metrics.txt'), 'w') as f:
        f.write(f"Accuracy: {test_accuracy:.4f}\n")
        f.write(f"Precision: {test_precision:.4f}\n")
        f.write(f"Recall: {test_recall:.4f}\n")
        f.write(f"F1 Score: {test_f1:.4f}\n")
        f.write(f"Auroc Score: {test_auroc:.4f}\n")

    return {  'params': params,
        'val_accuracy': val_accuracy,'val_precision': val_precision,
        'val_recall': val_recall,'val_f1': val_f1,
        'val_auroc':val_auroc,
        'test_accuracy': test_accuracy,'test_precision': test_precision,
        'test_recall': test_recall,'test_f1': test_f1,
        'test_auroc': test_auroc
    }