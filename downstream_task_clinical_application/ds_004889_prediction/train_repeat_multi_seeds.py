import os
import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from resnet import resnet10 as resnet_model
from model_utils import ADC3DMixedDataset

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    return running_loss / len(loader.dataset)

def eval_epoch(model, loader, device):
    model.eval()
    preds = []
    targets = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            preds.append(outputs.cpu().numpy())
            targets.append(labels.cpu().numpy())
    preds = np.concatenate(preds).reshape(-1)
    targets = np.concatenate(targets).reshape(-1)

    mae = np.mean(np.abs(preds - targets))
    # 计算Pearson相关系数
    if np.std(preds) > 0 and np.std(targets) > 0:
        corr = np.corrcoef(preds, targets)[0, 1]
    else:
        corr = np.nan
    return mae, corr

def main(real_train_img_dir, real_train_csv, gen_train_img_dir, gen_train_csv,
         val_img_dir, val_csv, test_img_dir, test_csv, model_save_dir,
         batch_size=4, ratio=0.5, lr=1e-4, epochs=100, device='cuda', seed=42):
    set_seed(seed)
    ratio_dir = f'ratio_{ratio}'
    save_dir = os.path.join(model_save_dir, ratio_dir)
    os.makedirs(save_dir, exist_ok=True)
    weight_path = os.path.join(save_dir, 'best_model.pth')

    train_ds = ADC3DMixedDataset(real_train_img_dir, real_train_csv, gen_train_img_dir, gen_train_csv, ratio=ratio,random_state=seed)
    val_ds   = ADC3DMixedDataset(val_img_dir, val_csv, '', '', ratio=0)
    test_ds  = ADC3DMixedDataset(test_img_dir, test_csv, '', '', ratio=0)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=8)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    model = resnet_model(inchan=1, num_classes=1).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_mae = float('inf')
    best_val_corr = None
    best_epoch = 0

    for epoch in range(1, epochs+1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_mae, val_corr = eval_epoch(model, val_loader, device)
        print(f"Epoch {epoch}/{epochs}: Train Loss={train_loss:.4f} | Val MAE={val_mae:.4f} | Val Corr={val_corr:.4f}")
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_val_corr = val_corr
            best_epoch = epoch
            torch.save(model.state_dict(), weight_path)
            print(f">>> New best model saved at {weight_path}")

    model.load_state_dict(torch.load(weight_path))
    test_mae, test_corr = eval_epoch(model, test_loader, device)
    print(f"Test MAE: {test_mae:.4f} | Test Corr: {test_corr:.4f}")

    return best_val_mae, best_val_corr, test_mae, test_corr, best_epoch

if __name__ == '__main__':
    results = []
    seeds = [41,42 ,43, 44, 45]  # 可以根据需要调整种子数量
    lr = 5e-5

    for seed in seeds:
        for ratio in np.arange(0, 3.01, 0.1):
            ratio = round(float(ratio), 2)
            print(f"\n=== Training with seed = {seed}, ratio = {ratio} ===\n")
            best_val_mae, best_val_corr, test_mae, test_corr, best_epoch = main(
                real_train_img_dir = '/Path_to/clinical_application_stroke/ds-004889-1.1.2/dwi_1d',
                real_train_csv     = '/Path_to/downstream_clinical_application_stroke/ds_4889_labels/training_dataset.csv',
                gen_train_img_dir  = '/Path_to/clinical_application_stroke/ds_004889_results/001-DiT-UKB-L-256-finetune_mode-full/evaluations/step_20000',
                gen_train_csv      = '/Path_to/downstream_clinical_application_stroke/ds_4889_labels/data_aug_file.csv',
                val_img_dir        = '/Path_to/clinical_application_stroke/ds-004889-1.1.2/dwi_1d',
                val_csv            = '/Path_to/downstream_clinical_application_stroke/ds_4889_labels/validation_dataset.csv',
                test_img_dir       = '/Path_to/clinical_application_stroke/ds-004889-1.1.2/dwi_1d',
                test_csv           = '/Path_to/downstream_clinical_application_stroke/ds_4889_labels/testing_dataset.csv',
                model_save_dir     = f'/Path_to/clinical_application_stroke/ds_004889_results_new/resnet_20k_multi_seed_new_{seed}_lr_{lr}',
                batch_size = 32,
                ratio = ratio,
                lr = lr,
                epochs = 50,
                device = 'cuda' if torch.cuda.is_available() else 'cpu',
                seed = seed
            )
            results.append({
                'seed': seed,
                'ratio': ratio,
                'best_val_mae': best_val_mae,
                'best_val_corr': best_val_corr,
                'test_mae': test_mae,
                'test_corr': test_corr,
                'best_epoch': best_epoch
            })
            pd.DataFrame(results).to_csv(
                f'/Path_to/clinical_application_stroke/ds_004889_results_new/resnet_20k_multi_seed_new_{seed}_lr_{lr}/results_multi_seed.csv',
                index=False
            )

    print("=== All Results ===")
    print(pd.DataFrame(results))
