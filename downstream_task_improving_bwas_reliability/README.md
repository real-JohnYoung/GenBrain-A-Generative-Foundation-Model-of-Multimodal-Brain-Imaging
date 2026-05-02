# Improving BWAS Reliability

Instruction for fine-tuning GenBrain to improve the reliability of brain-wide association studies (BWAS).

## Data Preparation

Prepare source modality images and corresponding disease labels, stored as 1D arrays using the MNI152 2mm mask (following the GenBrain pretraining preprocessing pipeline).

## Usage

**Step 1: Prepare your dataset**

Paired source images and disease labels in MNI152 2mm standard space (stored as 1D array, N_voxel=228,453).

**Step 2: Configure and run fine-tuning**

Taking Schizophrenia (SCZ) with smoothed VBM modality as an example. Edit the settings in `finetune_multisite_scz_one_site_vbm_step.py`, then run:

```bash
torchrun --nnodes=1 --nproc_per_node=1 finetune_multisite_scz_one_site_vbm_step.py
```

Code for other diseases (MDD, ASD) is also provided — adjust the disease label and modality settings in the corresponding script.

**Step 3: Inference**

Synthetic images are sampled by specifying combinations of covariates (e.g. age, sex, disease label):

```bash
python evaluation_ft_multisite_scz_one_site_vbm_avg_step.py
```

**Step 4: BWAS Analysis**

Synthetic cohorts are used alongside real data to perform BWAS, stabilizing effect-size estimates and improving reproducibility.

> Statistic analysis files will be uploaded soon.