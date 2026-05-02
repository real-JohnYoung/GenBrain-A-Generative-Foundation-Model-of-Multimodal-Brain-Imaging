# Cross-site Diagnosis

Instruction for fine-tuning GenBrain for cross-site diagnosis.

## Data Preparation

Prepare source modality images and corresponding disease labels, stored as 1D arrays using the MNI152 2mm mask (following the GenBrain pretraining preprocessing pipeline).

## Usage

**Step 1: Prepare your dataset**

Paired source images and disease labels in MNI152 2mm standard space (stored as 1D array, N_voxel=228,453).

**Step 2: Configure and run fine-tuning**

Taking Schizophrenia (SCZ) as an example. Edit the settings in `finetune_multisite_scz_T1_one_site_step.py`, then run:

```bash
torchrun --nnodes=1 --nproc_per_node=1 finetune_multisite_scz_T1_one_site_step.py
```

Code for other diseases is also provided — adjust the disease label and modality settings in the corresponding script.

**Step 3: Inference**

```bash
python evaluation_ft_multisite_scz_T1_step_avg.py
```

**Step 4: Diagnosis**

Use WMH-SynthSeg to extract regional brain volumetric features from both synthetic and real images. Mix synthetic and real features at varying ratios to train a LightGBM classifier for cross-site diagnosis.

Taking SCZ as an example, the cross-site diagnosis code is provided in [`downstream_task_cross_site_diagnosis/multisite_scz/`](multisite_scz/).