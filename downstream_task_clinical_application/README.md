# Clinical Application

Instruction for fine-tuning GenBrain for clinical applications.

## Data Preparation

Prepare source modality images and corresponding disease labels from clinical datasets, stored as 1D arrays using the MNI152 2mm mask (following the GenBrain pretraining preprocessing pipeline).

## Usage

**Step 1: Prepare your dataset**

Paired source images and disease labels in MNI152 2mm standard space (stored as 1D array, N_voxel=228,453).

**Step 2: Configure and run fine-tuning**

Taking acute stroke severity ([ds004889](https://openneuro.org/datasets/ds004889/versions/1.1.2)) as an example. Edit the settings in `finetune_model_4889_step.py`, then run:

```bash
torchrun --nnodes=1 --nproc_per_node=1 finetune_model_4889_step.py
```

Code for other clinical conditions is also provided — adjust the disease label and modality settings in the corresponding script.

**Step 3: Inference**

```bash
python evaluation_ft_4889_step.py
```

**Step 4: Diagnosis**

Mix synthetic and real images at varying ratios to train a clinical diagnosis model (e.g. ResNet). This augmentation strategy improves diagnostic performance under data scarcity.