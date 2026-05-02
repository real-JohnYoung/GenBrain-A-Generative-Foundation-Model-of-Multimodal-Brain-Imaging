# Image Enhancement

Instruction for fine-tuning GenBrain for image enhancement (denoising and motion correction).

## Data Preparation

Prepare paired data: corrupted images and raw images, both stored as 1D arrays using the MNI152 2mm mask (following the GenBrain pretraining preprocessing pipeline).

Preprocessing scripts (corrupted data simulation code) are provided in [`preprocess_file/image_enhancement/`](preprocess_file/image_enhancement/).

## Usage

**Step 1: Prepare your dataset**

Paired corrupted/raw data in MNI152 2mm standard space (stored as 1D array, N_voxel=228,453).

**Step 2: Configure and run fine-tuning**

Edit the settings in `train_enhancement_model_step.py`, then run:

```bash
torchrun --nnodes=1 --nproc_per_node=2 train_enhancement_model_step.py
```

**Step 3: Inference**

```bash
python evaluation_ft_enhancement_model_avg_step.py
```