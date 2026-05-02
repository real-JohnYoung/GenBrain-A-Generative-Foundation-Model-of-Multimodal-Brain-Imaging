# Image Super-resolution

Instruction for fine-tuning GenBrain for image super-resolution.

## Data Preparation

Prepare paired low-resolution and high-resolution images using the MNI152 standard space mask.

Preprocessing scripts: [`downstream_task_image_super_resolution/preprocess/preprocess.py`](preprocess/preprocess.py)

## Usage

**Step 1: Prepare your dataset**

Taking 2mm → 1mm super-resolution as an example: first apply nearest-neighbor interpolation to upsample the low-resolution image to 1mm space, then construct paired training data.

Since the pretraining voxel size is N_voxel=228,453, the 1mm image (N_voxel=1,827,095) is split into 8 non-overlapping parts (part index 0–7), each with its own voxel index of length 228,453 (boundary overlap is allowed). Each part is processed independently by the model.

**Step 2: Configure and run fine-tuning**

Edit the settings in `finetune_T1_super_resolution.py`, then run:

```bash
torchrun --nnodes=1 --nproc_per_node=2 finetune_T1_super_resolution.py
```

**Step 3: Inference**

Inference is performed part by part, then merged into the full 1mm volume:

```bash
python evaluation_T1_super_resolution_avg_step.py
```