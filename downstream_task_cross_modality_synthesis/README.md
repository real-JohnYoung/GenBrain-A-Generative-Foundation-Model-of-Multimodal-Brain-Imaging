# Cross-modality Synthesis

Instruction for fine-tuning GenBrain for cross-modality synthesis.

## Data Preparation

Prepare paired data: source modality and target modality images, both stored as 1D arrays using the MNI152 2mm mask (following the GenBrain pretraining preprocessing pipeline).

## Usage

**Step 1: Prepare your dataset**

Paired source/target modality data in MNI152 2mm standard space (stored as 1D array, N_voxel=228,453).

**Step 2: Configure and run fine-tuning**

Taking T1 → T2-FLAIR as an example. Edit the settings in `finetune_modality_translation_model_T1_T2_uncon_step.py`, then run:

```bash
torchrun --nnodes=1 --nproc_per_node=2 finetune_modality_translation_model_T1_T2_uncon_step.py
```

For 1mm resolution synthesis, use `finetune_modality_translation_model_T1_T2_1mm_uncon_step.py` and configure accordingly.

Code for other modality-to-modality translation directions is also provided — adjust the input/target modality settings in the corresponding script.

**Step 3: Inference**

```bash
python evaluation_ft_modality_translation_model_T1_FLAIR_avg_uncond_step_zic.py
```