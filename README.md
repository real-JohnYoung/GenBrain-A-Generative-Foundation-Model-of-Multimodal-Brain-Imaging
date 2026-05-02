<div align="center">

# GenBrain - A Generative Foundation Model of Multimodal Brain Imaging

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4.1-orange.svg)](https://pytorch.org/)
[![medRxiv](https://img.shields.io/badge/medRxiv-2025.12.19-red.svg)](https://www.medrxiv.org/content/10.64898/2025.12.19.25342614v1)

</div>

The version corresponding to the submitted manuscript is tagged as **v1.0**. This repository is currently being organized and further updates will be added over time.

---

## Overview

**GenBrain** is a generative foundation model designed for multimodal brain imaging.

GenBrain is pre-trained on large-scale neuroimaging data from the **UK Biobank** and evaluated across **81 independent sites**, demonstrating strong generalization across imaging modalities and downstream tasks.

This repository provides the core implementation of GenBrain, along with pretrained models for research and development in neuroimaging and AI-driven brain analysis.

---

## System Requirements

- **OS:** Ubuntu 22.04.5 LTS (kernel 3.10.0-1160)
- **Python:** 3.10.12
- **Libraries:** PyTorch 2.4.1+cu118, nibabel 5.3.2 — full details in `requirements.txt`
- **GPU:** NVIDIA A100-SXM, 80 GB memory

---

## Installation Guide

```bash
# Create conda environment if needed
conda create -n myenv python=3.10.12
conda activate myenv

# Install packages from requirements.txt
pip install -r requirements.txt
```

- Install time on a "normal" desktop computer takes about 10–20 minutes.
- A Docker image enabling direct execution of the code will be made available shortly.

---

## Demo

A demo for image enhancement is provided.  
Additional demos for other downstream tasks can be executed using the provided source code.  
Model weights can be downloaded from [Google Drive](#pretrained-weights) or obtained by contacting the corresponding author.

- **Instructions:** We assume that the corrupted images (T1w/T2-FLAIR modalities) have been registered to the MNI152 2mm standard space, and that the corresponding label files and model weights are properly configured. The demo can be run using:
  ```bash
  python run_inference.py
  ```
- **Expected output:** An enhanced MRI image.
- **Runtime:** On a standard desktop computer equipped with an NVIDIA GPU with more than 20 GB of memory, inference for a single sample using DDIM (50 steps) takes approximately 10–30 seconds.

---

## Instructions for Use

### GenBrain Pretraining

1. **Data Preprocessing:** UKB multimodal brain image datasets are non-linearly registered to the MNI152 2mm standard space (template provided in  [preprocess_file] folder, orientation: LAS). Brain voxels are then extracted according to nonzero indices in the template and saved as `.npy` files (stored as 1D array, N_voxel=228,453). *(Note: [FSL software](https://fsl.fmrib.ox.ac.uk/fsl/docs/index.html) is recommended for preprocessing)*

2. **Configure Pretraining Settings and Files:** Prepare model pretraining parameters, data file, and label file (including individual age, sex, and imaging modality); files are in the `labels/` directory. Phenotypic information and imaging modality details can be found in `data_info.json`.

3. **Run Training:** Multi-GPU training requires support for Distributed Data Parallel (NVIDIA A100 GPUs recommended):
   ```bash
   torchrun --nnodes=N_Node --nproc_per_node=N_GPU train.py
   torchrun --nnodes=1 --nproc_per_node=6 train.py    # Our setup: 1 node with 6 A100 GPUs
   ```
   Pretraining code is in the `pretraining_evaluation/` directory.

### GenBrain Fine-tuning

1. **Data Preprocessing**
   - Perform data preprocessing similar to the pretraining stage.
   - Ensure input formats, normalization, and any filtering steps match the pretraining pipeline.

2. **Load Pretrained Weights and Adapt Model Architecture**
   - Load GenBrain's pretrained weights.
   - Modify the architecture according to the target task:
     - **Image-level tasks:** e.g. adjust patchify layer.
     - **Disease label fine-tuning:** e.g. add disease-label embedder.
     - ...

3. **Run Fine-tuning and Inference:**
   ```bash
   torchrun --nnodes=N_Node --nproc_per_node=N_GPU finetune.py  # fine-tune
   python evaluate.py                                            # inference
   ```

4. **Note:** Our downstream task code can be referenced for implementation details and examples.

**Reproduction instructions:** Follow the default hyper-parameter settings in the source code or according to our paper.

---

## Supported Downstream Tasks

### 1. Image Enhancement
- Denoising and motion correction.
- Fine-tuned GenBrain v1.0 supports T1w and T2-FLAIR modalities.
- See detailed instructions [README.md](downstream_tasks/image_enhancement/README.md)

### 2. Cross-modality Synthesis
It can be used for:
- **Structural Synthesis** (e.g. T1w ↔ FLAIR)
- **Functional Synthesis:** rs-fMRI FC to task-based fMRI activations (e.g. 15 [seed-based](https://freesurfer.net/fswiki/CorticalParcellation_DU15NET) rs-fMRI FCs → "shapes" task contrast maps in UKB)
- **Structure-Function Synthesis:** dMRI scalar maps to rs-fMRI FC (e.g. 9 dMRI maps → Language Network FC)

### 3. Cross-site Diagnosis
- Fine-tune GenBrain on images with disease labels. Synthetic images and real images volumetric measures are extracted by WMH-SynthSeg. These volumetric features serve as quantitative inputs for a machine learning classification model (LightGBM).
- By using synthetic images' features, the machine learning classifier's generalizability can be improved in cross-site diagnosis.
- Specifically, this approach enhances the prediction of **Schizophrenia** and **Alzheimer's disease**.

### 4. Improving BWAS Reliability
- Fine-tune GenBrain on images with disease labels (modalities: VBM, seed-based rs-fMRI FC).
- Leveraging the population-level prior learned by GenBrain, synthetic images generated by the model improve the reliability of brain-wide association studies.
- Diseases: **Schizophrenia**, **Major Depressive Disorder**, **Autism Spectrum Disorder**.

### 5. Clinical Application
- Fine-tune GenBrain on clinical-grade images with disease labels. Synthetic images are added to real images at varying ratios to train the predictive model, improving its diagnostic performance.
- Specifically, this approach enhances the prediction of acute stroke severity ([ds004889](https://openneuro.org/datasets/ds004889/versions/1.1.2)) and chronic aphasia ([ds004884](https://openneuro.org/datasets/ds004884/versions/1.0.1)).

### 6. Image Super-resolution
- Fine-tune GenBrain for image super-resolution (e.g. 2 mm T1w → 1 mm T1w, operated in MNI152 standard space).
- The pipeline first applies nearest-neighbor interpolation to upsample the low-resolution image to the target resolution, followed by GenBrain for super-resolution.

---

## Pretrained Weights

You can download the pretrained weights here: [Google Drive Link](https://drive.google.com/drive/folders/1E9qbUd_KAVqukAzMyydR1YAHadBNfrDX)

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Citation

For usage of the code and associated manuscript, please cite:

```bibtex
@article{yang2025genbrain,
  title={GenBrain: A Generative Foundation Model of Multimodal Brain Imaging},
  author={Yang, Chang and Feng, Jianfeng and Beckmann, Christian F and Smith, Stephen M and Gong, Weikang},
  journal={medRxiv},
  pages={2025--12},
  year={2025},
  publisher={Cold Spring Harbor Laboratory Press}
}
```

[GenBrain: A Generative Foundation Model of Multimodal Brain Imaging](https://www.medrxiv.org/content/10.64898/2025.12.19.25342614v1)

---

## Contact

If you encounter any issues while using this code, please feel free to contact me. I will be happy to help.