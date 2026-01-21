# GenBrain - A Generative Foundation Model of Multimodal Brain Imaging
The version corresponding to the submitted manuscript is tagged as v1.0. This repository is currently being organized and further updates will be added over time.

## Overview

**GenBrain** is a generative foundation model designed for multimodal brain imaging. 

GenBrain is pre-trained on large-scale neuroimaging data from the **UK Biobank** and evaluated on **81 heterogeneous datasets**, demonstrating strong generalization across imaging modalities and downstream tasks.

This repository provides the core implementation of GenBrain, along with pretrained models for research and development in neuroimaging and AI-driven brain analysis.

## System Requirements

- OS: Ubuntu 22.04.5 LTS (kernel 3.10.0-1160)
- Python: 3.10.12
- Libraries: PyTorch 2.4.1+cu118, nibabel 5.3.2, more details see requirements.txt.
- GPU: NVIDIA A100-SXM, 80 GB memory

## Installation Guide
 - Install Instructions 

  ```bash
  # Create conda environment if needed.
  conda create -n myenv python=3.10.12
  conda activate myenv
  # Install packages from requirements.txt
  pip install -r requirements.txt
  ```
 - Install time on a "normal" desktop computer takes about 10~20 minutes.
 - A Docker image enabling direct execution of the code will be made available shortly.


## Demo

A demo for image enhancement is provided.  
Additional demos for other downstream tasks can be executed using the provided source code.  
Model weights can be downloaded from Google Drive or obtained by contacting the corresponding author.

- **Instructions:** We assume that the corrupted images (T1w/T2-FLAIR modalities) have been registered to the MNI152 2mm standard space, and that the corresponding label files and model weights are properly configured. 
   The demo can be run using:
  ```bash
  python run_inference.py
  ```
- **Expected output:** An enhanced MRI image.

- **Runtime:** On a standard desktop computer equipped with an NVIDIA GPU with more than 20 GB of memory, inference for a single sample using DDIM (50 steps) takes approximately 10–30 seconds.


## Instructions for Use

- **GenBrain Pre-training Instructions:**

  1. **Data Preprocessing:** UKB multimodal brain image dataset are non-linearly registered to the MNI152 2mm standard space. Brain voxels are then extracted according to nonzero indices in the template and saved as `.npy` files (stored as 1d array, N_voxel=228,453). Note: *([FSL software](https://fsl.fmrib.ox.ac.uk/fsl/docs/index.html) is recommended for preprocessing)*

  2. **Configure Pre-training Settings and Files:** Prepare model pre-training parameters, data file, and label file (including individual age, sex, and imaging modality), files are in `labels` directory. Phenotypic information and imaging modality details can be found in `data_info.json`. 
  
  3. **Running the Training:** Multi-GPU training requires support for Distributed Data Parallel (Recommend Nvidia A100 GPUs):
     ```bash
     torchrun --nnodes=N_Node --nproc_per_node=N_GPU train.py
     torchrun --nnodes=1 --nproc_per_node=6 train.py    # Our: 1 node with 6 Nvidia A100 GPUs
     ```
     Pre-training code is in `pretraining_evaluation` directory.

- **GenBrain Finetuning Instructions:**

  1. **Data Preprocessing**
     - Perform data preprocessing similar to the pre-training stage.
     - Ensure input formats, normalization, and any filtering steps match the pre-training pipeline.

  2. **Load Pre-trained Weights and Adapt Model Architecture**
     - Load GenBrain's pre-trained weights.
     - Modify the architecture according to the target task:
       - **Image-level tasks**: e.g. adjust patchify layer.
       - **Disease label finetune**: e.g add disease-label embedder.
       - ...
  3. **Running the Finetuning and Inference:**
     ```bash
     torchrun --nnodes=N_Node --nproc_per_node=N_GPU finetune.py # finetune
     python evaluate.py # inference
     ```
  4. **Note:** Our downstream task code can be referenced for implementation details and examples.
- **Reproduction instructions:** Follow the default hyper-parameter settings in the source code or according to our paper. 


## Support downstream tasks

1. **Image Enhancement**  
   - Denoising and motion correction.
   - Finetuned GenBrain v1.0 Support T1w and T2-FLAIR modalities. 

2. **Cross-modality Synthesis**  
   It can be used for :
   - Structural Synthesis (e.g. T1w<->FLAIR)
   - Functional Synthesis: rs-fMRI fc to task-based fMRI activations (e.g. 15 [seed-based](https://freesurfer.net/fswiki/CorticalParcellation_DU15NET) rs-fMRI fcs to "shapes" task contrast maps in UKB)
   - Structure-Function Synthesis: dMRI scalar maps to rs-fMRI fc (e.g. 9 dMRI maps to  Language Network-fc)
  
3. **Cross-site Diagnosis** 
   - Finetune GenBrain on images with disease labels. Synthetic images and real images volumetric measures are extracted by WMH-SynthSeg. These volumetric features served as quantitative inputs for a machine learning classification model(LightGBM).
   - By using synthetic images' features, machine learning classification model' generalizability can be improved in cross-site diagnosis.
  Specifically, this approach enhances the prediction of Schizophrenia and Alzheimer’s disease.

4. **Improving BWAS Reliability**  
   - Finetune GenBrain on images with disease labels (modalities like vbm, seed-based rs-fMRI fc).
   - Leveraging the population-level prior learned by GenBrain, synthetic images generated by the model improve the reliability of brain-wide association studies.
   - Diseases: Schizophrenia, Major Depressive Disorder, Autism Spectrum Disorder.

5. **Clinical Application**
   - Finetune GenBrain on clinical-grade images with disease labels. Synthetic images are added to real images at varying ratios to train the predictive model, improving its diagnostic performance.
   - Specifically, this approach enhances the prediction of acute stroke severity [ds004889](https://openneuro.org/datasets/ds004889/versions/1.1.2) and chronic aphasia [ds004884](https://openneuro.org/datasets/ds004884/versions/1.0.1).

6. **Image Super-resolution**  
   - Finetune GenBrain for image super-resolution task (e.g 2mm T1w -> 1mm T1w, operated in MNI152 standard space).
   - The pipeline first applies nearest-neighbor interpolation to upsample the low-resolution image to the target resolution, followed by GenBrain for super-resolution.


## Pretrained Weights

You can download the pre-trained weight here:  [Google Drive Link](https://drive.google.com/drive/folders/1ajTRlSGIfg9OXQtXX7TfBhVDKaMPrALM?usp=drive_link)


## License

This project is licensed under the MIT License.

## Citation
For usage of the code and associated manuscript, please cite [GenBrain: A Generative Foundation Model of Multimodal Brain Imaging](https://www.medrxiv.org/content/10.64898/2025.12.19.25342614v1).

## Contact
If you encounter any issues while using this code, please feel free to contact me. I will be happy to help.