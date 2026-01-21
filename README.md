# GenBrain - A Generative Foundation Model of Multimodal Brain Imaging

## Overview

**GenBrain** is a generative foundation model designed for multimodal brain imaging. 

GenBrain is pre-trained on large-scale neuroimaging data from the **UK Biobank** and evaluated on **81 heterogeneous datasets**, demonstrating strong generalization across imaging modalities and downstream tasks.

This repository provides the core implementation of GenBrain, along with pretrained models for research and development in neuroimaging and AI-driven brain analysis.


## System Requirements

## Installation guide


## Deomo
A demo for image enhancement is provided. 
Additional demos for other downstream tasks can be run using the provided source code. 
Model weights can be downloaded from Google Drive or obtained by contacting me.


## Instructions for use



## Support downstream tasks

1. **Image Enhancement**  
   - Denoising and motion correction.

2. **Cross-modality Synthesis**  
   It can be used for :
   - Structural Synthesis (e.g. T1w<->FLAIR)
   - Functional Synthesis: rs-fMRI fc to task-based fMRI activations (e.g. 15 [seed-based](https://freesurfer.net/fswiki/CorticalParcellation_DU15NET) rs-fMRI fcs to "shapes" task contrast maps in UKB)
   - Structure-Function Synthesis: dMRI scalar maps to rs-fMRI fc (e.g. 9 dMRI maps to  Language Network-fc)
  
3. **Cross-site diagnosis**  
4. **Improving BWAS realiability**  




5. **Clinical Application**
   - Finetune GenBrain on clinical-grade images with disease labels. Synthetic images are added to real images at varying ratios to train the predictive model, improving its diagnostic performance.
   - Specifically, this approach enhances the prediction of acute stroke severity (ds004889) and chronic aphasia (ds004884).

6. **Image Super-resolution**  
   - Finetune GenBrain for image super-resolution task (e.g 2mm T1w -> 1mm T1w, operated in MNI152 standard space).
   - The pipeline first applies nearest-neighbor interpolation to upsample the low-resolution image to the target resolution, followed by GenBrain for super-resolution.


## Pretrained Weights

You can download the pre-trained weight here:  [Google Drive Link](https://drive.google.com/drive/folders/1ajTRlSGIfg9OXQtXX7TfBhVDKaMPrALM?usp=drive_link)

---

## Directory Structure


## Contact
If you encounter any issues while using this code, please feel free to contact me. I will be happy to help.

