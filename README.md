# GenBrain - A Generative Foundation Model of Multimodal Brain Imaging

## Overview

**GenBrain** is a generative foundation model designed for multimodal brain imaging. 

GenBrain is pre-trained on large-scale neuroimaging data from the **UK Biobank** and evaluated on **81 heterogeneous datasets**, demonstrating strong generalization across imaging modalities and downstream tasks.

This repository provides the core implementation of GenBrain, along with pretrained models for research and development in neuroimaging and AI-driven brain analysis.

---
## Support downstream tasks

1. **Image Enhancement**  
   - Denoising and motion correction.

2. **Cross-modality Synthesis**  
   It can be used for :
   - Structural Synthesis (e.g.T1w<->FLAIR)
   - Functional Synthesis: rs-fMRI fc to task-based fMRI activations (e.g. 15 [seed-based](https://freesurfer.net/fswiki/CorticalParcellation_DU15NET) rs-fMRI fcs to "shapes" task contrast maps in UKB)
   - Structure-Function Synthesis: dMRI scalar maps to rs-fMRI fc (e.g. 9 dMRI maps to  Language Network-fc)
  
3. **Cross-site diagnosis**  
4. **Improving BWAS realiability**  
5. **Clinical Application**
6. **Image Super-resolution**  

## Pretrained Weights

You can download the pre-trained weight here:  [Google Drive Link](https://drive.google.com/drive/folders/1ajTRlSGIfg9OXQtXX7TfBhVDKaMPrALM?usp=drive_link)

---

## Directory Structure
