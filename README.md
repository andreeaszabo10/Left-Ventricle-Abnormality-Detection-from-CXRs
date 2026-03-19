# Left Ventricle Abnormality Detection from Chest X-Rays

Deep learning project for detecting left ventricular structural abnormalities from chest X-ray images.

This project focuses on identifying two cardiac conditions from routine chest radiographs:

- **SLVH** — Severe Left Ventricular Hypertrophy
- **DLV** — Dilated Left Ventricle

The goal is to explore whether chest X-rays contain enough visual information to support early detection of these abnormalities, using labels derived from echocardiography. The project compares a supervised CNN baseline with a transformer-based approach built on self-supervised pretrained features. :contentReference[oaicite:1]{index=1}

## Project Overview

Echocardiography is the standard tool for evaluating structural heart abnormalities, but it is not always available. Chest X-rays are cheaper, faster, and more accessible, so this project investigates whether they can be used for automatic screening of left ventricular abnormalities with deep learning. :contentReference[oaicite:2]{index=2}

The project includes:

- a **DenseNet121 baseline** that predicts echocardiographic measurements from chest X-rays
- a **ViT-Small transformer model** initialized with **Medical-MAE** pretrained weights
- binary classification experiments for **SLVH** and **DLV**
- comparison of preprocessing and augmentation strategies
- evaluation using **AUROC, precision, recall, and F1-score** :contentReference[oaicite:3]{index=3}

## Dataset

The project uses the **CheXchoNet** dataset, a multimodal dataset containing:

- **71,589 chest X-rays**
- **24,689 patients**
- paired echocardiography measurements:
  - IVSd
  - LVPWd
  - LVIDd
- demographic and clinical metadata such as age and sex :contentReference[oaicite:4]{index=4}

The original dataset is highly imbalanced, so balanced subsets were created for training and evaluation. For the transformer experiments, separate balanced binary datasets were built for SLVH and DLV, excluding combined-pathology samples. :contentReference[oaicite:5]{index=5}

## Methods

### 1. DenseNet121 Baseline

The baseline follows a supervised learning setup inspired by prior work:

- DenseNet121 pretrained on ImageNet
- predicts echocardiographic measurements from chest X-rays
- combines image features with age and sex
- outputs Gaussian parameters for IVSd, LVPWd, and LVIDd
- converts predicted measurements into disease probabilities using clinical thresholds :contentReference[oaicite:6]{index=6}

This baseline serves as a reference point for evaluating whether more modern feature extractors can improve performance.

### 2. Vision Transformer Approach

The second approach uses a **ViT-Small** model as a frozen feature extractor:

- initialized with **Medical-MAE** self-supervised pretrained weights
- pretrained on about **300,000 chest radiographs**
- transformer encoder frozen during training
- small **MLP classifier** trained on top for binary classification
- trained separately for **SLVH vs Healthy** and **DLV vs Healthy** :contentReference[oaicite:7]{index=7}

This setup removes explicit demographic inputs during representation learning and focuses on extracting meaningful visual signals directly from the chest X-rays.

## Preprocessing and Augmentation

The final transformer pipeline uses:

- normalization with mean **0.5** and std **0.5**
- random resized crop
- horizontal flip
- small rotations :contentReference[oaicite:8]{index=8}

These choices produced the most stable and best-performing results across experiments.

## Results

### Baseline Results

DenseNet121 achieved the following test AUROC values:

- **SLVH:** ~0.73
- **DLV:** ~0.75
- **Composite outcome:** ~0.79 :contentReference[oaicite:9]{index=9}

### Transformer Results

The ViT-based approach improved performance on both tasks:

- **SLVH:** **0.7807 AUROC**
- **DLV:** **0.7887 AUROC** :contentReference[oaicite:10]{index=10}

This suggests that self-supervised transformer features capture more relevant information from chest X-rays than the CNN baseline in this setting.

## Tech Stack

- **Python**
- **PyTorch**
- **Torchvision**
- **Vision Transformers**
- **DenseNet121**
- **Medical-MAE pretrained weights** :contentReference[oaicite:11]{index=11}

## Key Takeaways

- Chest X-rays contain useful signals for detecting left ventricular abnormalities
- A DenseNet121 baseline can learn meaningful patterns from radiographs
- A frozen self-supervised ViT encoder outperformed the CNN baseline
- Preprocessing and augmentation choices had a measurable impact on AUROC
- The project supports the use of deep learning as a possible screening aid for structural heart disease from routine imaging :contentReference[oaicite:12]{index=12}

## Limitations

Some important limitations of the project:

- balanced training subsets do not reflect real-world prevalence
- labels are derived from echocardiography thresholds and may contain noise
- data comes from a single center
- frozen encoders reduce overfitting but may limit adaptation to task-specific patterns
- image-only models may still encode demographic information indirectly :contentReference[oaicite:13]{index=13}

## Future Work

Possible next steps include:

- training on the full imbalanced dataset
- external validation on data from other hospitals
- careful fine-tuning of the transformer encoder
- multitask learning for multiple cardiac abnormalities
- stronger interpretability analysis and robustness testing :contentReference[oaicite:14]{index=14}

## Author

Cristina-Andreea Szabo
