# MultiFreq2Scale‑Fusion  
A unified multi-frequency–multi-scale network for multimodal medical image feature fusion

<!-- Badges (CI, License, PyPI) can go here -->

## 📖 Overview  
This repository implements the network proposed in “A Unified Multi‑Frequency–Multi‑Scale Network for Multimodal Medical Image Feature Fusion” (2025). Our model addresses both classification and segmentation by:

1. **Dual‑Branch Alignment**: The two branches (ResNet branch and PVTNet branch) extract different features respectively  
2. **Multi‑Frequency Feature Extraction (MFFE)**: Applying DCT basis filters to aligned features to extract complementary frequency details.  
3. **Multi‑Frequency & Multi‑Scale Fusion**: Cross-frequency and cross-scale fusion of global and local features, followed by concatenation + Conv+BN+ReLU for final feature reassembly.  
## 🏗️ Architecture Diagram
<img src="images/architecture.png" alt="MFSF²‑NET Architecture" style="width:60%;"/>

*Figure 1. MFSF²‑NET diagram.*

## 🎯 Key Features  
- ✅ New P‑SinkD dataset: multisource dermoscopy & CT images
- ✅ Classification on HAM10000 & ISIC2019 datasets  
- ✅ Segmentation on ISIC2018 & Kvasir‑SEG datasets  
- ✅ Configurable training and evaluation scripts  

## 📂 Datasets
All datasets are split in an 8:1:1 ratio for training, validation, and testing.

**Classification Datasets**

HAM10000: 10,015 dermoscopic images across seven categories.

ISIC2019: Over 25,000 dermoscopic images covering melanoma and non-melanoma.

P‑SinkD (Private): 836 dermoscopic + 3,344 reflectance confocal microscopy (RCM) images from 532 patients (Wuxi Second People’s Hospital, 2018–2024). Annotation by three board-certified dermatologists; consensus labels.

<img src="images/P-SinkD.png" alt="Dataset visualization" style="width:60%;"/>

*Figure 2. Dataset visualization.*

**Segmentation Datasets**

ISIC2018: 2,594 dermoscopic images with lesion masks.

Kvasir‑SEG: 1,000 polyp images with segmentation masks.

**Results**

<img src="images/classification.png" alt="Classification result diagram" style="width:60%;"/>

*Figure 3. Classification result.*

<img src="images/segmentation.png" alt="Segmentation result graph" style="width:60%;"/>

*Figure 4. Segmentation result.*

## 🔧 Future Updates

This repository will be continuously updated with new features, optimizations, and expanded dataset support.
