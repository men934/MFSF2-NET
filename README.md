# MultiFreq2Scale‑Fusion  
A unified multi-frequency–multi-scale network for multimodal medical image feature fusion

<!-- Badges (CI, License, PyPI) can go here -->

## 📖 Overview  
This repository implements the network proposed in “A Unified Multi‑Frequency–Multi‑Scale Network for Multimodal Medical Image Feature Fusion” (2025). Our model addresses both classification and segmentation by:

1. **Dual‑Branch Alignment**: Aligning ResNet and PVTNet feature maps via bilinear interpolation + 3×3 convolution, combining global and local representations.  
2. **Multi‑Frequency Feature Extraction (MFFE)**: Applying DCT basis filters to aligned features to extract complementary frequency details.  
3. **Multi‑Frequency & Multi‑Scale Fusion**: Cross-frequency and cross-scale fusion of global and local features, followed by concatenation + Conv+BN+ReLU for final feature reassembly.  

## 🎯 Key Features  
- ✅ Classification on HAM10000 & ISIC2019 datasets  
- ✅ Segmentation on ISIC2018 & Kvasir‑SEG datasets  
- ✅ New P‑SinkD dataset: multisource dermoscopy & CT images
- ✅ Configurable training and evaluation scripts  
