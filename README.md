# Avocado Ripening Stage Detection & Classification Pipeline
**Greenpod Labs Data Science Assignment-2**

## Overview
An end-to-end computer vision workflow comparing two distinct architectures for classifying avocado ripening stages (`Underripe`, `Breaking`, `Ripe`, `Overripe`):
1. **Approach 1 (Classical CV):** CIELAB color space ($a^*$ and $L^*$ channels) + Uniform Local Binary Patterns (LBP) texture descriptors + Random Forest Classifier.
2. **Approach 2 (Deep Learning):** Transfer Learning using MobileNetV2 with depthwise separable convolutions.

## Repository Contents
* `avocado_ripening_classification.ipynb`: Complete Google Colab notebook with EDA, data augmentation, training loops, and evaluation metrics.
* `main_demo.py`: Single-sample inference pipeline running both models.
* `Assignment2_Avocado_Report.pdf`: Comprehensive project report with comparative tables, latency analysis, and production question responses.

## Key Performance Results
* **Classical Pipeline (Random Forest):** 82.4% Accuracy, 0.798 Macro F1-score (~7 ms CPU latency).
* **Deep Learning (MobileNetV2):** 94.6% Accuracy, 0.941 Macro F1-score (~14 ms edge GPU latency).
