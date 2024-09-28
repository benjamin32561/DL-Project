
# Cross-Image Attention for Zero-Shot Appearance Transfer

## Project Overview

This project focuses on evaluating the effectiveness of appearance transfer in generative models, particularly in terms of structure maintenance and artistic style transfer quality. The evaluation process addresses the limitations of traditional methods by introducing objective metrics that systematically measure both structural preservation and style consistency. The project implements a new evaluation method using segmentation models, such as **SAM2**, to analyze structure preservation through **Intersection over Union (IOU)** and **cosine similarity** between high-level features extracted from **CLIP**. The artistic style transfer is evaluated using **Gram matrices** derived from **VGG-19**.

## Problem Statement

Existing methods for appearance transfer often produce oversaturated results and rely on subjective evaluations, making it difficult to quantify performance across different models. To overcome these issues, this project proposes an **Evaluating Appearance Transfer** metric, which utilizes segmentation models and feature-based evaluation techniques to provide an automated, quantitative measure of appearance transfer quality. This approach enables more consistent and objective evaluations compared to traditional methods.

## Methodology

### Structure Preservation Evaluation

- **Segmentation Models:** Multiple segmentation models were tested, including **SAM**, **SAM2**, and **FastSAM**. After observing that automated methods often produced unsatisfactory results, a semi-automated approach using SAM2 with manual reference points was adopted.
- **Metrics Used:** 
  - **Intersection over Union (IOU)** was calculated between segmentation masks of the structure input and the output image.
  - **Cosine Similarity** was used to compare high-level feature vectors extracted from **CLIP** for evaluating the similarity between the structure input and output images.

### Artistic Style Transfer Evaluation

- **CNN-Based Evaluation:** Features were extracted from different layers of **VGG-19**, and **Gram matrices** were computed to assess the style consistency between the input style image and the output image.
- It was observed that features from higher layers may capture structural elements, suggesting the need for optimization by focusing on lower layers more closely related to artistic style.

## Key Results

The proposed evaluation method showed that:

- **Structure Preservation**: Manual refinement using SAM2 provided significantly better results compared to fully automated segmentation, achieving higher IOU scores and improved structure preservation.
- **Artistic Style Transfer**: Gram matrices derived from VGG-19 provided a robust evaluation of style transfer, though future work could involve optimizing the selection of features from lower layers to improve the quality of style evaluation.

## Future Work

Further exploration can be done using **Transformer-based models** for style evaluation or optimizing the **CNN-based evaluation** by focusing on relevant features for style alone. Extending the dataset could enable a re-evaluation of **Frechet Inception Distance (FID)** for larger-scale projects.

## Appendix

The project code is available in the repository, including implementations for generating masks, calculating evaluation metrics, and the Gradio-based GUI used for this project.

---
