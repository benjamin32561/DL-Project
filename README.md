
# Cross-Image Attention for Zero-Shot Appearance Transfer

## Project Overview

This project addresses two critical challenges in the field of Cross-Image Attention for Zero-Shot Appearance Transfer: color saturation issues and the lack of objective evaluation metrics. The research explores innovative solutions to these problems through a two-pronged approach.

## Problem Statement

1.	To mitigate color saturation issues, we explore two main methods: a) Post-processing using Reinhard's Color Transfer algorithm to adjust the color characteristics of the output image. b) Developing a data-dependent β contrast factor based on entropy calculations to precisely control the contrast operation and prevent saturation effects.

2.	To objectively evaluate appearance transfer quality, we propose a novel evaluation framework that: a) Employs segmentation models, such as SAM2, to evaluate structure preservation using Intersection over Union (IOU). b) Utilizes cosine similarity between high-level features extracted using CLIP to assess structural similarity. c) Evaluates artistic style transfer quality using Gram matrices derived from VGG-19.

## Methodology

### Post-Processing Technique: Reinhard's Color Transfer
Reinhard's Color Transfer method was implemented as a post-processing step. This technique operates in the LAB color space, transferring color characteristics from a target image to a source image. 
To use the Reinhard Color Transfer method, run the saturation_correction.py script:
python saturation_correction.py --source <path_to_source_image> --target <path_to_target_image> --output <path_to_output_image>


### Dynamic Parameter Adjustment

A novel data-dependent β contrast factor was developed to refine the contrast operation and prevent saturation effects. This method leverages the statistical information contained in the attention map A, where each row represents the attention weights of a single spatial location in relation to all other spatial locations.
To use Dynamic Parameter Adjustment via the data-dependent β contrast factor, replace the original attention_utils.py in your project with the one provided in this repository, which includes the dynamic contrast adjustment functionality.


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
