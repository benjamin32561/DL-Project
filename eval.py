import torch
import torchvision.models as models
import numpy as np
from PIL import Image
from style_eval import calculate_style_scores_for_images
from structure_eval import calculate_iou
from FID_eval import calculate_per_image_fid, calculate_cosine_similarity  # Updated import
import clip

# Define the weights for each evaluation metric (equal weights for now)
# weights = [IOU_weight, FID_weight, Style_weight]
weights = [1.0, 1.0, 1.0]  # Equal weights

# Load the style images (replace with your actual style image paths)
style_images = [
    Image.open('style_image.png'),
]

# Load the structure images (replace with your actual structure image paths)
structure_images = [
    Image.open('structure_image.png'),
]

# Load the structure masks (ground truth masks)
structure_mask_images = [
    Image.open('structure_mask.png'),
]

# Load the output images (replace with actual paths of your output images)
image_list = [
    Image.open("generated_image.png"),
]  # Add your image paths

image_mask_list = [
    Image.open("generated_mask.png"),
]  # Add your image paths

# Use VGG19 for feature extraction (pre-trained on ImageNet)
cnn_model = models.vgg19(pretrained=True)

# Layers to be used for style loss computation (e.g., conv layers in VGG19)
selected_layers = [0, 5, 10, 19]

# Weights for the style loss corresponding to each selected layer
style_weights = [300000, 1000, 15, 3]

# Compute the style loss for the list of images and their corresponding style images
device = 'cuda' if torch.cuda.is_available() else 'cpu'
style_losses = calculate_style_scores_for_images(
    image_list,
    style_images,
    cnn_model,
    selected_layers,
    style_weights,
    device
)

print("Style losses for each image:", style_losses)

# Normalize style losses to get style scores (lower loss means better style matching)
style_scores = 1 / (1 + np.array(style_losses))  # Normalize to range (0, 1]

# Load ground truth masks and convert to numpy arrays
gt_masks = []
for mask_img in structure_mask_images:
    mask_np = np.array(mask_img.convert('L'))
    mask_np = (mask_np > 128).astype(float)
    gt_masks.append(mask_np)

# Generate result masks from output images (e.g., using a simple thresholding)
result_masks = []
for img in image_mask_list:
    gray_img = img.convert('L')  # Convert to grayscale
    mask_np = np.array(gray_img)
    # Apply a simple threshold to create a binary mask
    mask_np = (mask_np > 128).astype(float)
    result_masks.append(mask_np)

# Convert masks to numpy arrays
gt_masks_np = np.array(gt_masks)
result_masks_np = np.array(result_masks)

# Compute IoU scores
iou_scores = calculate_iou(gt_masks_np, result_masks_np)

print("IoU scores for each image:", iou_scores)

# Load reference images (e.g., structure images) and generated images (output images) for similarity computation
reference_images = []
for img in structure_images:
    img_np = np.array(img)
    reference_images.append(img_np)

generated_images = []
for img in image_list:
    img_np = np.array(img)
    generated_images.append(img_np)

# Load CLIP model for feature extraction
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_clip, preprocess_clip = clip.load("ViT-B/32", device=device)

# Compute per-image FID scores
fid_scores = calculate_per_image_fid(reference_images, generated_images, model_clip, device)
print("Per-image adapted FID scores for each image:", fid_scores)

similarity_scores = calculate_cosine_similarity(reference_images, generated_images, model_clip, device)
print("Cosine similarity scores for each image:", similarity_scores)

# Compute overall evaluation scores using the weights
# Ensure all scores are numpy arrays
iou_scores = np.array(iou_scores)
style_scores = np.array(style_scores)

# Choose whether to use fid_scores_normalized or similarity_scores_normalized
# For this example, we'll use cosine similarity
# If you prefer to use fid_scores_normalized, replace similarity_scores_normalized with fid_scores_normalized

# Compute the weighted sum of the scores
overall_scores = (
    weights[0] * iou_scores +
    weights[1] * similarity_scores +
    weights[2] * style_scores
) / sum(weights)

print("Overall evaluation scores for each image:", overall_scores)
