import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np

# CNN model to extract features (using a pre-trained VGG19 model)
class FeatureExtractor(nn.Module):
    """
    A class to extract features from specific layers of a pre-trained CNN.

    Args:
    - cnn (torch.nn.Module): A pre-trained CNN model (e.g., VGG19).
    - selected_layers (list of int): List of indices corresponding to the CNN layers
                                     from which features will be extracted.
    """
    def __init__(self, cnn, selected_layers):
        super(FeatureExtractor, self).__init__()
        self.cnn = cnn
        self.selected_layers = selected_layers
        
    def forward(self, x):
        """
        Forward pass through the CNN to extract features from the specified layers.

        Args:
        - x (torch.Tensor): The input image tensor.

        Returns:
        - features (list of torch.Tensor): The extracted feature maps from the specified layers.
        """
        features = []
        for i, layer in enumerate(self.cnn.features):
            x = layer(x)
            if i in self.selected_layers:
                features.append(x)
        return features

# Image preprocessing function for input to the CNN
def preprocess_image(image, size=256):
    """
    Preprocess an image for CNN input.

    Args:
    - image (PIL.Image): The input image to be preprocessed.
    - size (int): The size to which the image should be resized (default: 256).

    Returns:
    - torch.Tensor: A preprocessed image tensor with a batch dimension.
    """
    preprocess = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return preprocess(image).unsqueeze(0)  # Add batch dimension

# Function to convert a PyTorch tensor to a NumPy array
def tensor_to_np(tensor):
    """
    Convert a PyTorch tensor to a NumPy array.

    Args:
    - tensor (torch.Tensor): The input tensor.

    Returns:
    - np.ndarray: The corresponding NumPy array.
    """
    return tensor.squeeze(0).cpu().numpy()

# Function to extract features from a list of images
def extract_features(images, model, device):
    """
    Extract features from a list of images using a pre-trained CNN.

    Args:
    - images (list of PIL.Image): List of input images (PIL format).
    - model (FeatureExtractor): The CNN feature extractor.
    - device (str): The device to perform computation on ('cpu' or 'cuda').

    Returns:
    - features_list (list of list of np.ndarray): A list of feature maps for each image, where
                                                  each feature map is a NumPy array.
    """
    model.to(device)
    model.eval()  # Set the model to evaluation mode

    features_list = []
    with torch.no_grad():  # Disable gradient calculations
        for img in images:
            # Preprocess the image and move it to the appropriate device (CPU/GPU)
            img_tensor = preprocess_image(img).to(device)
            # Extract features from the CNN
            features = model(img_tensor)
            # Convert extracted features to NumPy arrays and store them
            features_list.append([tensor_to_np(f) for f in features])
    
    return features_list

# Gram matrix computation using NumPy
def gram_matrix_np(features, normalize=True):
    """
    Compute the Gram matrix from features (using NumPy).

    Args:
    - features (np.ndarray): Feature map of shape (C, H, W) for an image.
    - normalize (bool): Whether to normalize the Gram matrix by the number of neurons (H * W * C).

    Returns:
    - np.ndarray: Gram matrix of shape (C, C) for the input feature map.
    """
    C, H, W = features.shape
    features = features.reshape(C, H * W)  # Flatten the spatial dimensions
    gram = np.dot(features, features.T)  # Compute the Gram matrix

    if normalize:
        return gram / (H * W * C)  # Normalize by the total number of neurons
    else:
        return gram

# Function to compute the style score for each image
def style_score_np(feats_list, style_feats_list, style_weights):
    """
    Compute the style score between a set of images and the style reference image using Gram matrices.

    Args:
    - feats_list (list of np.ndarray): List of feature maps for each layer of the output image.
    - style_feats_list (list of np.ndarray): List of feature maps for each layer of the style reference image.
    - style_weights (list of float): Weights corresponding to each layer for the style score calculation.

    Returns:
    - float: Style score for the image, normalized to the range [0, 1], where 1 indicates the best match.
    """
    losses = []  # Initialize total loss for the image

    for i, (feats, style_feats) in enumerate(zip(feats_list, style_feats_list)):
        G = gram_matrix_np(feats)  # Gram matrix of the output image
        A = gram_matrix_np(style_feats)  # Gram matrix of the style reference image
        # Calculate the style loss as the weighted sum of squared differences between Gram matrices
        layer_loss = style_weights[i] * np.sum((G - A) ** 2)
        losses.append(layer_loss)
        
    loss = np.sum(losses)/np.sum(style_weights)  # Compute the total style loss

    # # Convert the total loss to a score between 0 and 1
    # min_loss = np.min(losses)
    # max_loss = np.max(losses)
    # normalized_losses = 1 - (losses - min_loss) / (max_loss - min_loss + 1e-5)
    
    # loss = np.sum(normalized_losses)

    return 1 / (1 + loss)  # Invert and normalize

# Main function to compute style scores for a list of images
def calculate_style_scores_for_images(images, style_images, cnn_model, selected_layers, style_weights, device='cpu'):
    """
    Extract features from a list of images and compute the style score for each image
    with respect to its corresponding style image.

    Args:
    - images (list of PIL.Image): List of output images to evaluate.
    - style_images (list of PIL.Image): List of style reference images, one per output image.
    - cnn_model (torch.nn.Module): A pre-trained CNN model (e.g., VGG19).
    - selected_layers (list of int): List of layer indices from which to extract features for style scoring.
    - style_weights (list of float): Weights for the style score at each layer.
    - device (str): The device to perform computation on ('cpu' or 'cuda').

    Returns:
    - np.ndarray: Style scores for each image in the range [0, 1], where 1 indicates the best match.
    """
    assert len(images) == len(style_images), "The number of output images must match the number of style images."

    # Initialize the feature extractor with the selected layers of the CNN
    feature_extractor = FeatureExtractor(cnn_model, selected_layers).to(device)
    
    # Extract features for the list of style images
    style_features_list = extract_features(style_images, feature_extractor, device)

    # Extract features for the list of output images
    image_features_list = extract_features(images, feature_extractor, device)

    # Compute style scores for each image and its corresponding style image
    scores = []
    for image_features, style_features in zip(image_features_list, style_features_list):
        score = style_score_np(image_features, style_features, style_weights)
        scores.append(score)

    return np.array(scores)
