import numpy as np
from scipy.linalg import sqrtm
import torch
from torchvision import transforms

def calculate_per_image_fid(reference_images, generated_images, model, device):
    """
    Calculates an adapted FID-like score between pairs of images.
    
    Parameters:
    - reference_images: list of np.ndarray of shape (H, W, C) for reference images.
    - generated_images: list of np.ndarray of shape (H, W, C) for generated images.
    - model: Pretrained model used to extract features.
    - device: 'cpu' or 'cuda'.
    
    Returns:
    - fid_scores: np.array of shape (N,) where each value is the adapted FID score for the corresponding image pair.
    """
    
    # Ensure both lists have the same length
    assert len(reference_images) == len(generated_images), "Both lists must have the same number of images"
    
    fid_scores = []
    
    # Transformation to preprocess images
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),  # Adjust if necessary
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    for ref_image, gen_image in zip(reference_images, generated_images):
        # Preprocess images
        ref_image_tensor = preprocess(ref_image).to(device)
        gen_image_tensor = preprocess(gen_image).to(device)
        
        # Extract features
        with torch.no_grad():
            ref_features = model.encode_image(ref_image_tensor.unsqueeze(0)).cpu().numpy().squeeze()
            gen_features = model.encode_image(gen_image_tensor.unsqueeze(0)).cpu().numpy().squeeze()
        
        # Compute means (since features are single vectors, means are the vectors themselves)
        mu_real = ref_features
        mu_gen = gen_features
        
        # Compute covariances (outer product of features)
        cov_real = np.outer(ref_features - mu_real, ref_features - mu_real)
        cov_gen = np.outer(gen_features - mu_gen, gen_features - mu_gen)
        
        # Compute the mean and covariance differences
        mean_diff = np.sum((mu_real - mu_gen) ** 2)
        cov_mean = sqrtm(cov_real @ cov_gen)
        
        # Handle numerical errors
        if np.iscomplexobj(cov_mean):
            cov_mean = cov_mean.real
        
        # Compute adapted FID
        fid = mean_diff + np.trace(cov_real + cov_gen - 2 * cov_mean)
        fid_scores.append(fid)
    
    fid_scores = np.array(fid_scores)
    return 1 / (1 + fid_scores)

def calculate_cosine_similarity(reference_images, generated_images, model, device):
    """
    Calculates the cosine similarity between pairs of images using a specified model for feature extraction.
    
    Parameters:
    - reference_images: list of np.ndarray of shape (H, W, C) for reference images.
    - generated_images: list of np.ndarray of shape (H, W, C) for generated images.
    - model: Pretrained model used to extract features.
    - device: 'cpu' or 'cuda'.
    
    Returns:
    - similarity_scores: np.array of shape (N,) where each value is the cosine similarity score for the corresponding image pair.
    """
    
    # Ensure both lists have the same length
    assert len(reference_images) == len(generated_images), "Both lists must have the same number of images"
    
    similarity_scores = []
    
    # Transformation to preprocess images
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),  # Adjust if necessary
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    for ref_image, gen_image in zip(reference_images, generated_images):
        # Preprocess images
        ref_image_tensor = preprocess(ref_image).to(device)
        gen_image_tensor = preprocess(gen_image).to(device)
        
        # Extract features
        with torch.no_grad():
            ref_features = model.encode_image(ref_image_tensor.unsqueeze(0)).cpu().numpy().squeeze()
            gen_features = model.encode_image(gen_image_tensor.unsqueeze(0)).cpu().numpy().squeeze()
        
        # Compute cosine similarity
        cosine_similarity = np.dot(ref_features, gen_features) / (np.linalg.norm(ref_features) * np.linalg.norm(gen_features))
        similarity_scores.append(cosine_similarity)
    
    return (np.array(similarity_scores)+1)/2
