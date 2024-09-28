import numpy as np

def calculate_iou(gt_masks, result_masks):
    """
    Calculate the Intersection over Union (IoU) between two sets of binary masks.

    Args:
        gt_masks (np.ndarray): Ground truth masks of shape (N, H, W), where N is the number of masks.
        result_masks (np.ndarray): Result masks of shape (N, H, W), where N is the number of masks.

    Returns:
        np.ndarray: IoU values for each mask of shape (N,).
    """
    # Ensure the masks are binary by thresholding
    gt_masks = (gt_masks > 0.5).astype(float)
    result_masks = (result_masks > 0.5).astype(float)
    
    # Calculate intersection and union
    intersection = np.sum(gt_masks * result_masks, axis=(1, 2))
    union = np.sum(gt_masks + result_masks, axis=(1, 2)) - intersection
    
    # Avoid division by zero
    union = np.maximum(union, 1e-6)
    
    # Calculate IoU
    iou = intersection / union
    
    return iou
