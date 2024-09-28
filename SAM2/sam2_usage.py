import torch
import cv2
import numpy as np
from typing import Optional, Union, Tuple, List

from matplotlib import pyplot as plt

from .sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from .sam2.build_sam import build_sam2
from .sam2.sam2_image_predictor import SAM2ImagePredictor

def show_anns(anns, borders=True):
    if len(anns) == 0:
        return
    # Sort annotations by area in descending order
    sorted_anns = sorted(anns, key=lambda x: x['area'], reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    # Get the shape of the first segmentation mask
    # Ensure segmentation is a numpy array
    first_segmentation = np.array(sorted_anns[0]['segmentation'])
    img = np.ones((first_segmentation.shape[0], first_segmentation.shape[1], 4))
    img[:, :, 3] = 0  # Set alpha channel to 0 initially

    for ann in sorted_anns:
        m = np.array(ann['segmentation'])  # Convert segmentation to numpy array if it isn't already
        color_mask = np.concatenate([np.random.random(3), [0.5]])  # Random color with 50% transparency
        img[m] = color_mask  # Apply color mask where segmentation is True
        if borders:
            import cv2
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            # Smooth the contours
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1)  # Draw contours

    ax.imshow(img)

def get_automatic_model(checkpoint: str, model_cfg: str, device="cuda"):
    sam2 = build_sam2(model_cfg, checkpoint, device=device)

    mask_generator = SAM2AutomaticMaskGenerator(sam2)
    return sam2, mask_generator

def auto_predict_mask(auto_predictor, image):
    masks = auto_predictor.generate(image)
    return masks

def build_model_and_predictor(checkpoint: str, model_cfg: str):
    model = build_sam2(model_cfg, checkpoint)
    predictor = SAM2ImagePredictor(model)
    return model, predictor

def predict_masks_for_multiple_boxes(
        predictor,
        image_input: Union[str, np.ndarray, torch.Tensor],
        boxes: Union[List[np.ndarray], np.ndarray],
        multimask_output: bool = True,
        return_logits: bool = False,
        normalize_coords: bool = True,
) -> np.ndarray:
    """
    Predict masks for the given input prompts using the SAM2 model for multiple bounding boxes.

    Parameters:
    - image_input (str, np.ndarray, torch.Tensor): Path to the image file, or the image as a numpy array or torch tensor.
    - boxes (List[np.ndarray] or np.ndarray): A list of length 4 arrays or a 2D numpy array with shape (N, 4) containing the box prompts in XYXY format.
    - multimask_output (bool): If true, the model will return three masks. Default is True.
    - return_logits (bool): If true, returns un-thresholded mask logits instead of a binary mask. Default is False.
    - normalize_coords (bool): If true, the point coordinates will be normalized to the range [0,1]. Default is True.

    Returns:
    - np.ndarray: A numpy array containing all the single segmentation results for each box.
    """
    # Load image
    if isinstance(image_input, str):
        image = cv2.imread(image_input)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    elif isinstance(image_input, np.ndarray):
        image_rgb = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB) if image_input.shape[-1] == 3 else image_input
    elif isinstance(image_input, torch.Tensor):
        image_rgb = image_input.permute(1, 2, 0).cpu().numpy() if image_input.shape[0] == 3 else image_input.cpu().numpy()
    else:
        raise ValueError("Unsupported image input type. Must be file path, numpy array, or torch tensor.")

    # Set image in predictor
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        predictor.set_image(image_rgb)

        # Prepare list to hold all segmentation results
        all_masks = []

        # Iterate over each box and perform prediction
        for box in boxes:
            masks, quality_scores, low_res_logits = predictor.predict(
                box=box,
                multimask_output=multimask_output,
                return_logits=return_logits,
                normalize_coords=normalize_coords
            )
            all_masks.append(masks)

        # Convert list of masks to a numpy array
        all_masks_np = np.stack(all_masks, axis=0)

    return all_masks_np

def predict_masks(
    predictor,
    image_input: Union[str, np.ndarray, torch.Tensor],
    point_coords: Optional[np.ndarray] = None,
    point_labels: Optional[np.ndarray] = None,
    box: Optional[np.ndarray] = None,
    mask_input: Optional[np.ndarray] = None,
    multimask_output: bool = True,
    return_logits: bool = False,
    normalize_coords: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Predict masks for the given input prompts using the SAM2 model.

    Parameters:
    - image_input (str, np.ndarray, torch.Tensor): Path to the image file, or the image as a numpy array or torch tensor.
    - point_coords (Optional[np.ndarray]): A Nx2 array of point prompts to the model. Each point is in (X,Y) in pixels.
    - point_labels (Optional[np.ndarray]): A length N array of labels for the point prompts. 1 indicates a foreground point and 0 indicates a background point.
    - box (Optional[np.ndarray]): A length 4 array given a box prompt to the model, in XYXY format.
    - mask_input (Optional[np.ndarray]): A low resolution mask input to the model, typically coming from a previous prediction iteration. Has form 1xHxW.
    - multimask_output (bool): If true, the model will return three masks. Default is True.
    - return_logits (bool): If true, returns un-thresholded mask logits instead of a binary mask. Default is False.
    - normalize_coords (bool): If true, the point coordinates will be normalized to the range [0,1]. Default is True. xyxy

    Returns:
    - Tuple[np.ndarray, np.ndarray, np.ndarray]: 
        - The output masks in CxHxW format, where C is the number of masks.
        - An array of length C containing the model's predictions for the quality of each mask.
        - An array of shape CxHxW, where C is the number of masks and H=W=256. These low resolution logits can be passed to a subsequent iteration as mask input.
    """
    # Load image
    if isinstance(image_input, str):
        image = cv2.imread(image_input)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    elif isinstance(image_input, np.ndarray):
        image_rgb = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB) if image_input.shape[-1] == 3 else image_input
    elif isinstance(image_input, torch.Tensor):
        image_rgb = image_input.permute(1, 2, 0).cpu().numpy() if image_input.shape[0] == 3 else image_input.cpu().numpy()
    else:
        raise ValueError("Unsupported image input type. Must be file path, numpy array, or torch tensor.")

    # Perform inference and prediction
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        predictor.set_image(image_rgb)
        masks, quality_scores, low_res_logits = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box,
            mask_input=mask_input,
            multimask_output=multimask_output,
            return_logits=return_logits,
            normalize_coords=normalize_coords
        )    

    return masks, quality_scores, low_res_logits
