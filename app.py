import tempfile
import gradio as gr
import numpy as np
from PIL import Image
import torch
import torchvision.models as models
import clip

# Import the ImagePrompter component
from gradio_image_prompter import ImagePrompter

# Import or define the SAM2 model and predictor functions
from SAM2.sam2_usage import build_model_and_predictor, predict_masks
from style_eval import calculate_style_scores_for_images
from structure_eval import calculate_iou
from FID_eval import calculate_cosine_similarity, calculate_per_image_fid

# Build the SAM2 model and predictor
checkpoint = "//raid/Ben/checkpoints/sam2_hiera_large.pt"
model_cfg = "//raid/Ben/DL Project/SAM2/sam2_configs/sam2_hiera_l.yaml"
model, predictor = build_model_and_predictor(checkpoint, model_cfg)

# Define the weights for each evaluation metric (equal weights for now)
weights = [1.0, 0.05, 1.0, 1.0]  # Equal weights

def parse_visual_prompt(points):
    boxes, neg_points, pos_points = [], [], []
    for point in points:
        if point[2] == 2 and point[-1] == 3:
            x1, y1, _, x2, y2, _ = point
            boxes.append([x1, y1, x2, y2])
        elif point[2] == 1 and point[-1] == 4:
            x, y, _, _, _, _ = point
            neg_points.append([x, y])
        elif point[2] == 0 and point[-1] == 4:
            x, y, _, _, _, _ = point
            pos_points.append([x, y])  # pos_points
    return boxes, pos_points, neg_points

def generate_masks_from_prompt(image_np, pos_points, neg_points):
    """Generate masks and scores for the given image and points."""
    masks, quality_scores = [], []
    if pos_points or neg_points:
        all_points = pos_points + neg_points
        point_coords = np.array(all_points)
        point_labels = np.array([1] * len(pos_points) + [0] * len(neg_points))
        masks_np, quality_scores_np, _ = predict_masks(
            predictor,
            image_input=image_np,
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=False,
            return_logits=False,
            normalize_coords=False
        )
        masks = [mask for mask in masks_np]
        quality_scores = quality_scores_np.tolist()
    return masks, quality_scores

def generate_single_image_masks(image_with_drawing):
    """Generate masks and scores for a single image with drawing."""
    if image_with_drawing is None or image_with_drawing["image"] is None:
        raise gr.Error("Please provide an image and drawing input.")

    image_np = np.array(image_with_drawing["image"])
    _, pos_points, neg_points = parse_visual_prompt(image_with_drawing["points"])

    masks, quality_scores = generate_masks_from_prompt(image_np, pos_points, neg_points)
    return masks, quality_scores, image_np

def create_colored_mask_image(masks, image_np):
    """Create a mask image where each mask is displayed in a different color."""
    if not masks:
        return Image.fromarray(image_np)

    mask_image = Image.fromarray(image_np).convert("RGBA")
    overlay = Image.new("RGBA", mask_image.size, (0, 0, 0, 0))

    for mask in masks:
        color = tuple(np.random.randint(0, 255, 3).tolist()) + (128,)
        mask_pil = Image.fromarray((mask * 255).astype(np.uint8), mode='L')
        color_image = Image.new("RGBA", mask_image.size, color)
        overlay = Image.composite(color_image, overlay, mask_pil)

    result_image = Image.alpha_composite(mask_image, overlay)
    return result_image.convert("RGB")

def generate_structure_masks(image_with_drawing):
    """Generate and display masks for the structure image."""
    structure_masks, structure_scores, structure_image_np = generate_single_image_masks(image_with_drawing)
    structure_mask_image = create_colored_mask_image(structure_masks, structure_image_np)
    return structure_mask_image, structure_masks, structure_scores, structure_image_np

def calculate_metrics(
    style_image,
    generated_image_with_drawing,
    generated_masks,
    generated_image_np,
    structure_image_with_drawing,
    structure_masks,
    structure_image_np
):
    if any(
        x is None
        for x in [
            style_image,
            generated_image_with_drawing,
            generated_masks,
            generated_image_np,
            structure_image_with_drawing,
            structure_masks,
            structure_image_np,
        ]
    ):
        raise gr.Error("Please provide all images and masks to calculate evaluation metrics.")

    # Convert the images from Gradio inputs (assume they are NumPy arrays)
    structure_image = Image.fromarray(structure_image_with_drawing["image"])
    style_image = Image.fromarray(style_image)
    generated_image = Image.fromarray(generated_image_with_drawing["image"])

    # Convert the style, structure, and generated images to lists for processing
    style_images = [style_image]
    structure_images = [structure_image]
    generated_images_list = [generated_image]

    # Combine masks for IoU calculation
    structure_mask_combined = np.sum(structure_masks, axis=0) > 0
    generated_mask_combined = np.sum(generated_masks, axis=0) > 0

    structure_masks_list = [structure_mask_combined.astype(np.float32)]
    generated_masks_list = [generated_mask_combined.astype(np.float32)]

    # Device setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Calculate Style Scores
    cnn_model = models.vgg19(pretrained=True).eval().to(device)  # Pre-trained VGG19 for feature extraction
    selected_layers = [0, 5, 10, 19]  # Layers for style score computation
    style_weights = [300000, 1000, 15, 3]  # Weights for style score layers

    style_scores = calculate_style_scores_for_images(
        generated_images_list,
        style_images,
        cnn_model,
        selected_layers,
        style_weights,
        device
    )

    # Calculate IoU
    iou_scores = calculate_iou(np.array(structure_masks_list), np.array(generated_masks_list))

    # Prepare images for FID and similarity calculation
    reference_images = []
    for img in structure_images:
        img_np = np.array(img)
        reference_images.append(img_np)

    generated_images_np = []
    for img in generated_images_list:
        img_np = np.array(img)
        generated_images_np.append(img_np)

    # Load CLIP model for feature extraction
    model_clip, preprocess_clip = clip.load("ViT-B/32", device=device)

    # Compute per-image FID scores
    fid_scores = calculate_per_image_fid(reference_images, generated_images_np, model_clip, device)
    # Normalize FID scores (lower is better, so invert and normalize)

    # Compute cosine similarity scores
    similarity_scores = calculate_cosine_similarity(reference_images, generated_images_np, model_clip, device)
    # Normalize similarity scores to range [0, 1]

    # Ensure all scores are numpy arrays
    iou_scores = np.array(iou_scores)
    fid_scores = np.array(fid_scores)
    style_scores = np.array(style_scores)
    similarity_scores = np.array(similarity_scores)

    # Compute overall evaluation scores
    overall_score = (
        weights[0] * iou_scores[0] +
        weights[1] * fid_scores[0] +
        weights[2] * style_scores[0] +
        weights[3] * similarity_scores[0]
    ) / sum(weights)

    # Prepare output metrics text
    metrics_text = (
        f"Style Score: {style_scores[0]:.4f}\n"
        f"IoU Score: {iou_scores[0]:.4f}\n"
        f"FID Score: {fid_scores[0]:.4f}\n"
        f"Cosine Similarity Score: {similarity_scores[0]:.4f}\n"
        f"Overall Score: {overall_score:.4f}"
    )

    return metrics_text

def download_mask(masks):
    if not masks:
        return None
    
    combined_mask = np.sum(masks, axis=0) > 0
    mask_image = Image.fromarray((combined_mask * 255).astype(np.uint8), mode='L')
    
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
        mask_image.save(tmp_file, format="PNG")
        tmp_file_path = tmp_file.name
    
    return tmp_file_path

with gr.Blocks() as demo:
    style_image = gr.Image(label="Style Image")

    with gr.Row():
        structure_image_with_drawing = ImagePrompter(label="Structure Image", scale=1)
        with gr.Column():
            structure_image_mask = gr.Image(label="Structure Image Mask")
            structure_image_mask_download_button = gr.DownloadButton("Download Mask")
    generate_structure_masks_button = gr.Button("Generate Structure Masks")

    with gr.Row():
        generated_image_with_drawing = ImagePrompter(label="Generated Image", scale=1)
        with gr.Column():
            generated_image_mask = gr.Image(label="Generated Image Mask")
            generated_image_mask_download_button = gr.DownloadButton("Download Mask")
    generate_generated_masks_button = gr.Button("Generate Generated Masks")

    run_button = gr.Button("Calculate Evaluation Metrics")
    metrics_output = gr.Textbox(label="Evaluation Metrics", lines=6)

    # Define gr.State variables
    generated_masks_state = gr.State()
    generated_scores_state = gr.State()
    generated_image_np_state = gr.State()

    structure_masks_state = gr.State()
    structure_scores_state = gr.State()
    structure_image_np_state = gr.State()

    # Generate masks for structure image when the button is clicked
    generate_structure_masks_button.click(
        fn=generate_structure_masks,
        inputs=[structure_image_with_drawing],
        outputs=[
            structure_image_mask,
            structure_masks_state,
            structure_scores_state,
            structure_image_np_state
        ]
    )

    # Generate masks for generated image when the button is clicked
    generate_generated_masks_button.click(
        fn=generate_structure_masks,
        inputs=[generated_image_with_drawing],
        outputs=[
            generated_image_mask,
            generated_masks_state,
            generated_scores_state,
            generated_image_np_state
        ]
    )

    # Calculate metrics when the run button is clicked
    run_button.click(
        fn=calculate_metrics,
        inputs=[
            style_image,
            generated_image_with_drawing,
            generated_masks_state,
            generated_image_np_state,
            structure_image_with_drawing,
            structure_masks_state,
            structure_image_np_state
        ],
        outputs=metrics_output
    )

    # Download mask for structure image
    structure_image_mask_download_button.click(
        fn=download_mask,
        inputs=[structure_masks_state],
        outputs=[structure_image_mask_download_button]
    )

    generated_image_mask_download_button.click(
        fn=download_mask,
        inputs=[generated_masks_state],
        outputs=[generated_image_mask_download_button]
    )

demo.launch()