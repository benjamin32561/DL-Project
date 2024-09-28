
# SAM 2: Segment Anything in Images and Videos

## Overview
SAM 2 is an advanced model for segmenting objects in images and videos. This repository provides functionalities for both image and video segmentation using the SAM 2 model.
Based on https://github.com/facebookresearch/segment-anything-2

## Download Models
Downloading models instructions can be found in https://github.com/facebookresearch/segment-anything-2

## Installation
To install the SAM 2 package, you need to have Python 3.10.0 or higher. You can install the required dependencies using pip.

### Build and Install SAM 2

```shell
pip install torch==2.4.0+cu121 torchvision==0.19.0+cu121 torchaudio==2.4.0+cu121 --index-url https://download.pytorch.org/whl/cu121
```

```bash
pip install -e .
python setup.py install
```

## Usage

### Image Segmentation

#### Example
```python
import os
import cv2
import numpy as np
from sam2_usage import predict_masks, save_image_with_overlay

# Define paths and initial points
image_path = "/path/to/your/image.jpg"
checkpoint = "/path/to/your/checkpoint.pth"
model_cfg = "/path/to/your/model_cfg.yaml"
point_coords = np.array([(1400, 330)], dtype=np.float32)
point_labels = np.array([1], dtype=np.int32)
output_path = "/path/to/save/output.jpg"

# Predict masks
masks, quality_scores, low_res_logits = predict_masks(
    image_input=image_path,
    checkpoint=checkpoint,
    model_cfg=model_cfg,
    point_coords=point_coords,
    point_labels=point_labels
)

# Save the image with overlays
save_image_with_overlay(
    image_input=image_path,
    masks=masks,
    point_coords=point_coords,
    point_labels=point_labels,
    output_path=output_path
)
```

### Video Segmentation

#### Example with Directory Path
```python
import os
import cv2
import numpy as np
import torch
from sam2_usage import predict_masks_video, save_video

# Define paths and initial inputs
video_dir = "/path/to/your/video_frames"
output_path = "/path/to/save/output.mp4"
checkpoint = "/path/to/your/checkpoint.pth"
model_cfg = "/path/to/your/model_cfg.yaml"
initial_points = {
    3: {'points': (np.array([[1400, 330]], dtype=np.float32), np.array([1], dtype=np.int32))}  # Example initial points and labels
}

# Create an initial mask
frame = cv2.imread(os.path.join(video_dir, "0000.jpg"))
frame_shape = frame.shape[:2]
frame_height, frame_width = frame_shape
mask = np.zeros(frame_shape, dtype=np.uint8)  # Create an empty mask with the frame's shape
mask[447:470, 1360:1420] = 1  # Set the bounding box area to 1
initial_masks = {
    4: {'mask': mask}  # Example initial mask
}
initial_inputs = {**initial_points, **initial_masks}

# Predict masks using the model
video_segments, frame_names, frame_height, frame_width = predict_masks_video(video_dir, checkpoint, model_cfg, initial_inputs)

# Save the output as a video
save_video(output_path, video_segments, frame_names, frame_height, frame_width, initial_inputs)
```

#### Example with List of Frame Paths
```python
import os
import cv2
import numpy as np
import torch
from sam2_usage import predict_masks_video, save_video

# Define paths and initial inputs
video_dir = "/path/to/your/video_frames"
frame_paths = [os.path.join(video_dir, f"{i:04d}.jpg") for i in range(len(os.listdir(video_dir)))]
output_path = "/path/to/save/output.mp4"
checkpoint = "/path/to/your/checkpoint.pth"
model_cfg = "/path/to/your/model_cfg.yaml"
initial_points = {
    3: {'points': (np.array([[1400, 330]], dtype=np.float32), np.array([1], dtype=np.int32))}  # Example initial points and labels
}

# Create an initial mask
frame = cv2.imread(frame_paths[0])
frame_shape = frame.shape[:2]
frame_height, frame_width = frame_shape
mask = np.zeros(frame_shape, dtype=np.uint8)  # Create an empty mask with the frame's shape
mask[447:470, 1360:1420] = 1  # Set the bounding box area to 1
initial_masks = {
    4: {'mask': mask}  # Example initial mask
}
initial_inputs = {**initial_points, **initial_masks}

# Predict masks using the model
video_segments, frame_names, frame_height, frame_width = predict_masks_video(frame_paths, checkpoint, model_cfg, initial_inputs)

# Save the output as a video
save_video(output_path, video_segments, frame_paths, frame_height, frame_width, initial_inputs)
```

#### Example with Numpy Array of Frames
```python
import os
import cv2
import numpy as np
import torch
from sam2_usage import predict_masks_video, save_video

# Define paths and initial inputs
video_dir = "/path/to/your/video_frames"
frame_paths = [os.path.join(video_dir, f"{i:04d}.jpg") for i in range(len(os.listdir(video_dir)))]
output_path = "/path/to/save/output.mp4"
checkpoint = "/path/to/your/checkpoint.pth"
model_cfg = "/path/to/your/model_cfg.yaml"
initial_points = {
    3: {'points': (np.array([[1400, 330]], dtype=np.float32), np.array([1], dtype=np.int32))}  # Example initial points and labels
}

# Create an initial mask
frame = cv2.imread(frame_paths[0])
frame_shape = frame.shape[:2]
frame_height, frame_width = frame_shape
mask = np.zeros(frame_shape, dtype=np.uint8)  # Create an empty mask with the frame's shape
mask[447:470, 1360:1420] = 1  # Set the bounding box area to 1
initial_masks = {
    4: {'mask': mask}  # Example initial mask
}
initial_inputs = {**initial_points, **initial_masks}

# Load frames as numpy array
numpy_frames = np.array([cv2.imread(frame_path) for frame_path in frame_paths])

# Predict masks using the model
video_segments, _ = predict_masks_video(numpy_frames, checkpoint, model_cfg, initial_inputs)

# Save the output as a video
save_video(output_path, video_segments, numpy_frames, frame_height, frame_width, initial_inputs)
```

#### Example with Tensor of Frames
```python
import os
import cv2
import numpy as np
import torch
from sam2_usage import predict_masks_video, save_video

# Define paths and initial inputs
video_dir = "/path/to/your/video_frames"
frame_paths = [os.path.join(video_dir, f"{i:04d}.jpg") for i in range(len(os.listdir(video_dir)))]
output_path = "/path/to/save/output.mp4"
checkpoint = "/path/to/your/checkpoint.pth"
model_cfg = "/path/to/your/model_cfg.yaml"
initial_points = {
    3: {'points': (np.array([[1400, 330]], dtype=np.float32), np.array([1], dtype=np.int32))}  # Example initial points and labels
}

# Create an initial mask
frame = cv2.imread(frame_paths[0])
frame_shape = frame.shape[:2]
frame_height, frame_width = frame_shape
mask = np.zeros(frame_shape, dtype=np.uint8)  # Create an empty mask with the frame's shape
mask[447:470, 1360:1420] = 1  # Set the bounding box area to 1
initial_masks = {
    4: {'mask': mask}  # Example initial mask
}
initial_inputs = {**initial_points, **initial_masks}

# Load frames as numpy array
numpy_frames = np.array([cv2.imread(frame_path) for frame_path in frame_paths])

# Convert to tensor
tensor_frames = torch.from_numpy(numpy_frames).permute(0, 3, 1, 2)  # Convert to tensor and permute to (num_frames, channels, height, width)

# Predict masks using the model
video_segments, _ = predict_masks_video(tensor_frames, checkpoint, model_cfg, initial_inputs)

# Save the output as a video
save_video(output_path, video_segments, tensor_frames, frame_height, frame_width, initial_inputs)
```
