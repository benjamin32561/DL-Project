import os
import cv2
import numpy as np
import torch

import PARAMETER
from sam2_usage import predict_masks, save_image_with_overlay, save_video, predict_masks_video
from util.video_to_numpy_array import get_video_frames

#### Image example usage #####
# image_path = "/raid/NehorayProjects/Ben/sprint/dataset/scene_1/0001.jpg"
# checkpoint = "/raid/NehorayProjects/Ben/sprint/segment-anything-2/checkpoints/sam2_hiera_large.pt"
# model_cfg = "sam2_hiera_l.yaml"
# point_coords = np.array([(1400, 330)], dtype=np.float32)
# point_labels = np.array([1], dtype=np.int32)
# output_path = "/raid/NehorayProjects/Ben/sprint/segment-anything-2/test.jpg"
#
# masks, quality_scores, low_res_logits = predict_masks(
#     image_input=image_path,
#     checkpoint=checkpoint,
#     model_cfg=model_cfg,
#     point_coords=point_coords,
#     point_labels=point_labels
# )
#
# save_image_with_overlay(
#     image_input=image_path,
#     masks=masks,
#     point_coords=point_coords,
#     point_labels=point_labels,
#     output_path=output_path
# )
#
# #### Video example usage with directory path ####
# video_dir = r"C:\Users\orior\PycharmProjects\VideoImageEnhancement\data\images\Shaback\sub\subsub"
# output_path = "/raid/NehorayProjects/Ben/sprint/segment-anything-2/test_dir.mp4"
# checkpoint = "C:\Users\orior\PycharmProjects\VideoImageEnhancement\Segmentation\SAM2\checkpoints\sam2_hiera_large.pt"
# model_cfg = "sam2_hiera_l.yaml"


initial_points = {
    3: {'points': (np.array([[320, 250]], dtype=np.float32), np.array([1], dtype=np.int32))}   # Example initial points and labels
}

# frame = cv2.imread(os.path.join(video_dir, "0000.jpg"))

#
# # # Predict masks using the model
# video_segments, frame_names = predict_masks_video(video_dir, checkpoint, model_cfg, initial_inputs)
#
# # # Save the output as a video
# save_video(output_path, video_segments, frame_names, frame_height, frame_width, initial_inputs)
#



#
#
# #### Video example usage with list of frame paths ####
# frame_paths = [os.path.join(video_dir, f"{i:04d}.jpg") for i in range(len(frame_names))]
# output_path = "/raid/NehorayProjects/Ben/sprint/segment-anything-2/test_list.mp4"
#
# # # Predict masks using the model
# video_segments, frame_names = predict_masks_video(frame_paths, checkpoint, model_cfg, initial_inputs)
#
# # # Save the output as a video
# save_video(output_path, video_segments, frame_paths, frame_height, frame_width, initial_inputs)




def numpy_unsqueeze(input_tensor, dim=-1):
    return np.expand_dims(input_tensor, dim)

def list_to_numpy(input_list):
    input_list = np.concatenate([numpy_unsqueeze(input_list[i], 0) for i in np.arange(len(input_list))])
    return input_list




#### Video example usage with numpy array of frames ####
# numpy_frames = np.array([cv2.imread(frame_path) for frame_path in frame_paths])

video_path = r"C:\Users\orior\PycharmProjects\VideoImageEnhancement\data\images\videos\Car_Going_Down\scene_0_resized_short_compressed.mp4"
video_list_of_frames = get_video_frames(video_path)[:5]
video_as_numpy = list_to_numpy(video_list_of_frames)

frame_shape = video_as_numpy[0].shape[:2]
frame_height, frame_width = frame_shape
mask = np.zeros(frame_shape, dtype=np.uint8)  # Create an empty mask with the frame's shape
# # XYXY: 1360 447 1420 470
mask[447:470, 1360:1420] = 1  # Set the bounding box area to 1
initial_masks = {
    4: {'mask': mask}  # Example initial mask
}
initial_inputs = {**initial_points, **initial_masks}


numpy_frames = video_as_numpy
output_path = "test_numpy.mp4"

checkpoint = PARAMETER.SAN_2_MODEL_HIERA_LARGE
model_cfg = PARAMETER.SAN_2_CONFIG_HIERA_LARGE

# Predict masks using the model
video_segments, _ = predict_masks_video(numpy_frames.copy(), checkpoint, model_cfg, initial_inputs)

# Save the output as a video
save_video(output_path, video_segments, numpy_frames, frame_height, frame_width, initial_inputs)






#
#
# #### Video example usage with tensor of frames ####
# tensor_frames = torch.from_numpy(numpy_frames).permute(0, 3, 1, 2)  # Convert to tensor and permute to (num_frames, channels, height, width)
# output_path = "/raid/NehorayProjects/Ben/sprint/segment-anything-2/test_tensor.mp4"
#
# # Predict masks using the model
# video_segments, _ = predict_masks_video(tensor_frames, checkpoint, model_cfg, initial_inputs)
#
# # Save the output as a video
# save_video(output_path, video_segments, tensor_frames, frame_height, frame_width, initial_inputs)
