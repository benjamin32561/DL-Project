import cv2
import numpy as np
import matplotlib.pyplot as plt



# Define paths to data
sorce_path = '/content/drive/My Drive/Deep Learning/final project/article_results'
target_path = '/content/drive/My Drive/Deep Learning/final project/article_results'

# Load the images
source = cv2.imread(sorce_path)
target = cv2.imread(target_path)


# Function to perform Reinhard color transfer
def color_transfer(source, target):
    # Convert images to Lab color space
    source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2Lab).astype(np.float32)
    target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2Lab).astype(np.float32)

    # Split the Lab channels
    l_s, a_s, b_s = cv2.split(source_lab)
    l_t, a_t, b_t = cv2.split(target_lab)

    # Calculate the mean and std dev of each channel in both source and target
    l_mean_s, l_std_s = l_s.mean(), l_s.std()
    a_mean_s, a_std_s = a_s.mean(), a_s.std()
    b_mean_s, b_std_s = b_s.mean(), b_s.std()

    l_mean_t, l_std_t = l_t.mean(), l_t.std()
    a_mean_t, a_std_t = a_t.mean(), a_t.std()
    b_mean_t, b_std_t = b_t.mean(), b_t.std()

    # Transfer the color by matching the mean and standard deviation
    l_s = (l_s - l_mean_s) * (l_std_t / l_std_s) + l_mean_t
    a_s = (a_s - a_mean_s) * (a_std_t / a_std_s) + a_mean_t
    b_s = (b_s - b_mean_s) * (b_std_t / b_std_s) + b_mean_t

    # Merge the channels and convert back to BGR color space
    result_lab = cv2.merge([l_s, a_s, b_s])

    result = cv2.cvtColor(result_lab.astype(np.uint8), cv2.COLOR_Lab2BGR)

    return result

def color_transfer_optimized(source, target):
    # Convert images to LAB color space
    source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype(np.float32)
    target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype(np.float32)

    # Split into L, A, B channels
    l_src, a_src, b_src = cv2.split(source_lab)
    l_tar, a_tar, b_tar = cv2.split(target_lab)

    # Compute mean and standard deviation for each channel in both images
    src_means = [np.mean(l_src), np.mean(a_src), np.mean(b_src)]
    src_stds = [np.std(l_src), np.std(a_src), np.std(b_src)]

    tar_means = [np.mean(l_tar), np.mean(a_tar), np.mean(b_tar)]
    tar_stds = [np.std(l_tar), np.std(a_tar), np.std(b_tar)]

    # Perform the transfer with scaling to avoid clipping
    # For each channel: new_value = (old_value - target_mean) * (src_std / tar_std) + src_mean

    for i, (l, a, b) in enumerate(zip([l_tar], [a_tar], [b_tar])):
        # L channel transfer, ensure no negative or out-of-bounds value
        l_tar = ((l_tar - tar_means[0]) * (src_stds[0] / (tar_stds[0] + 1e-5))) + src_means[0]
        l_tar = np.clip(l_tar, 0, 100)  # For LAB L channel, 0 to 100 range

        # A and B channels transfer, ensure within valid -128 to 127 range
        a_tar = ((a_tar - tar_means[1]) * (src_stds[1] / (tar_stds[1] + 1e-5))) + src_means[1]
        a_tar = np.clip(a_tar, -128, 127)  # LAB A channel: -128 to 127

        b_tar = ((b_tar - tar_means[2]) * (src_stds[2] / (tar_stds[2] + 1e-5))) + src_means[2]
        b_tar = np.clip(b_tar, -128, 127)  # LAB B channel: -128 to 127

    # Merge the adjusted LAB channels
    result_lab = cv2.merge([l_tar, a_tar, b_tar])

    # Convert LAB back to RGB color space
    result_rgb = cv2.cvtColor(result_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

    return result_rgb


# color transfer
result = color_transfer(source, target)

# color transfer optimized
result2 = color_transfer_optimized(source, target)


# Create subplots
fig, axes = plt.subplots(2, 2, figsize=(10, 10))

# Original Image
axes[0, 0].imshow(cv2.cvtColor(source, cv2.COLOR_BGR2RGB)) 
axes[0, 0].set_title('Original Image')
axes[0, 0].axis('off')

# Target Image
axes[0, 1].imshow(cv2.cvtColor(target, cv2.COLOR_BGR2RGB))
axes[0, 1].set_title('Target Image')
axes[0, 1].axis('off')

# Transferred Image
axes[1, 0].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
axes[1, 0].set_title('Transferred Image')
axes[1, 0].axis('off')

# Transferred Optimized Image
axes[1, 1].imshow(cv2.cvtColor(result2, cv2.COLOR_BGR2RGB))
axes[1, 1].set_title('Transferred Optimized Image')
axes[1, 1].axis('off')

plt.tight_layout()
plt.show()