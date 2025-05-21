import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage.measure import label, regionprops
from skimage.morphology import closing,disk, dilation

def mask_outside_regions(original_image, segmented_image, padding,threshold,disk_size):
    # Resize original image
    original_image = cv2.resize(original_image, (1600, 1067))
    
    # Convert segmented image to grayscale and threshold
    gray = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    binary = binary > 0  # Convert to boolean
    
    # Label connected components
    labeled = label(binary)
    props = regionprops(labeled)

    # Create a black mask same size as original image
    mask = np.zeros(original_image.shape[:2], dtype=np.uint8)  # single channel mask
       
    # For each region, set mask pixels inside padded bounding box to 255 (white)
    for prop in props:
        minr, minc, maxr, maxc = prop.bbox
        
        # Expand bounding box with padding and clamp to image boundaries
        minr = max(minr - padding, 0)
        minc = max(minc - padding, 0)
        maxr = min(maxr + padding, original_image.shape[0])
        maxc = min(maxc + padding, original_image.shape[1])
        
        mask[minr:maxr, minc:maxc] = 255  # white region in mask
    # Apply mask to original image: keep pixels inside boxes, zero elsewhere
    # If original image has 3 channels, apply mask to all channels
    masked_image = cv2.bitwise_and(original_image, original_image, mask=mask)
    new_img_gray = cv2.cvtColor(masked_image, cv2.COLOR_RGB2GRAY)
    

    # # Create a refined mask where pixel intensity is below 150
    refined_mask = (new_img_gray < threshold).astype(np.uint8) * 255  # convert boolean to uint8
    closed = closing(refined_mask,disk(disk_size))

    # # Apply the refined mask to the already masked image
    masked_image = cv2.bitwise_and(masked_image, masked_image, mask=closed)
    
    # Optional: display the masked image
    plt.imshow(masked_image)
    plt.axis('off')
    plt.show()
    
    return masked_image


def keep_only_brown(image):
    # Define RGB range for brown color
    lower_brown = np.array([60, 30, 0])   # Lower bound for R, G, B
    upper_brown = np.array([160, 110, 60])  # Upper bound for R, G, B

    # Create a mask for brown color
    mask = cv2.inRange(image, lower_brown, upper_brown)

    # Bitwise-AND mask and original image
    brown_only = cv2.bitwise_and(image, image, mask=mask)

    return brown_only