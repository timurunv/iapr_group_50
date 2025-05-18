import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage.measure import label, regionprops

def region_growing(original_image,segmented_image):    

    # Convert to grayscale
    original_image = cv2.resize(original_image, (1600,1067))
    gray = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2GRAY)

    # Threshold to binary
    _, binary = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    binary = binary > 0  # Convert to boolean

    # Label the binary image
    labeled = label(binary)

    # Extract region properties
    props = regionprops(labeled)

    # Plot original image with bounding boxes
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(segmented_image)

    padding = 20  # pixels  
    cropped_images = []
    for prop in props:
        minr, minc, maxr, maxc = prop.bbox

        # Expand the bounding box with padding, while keeping bounds within image size
        minr = max(minr - padding, 0)
        minc = max(minc - padding, 0)
        maxr = min(maxr + padding, segmented_image.shape[0])
        maxc = min(maxc + padding, segmented_image.shape[1])
        crop = original_image[minr:maxr, minc:maxc]
        cropped_images.append(crop)

    return(cropped_images)