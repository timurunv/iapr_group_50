import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage.measure import label, regionprops

def region_growing(segmented_image):    

    # Convert to grayscale
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

    padding = 5  # pixels

    for prop in props:
        minr, minc, maxr, maxc = prop.bbox

        # Expand the bounding box with padding, while keeping bounds within image size
        minr = max(minr - padding, 0)
        minc = max(minc - padding, 0)
        maxr = min(maxr + padding, segmented_image.shape[0])
        maxc = min(maxc + padding, segmented_image.shape[1])
        # Draw rectangle
        rect = patches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)

    ax.set_axis_off()
    plt.tight_layout()
    plt.show()