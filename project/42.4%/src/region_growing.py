import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage.measure import label, regionprops
from skimage.morphology import closing,disk, dilation

def mask_outside_regions(original_image, segmented_image, padding,threshold,disk_size):
    original_image = cv2.resize(original_image, (1600, 1067))
    
    gray = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    binary = binary > 0
    
    labeled = label(binary)
    props = regionprops(labeled)

    mask = np.zeros(original_image.shape[:2], dtype=np.uint8)
       
    for prop in props:
        minr, minc, maxr, maxc = prop.bbox
        
        minr = max(minr - padding, 0)
        minc = max(minc - padding, 0)
        maxr = min(maxr + padding, original_image.shape[0])
        maxc = min(maxc + padding, original_image.shape[1])
        
        mask[minr:maxr, minc:maxc] = 255
    masked_image = cv2.bitwise_and(original_image, original_image, mask=mask)
    new_img_gray = cv2.cvtColor(masked_image, cv2.COLOR_RGB2GRAY)
    
    refined_mask = (new_img_gray < threshold).astype(np.uint8) * 255 
    closed = closing(refined_mask,disk(disk_size))
    masked_image = cv2.bitwise_and(masked_image, masked_image, mask=closed)

    
    return masked_image
