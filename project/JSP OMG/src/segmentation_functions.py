import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from src.chocolates import *
import cv2 

def segmentation_clean_background(img):
    # --- 1. Load and Resize Image ---
    img = cv2.resize(img, (1600,1067))

    # --- 2. Convert to Grayscale ---
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # --- 3. Blur to Reduce Noise ---
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # --- 4. Canny Edge Detection ---
    edges = cv2.Canny(blurred, threshold1=20, threshold2=50)

    # --- 5. Dilate to Connect Broken Edges ---
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated = cv2.dilate(edges, kernel, iterations=1)

    # --- 6. Morphological Closing to Fill Gaps ---
    closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel, iterations=2)

    # --- 7. Remove Small Objects by Area Filtering ---
    # Find all contours
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # --- 7. Replace small contour filtering with ellipse fitting ---
    mask = np.zeros_like(closed)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        # Skip tiny noise
        if area < 2000:
            if len(cnt) >= 5:
                ellipse = cv2.fitEllipse(cnt)
                (x, y), (MA, ma), angle = ellipse
                if MA > 5 and ma > 5:
                    cv2.ellipse(mask, ellipse, 255, -1)
        else:
            # If not enough points, fall back to filled contour
            cv2.drawContours(mask, [cnt], -1, 255, thickness=cv2.FILLED)

    # --- 8. Apply Mask to Original Image ---
    segmented_objects = cv2.bitwise_and(img, img, mask=mask)

    return segmented_objects


def segmentation_background_stylo(img):
    # --- 1. Load and Resize Image ---
    img = cv2.resize(img, (1600,1067))

    # --- 2. Convert to HSV ---
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV_FULL)
    value = hsv[:, :, 1]


    # --- 3. Thresholding ---
    thresh = 100
    value_thresh = np.zeros_like(value)
    value_thresh[value > thresh] = value[value > thresh]

    # --- 4. Canny Edge Detection ---
    edges = cv2.Canny(value_thresh, 200, 250)

    # --- 5. Dilate to Connect Broken Edges ---
    radius = 2
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1))
    dilated = cv2.dilate(edges, kernel, iterations=1)

    # --- 7. Remove Small Objects by Area Filtering ---
    # Find all contours
    contours, hierarchy = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # --- 7. Replace small contour filtering with ellipse fitting ---
    mask = np.zeros_like(dilated)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 200000:
            cv2.drawContours(mask, [cnt], -1, 255, thickness=cv2.FILLED)

    # Separate objects
    kernel = np.ones((40, 40), np.uint8) #TODO à voir si on peut réduire la taille du kernel
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # --- 8. Apply Mask to Original Image ---
    segmented_objects = cv2.bitwise_and(img, img, mask=mask)

    return segmented_objects

#Pour background_flowers_simples: (1.0, 1.0, 0.0) avec thresh = 80

def segmentation_weighted(img, weights=(1.0, 1.0, 1.0)):
    # --- 1. Load and Resize Image ---
    img = cv2.resize(img, (1600,1067))

    # --- 2. Convert to Grayscale ---
    masked = chocolate_masking_weighted(img, 20, weights)
    
    value = cv2.cvtColor(masked, cv2.COLOR_BGR2HSV)[:, :, 1]
    #plt.imshow(value, cmap='gray')

    thresh = 80 #80 best so far
    value_thresh = (value > thresh).astype(np.uint8) * 255
    #plt.imshow(value_thresh, cmap='gray')

    # # --- 6. Erode to Remove Small Noise ---
    radius = 2
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1))
    eroded = cv2.erode(value_thresh, kernel, iterations=1)
    #plt.imshow(eroded)

    # --- 6. Morphological Closing to Fill Gaps ---
    # radius = 2
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1))
    # closed = cv2.morphologyEx(eroded, cv2.MORPH_CLOSE, kernel, iterations=2)
    # plt.imshow(closed)

    # Closing
    # radius = 4
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1))
    # closed = cv2.morphologyEx(eroded, cv2.MORPH_CLOSE, kernel, iterations=2)
    # plt.imshow(closed, cmap='gray')

    # --- 5. Dilate to Connect Broken Edges ---
    radius = 3
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1))
    dilated = cv2.dilate(eroded, kernel, iterations=1)
    #plt.imshow(dilated, cmap='gray')

    # --- 4. Canny Edge Detection ---
    edges = cv2.Canny(dilated, 200, 250) #50, 150
    #plt.imshow(edges, cmap='gray')

    # --- 5. Dilate to Connect Broken Edges ---
    # radius = 2
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1))
    # dilated = cv2.dilate(edges, kernel, iterations=1)
    #plt.imshow(dilated, cmap='gray')

    # --- 6. Morphological Closing to Fill Gaps ---
    radius = 4
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    #plt.imshow(closed, cmap='gray')

    # --- 7. Remove Small Objects by Area Filtering ---
    # Find all contours
    contours, hierarchy = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


    #plot contours
    # cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
    # cv2.imshow("Contours", img)
    # plt.imshow(img)

    # --- 7. Replace small contour filtering with ellipse fitting ---
    mask = np.zeros_like(closed)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 200000:
            cv2.drawContours(mask, [cnt], -1, 255, thickness=cv2.FILLED)
            if area < 2000:
                if len(cnt) >= 5:
                    ellipse = cv2.fitEllipse(cnt)
                    (x, y), (MA, ma), angle = ellipse
                    if MA > 5 and ma > 5:
                        cv2.ellipse(mask, ellipse, 255, -1)

    # Separate objects
    kernel = np.ones((40, 40), np.uint8) #TODO à voir si on peut réduire la taille du kernel
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # --- 8. Apply Mask to Original Image ---
    segmented_objects = cv2.bitwise_and(img, img, mask=mask)

    return segmented_objects


def segmentation_orange_book(img):
    # --- 1. Load and Resize Image ---
    img = cv2.resize(img, (1600,1067))

    # --- 2. Convert to HSV ---
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV_FULL)
    value = hsv[:, :, 2]

    # Gaussian blur to reduce noise
    value = cv2.GaussianBlur(value, (5, 5), 0) #(5,5)
    #plt.imshow(value, cmap='gray')

    # --- 4. Canny Edge Detection ---
    edges = cv2.Canny(value, 50, 100) #(50, 100)
    #plt.imshow(edges, cmap='gray')

    # --- 5. Dilate to Connect Broken Edges ---
    radius = 2 #2
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1))
    dilated = cv2.dilate(edges, kernel, iterations=1)
    #plt.imshow(dilated, cmap='gray')

    # --- 6. Remove small objects ---

    # Label connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(dilated, connectivity=8)

    # Set a minimum size threshold (in pixels)
    min_size = 2000  # adjust as needed

    # Create output image
    output = np.zeros(dilated.shape, dtype=np.uint8)

    for i in range(1, num_labels):  # skip label 0 (background)
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            output[labels == i] = 255

    #plt.imshow(output, cmap='gray')

    contours, hierarchy = cv2.findContours(output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # --- 7. Replace small contour filtering with ellipse fitting ---
    mask = np.zeros_like(dilated)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 30000:  # adjust the area threshold as needed
            cv2.drawContours(mask, [cnt], -1, 255, thickness=cv2.FILLED)
            
    #plt.imshow(mask, cmap='gray')

    # contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # for cnt in contours:
    #     area = cv2.contourArea(cnt)
    #     if area < 30000:  # adjust the area threshold as needed
    #         cv2.drawContours(mask, [cnt], -1, 255, thickness=cv2.FILLED)

    #plt.imshow(mask, cmap='gray')

    # Close objects
    radius = 4
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    # # Dilate to connect broken edges
    # radius = 2
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1))
    # mask = cv2.dilate(mask, kernel, iterations=1)

    #plt.imshow(mask, cmap='gray')

    # Separate objects
    radius = 40
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    #plt.imshow(mask, cmap='gray')

    # --- 8. Apply Mask to Original Image ---
    segmented_objects = cv2.bitwise_and(img, img, mask=mask)

    return segmented_objects