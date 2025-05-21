import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from src.chocolates import *
import cv2 

def segmentation_clean_background(img):
    img = cv2.resize(img, (1600,1067))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, threshold1=20, threshold2=50)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated = cv2.dilate(edges, kernel, iterations=1)
    closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(closed)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 2000:
            if len(cnt) >= 5:
                ellipse = cv2.fitEllipse(cnt)
                (x, y), (MA, ma), angle = ellipse
                if MA > 5 and ma > 5:
                    cv2.ellipse(mask, ellipse, 255, -1)
        else:
            cv2.drawContours(mask, [cnt], -1, 255, thickness=cv2.FILLED)

    segmented_objects = cv2.bitwise_and(img, img, mask=mask)

    return segmented_objects

def segmentation_orange_book(img):
    img = cv2.resize(img, (1600,1067))
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV_FULL)
    value = hsv[:, :, 2]
    value = cv2.GaussianBlur(value, (5, 5), 0) 
    edges = cv2.Canny(value, 50, 100) 

    radius = 2 
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1))
    dilated = cv2.dilate(edges, kernel, iterations=1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(dilated, connectivity=8)
    min_size = 2000
    output = np.zeros(dilated.shape, dtype=np.uint8)

    for i in range(1, num_labels): 
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            output[labels == i] = 255

    contours, hierarchy = cv2.findContours(output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    mask = np.zeros_like(dilated)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 30000: 
            cv2.drawContours(mask, [cnt], -1, 255, thickness=cv2.FILLED)
            
    radius = 4
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    radius = 40
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    segmented_objects = cv2.bitwise_and(img, img, mask=mask)

    return segmented_objects

def segmentation_sac(img):
    img = cv2.resize(img, (1600,1067))

    masked = chocolate_masking_weighted(img, 5, (1, 1, 0))

    hsv = cv2.cvtColor(masked, cv2.COLOR_RGB2HSV_FULL)
    value = hsv[:, :, 1]
    thresh = 75
    value_thresh = np.zeros_like(value)
    value_thresh[value > thresh] = value[value > thresh]

    edges = cv2.Canny(value_thresh, 200, 250)

    radius = 5
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1))
    dilated = cv2.dilate(edges, kernel, iterations=1)

    contours, hierarchy = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(dilated)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 200000:
            cv2.drawContours(mask, [cnt], -1, 255, thickness=cv2.FILLED)

    radius = 40
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    segmented_objects = cv2.bitwise_and(img, img, mask=mask)

    return segmented_objects

def segmentation_sachet(img):
    img = cv2.resize(img, (1600,1067))

    masked = chocolate_masking_weighted(img, 5, (1.0, 1.0, 0.0))

    hsv = cv2.cvtColor(masked, cv2.COLOR_RGB2HSV_FULL)
    value = hsv[:, :, 1]

    thresh = 65
    value_thresh = np.zeros_like(value)
    value_thresh[value > thresh] = value[value > thresh]

    edges = cv2.Canny(value_thresh, 200, 250)
    radius = 3
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1))
    dilated = cv2.dilate(edges, kernel, iterations=1)

    contours, hierarchy = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    mask = np.zeros_like(dilated)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 200000:
            cv2.drawContours(mask, [cnt], -1, 255, thickness=cv2.FILLED)

    radius = 40
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    segmented_objects = cv2.bitwise_and(img, img, mask=mask)

    return segmented_objects

