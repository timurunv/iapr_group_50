import numpy as np
import pandas as pd
import cv2

chocolate_colors = pd.read_csv('src/concatenated.csv')[['H', 'S', 'V']]

def chocolate_masking_weighted(img, threshold=30, weights=(1.0, 1.0, 0.0)):

    hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV_FULL).astype(np.float32)
    
    mask_total = np.zeros(hsv_img.shape[:2], dtype=np.uint8)
    
    w = np.array(weights).reshape((1, 1, 3))

    for index, row in chocolate_colors.iloc[::5000].iterrows():
        diff = hsv_img - np.array(row, dtype=np.float32) 
        weighted_diff = diff ** 2 * w
        dist = np.sqrt(np.sum(weighted_diff, axis=2))
        mask = (dist < threshold).astype(np.uint8) * 255
        mask_total = cv2.bitwise_or(mask_total, mask)

    result = cv2.bitwise_and(img, img, mask=mask_total)
    
    return result

def chocolate_masking(img,threshold=50):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV_FULL)
    hs = hsv[:, :, :2]
    mask_total = np.zeros(hsv.shape[:2], dtype=np.uint8)

    for index, row in chocolate_colors.iloc[::1000].iterrows():
        target_hs = np.array(row, dtype=np.float32) 
        diff = hs - target_hs
        distance = np.linalg.norm(diff, axis=2)

        mask = (distance < threshold).astype(np.uint8) * 255
        mask_total = cv2.bitwise_or(mask_total, mask)

    result = cv2.bitwise_and(img, img, mask=mask_total)
    
    return result