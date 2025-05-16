import numpy as np
import pandas as pd
import cv2

#Constants file with the mean hsv values for each chocolate
AMANDINA_COLOR = (np.float64(58.40766694146743), np.float64(66.15457543281121), np.float64(152.40210222588624))
ARABIA_COLOR = (np.float64(38.59955068800899), np.float64(112.08010390339793), np.float64(94.69931199101376))
COMTESSE_COLOR = (np.float64(52.0501003440367), np.float64(27.74534116972477), np.float64(201.78411697247705))
CREME_BRULEE_COLOR = (np.float64(40.16990683397059), np.float64(76.00694706230814), np.float64(167.60035543109484))
JELLY_BLACK_COLOR = (np.float64(117.30848146267141), np.float64(64.72910106653123), np.float64(78.65027932960894))
JELLY_BLACK_COLOR2 = (np.float64(21), np.float64(41), np.float64(13))
JELLY_BLACK_COLOR3 = (8.08, 86.12, 47.50)       
JELLY_BLACK_COLOR4 = (111.14, 58.45, 156.19)  
JELLY_BLACK_COLOR5 = (140.45, 40.01, 75.53)
JELLY_MILK_COLOR = (np.float64(71.73325266747332), np.float64(77.95523044769553), np.float64(118.86547134528655))
JELLY_WHITE_COLOR = (np.float64(35.59790973871734), np.float64(36.9237054631829), np.float64(190.95562945368172))
NOBLESSE_COLOR = (np.float64(29.248012816562944), np.float64(121.59116396574034), np.float64(131.97541438166246))
NOIR_AUTHENTIQUE_COLOR = (np.float64(44.606715806715805), np.float64(100.98252798252798), np.float64(102.75080535080535))
PASSION_AU_LAIT_COLOR = (np.float64(26.783860265417644), np.float64(83.06542740046838), np.float64(124.0754781420765))
STRACCIATELA_COLOR = (np.float64(67.05495676149893), np.float64(76.45901742072941), np.float64(101.75084597067301))
TENTATION_NOIR_COLOR = (np.float64(75.19297608234938), np.float64(81.41132303966091), np.float64(89.20629730547986))
TRIANGOLO_COLOR = (np.float64(41.55256381549485), np.float64(96.41519256605463), np.float64(114.78549037169726))

chocolate_mean_colors = [
    AMANDINA_COLOR,
    ARABIA_COLOR,
    COMTESSE_COLOR,
    CREME_BRULEE_COLOR,
    JELLY_BLACK_COLOR,
    JELLY_BLACK_COLOR2,
    JELLY_BLACK_COLOR3,
    JELLY_BLACK_COLOR4,
    JELLY_BLACK_COLOR5,
    JELLY_MILK_COLOR,
    JELLY_WHITE_COLOR,
    NOBLESSE_COLOR,
    NOIR_AUTHENTIQUE_COLOR,
    PASSION_AU_LAIT_COLOR,
    STRACCIATELA_COLOR,
    TENTATION_NOIR_COLOR,
    TRIANGOLO_COLOR
]

chocolate_colors = pd.read_csv('C:/Users/timur/OneDrive/Documents/GitHub/iapr_group_50/project/hsv_samples/concatenated.csv')[['H', 'S', 'V']]

def chocolate_masking_weighted(img, threshold=30, weights=(1.0, 1.0, 0.0)):
    """
    Crée un masque avec une distance HSV pondérée.

    Args:
        img (np.ndarray): Image RGB.
        threshold (float): Seuil sur la distance pondérée.
        weights (tuple): Poids (H, S, V) pour la distance.

    Returns:
        np.ndarray: Masque binaire (uint8).
    """
    hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV_FULL).astype(np.float32)
    
    mask_total = np.zeros(hsv_img.shape[:2], dtype=np.uint8)
    
    # Étendre les poids pour faire un broadcast
    w = np.array(weights).reshape((1, 1, 3))

    for index, row in chocolate_colors.iloc[::5000].iterrows():
        diff = hsv_img - np.array(row, dtype=np.float32)  # shape (H, W, 3)
        weighted_diff = diff ** 2 * w
        dist = np.sqrt(np.sum(weighted_diff, axis=2))
        mask = (dist < threshold).astype(np.uint8) * 255
        mask_total = cv2.bitwise_or(mask_total, mask)

    result = cv2.bitwise_and(img, img, mask=mask_total)
    
    return result

def chocolate_mean_masking_weighted(img, threshold=30, weights=(1.0, 1.0, 0.0)):
    """
    Crée un masque avec une distance HSV pondérée.

    Args:
        img (np.ndarray): Image RGB.
        threshold (float): Seuil sur la distance pondérée.
        weights (tuple): Poids (H, S, V) pour la distance.

    Returns:
        np.ndarray: Masque binaire (uint8).
    """
    hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV_FULL).astype(np.float32)
    
    mask_total = np.zeros(hsv_img.shape[:2], dtype=np.uint8)
    
    # Étendre les poids pour faire un broadcast
    w = np.array(weights).reshape((1, 1, 3))

    for row in chocolate_mean_colors:
        diff = hsv_img - np.array(row, dtype=np.float32)  # shape (H, W, 3)
        weighted_diff = diff ** 2 * w
        dist = np.sqrt(np.sum(weighted_diff, axis=2))
        mask = (dist < threshold).astype(np.uint8) * 255
        mask_total = cv2.bitwise_or(mask_total, mask)

    result = cv2.bitwise_and(img, img, mask=mask_total)
    
    return result

def chocolate_masking(img,threshold=50):
    """
    Apply a mask to the image to isolate chocolate colors.
    Args:
        img (numpy.ndarray): Input image in RGB format.
        threshold (int): Threshold (max distance to chocolate colors) for color matching.
    Returns:
        numpy.ndarray: Masked image (HSV) with chocolate colors.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV_FULL)
    hs = hsv[:, :, :2]
    mask_total = np.zeros(hsv.shape[:2], dtype=np.uint8)

    for index, row in chocolate_colors.iloc[::1000].iterrows():
        target_hs = np.array(row, dtype=np.float32)  # Only H and S
        diff = hs - target_hs
        distance = np.linalg.norm(diff, axis=2)

        mask = (distance < threshold).astype(np.uint8) * 255
        mask_total = cv2.bitwise_or(mask_total, mask)

    result = cv2.bitwise_and(img, img, mask=mask_total)
    
    return result
    

    #ESSAYER LA MEME CHOSE AVEC RGB ET NON HSV