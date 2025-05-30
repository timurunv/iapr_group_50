# IMPORTS FROM LABS #
import sys 
assert (sys.version_info.major == 3) and (sys.version_info.minor == 9)
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from skimage.color import rgb2hsv
from skimage.morphology import closing, opening, disk, remove_small_holes, remove_small_objects, binary_dilation, binary_erosion, binary_closing
from skimage.transform import rotate, resize, warp
from sklearn.metrics.pairwise import euclidean_distances
from skimage.measure import regionprops, find_contours
from skimage.feature import canny
from skimage import io, color
from skimage import segmentation, measure, exposure

import cv2 
import numpy as np

import platform
from tqdm import tqdm
from typing import Optional, Callable
from sklearn.metrics import accuracy_score, f1_score
from sklearn.covariance import LedoitWolf

import os
from skimage.measure import label, regionprops
from skimage.color import rgb2gray

from skimage.util import img_as_ubyte

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def cluster1_1class(cropped_choc) :
    choc_contour = []
    img = np.array(cropped_choc)
    img = np.mean(img, axis=2)
    binary = img > 0
    contours = find_contours(binary, level=0.5)
    if contours:
        choc_contour = np.fliplr(max(contours, key=lambda x: x.shape[0]))
    choc_contour = choc_contour.astype(np.float32)

    if choc_contour.ndim == 2:
        choc_contour = choc_contour.astype(np.int32).reshape(-1, 1, 2)

    mask = np.zeros(cropped_choc.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [choc_contour], -1, color=255, thickness=-1)
    mean_color = cv2.mean(cropped_choc, mask=mask) 
    mean_color_rgb = mean_color[:3][::-1]

    img = np.array(cropped_choc)
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [choc_contour], -1, color=255, thickness=-1)
    hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    mean_hsv = cv2.mean(hsv_img, mask=mask)
    mean_hsv = mean_hsv[:3]

    img = np.array(cropped_choc)
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [choc_contour], -1, color=255, thickness=-1)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    masked = cv2.bitwise_and(gray, gray, mask=mask)
    mean_intensity = np.mean(masked[mask > 0])
    rms_contrast = np.sqrt(np.mean((masked[mask > 0] - mean_intensity) ** 2))

    gray = cv2.cvtColor(cropped_choc, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, threshold1=50, threshold2=150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=50,
                            minLineLength=30, maxLineGap=10)
    line_img = cropped_choc.copy()
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    stripe_count = len(lines) if lines is not None else 0

    perimeter = cv2.arcLength(choc_contour, closed=True)

    area = cv2.contourArea(choc_contour)

    circularity = 4*np.pi * area / perimeter**2

    X_rgb_peri_area_circu = np.array([
        [118.15, 140.1, 164.54, 566.9, 18405.0, 0.7196], #, 27.86, 74.94, 165.25, 35.71, 14],  # class 1
        [63.66, 72.41, 98.92, 549.3, 17558.0, 0.7313], #, 30.93, 99.52, 101.26, 35.41, 0],     # class 2
        [86.12, 97.97, 121.84, 628.8, 20334.0, 0.6461], #, 18.59, 81.83, 122.23, 38.51, 5]
        [72.52, 81.16, 111.48, 584.0, 17327.5, 0.6384]
    ])
    y = np.array([1, 2, 3, 4])

    combined = np.hstack((mean_color_rgb, perimeter, area, circularity)) 

    scaler = StandardScaler()
    weights = np.array([0.8, 0.8, 0.8, 1, 1])
    X_scaled = scaler.fit_transform(X_rgb_peri_area_circu) 
    combined_scaled = scaler.transform([combined]) 

    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_scaled, y)
    new_sample = np.array(combined_scaled)
    predicted_class = knn.predict(new_sample)
    
    if predicted_class == 1 :
        return "Crème Brulée"
    if predicted_class == 2 :
        return "Noir authentique"
    if predicted_class == 3 :
        return "Passion au lait"
    if predicted_class == 4 :
        return "Triangolo"
    
    return "Unable to determine"

def cluster1_2class(chocolate) :
    choc_contour = []
    img = np.array(chocolate)
    img = np.mean(img, axis=2)
    binary = img > 0
    contours = find_contours(binary, level=0.5)
    if contours:
        choc_contour = np.fliplr(max(contours, key=lambda x: x.shape[0]))
    choc_contour = choc_contour.astype(np.float32)

    if choc_contour.ndim == 2:
        choc_contour = choc_contour.astype(np.int32).reshape(-1, 1, 2)

    # RGB
    mask = np.zeros(chocolate.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [choc_contour], -1, color=255, thickness=-1)
    mean_color = cv2.mean(chocolate, mask=mask)
    # Convert BGR to RGB if needed
    mean_color_rgb = mean_color[:3][::-1]

    # HSV
    img = np.array(chocolate)
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [choc_contour], -1, color=255, thickness=-1)
    hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    mean_hsv = cv2.mean(hsv_img, mask=mask)
    mean_hsv = mean_hsv[:3]

    # Texture
    img = np.array(chocolate)
    gray = img_as_ubyte(rgb2gray(img))

    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = np.log1p(np.abs(fshift))
    texture = np.var(magnitude_spectrum)

    # Rectangularity
    rect = cv2.minAreaRect(choc_contour)
    box = cv2.boxPoints(rect)
    box = np.int32(box)

    area_contour = cv2.contourArea(choc_contour)
    area_box = cv2.contourArea(box)
    rectangularity = area_contour / (area_box + 1e-5)

    # Contrast
    img = np.array(chocolate)
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [choc_contour], -1, color=255, thickness=-1)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    masked = cv2.bitwise_and(gray, gray, mask=mask)
    mean_intensity = np.mean(masked[mask > 0])
    rms_contrast = np.sqrt(np.mean((masked[mask > 0] - mean_intensity) ** 2))

    # Stripe detection
    gray = cv2.cvtColor(chocolate, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, threshold1=50, threshold2=150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=50,
                            minLineLength=30, maxLineGap=10)
    line_img = chocolate.copy()
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    stripe_count = len(lines) if lines is not None else 0

    if stripe_count <= 20 and stripe_count >= 8 :
        return "Straciatella"


    # Reference
    X_rgb_hsv_text_rect_cont = np.array([ 
        [116.85, 133.55, 153.15, 40.87, 69.17, 153.67, 0.93, 0.75, 43.92],      # class 1
        [51.69, 62.39, 89.09, 26.53, 118.61, 89.53, 0.7781, 0.7882, 34.08],     # class 2
        [181.48, 198.41, 202.34, 32.86, 29.1, 203.02, 0.8277, 0.7769, 27.39],   # class 3
        [65.061, 89.55, 129.29, 17.28, 127.29, 129.40, 0.7771, 0.7616, 27.15],  # class 4
        #[72.46, 75.92, 95.81, 47.73, 81.07, 96.69, 0.7851, 0.8144, 32.16],     # class 5
        [65.9, 68.21, 87.37, 51.97, 80.02, 87.68, 0.8273, 0.7967, 36.41],       # class 6
        [70.04, 78.85, 111.07, 27.55, 102.75, 111.14, 0.7792, 0.6865, 28.62],   # class 7 
    ])
    
    y = np.array([1, 2, 3, 4, 6, 7]) # ,5

    combined = np.hstack((mean_color_rgb, mean_hsv, texture, rectangularity, rms_contrast))
    scaler = StandardScaler()
    weights = np.array([1, 1, 1, 1, 0.8, 0.8, 0.6, 0.6, 0.3])
    X_scaled = scaler.fit_transform(X_rgb_hsv_text_rect_cont)
    combined_scaled = scaler.transform([combined])

    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_scaled, y)
    new_sample = np.array(combined_scaled)
    predicted_class = knn.predict(new_sample)

    if predicted_class == 1 :
        return "Amandina"
    if predicted_class == 2 :
        return "Arabia"
    if predicted_class == 3 :
        return "Comtesse"
    if predicted_class == 4 :
        return "Noblesse"
    if predicted_class == 6 :
        return "Tentation noir"
    if predicted_class == 7 :
        return "Triangolo"
    
    return "Unable to determine"

def cluster2_class(chocolate) :
    choc_contour = []
    img = np.array(chocolate)
    img = np.mean(img, axis=2)
    binary = img > 0
    contours = find_contours(binary, level=0.5)
    if contours:
        choc_contour = np.fliplr(max(contours, key=lambda x: x.shape[0]))
    choc_contour = choc_contour.astype(np.float32)

    if choc_contour.ndim == 2:
        choc_contour = choc_contour.astype(np.int32).reshape(-1, 1, 2)

    # RGB
    mask = np.zeros(chocolate.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [choc_contour], -1, color=255, thickness=-1)
    mean_color = cv2.mean(chocolate, mask=mask) 
    mean_color_rgb = mean_color[:3][::-1]

    # HSV
    img = np.array(chocolate)
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [choc_contour], -1, color=255, thickness=-1)
    hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    mean_hsv = cv2.mean(hsv_img, mask=mask)
    mean_hsv = mean_hsv[:3]
    
    # Contrast
    img = np.array(chocolate)
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [choc_contour], -1, color=255, thickness=-1)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    masked = cv2.bitwise_and(gray, gray, mask=mask)
    mean_intensity = np.mean(masked[mask > 0])
    rms_contrast = np.sqrt(np.mean((masked[mask > 0] - mean_intensity) ** 2))

    # Reference
    X_rgb_hsv_cont = np.array([ 
        [68.76, 64.52, 71.55, 80.87, 63.44, 77.08, 43.00],      # class 1
        [87.51, 92.55, 115.25, 49.40, 76.30, 116.34, 37.16],     # class 2
        [160.69, 180.14, 187.04, 24.54, 36.21, 187.2, 34.51],   # class 3
    ])
    
    y = np.array([1, 2, 3])

    combined = np.hstack((mean_color_rgb, mean_hsv, rms_contrast))

    scaler = StandardScaler()
    weights = np.array([1, 1, 1, 1, 1, 1, 0.3]) 
    X_scaled = scaler.fit_transform(X_rgb_hsv_cont) * weights
    combined_scaled = scaler.transform([combined]) * weights
    # KNN
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_scaled, y)
    new_sample = np.array(combined_scaled)
    predicted_class = knn.predict(new_sample)

    if predicted_class == 1 :
        return "Jelly Black"
    if predicted_class == 2 :
        return "Jelly Milk"
    if predicted_class == 3 :
        return "Jelly White"
    
    return "Unable to determine"

def choc_classifier(chocolate) :
    choc_class = ''

    img = np.array(chocolate)
    mask = np.any(img > 10, axis=2)
    if np.sum(mask) == 0:
        return 'Ignored: All black'

    avg_color = np.mean(img[mask], axis=0)

    box1 = np.array([38.08589697, 48.86284786, 73.21062837])
    box2 = np.array([50.47642587, 57.81508881, 64.93243105])
    box3 = np.array([41.73481665, 43.04650497, 40.80567227])
    magnet1 = np.array([44.36397149, 49.19480626, 52.67076268])
    magnet2 = np.array([87.38750211, 119.24561898, 110.90246239])
    magnet3 = np.array([52.07047985, 61.89227378, 74.86243147])
    magnet4 = np.array([44.84981227, 152.63533963, 109.03447491])
    magnet5 = np.array([44.58164251, 55.60676329, 82.78636608])
    magnet6 = np.array([41.67514188, 60.42372304, 99.57775255])
    
    threshold = 10
    if (np.linalg.norm(avg_color - box1) < threshold or
        np.linalg.norm(avg_color - box2) < threshold or
        np.linalg.norm(avg_color - box3) < threshold or
        np.linalg.norm(avg_color - magnet1) < threshold or
        np.linalg.norm(avg_color - magnet2) < threshold or
        np.linalg.norm(avg_color - magnet3) < threshold or
        np.linalg.norm(avg_color - magnet4) < threshold or
        np.linalg.norm(avg_color - magnet5) < threshold or
        np.linalg.norm(avg_color - magnet6) < threshold):
        return 'Ignored: Color too close to black boxes or magnets'
    
    choc_contour = []
    img = np.array(chocolate)
    img = np.mean(img, axis=2)
    binary = img > 0
    contours = find_contours(binary, level=0.5)
    if contours:
        choc_contour = np.fliplr(max(contours, key=lambda x: x.shape[0]))

    choc_contour = choc_contour.astype(np.float32)
    ratio = cv2.contourArea(choc_contour)/cv2.arcLength(choc_contour, closed=True)
    compacity = cv2.contourArea(choc_contour)**2/cv2.arcLength(choc_contour, closed=True)
    cluster = 0

    if compacity > 310000 : #350000
        cluster = 1
    else :
        cluster = 2

    if ratio > 33.4 and cluster == 1:
        cluster = 11
    if ratio <= 33.4 and cluster == 1:
        cluster = 12
    
    if cluster == 11 :
        choc_class = cluster1_1class(chocolate)

    if cluster == 12 :
        choc_class = cluster1_2class(chocolate)

    if cluster == 2 :
        choc_class = cluster2_class(chocolate)

    
    return choc_class

def classification(segmented_image) :
    # Jelly White, Jelly Milk, Jelly Black, Amandina, Crème brulée, Triangolo, Tentation noir, Comtesse, Noblesse, Noir authentique, Passion au lait, Arabia, Stracciatella
    chocolate_count = [0,0,0,0,0,0,0,0,0,0,0,0,0]

    masked_img = np.mean(segmented_image, axis=2) > 0
    masked_img = remove_small_objects(masked_img, min_size=5000)
    
    labeled_mask = label(masked_img)
    regions = regionprops(labeled_mask)

    if not regions:
        return chocolate_count

    for region in regions:
        minr, minc, maxr, maxc = region.bbox
        isolated_mask = labeled_mask == region.label

        img_crop = segmented_image[minr-1:maxr+1, minc-1:maxc+1]
        mask_crop = isolated_mask[minr-1:maxr+1, minc-1:maxc+1]

        isolated_img = np.zeros_like(img_crop)
        isolated_img[mask_crop] = img_crop[mask_crop]

        footprint = disk(15)
        smoothed_mask = opening(mask_crop, footprint=footprint)
        isolated_img = np.zeros_like(img_crop)
        for c in range(3):
            isolated_img[..., c] = img_crop[..., c] * smoothed_mask

        choc_contour = []
        binary = isolated_mask > 0

        contours = find_contours(binary, level=0.5)
        if contours:
            choc_contour = np.fliplr(max(contours, key=lambda x: x.shape[0]))

        choc_contour = choc_contour.astype(np.float32)
        area = cv2.contourArea(choc_contour)

        if (isolated_img.shape[0] < 2 or isolated_img.shape[1] < 2) : 
            continue
    
        if (isolated_img.shape[0] > 350 or isolated_img.shape[1] > 350) : 
            continue

        if isolated_img.shape[0] < 95 and isolated_img.shape[1] < 95 :
            continue

        if (isolated_img.shape[0] > 230 or isolated_img.shape[1] > 230 or area > 20000) : 
            region_crop = isolated_mask[minr:maxr, minc:maxc].astype(np.uint8)
            chocolate_crop = segmented_image[minr:maxr, minc:maxc]

            # --- WATERSHED START ---
            dist = cv2.distanceTransform(region_crop * 255, cv2.DIST_L2, 5)
            _, sure_fg = cv2.threshold(dist, 0.9 * dist.max(), 255, 0)
            sure_fg = np.uint8(sure_fg)
            unknown = cv2.subtract(region_crop * 255, sure_fg)

            _, markers = cv2.connectedComponents(sure_fg)
            markers = markers + 1
            markers[unknown == 255] = 0

            chocolate_crop_color = chocolate_crop.copy()
            markers = cv2.watershed(chocolate_crop_color, markers)

            isolated_imgs = []
            for label_val in np.unique(markers):
                if label_val <= 1:
                    continue

                mask_ws = (markers == label_val).astype(np.uint8)
                if np.sum(mask_ws) < 10000:
                    continue

                masked_choco = np.zeros_like(chocolate_crop_color)
                for c in range(3):
                    masked_choco[..., c] = chocolate_crop_color[..., c] * mask_ws

                isolated_imgs.append(masked_choco)
            # --- WATERSHED END ---
            for isolated in isolated_imgs:
                choc_class = choc_classifier(isolated)

                if choc_class == "Jelly White":
                    chocolate_count[0] += 1
                if choc_class == "Jelly Milk":
                    chocolate_count[1] += 1
                if choc_class == "Jelly Black":
                    chocolate_count[2] += 1
                if choc_class == "Amandina":
                    chocolate_count[3] += 1
                if choc_class == "Crème Brulée":
                    chocolate_count[4] += 1
                if choc_class == "Triangolo":
                    chocolate_count[5] += 1
                if choc_class == "Tentation noir":
                    chocolate_count[6] += 1
                if choc_class == "Comtesse":
                    chocolate_count[7] += 1
                if choc_class == "Noblesse":
                    chocolate_count[8] += 1
                if choc_class == "Noir authentique":
                    chocolate_count[9] += 1
                if choc_class == "Passion au lait":
                    chocolate_count[10] += 1
                if choc_class == "Arabia":
                    chocolate_count[11] += 1
                if choc_class == "Straciatella":
                    chocolate_count[12] += 1

            continue

        # Classify the chocolate
        choc_class = choc_classifier(isolated_img)

        if choc_class == "Jelly White":
            chocolate_count[0] += 1
        if choc_class == "Jelly Milk":
            chocolate_count[1] += 1
        if choc_class == "Jelly Black":
            chocolate_count[2] += 1
        if choc_class == "Amandina":
            chocolate_count[3] += 1
        if choc_class == "Crème Brulée":
            chocolate_count[4] += 1
        if choc_class == "Triangolo":
            chocolate_count[5] += 1
        if choc_class == "Tentation noir":
            chocolate_count[6] += 1
        if choc_class == "Comtesse":
            chocolate_count[7] += 1
        if choc_class == "Noblesse":
            chocolate_count[8] += 1
        if choc_class == "Noir authentique":
            chocolate_count[9] += 1
        if choc_class == "Passion au lait":
            chocolate_count[10] += 1
        if choc_class == "Arabia":
            chocolate_count[11] += 1
        if choc_class == "Straciatella":
            chocolate_count[12] += 1


    

    return chocolate_count