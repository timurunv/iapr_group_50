import os
import numpy as np
import cv2
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def compute_average_hsv_from_rgb(image_rgb):
    image_hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    avg_hsv = np.mean(image_hsv, axis=(0, 1))
    return avg_hsv

def load_features(img_rgb):
    features = []
    avg_hsv = compute_average_hsv_from_rgb(img_rgb)
    features.append(avg_hsv)
    return np.array(features)


