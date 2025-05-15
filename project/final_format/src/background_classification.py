import os
import numpy as np
import cv2
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def compute_average_hsv_from_rgb(image_rgb):
    """Compute average HSV color from an RGB image (converted internally)."""
    image_hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    avg_hsv = np.mean(image_hsv, axis=(0, 1))
    return avg_hsv

def load_features_from_list(images_rgb_list):
    """Compute average HSV features from a list of RGB images."""
    features = []

    for i, image_rgb in enumerate(images_rgb_list):
        if image_rgb is None or not isinstance(image_rgb, np.ndarray):
            print(f"Invalid image at index {i}")
            continue

        avg_hsv = compute_average_hsv_from_rgb(image_rgb)
        features.append(avg_hsv)

    return np.array(features)

def cluster_images(features, n_clusters=6):
    """Run KMeans clustering on HSV feature vectors."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(features)
    return labels, kmeans.cluster_centers_

def show_cluster_examples(clustered_images, cluster_id, max_images=5):
    """Show example RGB images from a specific cluster."""
    images = clustered_images[cluster_id][:max_images]

    plt.figure(figsize=(15, 3))
    for i, img in enumerate(images):
        plt.subplot(1, len(images), i + 1)
        plt.imshow(img)
        plt.axis('off')
    plt.suptitle(f"Images from Cluster {cluster_id}")
    plt.show()

