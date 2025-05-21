
from src.project_utils import *
import src.segmentation_functions as sf
from src.classification import *
from src.background_classification import *
from src.region_growing import *
import pandas as pd
import matplotlib.pyplot as plt
from skimage.morphology import closing, disk,dilation,remove_small_objects,erosion, remove_small_holes
import joblib
import argparse

def main():
    parser = argparse.ArgumentParser(description="Run image segmentation and classification pipeline.")
    parser.add_argument("test_path_ref", type=str, help="Path to the test images folder")
    args = parser.parse_args()
    test_images, test_images_ref = load_images(args.test_path_ref)

    df = pd.read_csv('src/sample_submission.csv')
    kmeans = joblib.load("src/kmeans_background_model.pkl")

    
    for i,image in enumerate(test_images):
        features = load_features(image)
        label = kmeans.predict(features)
        if label == 0:
            segmented_image = sf.segmentation_clean_background(image)
            row = np.array(classification(segmented_image))
            if np.sum(row) <= 4:
                gray_segmented = cv2.cvtColor(segmented_image,cv2.COLOR_RGB2GRAY)
                gray_segmented = (gray_segmented > 10) & (gray_segmented < 100)
                removed = remove_small_objects(gray_segmented,2000)
                closed = closing(removed,disk(9))
                eroded = erosion(closed,disk(7))
                dilated = dilation(eroded,disk(7)).astype(np.uint8) * 255
                segmented_image = cv2.bitwise_and(segmented_image, segmented_image, mask=dilated)
                row += np.array(classification(segmented_image))
                if np.sum(row) <= 3:
                    segmented_image = sf.segmentation_sachet(image)
                    row += np.array(classification(segmented_image))
        elif label == 1:
            segmented_image = sf.segmentation_clean_background(image)
            row = classification(segmented_image)
            if np.sum(row) <= 3:
                segmented_image = sf.segmentation_orange_book(segmented_image)
                row = classification(segmented_image)
        elif label == 2:
            segmented_image = sf.segmentation_clean_background(image)
            row = classification(segmented_image)
        elif label == 3:
            segmented_image = sf.segmentation_clean_background(image)
            image_hsv = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2HSV)
            lower_blue = np.array([75, 20, 20])  
            upper_blue = np.array([145, 255, 255])
            blue_mask = cv2.inRange(image_hsv, lower_blue, upper_blue)
            non_blue_mask = cv2.bitwise_not(blue_mask)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
            non_blue_mask_closed = cv2.morphologyEx(non_blue_mask, cv2.MORPH_CLOSE, kernel)
            segmented_image = cv2.bitwise_and(segmented_image, segmented_image, mask=non_blue_mask_closed)
            gray_segmented = cv2.cvtColor(segmented_image,cv2.COLOR_RGB2GRAY)
            gray_segmented = (gray_segmented > 10) & (gray_segmented < 100)
            removed = remove_small_objects(gray_segmented,2000)
            closed = closing(removed,disk(9))
            eroded = erosion(closed,disk(5))
            removed = remove_small_objects(eroded,6000)
            filled = remove_small_holes(removed,5000)
            closed = closing(filled,disk(11))
            dilated = dilation(closed,disk(9)).astype(np.uint8) * 255
            segmented_image = cv2.bitwise_and(segmented_image, segmented_image, mask=dilated)
            row = classification(segmented_image)
        elif label == 4:
            segmented_image = sf.segmentation_clean_background(image)
            row = classification(segmented_image)
            if np.sum(row) <= 3:
                segmented_image = sf.segmentation_sac(segmented_image)
                segmented_image = mask_outside_regions(image,segmented_image,padding = 10,threshold=130,disk_size=9)
                row = classification(segmented_image)
        elif label == 5:
            segmented_image = sf.segmentation_clean_background(image)
            row = np.array(classification(segmented_image))
            if np.sum(row) <= 3:
                gray_segmented = cv2.cvtColor(segmented_image,cv2.COLOR_RGB2GRAY)
                gray_segmented = (gray_segmented > 10) & (gray_segmented < 100)
                removed = remove_small_objects(gray_segmented,2000)
                closed = closing(removed,disk(9))
                eroded = erosion(closed,disk(7))
                dilated = dilation(eroded,disk(7)).astype(np.uint8) * 255
                segmented_image = cv2.bitwise_and(segmented_image, segmented_image, mask=dilated)
                row += np.array(classification(segmented_image))

        image_ref = int(test_images_ref[i])
        row = np.insert(row,0,image_ref)
        df.loc[df.id == image_ref,:] = row

    df.set_index('id').to_csv('submission_final.csv')

if __name__ == "__main__":
    main()