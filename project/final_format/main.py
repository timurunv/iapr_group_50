#Imports
from src.project_utils import *


def main():
    #CHANGE THE PATH REF 
    reference_path_ref = '/Users/louiscuendet/Documents/EPFL NEURO-X /Image Analysis and Pattern Recognition/dataset_project_iapr2025/references'
    train_path_ref = '/Users/louiscuendet/Documents/EPFL NEURO-X /Image Analysis and Pattern Recognition/dataset_project_iapr2025/train'
    test_path_ref = '/Users/louiscuendet/Documents/EPFL NEURO-X /Image Analysis and Pattern Recognition/dataset_project_iapr2025/test'
    reference_images = load_images(reference_path_ref)
    train_images = load_images(train_path_ref)
    test_images = load_images(test_path_ref)
    #segmented_reference_images = segmentation(reference_images)
    #segmented_train_images = segmentation(train_images)
    #segmented_test_images = segmentation(test_images)
    


if __name__ == "__main__":
    main()