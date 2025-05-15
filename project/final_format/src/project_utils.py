import matplotlib.image as mpimg
import os

def load_images(path_ref):
        
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
    jpg_files = [f for f in os.listdir(path_ref) if f.lower().endswith(valid_exts)]
    n_images = len(jpg_files)

    loaded_images = []

    for i in range (n_images):
        img_path = os.path.join(path_ref, jpg_files[i])
        img = mpimg.imread(img_path)
        loaded_images.append(img)

    return loaded_images