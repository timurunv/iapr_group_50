import matplotlib.image as mpimg
import os

def load_images(path_ref):
        
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
    jpg_files = [f for f in os.listdir(path_ref) if f.lower().endswith(valid_exts)]
    n_images = len(jpg_files)

    loaded_images = []
    images_ref = []

    for i in range (n_images):
        img_path = os.path.join(path_ref, jpg_files[i])
        img = mpimg.imread(img_path)
        name = os.path.splitext(jpg_files[i])[0] 
        clean_name = name.lstrip('L')

        loaded_images.append(img)
        images_ref.append(clean_name)

    return loaded_images,images_ref