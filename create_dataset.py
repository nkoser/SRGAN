import glob
import os
import shutil
from os.path import join

from PIL import Image


def copy_images(source_files, target_folder):
    # Ensure the target folder exists, create it if it doesn't
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    i = 0
    for source_file in source_files:
        # Check if the source file exists
        if os.path.isfile(source_file):
            # Construct the full path for the target file
            with Image.open(source_file) as img:
                width, height = img.size
                if width >= 88 and height >= 88:
                    # Construct the full path for the target file
                    target_file = os.path.join(target_folder, os.path.basename(source_file))
                    # Copy the file to the target folder
                    shutil.copy2(source_file, target_file)
                    print(f"Copied {source_file} to {target_file}")
                else:
                    i+=1
                    print(f"File {source_file} dimensions are too small: {width}x{height}")

    print(f'{i} files have a to small size')


if __name__ == "__main__":
    # List of source image files

    root_path = r'/views/data-acc/RW/sukin707/superresolution/archive/VOC2012_train_val/VOC2012_train_val/JPEGImages'
    img_files = glob.glob(join(root_path, '*.jpg'))

    print(len(img_files))

    train_img = img_files[:16700]
    print(len(train_img))
    val_imgs = img_files[16700:]

    # Target folder
    target_train_folder = r'/views/data-acc/RW/sukin707/superresolution/data/VOC2012_train'
    target_val_folder = r'/views/data-acc/RW/sukin707/superresolution/data/VOC2012_val'

    os.makedirs(target_train_folder, exist_ok=True)
    os.makedirs(target_val_folder,exist_ok=True)

    # Copy the images
    copy_images(train_img, target_train_folder)
    print(f"Images have been successfully copied to {target_train_folder}.")
    copy_images(val_imgs,target_val_folder)
