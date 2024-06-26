import glob
import os
import shutil
from os.path import join

import torch
from PIL import Image
from matplotlib import pyplot as plt

from utils import create_directories


def save_patches(files, save_lr, save_hr):
    for ind, file in enumerate(files):
        x = torch.load(file)
        k = x['k_ct']
        x = x['x_ct']
        x = normalize(x)
        k = normalize(k)
        for l in range(k.shape[0]):
            img_lr = k[:, :, l]
            img_hr = x[:, :, (l * 4) - 1]

            if img_hr.max() > 0 and img_lr.max() > 0:
                plt.imsave(join(save_lr, f'img_{ind}_{l}.png'), arr=img_lr, cmap='gray')
                plt.imsave(join(save_hr, f'img_{ind}_{l}.png'), arr=img_hr, cmap='gray')

                print('Save Batch')
def normalize(t):
    return (t - t.min()) / (t.max() - t.min())

if __name__ == "__main__":
    # List of source image files

    root_path = r'/views/data-acc/RW/sukin707/superresolution/data/Patches'

    train_files = glob.glob(join(root_path, 'training', '*.pt'))
    val_files = glob.glob(join(root_path, 'validation', '*.pt'))

    # Target folder
    target_train_folder = r'/views/data-acc/RW/sukin707/superresolution/data/UKE_train/low_res'
    target_train_hr_folder = r'/views/data-acc/RW/sukin707/superresolution/data/UKE_train/high_res'
    target_val_folder = r'/views/data-acc/RW/sukin707/superresolution/data/UKE_val/low_res'
    target_val_hr_folder = r'/views/data-acc/RW/sukin707/superresolution/data/UKE_val/high_res'

    create_directories([target_val_hr_folder, target_val_folder, target_train_hr_folder, target_train_folder])

    save_patches(train_files, target_train_folder, target_train_hr_folder)
