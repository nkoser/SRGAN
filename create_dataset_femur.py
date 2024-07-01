import glob
import os
import shutil
from os.path import join

import torch
from PIL import Image
from matplotlib import pyplot as plt

from data_utils import save_array_as_nifti
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
                #plt.imsave(join(save_lr, f'img_{ind}_{l}.pt'), arr=img_lr, cmap='gray')
                #plt.imsave(join(save_hr, f'img_{ind}_{l}.pt'), arr=img_hr, cmap='gray')
                #torch.save(img_lr, join(save_lr, f'img_{ind}_{l}.pt'))
                #torch.save(img_hr, join(save_hr, f'img_{ind}_{l}.pt'))
                save_array_as_nifti(img_lr, join(save_lr, f'img_{ind}_{l}.nii'))
                save_array_as_nifti(img_hr, join(save_hr, f'img_{ind}_{l}.nii'))
                print('Save Batch')
def normalize(t):
    return (t - t.min()) / (t.max() - t.min())

if __name__ == "__main__":
    # List of source image files

    root_path = r'/views/data-acc/RW/sukin707/superresolution/data/Patches'

    train_files = glob.glob(join(root_path, 'training', '*.pt'))
    val_files = glob.glob(join(root_path, 'validation', '*.pt'))

    # Target folder
    target_train_folder = r'/views/data-acc/RW/sukin707/superresolution/data/UKE_dcm_train/low_res'
    target_train_hr_folder = r'/views/data-acc/RW/sukin707/superresolution/data/UKE_dcm_train/high_res'
    target_val_folder = r'/views/data-acc/RW/sukin707/superresolution/data/UKE_dcm_val/low_res'
    target_val_hr_folder = r'/views/data-acc/RW/sukin707/superresolution/data/UKE_dcm_val/high_res'

    create_directories([target_val_hr_folder, target_val_folder, target_train_hr_folder, target_train_folder])

    save_patches(val_files, target_val_folder, target_val_hr_folder)
