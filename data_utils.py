import os.path
from abc import ABC, abstractmethod
from os import listdir
from os.path import join

import SimpleITK as sitk
from PIL import Image
from monai import transforms
from monai.data import CacheDataset, Dataset
from monai.utils import InterpolateMode
from torchvision.transforms import ToTensor, CenterCrop, Resize

from custom_transforms import NumChanneld, ValidResized


def is_image_file(filename):
    return any(
        filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG', '.pt', '.nii'])


def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


def save_array_as_nifti(array, file_name):
    sitk_image = sitk.GetImageFromArray(array)

    # Write the SimpleITK image to a DICOM file
    sitk.WriteImage(sitk_image, file_name)


class MONAIDataset(ABC):

    def __init__(self, train_path, val_path, config, device):
        self.device = device
        self.config = config
        self.num_channel = config['num_channel']
        self.batch_size = config['batch_size']
        self.train_datalist = self.build_list(train_path)
        self.val_datalist = self.build_list(val_path)

    def build_list(self, path):
        filenames = [join(path, x) for x in listdir(path) if is_image_file(x)]
        return [{"image": item} for item in filenames if is_image_file(filename=item)]

    @abstractmethod
    def get_val_dataset(self):
        pass

    @abstractmethod
    def get_train_dataset(self):
        pass


class VOC2012(MONAIDataset):

    def __init__(self, train_path, val_path, config, device):
        super().__init__(train_path, val_path, config, device=device)

    def get_train_dataset(self):
        train_ds = CacheDataset(data=self.train_datalist,
                                transform=transforms.Compose(monai_transform(self.config['crop_size'],
                                                                             self.config['upscale_factor'],
                                                                             self.num_channel,
                                                                             self.device)), cache_rate=0.01)
        return train_ds

    def get_val_dataset(self):
        val_ds = CacheDataset(data=self.val_datalist,
                              transform=transforms.Compose(monai_val_transform(self.config['upscale_factor'],
                                                                               self.num_channel,
                                                                               self.device)))
        return val_ds


def monai_transform(crop_size, upscale_factor, num_channels, device):
    crop_size = calculate_valid_crop_size(crop_size, upscale_factor)
    low_img_size = (crop_size // upscale_factor, crop_size // upscale_factor)
    composed_transform = [
        transforms.LoadImaged(keys=["image"]),
        transforms.ToTensord(keys=['image'], device=device),
        transforms.EnsureChannelFirstd(keys=["image"]),
        NumChanneld(keys=["image"], num_channel=num_channels),
        transforms.ScaleIntensityd(keys=["image"], minv=0, maxv=1),
        transforms.RandSpatialCropD(keys=['image'], roi_size=(crop_size, crop_size)),
        transforms.CopyItemsd(keys=["image"], times=1, names=["low_res_image"]),
        transforms.Resized(keys=["low_res_image"], spatial_size=low_img_size, mode=InterpolateMode.BICUBIC),
    ]

    return composed_transform


def monai_val_transform(upscale_factor, num_channels, device):
    composed_transform = [
        transforms.LoadImaged(keys=["image"]),
        transforms.ToTensord(keys=['image'], device=device),
        transforms.EnsureChannelFirstd(keys=["image"]),
        NumChanneld(keys=["image"], num_channel=num_channels),
        transforms.ScaleIntensityd(keys=["image"], minv=0, maxv=1),
        transforms.CopyItemsd(keys=["image"], times=1, names=["low_res_image"]),
        ValidResized(keys=["image"], upscale_factor=upscale_factor)]
    return composed_transform


class UKE:
    def __init__(self, train_path, val_path, train_lr_path, val_lr_path, config, device):
        self.device = device
        self.config = config
        self.train_dict = self.build_list(train_lr_path, train_path)
        self.val_dict = self.build_list(val_lr_path, val_path)[:1000]

    def build_list(self, lr_path, hr_path):
        lr, hr = self.__test_paired(lr_path, hr_path)
        filenames_lr = [join(lr_path, x) for x in lr if is_image_file(x)]
        filenames_hr = [join(hr_path, x) for x in hr if is_image_file(x)]

        return [{"image": item, 'low_res_image': lr_item} for item, lr_item, in zip(filenames_hr, filenames_lr)]

    def __test_paired(self, lr_path, hr_path):
        lr = listdir(lr_path)
        hr = listdir(hr_path)
        for x, y in zip(lr, hr):
            assert os.path.basename(x) == os.path.basename(y)
        return lr, hr

    def get_train_dataset(self):
        t = transforms.Compose([
            transforms.LoadImaged(keys=['image', 'low_res_image']),
            transforms.ToTensord(keys=['image', 'low_res_image'], device=self.device),
            transforms.EnsureChannelFirstd(keys=['image', 'low_res_image']),
            NumChanneld(keys=['image', 'low_res_image'], num_channel=self.config['num_channel']),
            #transforms.RandFlipd(keys=['image', 'low_res_image'])

        ])
        return CacheDataset(self.train_dict, t, cache_rate=self.config['cache_rate'])

    def get_val_dataset(self):
        t = transforms.Compose([
            transforms.LoadImaged(keys=['image', 'low_res_image']),
            transforms.ToTensord(keys=['image', 'low_res_image'], device=self.device),
            transforms.EnsureChannelFirstd(keys=['image', 'low_res_image']),
            NumChanneld(keys=['image', 'low_res_image'], num_channel=self.config['num_channel']),
            transforms.CopyItemsD(keys=['low_res_image'], times=1, names=['recover_hr']),
            transforms.Resized(keys=['recover_hr'], spatial_size=(128, 128), mode=InterpolateMode.BICUBIC),
            # transforms.RandFlipd(keys=['image', 'low_res_image', 'recover_hr'])
        ])
        return CacheDataset(self.val_dict, t, cache_rate=self.config['cache_rate'])


class UKEHR:
    def __init__(self, train_path, val_path, config, device):
        self.device = device
        self.config = config
        self.train_dict = self.build_list(train_path)
        self.val_dict = self.build_list(val_path)[:1000]

    def build_list(self, hr_path):
        filenames_hr = [join(hr_path, x) for x in listdir(hr_path) if is_image_file(x)]
        return [{"image": item} for item in filenames_hr]

    def get_train_dataset(self):
        t = transforms.Compose([
            transforms.LoadImaged(keys=['image']),
            transforms.ToTensord(keys=['image'], device=self.device),
            transforms.EnsureChannelFirstd(keys=['image']),
            NumChanneld(keys=['image'], num_channel=self.config['num_channel']),
            transforms.CopyItemsD(keys=['image'], times=1, names=['low_res_image']),
            transforms.Resized(keys=['low_res_image'], spatial_size=(32, 32), mode=InterpolateMode.BILINEAR)

        ])
        return CacheDataset(self.train_dict, t, cache_rate=1)

    def get_val_dataset(self):
        t = transforms.Compose([
            transforms.LoadImaged(keys=['image']),
            transforms.ToTensord(keys=['image'], device=self.device),
            transforms.EnsureChannelFirstd(keys=['image']),
            NumChanneld(keys=['image'], num_channel=self.config['num_channel']),
            transforms.CopyItemsD(keys=['image'], times=1, names=['low_res_image']),
            transforms.Resized(keys=['low_res_image'], spatial_size=(32, 32), mode=InterpolateMode.BILINEAR),
            transforms.CopyItemsD(keys=['low_res_image'], times=1, names=['recover_hr']),
            transforms.Resized(keys=['recover_hr'], spatial_size=(128, 128), mode=InterpolateMode.BICUBIC)
        ])
        return CacheDataset(self.val_dict, t, cache_rate=1)


def display_transform():
    return transforms.Compose([
        Resize(400),
        CenterCrop(400),
    ])


class TestDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, upscale_factor):
        super(TestDatasetFromFolder, self).__init__()
        self.lr_path = dataset_dir + '/SRF_' + str(upscale_factor) + '/data/'
        self.hr_path = dataset_dir + '/SRF_' + str(upscale_factor) + '/target/'
        self.upscale_factor = upscale_factor
        self.lr_filenames = [join(self.lr_path, x) for x in listdir(self.lr_path) if is_image_file(x)]
        self.hr_filenames = [join(self.hr_path, x) for x in listdir(self.hr_path) if is_image_file(x)]

    def __getitem__(self, index):
        image_name = self.lr_filenames[index].split('/')[-1]
        lr_image = Image.open(self.lr_filenames[index])
        w, h = lr_image.size
        hr_image = Image.open(self.hr_filenames[index])
        hr_scale = Resize((self.upscale_factor * h, self.upscale_factor * w), interpolation=Image.BICUBIC)
        hr_restore_img = hr_scale(lr_image)
        return image_name, ToTensor()(lr_image), ToTensor()(hr_restore_img), ToTensor()(hr_image)

    def __len__(self):
        return len(self.lr_filenames)
