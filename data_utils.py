from os import listdir
from os.path import join

import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize
import torch.nn.functional as F

from utils import file_ends_with


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG', '.pt'])


def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


def train_hr_transform(crop_size):
    return Compose([
        RandomCrop(crop_size),
        ToTensor(),
    ])


def train_lr_transform(crop_size, upscale_factor):
    return Compose([
        ToPILImage(),
        Resize(crop_size // upscale_factor, interpolation=Image.BICUBIC),
        ToTensor()
    ])


def display_transform():
    return Compose([
        ToPILImage(),
        Resize(400),
        CenterCrop(400),
        ToTensor()
    ])


class TrainDataSetUKEHR(Dataset):
    def __init__(self, high_res_dir):
        (TrainDataSetUKEHR, self).__init__()
        self.high_res_dir = high_res_dir
        self.image_hr_filenames = [join(high_res_dir, x) for x in listdir(high_res_dir) if is_image_file(x)]

    def __getitem__(self, index):
        # print(self.image_hr_filenames[index])
        hr_image = torch.tensor(torch.load(self.image_hr_filenames[index]))  # [:3, :, :]
        lr_image = resize(hr_image, 32)
        return lr_image.unsqueeze(0).float(), hr_image.unsqueeze(0).float()

    def __len__(self):
        return len(self.image_hr_filenames)


class ValDataSetUKEHR(Dataset):
    def __init__(self, high_res_dir):
        (ValDataSetUKEHR, self).__init__()
        self.high_res_dir = high_res_dir
        self.image_hr_filenames = [join(high_res_dir, x) for x in listdir(high_res_dir) if is_image_file(x)]

    def __getitem__(self, index):
        hr_image = torch.tensor(torch.load(self.image_hr_filenames[index]))
        lr_image = resize(hr_image, 32)
        hr_recover = resize(lr_image, 128)
        return lr_image.unsqueeze(0).float(), hr_recover.unsqueeze(0).float(), hr_image.unsqueeze(0).float()

    def __len__(self):
        return len(self.image_hr_filenames)


class TrainDataSetUKE(Dataset):
    def __init__(self, low_res_dir, high_res_dir):
        (TrainDataSetUKE, self).__init__()
        self.high_res_dir = high_res_dir
        self.low_res_dir = low_res_dir
        self.image_hr_filenames = [join(high_res_dir, x) for x in listdir(high_res_dir) if is_image_file(x)]
        self.image_lr_filenames = [join(low_res_dir, x) for x in listdir(low_res_dir) if is_image_file(x)]
        self.hr_transform = ToTensor()
        self.lr_transform = ToTensor()
        self.file_ext = file_ends_with(self.image_hr_filenames[0])

    def __getitem__(self, index):

        # print(Image.open(self.image_hr_filenames[index]).getextrema())
        if self.file_ext != 'pt':
            hr_image = self.hr_transform(Image.open(self.image_hr_filenames[index]))[:3, :, :]
            lr_image = self.lr_transform(Image.open(self.image_lr_filenames[index]))[:3, :, :]
        else:
            hr_image = self.hr_transform(torch.load(self.image_hr_filenames[index])).float()
            lr_image = self.lr_transform(torch.load(self.image_lr_filenames[index])).float()
        return lr_image, hr_image

    def __len__(self):
        return len(self.image_hr_filenames)


class ValDataSetUKE(Dataset):
    def __init__(self, low_res_dir, high_res_dir):
        (ValDataSetUKE, self).__init__()
        self.high_res_dir = high_res_dir
        self.low_res_dir = low_res_dir
        self.image_hr_filenames = [join(high_res_dir, x) for x in listdir(high_res_dir) if is_image_file(x)]
        self.image_lr_filenames = [join(low_res_dir, x) for x in listdir(low_res_dir) if is_image_file(x)]
        self.hr_transform = ToTensor()
        self.lr_transform = ToTensor()
        self.file_ext = file_ends_with(self.image_hr_filenames[0])

    def __getitem__(self, index):
        if self.file_ext != 'pt':
            hr_image = self.hr_transform(Image.open(self.image_hr_filenames[index]))[:3, :, :]
            lr_image = self.lr_transform(Image.open(self.image_lr_filenames[index]))[:3, :, :]
        else:
            hr_image = self.hr_transform(torch.load(self.image_hr_filenames[index])).float()
            lr_image = self.lr_transform(torch.load(self.image_lr_filenames[index])).float()
        return lr_image, lr_image, hr_image

    def __len__(self):
        return len(self.image_hr_filenames)


class TrainDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, crop_size, upscale_factor,num_channel):
        super(TrainDatasetFromFolder, self).__init__()
        self.num_channel = num_channel
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]
        crop_size = calculate_valid_crop_size(crop_size, upscale_factor)
        self.hr_transform = train_hr_transform(crop_size)
        self.lr_transform = train_lr_transform(crop_size, upscale_factor)

    def __getitem__(self, index):
        hr_image = self.hr_transform(Image.open(self.image_filenames[index]))[:self.num_channel]
        lr_image = self.lr_transform(hr_image)[:self.num_channel]
        return lr_image, hr_image

    def __len__(self):
        return len(self.image_filenames)


class ValDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, upscale_factor,num_channel):
        super(ValDatasetFromFolder, self).__init__()
        self.num_channel = num_channel
        self.upscale_factor = upscale_factor
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]

    def __getitem__(self, index):
        hr_image = Image.open(self.image_filenames[index])
        w, h = hr_image.size
        crop_size = calculate_valid_crop_size(min(w, h), self.upscale_factor)
        lr_scale = Resize(crop_size // self.upscale_factor, interpolation=Image.BICUBIC)
        hr_scale = Resize(crop_size, interpolation=Image.BICUBIC)
        hr_image = CenterCrop(crop_size)(hr_image)
        lr_image = lr_scale(hr_image)
        hr_restore_img = hr_scale(lr_image)
        return ToTensor()(lr_image)[:self.num_channel], ToTensor()(hr_restore_img)[:self.num_channel], ToTensor()(hr_image)[:self.num_channel]

    def __len__(self):
        return len(self.image_filenames)


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


def resize(img, size):
    tensor = img.unsqueeze(0).unsqueeze(0)  # Form: [1, 1, 128, 128]

    resized_tensor = F.interpolate(tensor, size=(size, size), mode='bilinear', align_corners=False)

    # Entferne die hinzugefügten Dimensionen


    #print(resized_tensor.shape)
    return resized_tensor.squeeze(0).squeeze()
