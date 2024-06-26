from os import listdir
from os.path import join

from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])


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


def to_tensor():
    return Compose([ToTensor()])


class TrainDataSetUKEHR(Dataset):
    def __init__(self, high_res_dir):
        (TrainDataSetUKEHR, self).__init__()
        self.high_res_dir = high_res_dir
        self.image_hr_filenames = [join(high_res_dir, x) for x in listdir(high_res_dir) if is_image_file(x)]

    def __getitem__(self, index):
        #print(self.image_hr_filenames[index])
        hr_image = Image.open(self.image_hr_filenames[index])  # [:3, :, :]
        lr_image = Resize(32)(hr_image)
        return ToTensor()(lr_image)[:3, :, :], ToTensor()(hr_image)[:3, :, :]

    def __len__(self):
        return len(self.image_hr_filenames)


class ValDataSetUKEHR(Dataset):
    def __init__(self, high_res_dir):
        (ValDataSetUKEHR, self).__init__()
        self.high_res_dir = high_res_dir
        self.image_hr_filenames = [join(high_res_dir, x) for x in listdir(high_res_dir) if is_image_file(x)]

    def __getitem__(self, index):
        hr_image = Image.open(self.image_hr_filenames[index])
        lr_image = Resize(32)(hr_image)
        hr_recover = Resize((128))(lr_image)
        return ToTensor()(lr_image)[:3, :, :], ToTensor()(hr_recover)[:3, :, :], ToTensor()(hr_image)[:3, :, :]

    def __len__(self):
        return len(self.image_hr_filenames)


class TrainDataSetUKE(Dataset):
    def __init__(self, low_res_dir, high_res_dir):
        (TrainDataSetUKE, self).__init__()
        self.high_res_dir = high_res_dir
        self.low_res_dir = low_res_dir
        self.image_hr_filenames = [join(high_res_dir, x) for x in listdir(high_res_dir) if is_image_file(x)]
        self.image_lr_filenames = [join(low_res_dir, x) for x in listdir(low_res_dir) if is_image_file(x)]
        self.hr_transform = to_tensor()
        self.lr_transform = to_tensor()

    def __getitem__(self, index):
        hr_image = self.hr_transform(Image.open(self.image_hr_filenames[index]))[:3, :, :]
        lr_image = self.lr_transform(Image.open(self.image_lr_filenames[index]))[:3, :, :]
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
        self.hr_transform = to_tensor()
        self.lr_transform = to_tensor()

    def __getitem__(self, index):
        hr_image = self.hr_transform(Image.open(self.image_hr_filenames[index]))[:3, :, :]
        lr_image = self.lr_transform(Image.open(self.image_lr_filenames[index]))[:3, :, :]
        # print(hr_image, lr_image)
        return lr_image, lr_image, hr_image

    def __len__(self):
        return len(self.image_hr_filenames)


class TrainDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, crop_size, upscale_factor):
        super(TrainDatasetFromFolder, self).__init__()
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]
        crop_size = calculate_valid_crop_size(crop_size, upscale_factor)
        self.hr_transform = train_hr_transform(crop_size)
        self.lr_transform = train_lr_transform(crop_size, upscale_factor)

    def __getitem__(self, index):
        hr_image = self.hr_transform(Image.open(self.image_filenames[index]))
        lr_image = self.lr_transform(hr_image)
        return lr_image, hr_image

    def __len__(self):
        return len(self.image_filenames)


class ValDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, upscale_factor):
        super(ValDatasetFromFolder, self).__init__()
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
        return ToTensor()(lr_image), ToTensor()(hr_restore_img), ToTensor()(hr_image)

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
