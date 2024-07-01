from os.path import join
from typing import Mapping, Hashable

from PIL import Image
import torch
from matplotlib import pyplot as plt
from monai.config import KeysCollection
from monai.data import CacheDataset, DataLoader
from monai.transforms.inverse import InvertibleTransform
from monai.transforms.transform import LazyTransform, MapTransform
import monai
from monai.utils import InterpolateMode
from torchvision.transforms import ToTensor, CenterCrop


def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


class NumChanneld(MapTransform, InvertibleTransform, LazyTransform):
    """
   """

    def __init__(
            self,
            keys: KeysCollection,
            num_channel: int,
            lazy: bool = False,
    ) -> None:
        MapTransform.__init__(self, keys)
        LazyTransform.__init__(self, lazy=lazy)
        self.num_channel = num_channel

    @LazyTransform.lazy.setter  # type: ignore
    def lazy(self, val: bool) -> None:
        raise NotImplementedError()

    def __call__(self, data: Mapping[Hashable, torch.Tensor], lazy: bool | None = None) -> dict[Hashable, torch.Tensor]:
        """
        Returns:
            a dictionary containing the transformed data, as well as any other data present in the dictionary
        """
        d = dict(data)
        for key, val in d.items():
            if val.shape[0] != self.num_channel:
                if val.shape[0] == 1:
                    d[key] = val.repeat(self.num_channel, 1, 1)
                elif val.shape[0] > self.num_channel:
                    d[key] = val[:self.num_channel, :, :]
        return d


class ValidResized(MapTransform, InvertibleTransform, LazyTransform):
    """
  """

    def __init__(
            self,
            keys: KeysCollection,
            upscale_factor: int,
            lazy: bool = False,
    ) -> None:
        MapTransform.__init__(self, keys)
        LazyTransform.__init__(self, lazy=lazy)
        self.upscale_factor = upscale_factor

    @LazyTransform.lazy.setter  # type: ignore
    def lazy(self, val: bool) -> None:
        raise NotImplementedError()

    def __call__(self, data: Mapping[Hashable, torch.Tensor], lazy: bool | None = None) -> dict[Hashable, torch.Tensor]:
        """
       Returns:
           a dictionary containing the transformed data, as well as any other data present in the dictionary
       """
        d = dict(data)
        _, w, h = d['image'].shape
        crop_size = calculate_valid_crop_size(min(w, h), self.upscale_factor)
        d['image'] = CenterCrop(crop_size)(d['image'])
        d['low_res_image'] = monai.transforms.Resize(spatial_size=(crop_size//self.upscale_factor,
                                                                   crop_size//self.upscale_factor),
                                                     mode=InterpolateMode.BICUBIC)(d['low_res_image'])
        d['recover_hr'] = monai.transforms.Resize(spatial_size=(crop_size, crop_size),
                                                  mode=InterpolateMode.BICUBIC)(d['low_res_image'])
        return d
#

# tr = monai.transforms.Compose([
#     monai.transforms.LoadImaged(keys=['image']),
#     monai.transforms.EnsureChannelFirstD(keys=['image']),
#     NumChanneld(keys=['image'], num_channel=1) ,
#
# ])
# pil_img = Image.open(join('/views/data-acc/RW/sukin707/superresolution/data', 'VOC2012_train', '2012_004329.jpg'))
# print(pil_img.size)
# print(ToTensor()(pil_img).shape)
#
# train_ds = CacheDataset(
#     data=[{'image': join('/views/data-acc/RW/sukin707/superresolution/data', 'VOC2012_train', '2012_004329.jpg')}],
#     transform=tr)
# train_dl = DataLoader(train_ds, batch_size=1)
# for x in train_dl:
#     print(x['image'].shape, x['image'].min(), x['image'].max())
#     print(x['image'][0].permute(2, 1, 0).shape)
#     plt.imshow(x['image'][0].permute(1, 2, 0).type(torch.int64))
#     plt.show()
