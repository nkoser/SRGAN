import math
import torch
from PIL import Image
from matplotlib import pyplot as plt
from torch import nn
from torchvision.models import vgg16
from torchvision.transforms import ToTensor


class IdentityConv2(nn.Module):
    def __init__(self, in_channels=1, out_channels=3, padding=1, kernel_size=3):
        super(IdentityConv2, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.in_channels = in_channels
        self.out_channels = out_channels

        # init
        wts = torch.zeros(1, 1, kernel_size, kernel_size)
        nn.init.dirac_(wts)
        wts = wts.repeat(self.out_channels, self.in_channels, 1, 1)

        self.conv_layer = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=self.kernel_size, bias=False,
                                    padding=self.padding, groups=1)  # self.out_channels)

        with torch.no_grad():
            self.conv_layer.weight.copy_(wts)

    def forward(self, x):
        return self.conv_layer(x)


# img = Image.open('images/1.png')
# img = ToTensor()(img)[:3].unsqueeze(0)
# print(img.shape, img.min(),img.max())
# plt.imshow(img[0].permute(1, 2, 0))
# plt.show()
# img = IdentityConv2(in_channels=3, out_channels=3, kernel_size=3, padding=1)(img)
# print(img.shape)
# plt.imshow(img[0].detach().permute(1, 2, 0))
# plt.show()


class Generator(nn.Module):
    def __init__(self, scale_factor, num_channel=1):
        self.num_channel = num_channel
        upsample_block_num = int(math.log(scale_factor, 2))

        super(Generator, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(self.num_channel, 64, kernel_size=9, padding=4),
            nn.PReLU()
        )
        self.block2 = ResidualBlock(64)
        self.block3 = ResidualBlock(64)
        self.block4 = ResidualBlock(64)
        self.block5 = ResidualBlock(64)
        self.block6 = ResidualBlock(64)
        self.block7 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64)
        )
        block8 = [UpsampleBLock(64, 2) for _ in range(upsample_block_num)]
        block8.append(nn.Conv2d(64, self.num_channel, kernel_size=9, padding=4))
        self.block8 = nn.Sequential(*block8)

    def forward(self, x):
        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)
        block6 = self.block6(block5)
        block7 = self.block7(block6)
        block8 = self.block8(block1 + block7)

        return (torch.tanh(block8) + 1) / 2


class Discriminator(nn.Module):
    def __init__(self, num_channel = 1):
        super(Discriminator, self).__init__()
        self.num_channel = num_channel
        self.net = nn.Sequential(
            nn.Conv2d(self.num_channel, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1, kernel_size=1)
        )

    def forward(self, x):
        batch_size = x.size(0)
        return torch.sigmoid(self.net(x).view(batch_size))


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)

        return x + residual


class UpsampleBLock(nn.Module):
    def __init__(self, in_channels, up_scale):
        super(UpsampleBLock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * up_scale ** 2, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(up_scale)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x
