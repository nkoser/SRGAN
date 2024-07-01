# Copyright Lornatang. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class ResidualDenseBlock(nn.Module):
    r"""Achieves densely connected convolutional layers.
    `Densely Connected Convolutional Networks <https://arxiv.org/pdf/1608.06993v5.pdf>` paper.

    Args:
        channels (int): The number of channels in the input image.
        growth_channels (int): The number of channels that increase in each layer of convolution.
    """

    def __init__(self, channels: int, growth_channels: int) -> None:
        super(ResidualDenseBlock, self).__init__()
        self.conv_1 = nn.Conv2d(channels + growth_channels * 0, growth_channels, 3, stride=1, padding=1)
        self.conv_2 = nn.Conv2d(channels + growth_channels * 1, growth_channels, 3, stride=1, padding=1)
        self.conv_3 = nn.Conv2d(channels + growth_channels * 2, growth_channels, 3, stride=1, padding=1)
        self.conv_4 = nn.Conv2d(channels + growth_channels * 3, growth_channels, 3, stride=1, padding=1)
        self.conv_5 = nn.Conv2d(channels + growth_channels * 4, channels, 3, stride=1, padding=1)

        self.leaky_relu = nn.LeakyReLU(0.2, True)
        self.identity = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out_1 = self.leaky_relu(self.conv_1(x))
        out_2 = self.leaky_relu(self.conv_2(torch.cat([x, out_1], 1)))
        out_3 = self.leaky_relu(self.conv_3(torch.cat([x, out_1, out_2], 1)))
        out_4 = self.leaky_relu(self.conv_4(torch.cat([x, out_1, out_2, out_3], 1)))
        out_5 = self.identity(self.conv_5(torch.cat([x, out_1, out_2, out_3, out_4], 1)))
        out = torch.mul(out_5, 0.2)
        return torch.add(out, identity)


class ResidualResidualDenseBlock(nn.Module):
    r"""Multi-layer residual dense convolution block.

    Args:
        channels (int): The number of channels in the input image.
        growth_channels (int): The number of channels that increase in each layer of convolution.
    """

    def __init__(self, channels: int, growth_channels: int) -> None:
        super(ResidualResidualDenseBlock, self).__init__()
        self.rdb_1 = ResidualDenseBlock(channels, growth_channels)
        self.rdb_2 = ResidualDenseBlock(channels, growth_channels)
        self.rdb_3 = ResidualDenseBlock(channels, growth_channels)

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.rdb_1(x)
        out = self.rdb_2(out)
        out = self.rdb_3(out)
        out = torch.mul(out, 0.2)
        return torch.add(out, identity)


class RRDBNet(nn.Module):
    def __init__(
            self,
            in_channels: int = 3,
            out_channels: int = 3,
            channels: int = 64,
            growth_channels: int = 32,
            num_rrdb: int = 15, #23,
            upscale_factor: int = 4,
    ) -> None:
        super(RRDBNet, self).__init__()
        assert upscale_factor in (2, 3, 4), "Upscale factor should be 2, 3 or 4."
        self.upscale_factor = upscale_factor

        # The first layer of convolutional layer
        self.conv_1 = nn.Conv2d(in_channels, channels, (3, 3), (1, 1), (1, 1))

        # Feature extraction backbone network
        trunk = []
        for _ in range(num_rrdb):
            trunk.append(ResidualResidualDenseBlock(channels, growth_channels))
        self.trunk = nn.Sequential(*trunk)

        # After the feature extraction network, reconnect a layer of convolutional blocks
        self.conv_2 = nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1))

        # Up-sampling convolutional layer
        if self.upscale_factor == 2 or self.upscale_factor == 3:
            self.up_sampling_1 = nn.Sequential(
                nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1)),
                nn.LeakyReLU(0.2, True),
            )
        elif self.upscale_factor == 4:
            self.up_sampling_1 = nn.Sequential(
                nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1)),
                nn.LeakyReLU(0.2, True),
            )
            self.up_sampling_2 = nn.Sequential(
                nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1)),
                nn.LeakyReLU(0.2, True),
            )
        else:  # 8
            self.up_sampling_1 = nn.Sequential(
                nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1)),
                nn.LeakyReLU(0.2, True),
            )
            self.up_sampling_2 = nn.Sequential(
                nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1)),
                nn.LeakyReLU(0.2, True),
            )
            self.up_sampling_3 = nn.Sequential(
                nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1)),
                nn.LeakyReLU(0.2, True),
            )

        # Output layer
        self.conv_3 = nn.Sequential(
            nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1)),
            nn.LeakyReLU(0.2, True),
        )

        # Output layer
        self.conv_4 = nn.Conv2d(channels, out_channels, (3, 3), (1, 1), (1, 1))

        initialize_weights(self.modules())

    def forward(self, x: Tensor) -> Tensor:
        conv_1 = self.conv_1(x)
        x = self.trunk(conv_1)
        x = self.conv_2(x)
        x = torch.add(conv_1, x)

        if self.upscale_factor == 2:
            x = self.up_sampling_1(F.interpolate(x, scale_factor=2, mode="nearest"))
        elif self.upscale_factor == 3:
            x = self.up_sampling_1(F.interpolate(x, scale_factor=3, mode="nearest"))
        elif self.upscale_factor == 4:
            x = self.up_sampling_1(F.interpolate(x, scale_factor=2, mode="nearest"))
            x = self.up_sampling_2(F.interpolate(x, scale_factor=2, mode="nearest"))
        else:  # 8
            x = self.up_sampling_1(F.interpolate(x, scale_factor=2, mode="nearest"))
            x = self.up_sampling_2(F.interpolate(x, scale_factor=2, mode="nearest"))
            x = self.up_sampling_3(F.interpolate(x, scale_factor=2, mode="nearest"))

        x = self.conv_3(x)
        x = self.conv_4(x)

        x = torch.clamp_(x, 0.0, 1.0)

        return x

def initialize_weights(modules: Any):
    r"""Initializes the weights of the model.

     Args:
         modules: The model to be initialized.
     """
    for module in modules:
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight)
            module.weight.data *= 0.1
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)