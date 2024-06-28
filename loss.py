import torch
from torch import nn
from torchvision.models.vgg import vgg16, vgg19
from torchvision.transforms import Normalize

from model import IdentityConv2


class GeneratorLoss(nn.Module):
    def __init__(self, perceptual_weight, adv_weight, mse_weight, tv_weight):
        super(GeneratorLoss, self).__init__()
        self.tv_weight = tv_weight
        self.mse_weight = mse_weight
        self.adv_weight = adv_weight
        self.perceptual_weight = perceptual_weight
        vgg = vgg16(pretrained=True)

        # layers = [IdentityConv2()]
        # layers.extend(list(vgg.features)[1:31])
        # loss_network = nn.Sequential(*layers)

        loss_network = nn.Sequential(*list(vgg.features)[:31]).eval()

        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network
        self.mse_loss = nn.MSELoss()
        self.tv_loss = TVLoss()

    def forward(self, out_labels, out_images, target_images, only_gen):
        out_images = out_images.repeat(1, 3, 1, 1)
        target_images = target_images.repeat(1, 3, 1, 1)
        # Adversarial Loss
        adversarial_loss = torch.mean(1 - out_labels)
        # Perception Loss
        perception_loss = self.mse_loss(self.loss_network(normalize_vgg(out_images)),
                                        self.loss_network(normalize_vgg(target_images)))
        # Image Loss
        image_loss = self.mse_loss(out_images, target_images)
        # TV Loss
        tv_loss = self.tv_loss(out_images)
        if only_gen:
            return self.mse_weight * image_loss + self.perceptual_weight * perception_loss + self.tv_weight * tv_loss
        else:
            return self.adv_weight * adversarial_loss + self.mse_weight * image_loss + self.perceptual_weight * perception_loss + self.tv_weight * tv_loss


class EGeneratorLoss(GeneratorLoss):
    def __init__(self, perceptual_weight, adv_weight, mse_weight, tv_weight):
        super(EGeneratorLoss, self).__init__(perceptual_weight, adv_weight, mse_weight, tv_weight)
        self.loss_network = vgg19(pretrained=True).features[:35].eval()
        for param in self.loss_network.parameters():
            param.requires_grad = False

        self.l1_criterion = nn.L1Loss()
        self.l1_weight = self.mse_weight

    def forward(self, out_labels, out_images, target_images, only_gen):
        adversarial_loss = self.adv_weight * -out_labels
        vgg_loss = self.mse_loss(self.loss_network(out_images), self.loss_network(target_images))
        l1_loss = self.l1_criterion(out_images, target_images)
        loss = vgg_loss + self.l1_weight * l1_loss + adversarial_loss
        return loss


class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]


def gradient_penalty(critic, real, fake, device):
    batch_size, channels, height, width = real.shape
    alpha = torch.rand((batch_size, 1, 1, 1)).repeat(1, channels, height, width).to(device)
    interpolated_images = real * alpha + fake.detach() * (1 - alpha)
    interpolated_images.requires_grad_(True)

    mixed_scores = critic(interpolated_images)

    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)

    return gradient_penalty


def normalize_vgg(x):
    return Normalize([0.48235, 0.45882, 0.40784], [0.00392156862745098, 0.00392156862745098, 0.00392156862745098])(x)


if __name__ == "__main__":
    pass
    # print(g_loss)
