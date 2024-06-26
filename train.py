import argparse
import os
from os.path import join

from esrgan import EGenerator, EDiscriminator

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from math import log10

import pandas as pd
import torch.optim as optim
import torch.utils.data
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

import pytorch_ssim
from data_utils import TrainDatasetFromFolder, ValDatasetFromFolder, display_transform, TrainDataSetUKE, ValDataSetUKE, \
    TrainDataSetUKEHR, ValDataSetUKEHR
from loss import GeneratorLoss, EGeneratorLoss, gradient_penalty
from model import Generator, Discriminator
from utils import read_yaml_file, get_device

# parser = argparse.ArgumentParser(description='Train Super Resolution Models')
# parser.add_argument('--crop_size', default=88, type=int, help='training images crop size')
# parser.add_argument('--upscale_factor', default=4, type=int, choices=[2, 4, 8],
# help='super resolution upscale factor')
# parser.add_argument('--num_epochs', default=100, type=int, help='train epoch number')


if __name__ == '__main__':

    # replace through config
    opt_dict = read_yaml_file('config/config.yml')

    CROP_SIZE = opt_dict['crop_size']
    UPSCALE_FACTOR = opt_dict['upscale_factor']
    NUM_EPOCHS = opt_dict['num_epochs']

    DATA_DIR = opt_dict['data_dir']
    ROOT_DIR = opt_dict['root_dir']
    RESULTS_DIR = join(ROOT_DIR, opt_dict['exp_name'])

    GAN = opt_dict['gan']

    PERCEPTUAL_WEIGHT = opt_dict['perceptual_weight']
    ADV_WEIGHT = opt_dict['adv_weight']
    MSE_WEIGHT = opt_dict['mse_weight']
    TV_WEIGHT = opt_dict['tv_weight']

    GP_WEIGHT = opt_dict['gp_weight']
    WARM_UP = opt_dict['warm_up']

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # get device
    device = get_device()

    if opt_dict['dataset'] == 'VOC2012':
        train_set = TrainDatasetFromFolder(join(DATA_DIR, 'VOC2012_train'), crop_size=CROP_SIZE,
                                           upscale_factor=UPSCALE_FACTOR)
        val_set = ValDatasetFromFolder(join(DATA_DIR, 'VOC2012_val'), upscale_factor=UPSCALE_FACTOR)
    elif opt_dict['dataset'] == 'UKE':
        train_set = TrainDataSetUKE(join(DATA_DIR, 'UKE_train', 'low_res'), join(DATA_DIR, 'UKE_train', 'high_res'))
        val_set = ValDataSetUKE(join(DATA_DIR, 'UKE_val', 'low_res'), join(DATA_DIR, 'UKE_val', 'high_res'))
    elif opt_dict['dataset'] == 'UKEHR':
        train_set = TrainDataSetUKEHR(join(DATA_DIR, 'UKE_train', 'high_res'))
        val_set = ValDataSetUKEHR(join(DATA_DIR, 'UKE_val', 'high_res'))
    else:
        raise ValueError('Please use one of the following: VOC2012,UKE or UKEHR')

    train_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=64, shuffle=True)
    val_loader = DataLoader(dataset=val_set, num_workers=4, batch_size=1, shuffle=False)

    if GAN == 'srgan':
        netG = Generator(UPSCALE_FACTOR).to(device)
        netD = Discriminator().to(device)
    elif GAN == 'esrgan':
        netG = EGenerator().to(device)
        netD = Discriminator().to(device)  # EDiscriminator().to(device)
    else:
        raise ValueError('Please use one of the following: srgan or esrgan')

    print('# generator parameters:', sum(param.numel() for param in netG.parameters()))
    print('# discriminator parameters:', sum(param.numel() for param in netD.parameters()))

    # if GAN == 'srgan':
    generator_criterion = GeneratorLoss(tv_weight=TV_WEIGHT,
                                        mse_weight=MSE_WEIGHT,
                                        perceptual_weight=PERCEPTUAL_WEIGHT,
                                        adv_weight=ADV_WEIGHT).to(device)
    # elif GAN == 'esrgan':
    #    generator_criterion = EGeneratorLoss(tv_weight=TV_WEIGHT,
    #                                         mse_weight=MSE_WEIGHT,
    #                                         perceptual_weight=PERCEPTUAL_WEIGHT,
    #                                         adv_weight=ADV_WEIGHT).to(device)
    optimizerG = optim.Adam(netG.parameters())
    optimizerD = optim.Adam(netD.parameters())

    results = {'d_loss': [], 'g_loss': [], 'd_score': [], 'g_score': [], 'psnr': [], 'ssim': []}

    for epoch in range(1, NUM_EPOCHS + 1):
        train_bar = tqdm(train_loader)
        running_results = {'batch_sizes': 0, 'd_loss': 0, 'g_loss': 0, 'd_score': 0, 'g_score': 0}

        netG.train()
        netD.train()
        for data, target in train_bar:
            ##g_update_first = True
            batch_size = data.size(0)
            running_results['batch_sizes'] += batch_size

            z = Variable(data)
            z = z.to(device)

            if WARM_UP < epoch:
                ############################
                # (1) Update D network: maximize D(x)-1-D(G(z))
                ###########################
                real_img = Variable(target)
                real_img = real_img.to(device)
                fake_img = netG(z)

                # if GAN == 'esrgan':
                #    gp = gradient_penalty(netD, real_img, fake_img, device=device)

                netD.zero_grad()
                real_out = netD(real_img).mean()
                fake_out = netD(fake_img).mean()
                # if GAN == 'esrgan':
                # d_loss = -real_out - fake_out + GP_WEIGHT * gp
                # elif GAN == 'srgan':
                d_loss = 1 - real_out + fake_out
                # else:
                # raise ValueError('Unknown GAN type')

                d_loss.backward(retain_graph=True)
                optimizerD.step()

            ############################
            # (2) Update G network: minimize 1-D(G(z)) + Perception Loss + Image Loss + TV Loss
            ###########################
            netG.zero_grad()
            ## The two lines below are added to prevent runtime error in Google Colab ##
            fake_img = netG(z)
            fake_out = netD(fake_img).mean()
            ##
            g_loss = generator_criterion(fake_out, fake_img, real_img)
            g_loss.backward()

            fake_img = netG(z)
            fake_out = netD(fake_img).mean()

            optimizerG.step()

            # loss for current batch before optimization 
            running_results['g_loss'] += g_loss.item() * batch_size
            running_results['g_score'] += fake_out.item() * batch_size

            if WARM_UP < epoch:
                running_results['d_loss'] += d_loss.item() * batch_size
                running_results['d_score'] += real_out.item() * batch_size

            if WARM_UP < epoch:
                train_bar.set_description(desc='[%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f' % (
                    epoch, NUM_EPOCHS, running_results['d_loss'] / running_results['batch_sizes'],
                    running_results['g_loss'] / running_results['batch_sizes'],
                    running_results['d_score'] / running_results['batch_sizes'],
                    running_results['g_score'] / running_results['batch_sizes']))
            else:
                train_bar.set_description(desc='[%d/%d] Loss_G: %.4f D(G(z)): %.4f' % (epoch, NUM_EPOCHS,
                                                                                       running_results['g_loss'] /
                                                                                       running_results['batch_sizes'],
                                                                                       running_results['g_score'] /
                                                                                       running_results['batch_sizes']))

        netG.eval()
        out_path = join(RESULTS_DIR, 'training_results/SRF_' + str(UPSCALE_FACTOR) + '/')
        os.makedirs(out_path, exist_ok=True)

        with torch.no_grad():
            val_bar = tqdm(val_loader)
            valing_results = {'mse': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'batch_sizes': 0}
            val_images = []
            for val_lr, val_hr_restore, val_hr in val_bar:
                batch_size = val_lr.size(0)
                valing_results['batch_sizes'] += batch_size
                lr = val_lr.to(device)
                hr = val_hr.to(device)
                sr = netG(lr)
                if hr.max() > 0:
                    batch_mse = ((sr - hr) ** 2).data.mean()
                    valing_results['mse'] += batch_mse * batch_size
                    batch_ssim = pytorch_ssim.ssim(sr, hr).item()
                    valing_results['ssims'] += batch_ssim * batch_size
                    # print(hr.max(), valing_results['mse'], valing_results['batch_sizes'])
                    valing_results['psnr'] = 10 * log10(
                        (hr.max() ** 2) / (valing_results['mse'] / valing_results['batch_sizes']))
                    valing_results['ssim'] = valing_results['ssims'] / valing_results['batch_sizes']
                    val_bar.set_description(
                        desc='[converting LR images to SR images] PSNR: %.4f dB SSIM: %.4f' % (
                            valing_results['psnr'], valing_results['ssim']))

                    val_images.extend(
                        [display_transform()(val_hr_restore.squeeze(0)), display_transform()(hr.data.cpu().squeeze(0)),
                         display_transform()(sr.data.cpu().squeeze(0))])
            val_images = torch.stack(val_images)
            val_images = torch.chunk(val_images, val_images.size(0) // 15)
            val_save_bar = tqdm(val_images, desc='[saving training results]')
            index = 1
            for image in val_save_bar:
                image = utils.make_grid(image, nrow=3, padding=5)
                utils.save_image(image, out_path + 'epoch_%d_index_%d.png' % (epoch, index), padding=5)
                index += 1
                if index == 6:
                    break

                    # save model parameters
        torch.save(netG.state_dict(), join(RESULTS_DIR, 'netG_epoch_%d_%d.pth' % (UPSCALE_FACTOR, epoch)))

        if WARM_UP < epoch:
            torch.save(netD.state_dict(), join(RESULTS_DIR, 'netD_epoch_%d_%d.pth' % (UPSCALE_FACTOR, epoch)))
            # save loss\scores\psnr\ssim
            results['d_loss'].append(running_results['d_loss'] / running_results['batch_sizes'])
            results['d_score'].append(running_results['d_score'] / running_results['batch_sizes'])
        results['g_loss'].append(running_results['g_loss'] / running_results['batch_sizes'])
        results['g_score'].append(running_results['g_score'] / running_results['batch_sizes'])
        results['psnr'].append(valing_results['psnr'])
        results['ssim'].append(valing_results['ssim'])

        if epoch % 10 == 0 and epoch != 0:
            out_path = join(RESULTS_DIR, 'statistics/')
            os.makedirs(out_path, exist_ok=True)
            if WARM_UP < epoch:
                data_frame = pd.DataFrame(
                    data={'Loss_G': results['g_loss'],
                          'Score_G': results['g_score'], 'PSNR': results['psnr'], 'SSIM': results['ssim']},
                    index=range(1, epoch + 1))
            else:
                data_frame = pd.DataFrame(
                    data={'Loss_D': results['d_loss'], 'Loss_G': results['g_loss'], 'Score_D': results['d_score'],
                          'Score_G': results['g_score'], 'PSNR': results['psnr'], 'SSIM': results['ssim']},
                    index=range(1, epoch + 1))
            data_frame.to_csv(out_path + 'srf_' + str(UPSCALE_FACTOR) + '_train_results.csv', index_label='Epoch')
