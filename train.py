import argparse
import os
from os.path import join

from loss import GeneratorLoss

os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3'
from monai.data import DataLoader
from esrgan import RRDBNet
from math import log10
import pandas as pd
import torch.optim as optim
import torch.utils.data
import torchvision.utils as utils
from torch.autograd import Variable
from tqdm import tqdm
import pytorch_ssim
from data_utils import display_transform, VOC2012, UKE, UKEHR
from model import Generator, Discriminator
from utils import read_yaml_file, get_device


def load_gen(opt_dict, device):
    if opt_dict['dataset'] == 'VOC2012':
        voc2012_dataset = VOC2012(join(DATA_DIR, 'VOC2012_train'), join(DATA_DIR, 'VOC2012_val'), opt_dict, device)
        train_set = voc2012_dataset.get_train_dataset()
        val_set = voc2012_dataset.get_val_dataset()
    elif opt_dict['dataset'] == 'UKE':

        uke = UKE(train_path=join(DATA_DIR, 'UKE_dcm_train', 'high_res'),
                  train_lr_path=join(DATA_DIR, 'UKE_dcm_train', 'low_res'),
                  val_lr_path=join(DATA_DIR, 'UKE_dcm_val', 'low_res'),
                  val_path=join(DATA_DIR, 'UKE_dcm_val', 'high_res'),
                  config=opt_dict,
                  device=device)
        train_set = uke.get_train_dataset()
        val_set = uke.get_val_dataset()

    elif opt_dict['dataset'] == 'UKEHR':

        ukehr = UKEHR(train_path=join(DATA_DIR, 'UKE_dcm_train', 'high_res'),
                      val_path=join(DATA_DIR, 'UKE_dcm_val', 'high_res'),
                      config=opt_dict,
                      device=device)
        train_set = ukehr.get_train_dataset()
        val_set = ukehr.get_val_dataset()
    else:
        raise ValueError('Please use one of the following: VOC2012,UKE or UKEHR')

    train_loader = DataLoader(dataset=train_set, num_workers=0, batch_size=opt_dict['batch_size'], shuffle=True)
    val_loader = DataLoader(dataset=val_set, num_workers=0, batch_size=1, shuffle=False)

    return train_loader, val_loader


def load_gan(opt_dict):
    if opt_dict['gan'] == 'srgan':
        netG = Generator(UPSCALE_FACTOR, num_channel=NUM_CHANNEL).to(device)
        netD = Discriminator(num_channel=NUM_CHANNEL).to(device)
    elif opt_dict['gan'] == 'esrgan':
        netG = RRDBNet(upscale_factor=UPSCALE_FACTOR, in_channels=NUM_CHANNEL, out_channels=NUM_CHANNEL).to(device)
        netD = Discriminator(num_channel=NUM_CHANNEL).to(device)  # EDiscriminator().to(device)
    else:
        raise ValueError('Please use one of the following: srgan or esrgan')
    print('# generator parameters:', sum(param.numel() for param in netG.parameters()))
    print('# discriminator parameters:', sum(param.numel() for param in netD.parameters()))

    return netG, netD


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

    GP_WEIGHT = opt_dict['gp_weight']
    WARM_UP = opt_dict['warm_up']

    NUM_CHANNEL = opt_dict['num_channel']

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # get device
    device = get_device()

    # Load train_loader and validation loader
    train_loader, val_loader = load_gen(opt_dict, device)

    # Load GAN Network
    netG, netD = load_gan(opt_dict)

    generator_criterion = GeneratorLoss(opt_dict).to(device)

    optimizerG = optim.Adam(netG.parameters(), lr=opt_dict['lr_g'])
    optimizerD = optim.Adam(netD.parameters(), lr=opt_dict['lr_d'])

    results = {'d_loss': [], 'g_loss': [], 'd_score': [], 'g_score': [], 'psnr': [], 'ssim': []}

    for epoch in range(1, NUM_EPOCHS + 1):
        train_bar = tqdm(train_loader)
        running_results = {'batch_sizes': 0, 'd_loss': 0, 'g_loss': 0, 'd_score': 0, 'g_score': 0}

        netG.train()
        netD.train()
        for batch in train_bar:
            target = batch['image']
            data = batch['low_res_image']

            batch_size = data.size(0)
            running_results['batch_sizes'] += batch_size

            z = Variable(data)
            real_img = Variable(target)

            if WARM_UP < epoch:
                ############################
                # (1) Update D network: maximize D(x)-1-D(G(z))
                ###########################
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
            g_loss = generator_criterion(fake_out, fake_img, real_img, epoch > WARM_UP)
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
            else:
                running_results['d_loss'] += 0
                running_results['d_score'] += 0

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
            for batch in val_bar:
                val_lr = batch['low_res_image']
                val_hr_restore = batch['recover_hr']
                val_hr = batch['image']
                batch_size = val_hr.size(0)

                valing_results['batch_sizes'] += batch_size
                lr = val_lr
                hr = val_hr

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
                        [display_transform()(val_hr_restore)[0, :NUM_CHANNEL, :, :].cpu(),
                         display_transform()(hr.data)[0, :NUM_CHANNEL, :, :].cpu(),
                         display_transform()(sr.data)[0, :NUM_CHANNEL, :, :].cpu()])
            val_images = torch.stack(val_images)
            val_images = torch.chunk(val_images, val_images.size(0) // 15)
            val_save_bar = tqdm(val_images, desc='[saving training results]')
            index = 1
            if epoch % 5 == 0 and epoch != 0:
                for image in val_save_bar:
                    image = utils.make_grid(image, nrow=3, padding=5)
                    utils.save_image(image, out_path + 'epoch_%d_index_%d.png' % (epoch, index), padding=5)
                    index += 1
                    if index == 6:
                        break

        if epoch % 50 == 0 and epoch != 0:  # save model parameters
            torch.save(netG.state_dict(), join(RESULTS_DIR, 'netG_epoch_%d_%d.pth' % (UPSCALE_FACTOR, epoch)))

        if WARM_UP < epoch:
            torch.save(netD.state_dict(), join(RESULTS_DIR, 'netD_epoch_%d_%d.pth' % (UPSCALE_FACTOR, epoch)))
            # save loss\scores\psnr\ssim
            results['d_loss'].append(running_results['d_loss'] / running_results['batch_sizes'])
            results['d_score'].append(running_results['d_score'] / running_results['batch_sizes'])
        else:
            results['d_loss'].append(0)
            results['d_score'].append(0)

        results['g_loss'].append(running_results['g_loss'] / running_results['batch_sizes'])
        results['g_score'].append(running_results['g_score'] / running_results['batch_sizes'])
        results['psnr'].append(valing_results['psnr'])
        results['ssim'].append(valing_results['ssim'])

        if epoch % 10 == 0 and epoch != 0:
            out_path = join(RESULTS_DIR, 'statistics/')
            os.makedirs(out_path, exist_ok=True)
            data_frame = pd.DataFrame(
                data={'Loss_D': results['d_loss'], 'Loss_G': results['g_loss'], 'Score_D': results['d_score'],
                      'Score_G': results['g_score'], 'PSNR': results['psnr'], 'SSIM': results['ssim']},
                index=range(1, epoch + 1))
            data_frame.to_csv(out_path + 'srf_' + str(UPSCALE_FACTOR) + '_train_results.csv', index_label='Epoch')
