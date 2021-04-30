import torch
import os
import torchvision.utils as utils
from pixelcnnpp.pixelcnnpp import (PixelCNN, load_part_of_model)

from functools import partial
import dataset
from pixelcnnpp.samplers import *
from torchvision.utils import save_image
import shutil
import torch.nn as nn
import torch.nn.functional as F
import numbers
import math
import matplotlib.pyplot as plt
import time

import torchvision
import torch.autograd as autograd


class PixelCNNPPSamplerRunner(object):
    def __init__(self, args, config):
        self.args = args
        self.config = config

    def logit_transform(self, image, lambd=1e-6):
        image = .5 * image + .5
        image = lambd + (1 - 2 * lambd) * image
        latent_image = torch.log(image) - torch.log1p(-image)
        ll = F.softplus(-latent_image).sum() + F.softplus(latent_image).sum() + np.prod(
            image.shape) * (np.log(1 - 2 * lambd) + np.log(.5))
        nll = -ll
        return latent_image, nll

    def train(self):
        assert not self.config.ema, "ema sampling is not supported now"
        self.load_pixelcnnpp()
        self.sample()

    def sample(self):
        sample_batch_size = self.config.batch_size
        self.ar_model.eval()
        model = partial(self.ar_model, sample=True)

        rescaling_inv = lambda x: .5 * x + .5
        rescaling = lambda x: (x - .5) * 2.
        if self.config.dataset == 'CIFAR10' or self.config.dataset == 'celeba':
            x = torch.zeros(sample_batch_size, 3, 32, 32, device=self.config.device)
            clamp = False
            bisection_iter = 20
            basic_sampler = partial(sample_from_discretized_mix_logistic_inverse_CDF, model=model,
                                                nr_mix=self.config.nr_logistic_mix, clamp=clamp,
                                                bisection_iter=bisection_iter)
            # basic_sampler = lambda x: sample_from_discretized_mix_logistic(x, model,
            #                                                                   self.config.nr_logistic_mix,
            #                                                                   clamp=clamp)

        elif 'MNIST' in self.config.dataset:
            x = torch.zeros(sample_batch_size, 1, 28, 28, device=self.config.device)
            clamp = False
            bisection_iter = 30
            basic_sampler = partial(sample_from_discretized_mix_logistic_inverse_CDF_1d, model=model,
                                    nr_mix=self.config.nr_logistic_mix, clamp=clamp,
                                    bisection_iter=bisection_iter)
            # basic_sampler = lambda x: sample_from_discretized_mix_logistic_1d(x, model, self.config.nr_logistic_mix,
            #                                                               clamp=clamp)

        os.makedirs(os.path.join(self.args.log, 'images'), exist_ok=True)
        os.makedirs(os.path.join(self.args.log, 'samples'), exist_ok=True)

        def sigmoid_transform(samples, lambd=1e-6):
            samples = torch.sigmoid(samples)
            samples = (samples - lambd) / (1 - 2 * lambd)
            return samples

        if self.config.reverse_sampling:
            noisy = torch.load(self.config.noisy_samples_path)

            if self.config.with_logit is True:
                images_concat = torchvision.utils.make_grid(sigmoid_transform(noisy), nrow=int(self.config.batch_size ** 0.5), padding=0,
                                                        pad_value=0)
            else:
                images_concat = torchvision.utils.make_grid(rescaling_inv(noisy), nrow=int(self.config.batch_size ** 0.5), padding=0,
                                                        pad_value=0)
            torchvision.utils.save_image(images_concat, os.path.join(self.args.log, 'images', "original_noisy.png"))

        # Sampling from current model
        with torch.no_grad():
            images_array = []
            with torch.no_grad():
                for it in range(self.config.iteration):
                    if self.config.reverse_sampling:
                        x = noisy[it * self.config.batch_size: (it+1) * self.config.batch_size]
                        x = torch.cat([x, x], dim=2)
                    else:
                        x = torch.randn_like(x)

                    x = x.cuda(non_blocking=True)
                    for i in range(-x.shape[-1], 0, 1):
                        for j in range(x.shape[-1]):
                            print(it, i, j, flush=True)
                            samples = basic_sampler(x)
                            x[:, :, i, j] = samples[:, :, i, j]

                        if it == 0:
                            if self.config.with_logit is True:
                                images_concat = torchvision.utils.make_grid(sigmoid_transform(x), nrow=int(x.shape[0] ** 0.5),
                                                                        padding=0, pad_value=0)
                            else:
                                images_concat = torchvision.utils.make_grid(rescaling_inv(x)[:, :, -x.shape[-1]:, :], nrow=int(x.shape[0] ** 0.5),
                                                                        padding=0, pad_value=0)
                            torchvision.utils.save_image(images_concat, os.path.join(self.args.log, 'images', "samples.png"))
                    images_array.append(x)
                    torch.save(torch.cat(images_array, dim=0), os.path.join(self.args.log, 'samples', "samples_{}.pth".format(self.config.dataset)))

                torch.save(torch.cat(images_array, dim=0), os.path.join(self.args.log, 'samples', "samples_{}.pth".format(self.config.dataset)))


    def load_pixelcnnpp(self):
        def load_parallel(path=self.config.ckpt_path, loc=self.config.device):
            checkpoint = torch.load(path, map_location=loc)[0]
            state_dict = checkpoint
            for k in list(state_dict.keys()):
                if not k.startswith('module.'):
                    # remove prefix
                    state_dict["module."+k] = state_dict[k]
                    # delete renamed or unused k
                del state_dict[k]
            return state_dict

        obs = (1, 56, 28) if 'MNIST' in self.config.dataset else (3, 32, 32)
        input_channels = obs[0]

        model = PixelCNN(self.config)
        model = model.to(self.config.device)
        model = torch.nn.DataParallel(model)

        back_compat = False
        if back_compat:
            load_part_of_model(model, self.config.ckpt_path, back_compat=True)
        else:
            model.load_state_dict(torch.load(self.config.ckpt_path, map_location=self.config.device)[0])

        print('model parameters loaded')
        self.ar_model = model

    def test(self):
        pass
