import torch
import os
import torchvision.utils as utils
from pixelcnnpp.pixelcnnpp import (PixelCNN,
                                   load_part_of_model,
                                   mix_logistic_loss_1d,
                                   mix_logistic_loss,
                                   )

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



class PixelCNNPPGradientSamplerRunner(object):
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

        elif 'MNIST' in self.config.dataset:
            x = torch.zeros(sample_batch_size, 1, 28, 28, device=self.config.device)
            clamp = False
            bisection_iter = 30
            basic_sampler = partial(sample_from_discretized_mix_logistic_inverse_CDF_1d, model=model,
                                    nr_mix=self.config.nr_logistic_mix, clamp=clamp,
                                    bisection_iter=bisection_iter)

        # if os.path.exists(os.path.join(self.args.log, 'images')):
        #     shutil.rmtree(os.path.join(self.args.log, 'images'))
        #
        # if os.path.exists(os.path.join(self.args.log, 'samples')):
        #     shutil.rmtree(os.path.join(self.args.log, 'samples'))

        os.makedirs(os.path.join(self.args.log, 'images'), exist_ok=True)
        os.makedirs(os.path.join(self.args.log, 'samples'), exist_ok=True)


        def sigmoid_transform(samples, lambd=1e-6):
            samples = torch.sigmoid(samples)
            samples = (samples - lambd) / (1 - 2 * lambd)
            return samples

        import pickle
        import torch.autograd as autograd

        noisy = torch.load(self.config.noisy_samples_path)
        if self.config.with_logit is True:
            torchvision.utils.save_image(sigmoid_transform(noisy), os.path.join(self.args.log, 'images', "noisy_samples.png"))
        else:
            torchvision.utils.save_image(rescaling_inv(noisy), os.path.join(self.args.log, 'images', "noisy_samples.png"))
        print(noisy.shape)
        # Sampling from current model

        images_array = []
        for it in range(self.config.iteration):
            print("{}/{}".format(it, self.config.iteration))
            x = noisy[it * self.config.batch_size: (it + 1) * self.config.batch_size]
            # x.requires_grad_(True)
            # output = model(x) #.detach()
            # # x.requires_grad_(True)

            output = model(x).detach()
            x.requires_grad_(True)

            if x.shape[1] == 1:
                log_pdf = mix_logistic_loss_1d(x, output, likelihood=True)
            else:
                log_pdf = mix_logistic_loss(x, output, likelihood=True)

            score = autograd.grad(log_pdf.sum(), x, create_graph=True)[0]
            x = x + self.config.noise ** 2 * score
            x = x.detach().data

            if it == 0:
                if self.config.with_logit is True:
                    images_concat = torchvision.utils.make_grid(sigmoid_transform(x), nrow=int(x.shape[0] ** 0.5),
                                                            padding=0, pad_value=0)
                else:
                    images_concat = torchvision.utils.make_grid(rescaling_inv(x)[:, :, -x.shape[-1]:, :], nrow=int(x.shape[0] ** 0.5),
                                                            padding=0, pad_value=0)
                torchvision.utils.save_image(images_concat, os.path.join(self.args.log, 'images', "gradient_denoised_samples.png"))
            images_array.append(x.data.cpu())
            del(score)

            torch.save(torch.cat(images_array, dim=0), os.path.join(self.args.log, 'samples', "gradient_denoised_samples_{}.pkl".format(self.config.dataset)))

        torch.save(torch.cat(images_array, dim=0),
                       os.path.join(self.args.log, 'samples', "gradient_denoised_samples_{}.pkl".format(self.config.dataset)))


    def load_pixelcnnpp(self):
        def load_parallel(path=self.config.ckpt_path, loc=self.config.device):
            checkpoint = torch.load(path, map_location=loc)[0]
            state_dict = checkpoint
            for k in list(state_dict.keys()):
                if not k.startswith('module.'):
                    #         # remove prefix
                    state_dict["module."+k] = state_dict[k]
                #     # delete renamed or unused k
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
