import torch as th
import torch.nn as nn

import numpy as np
from functools import partial

from .models import Encoder, Decoder, Score
from .utils import jvp

class ContGAN(nn.Module):
    def __init__(self, image_shape, decoder_type):
        super().__init__()

        self.image_shape = image_shape
        self.encoder = Encoder(image_shape=image_shape)
        self.decoder = Decoder(image_shape=image_shape, decoder_type=decoder_type)
        self.score = Score(image_shape=image_shape)
    
    def loss(self, x_0, t):
        t.requires_grad_()
        encoder = partial(self.encoder, x_0=x_0)
        x_t, x_t_grad = jvp(encoder, t)

        x_hat, eps_hat = self.decoder(x_t=x_t, t=t)

        t_ = t.detach().clone().requires_grad_()
        encoder_ = partial(self.encoder, x_0=x_hat, eps=eps_hat)
        _, x_t_hat_grad = jvp(encoder_, t_)

        score = self.score(x_t=x_t, t=t)
        return (score * (x_t_hat_grad - x_t_grad)).mean(dim=0).sum()
    
    def encoder_decoder_loss(self, x_0, t):
        self.encoder.train()
        self.decoder.train()
        self.score.eval()

        return self.loss(x_0, t)
    
    def score_loss(self, x_0, t):
        self.encoder.eval()
        self.decoder.eval()
        self.score.train()

        return -self.loss(x_0, t)
    
    def generate_(self, x_1, T):
        dt = 1. / T
        x_t = x_1

        latent_trajectories = [x_1.detach().cpu()]
        prediction_trajectories = []

        for tm in np.linspace(1., dt, T):
            t = th.full((x_1.shape[0],), tm, device=x_1.device)
            
            with th.no_grad():
                x_hat, eps_hat = self.decoder(x_t=x_t, t=t)
                prediction_trajectories.append(x_hat.detach().cpu())

            t_ = t.detach().clone().requires_grad_()
            encoder_ = partial(self.encoder, x_0=x_hat, eps=eps_hat)
            _, x_t_hat_grad = jvp(encoder_, t_, create_graph=False)

            x_t = x_t - dt * x_t_hat_grad.detach()
            latent_trajectories.append(x_t.detach().cpu())

        return x_t.clip(-1., 1.), latent_trajectories, prediction_trajectories
    
    def generate(self, n, device, T=100):
        x_1 = th.randn((n, *self.image_shape), device=device)
        return self.generate_(x_1, T=T)