import torch as th
import torch.nn as nn

import numpy as np

from .models import Encoder, Direction, Score

class ContGAN(nn.Module):
    def __init__(self, image_shape, score_condition):
        super().__init__()

        self.image_shape = image_shape
        self.encoder = Encoder(image_shape=image_shape)
        self.direction = Direction(image_shape=image_shape)
        self.score = Score(image_shape=image_shape, condition=score_condition)
    
    def encoder_direction_loss(self, x_0, x_1, t):
        self.encoder.train()
        self.direction.train()
        self.score.eval()

        x_t = self.encoder(x_0=x_0, x_1=x_1, t=t)
        direction = self.direction(x_t=x_t, t=t)
        score = self.score(x_0=x_0, x_1=x_1, x_t=x_t, t=t)

        return (score * (direction - (x_1 - x_0))).mean(dim=0).sum()
    
    def score_loss(self, x_0, x_1, t):
        self.encoder.eval()
        self.direction.eval()
        self.score.train()

        with th.no_grad():
            x_t = self.encoder(x_0=x_0, x_1=x_1, t=t)
            direction = self.direction(x_t=x_t, t=t)
            
        score = self.score(x_0=x_0, x_1=x_1, x_t=x_t, t=t)

        return -(score * (direction - (x_1 - x_0))).mean(dim=0).sum()
    
    def generate_(self, x_1, T):
        dt = 1. / T
        x_t = x_1

        latent_trajectories = [x_1.detach().cpu()]
        for tm in np.linspace(1., dt, T):
            t = th.full((x_1.shape[0],), tm, device=x_1.device)

            with th.no_grad():
                direction = self.direction(x_t=x_t, t=t)
            
            x_t = x_t - dt * direction.detach()
            latent_trajectories.append(x_t.detach().cpu())

        return x_t.clip(-1., 1.), latent_trajectories
    
    def generate(self, n, device, T=100):
        x_1 = th.randn((n, *self.image_shape), device=device)
        return self.generate_(x_1, T=T)