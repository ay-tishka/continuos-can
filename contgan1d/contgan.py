import torch as th
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint

import numpy as np

from .unet import Unet

class ContGAN(nn.Module):
    def __init__(self):
        super().__init__()

        self.direction = Unet()
        self.score = Unet(bounded=True)

    def x_t(self, x_0, x_1, t):
        t = t.reshape(-1, 1)
        return (1. - t) * x_0 + t * x_1
    
    def direction_loss(self, x_0, x_1, t):
        self.direction.train()
        self.score.eval()

        x_t = self.x_t(x_0, x_1, t)
        direction = self.direction(t, x_t)
        score = self.score(t, x_t)

        return (score * (direction - (x_1 - x_0))).mean(dim=0).sum()
    
    def score_loss(self, x_0, x_1, t):
        self.direction.eval()
        self.score.train()

        x_t = self.x_t(x_0, x_1, t)
        with th.no_grad():
            direction = self.direction(t, x_t)
        score = self.score(t, x_t)

        return -(score * (direction - (x_1 - x_0))).mean(dim=0).sum()
    
    @th.no_grad()
    def ode_(self, x_1):
        self.direction.eval()
        return odeint(self.direction, x_1, th.tensor(np.linspace(1., 0., 10), device=x_1.device))
    
    @th.no_grad()
    def ode(self, n, device):
        x_1 = th.randn((n, 1), device=device)
        return self.ode_(x_1)