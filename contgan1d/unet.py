import torch as th
import torch.nn as nn

class Unet(nn.Module):
    def __init__(self, bounded=False):
        super().__init__()

        self.bounded = bounded
        self.model = nn.Sequential(
            nn.Linear(2, 64),
            nn.LayerNorm((64,)),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.LayerNorm((64,)),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.LayerNorm((64,)),
            nn.GELU(),
            nn.Linear(64, 1)
        )

    def forward(self, t, x):
        t = t.reshape(-1, 1)
        inp = th.concat([x, t + th.zeros_like(x)], dim=1)

        out = self.model(inp)

        if self.bounded:
            norm = out.norm(p=2, dim=1, keepdim=True)
            out = out / norm.clamp_min(1e-8) * th.tanh(norm.clamp_min(1e-8))
        return out