import torch as th
import torch.nn as nn

from .unet import UNetModel

class Unet(UNetModel):
    def __init__(self, image_size, in_channels, out_channels):
        super().__init__(
            image_size = image_size, 
            in_channels = in_channels,
            model_channels = 64, 
            out_channels = out_channels, 
            num_res_blocks = 1, 
            attention_resolutions = (16,), 
            dropout = 0.0, 
            channel_mult = (1, 2, 2, 2), 
            conv_resample = True, 
            dims = 2, 
            num_classes = None, 
            use_checkpoint = False, 
            use_fp16 = False, 
            num_heads = 4, 
            num_head_channels = -1, 
            num_heads_upsample = -1, 
            use_scale_shift_norm = True, 
            resblock_updown = False, 
            use_new_attention_order = False
        )

class ResidualUnet(nn.Module):
    def __init__(self, image_size, in_channels, out_channels):
        super().__init__()

        self.model = Unet(
            image_size = image_size, 
            in_channels = in_channels, 
            out_channels = out_channels
        )
        self.residual = nn.Conv2d(in_channels, out_channels, *(1, 1, 0)) if in_channels != out_channels else nn.Identity()
    def forward(self, x, t):
        h = self.residual(x)
        return h + self.model(x, t)

class Encoder(nn.Module):
    def __init__(self, image_shape):
        super().__init__()

        assert image_shape[1] == image_shape[2]
        self.image_shape = image_shape
        self.model = ResidualUnet(
            image_size = image_shape[1],
            in_channels = image_shape[0] * 2,
            out_channels = image_shape[0]
        )
    
    def forward(self, t, x_0, eps=None):
        if eps is None:
            eps = th.randn_like(x_0)

        inp = th.concat([x_0, eps], dim=1)
        shift = self.model(inp, t)

        t = t.reshape(-1, 1, 1, 1)
        assert t.requires_grad
        return (1. - t) * x_0 + t * eps + (1. - t) * t * shift
    
class Decoder(nn.Module):
    def __init__(self, image_shape, decoder_type):
        super().__init__()

        assert image_shape[1] == image_shape[2]
        self.image_shape = image_shape

        self.decoder_type = decoder_type
        if self.decoder_type == "fixed":
            in_channels = image_shape[0]
        elif self.decoder_type == "noisy":
            in_channels = image_shape[0] * 2
        elif self.decoder_type == "double_noisy":
            in_channels = image_shape[0] * 3
        else:
            raise ValueError("No such decoder type.")
        
        self.model = ResidualUnet(
            image_size = image_shape[1],
            in_channels = in_channels,
            out_channels = image_shape[0] * 2
        )
    
    def forward(self, x_t, t):
        if self.decoder_type == "fixed":
            inp = x_t
        elif self.decoder_type == "noisy":
            eps = th.randn_like(x_t)
            inp = th.concat([x_t, eps], dim=1)
        elif self.decoder_type == "double_noisy":
            eps = th.randn(x_t.shape[0], x_t.shape[1] * 2, *(x_t.shape[2:]), device=x_t.device)
            inp = th.concat([x_t, eps], dim=1)

        out = self.model(inp, t)
        out, eps_out = out[:, :-self.image_shape[0], :, :], out[:, -self.image_shape[0]:, :, :]
        return out, eps_out

class Score(nn.Module):
    def __init__(self, image_shape):
        super().__init__()

        assert image_shape[1] == image_shape[2]
        self.image_shape = image_shape
        self.model = ResidualUnet(
            image_size = image_shape[1],
            in_channels = image_shape[0],
            out_channels = image_shape[0]
        )

    def forward(self, x_t, t):
        out = self.model(x_t, t)

        norm = out.reshape(out.shape[0], -1).norm(p=2, dim=1).reshape(out.shape[0], 1, 1, 1)
        out = out / norm.clamp_min(1e-8) * th.tanh(norm.clamp_min(1e-8))
        return out
