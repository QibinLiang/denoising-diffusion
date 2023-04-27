import torch as tr
import torch.nn as nn
from typing import Dict
import torchvision.transforms.functional as transform_func

class CNNBlock(nn.Module):
    """A base CNN block for Up and Down layers

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        kernel_size (int, optional): kernel size. Defaults to 3.
        stride (int, optional): stride. Defaults to 1.
        bias (bool, optional): bias. Defaults to False.
        padding (int, optional): padding. Defaults to 1.
        activation (nn.Module, optional): activation function. Defaults to nn.ReLU.

    """
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: int = 3, 
        stride: int = 1, 
        bias: bool = False,
        padding: int = 1,
        activation: tr.nn.Module = nn.ReLU,
        ):
        super(CNNBlock, self).__init__()

        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    in_channels[i], 
                    out_channels[i], 
                    kernel_size, 
                    stride, 
                    bias=bias, 
                    padding=padding),
                nn.BatchNorm2d(out_channels[i]),
                activation()
            )
            for i in range(len(in_channels))
        ])
    
    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
        return x
    
class DownSampling(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.downsampler = nn.MaxPool2d(2)

    def forward(self, x):
        return self.downsampler(x)
    
class UpSampling(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            ) -> None:
        super().__init__()
        self.upsampler = nn.ConvTranspose2d(in_channels, out_channels, 2, 2)
    
    def forward(self, x):
        return self.upsampler(x)

def shortcut(
        down_feat: tr.Tensor, 
        up_feat: tr.Tensor
        ):
    """ This function is used to crop the downsampled feature and 
        concatenate it with the upsampled feature

    Args:
        down_feat (torch.Tensor): downsampled feature
        up_feat (torch.Tensor): upsampled feature

    Returns:
        torch.Tensor: concatenated feature
    """
    B_down, C_down, H_down, W_down = down_feat.shape
    B_up, C_up, H_up, W_up = up_feat.shape
    # compute the cropping size 
    h_top = (H_down - H_up) // 2
    w_left = (W_down - W_up) //2
    # crop the feature
    cropped_down_feat = transform_func.crop(down_feat, h_top, w_left, H_up, W_up)
    feat = tr.concat((cropped_down_feat, up_feat), dim=1)
    return feat

class Down(nn.Module):
    def __init__(self, config: Dict) -> None:
        super().__init__()
        self.layers = tr.nn.ModuleList()
        for item in zip(config['in_channels'], config['out_channels']):
            in_channel, out_channel = item
            self.layers.append(CNNBlock(in_channel, out_channel))
            self.layers.append(DownSampling())

        use_time_emb = config['use_time_emb']
        if use_time_emb:
            feat_dim = config['feat_dim']
            emb_dim = config['emb_dim']
            down_inchnnels = config['in_channels']
            n_channel = down_inchnnels[0][0]
            w = h = feat_dim[-1]
            self.bottle_neck_shape = (int(n_channel//2), w, h)
            self.time_emb_proj = TimeEmbProj(int(n_channel//2) * w *h, emb_dim)
    
    def forward(self, x: tr.Tensor, t: tr.Tensor = None):
        x_memory = []
        if t is not None:
            t = self.time_emb_proj(t)
            t = t.reshape(-1, *self.bottle_neck_shape)
            x = tr.cat((x, t), dim=1)
        for _, layer in enumerate(self.layers):
            x_ = layer(x)
            if _ % 2 == 0:
                x_memory.insert(0, x_)
            x = x_
        return x, x_memory

class Up(nn.Module):
    def __init__(self, config: Dict) -> None:
        super().__init__()
        self.layers = tr.nn.ModuleList()
        for item in zip(config['in_channels'], config['out_channels']):
            in_channel, out_channel = item
            self.layers.append(UpSampling(*in_channel))
            self.layers.append(CNNBlock(in_channel, out_channel))
    
    def forward(self, x: tr.Tensor, x_down: tr.Tensor):
        for _, layer in enumerate(self.layers):
            x = layer(x)
            if _ % 2 == 0:
                x = shortcut(x_down[_ // 2], x)
        return x
    
class BottleNeck(nn.Module):
    def __init__(self, config: Dict) -> None:
        super().__init__()
        self.conv = CNNBlock(config['in_channels'], config['out_channels'])
        use_time_emb = config['use_time_emb']
        if use_time_emb:
            feat_dim = config['feat_dim']
            emb_dim = config['emb_dim']
            bn_inchnnels = config['in_channels']
            n_layers = len(bn_inchnnels)
            n_channel = bn_inchnnels[0]
            w = h = feat_dim[-1] // (2 ** 3)
            self.bottle_neck_shape = (int(n_channel//2), w, h)
            self.time_emb_proj = TimeEmbProj(int(n_channel//2) * w *h, emb_dim)
    
    def forward(self, x, t):
        if t is not None:
            t = self.time_emb_proj(t)
            t = t.reshape(-1, *self.bottle_neck_shape)
            x = tr.cat((x, t), dim=1)
        return self.conv(x)
    
class UNet(nn.Module):
    """ UNet model for modeling the posterior distribution of $x_{t-1}$ given $x_{t}$ 
        (https://arxiv.org/abs/1505.04597) 
    
    Args:
        config (Dict): configuration dictionary

    Usage:
        >>> unet = UNet(config)
        >>> x = torch.randn(1, 3, 256, 256)
        >>> x = unet(x)
        >>> x.shape
        torch.Size([1, 3, 256, 256])
        
    """
    def __init__(self, config: Dict) -> None:
        super().__init__()
        self.down = Down(config['down'])
        self.up = Up(config['up'])
        self.bottle_neck = BottleNeck(config['bottle_neck'])
        self.class_conv = nn.Conv2d(config['layer_out_dim'], config['out_dim'], 1, 1)
    
    def forward(self, x, t=None):
        x, xs = self.down(x, t)
        x = self.bottle_neck(x, t)
        x = self.up(x, xs)
        x = self.class_conv(x)
        return x

class TimeEmbProj(nn.Module):
    def __init__(self, proj_dim, time_emb_dim=1024) -> None:
        super().__init__()
        self.proj_1 = nn.Linear(time_emb_dim, proj_dim, bias=False)
        self.norm = nn.LayerNorm(proj_dim)
        self.proj_2 = nn.Linear(proj_dim, proj_dim, bias=False)
    
    def forward(self, x):
        x = self.proj_1(x)
        x = self.norm(x)
        x = self.proj_2(x)
        return x

class UNetWithEmbedding(UNet):
    def __init__(self) -> None:
        super(UNetWithEmbedding, UNet).__init__()