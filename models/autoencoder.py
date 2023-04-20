import torch as tr
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Literal

class AutoEncoder(nn.Module):
    def __init__(self, config: Dict) -> None:
        super().__init__()
        enc_config = config['encoder']
        dec_config = config['decoder']
        self.encoder = Encoder(enc_config)
        self.decoder = Decoder(dec_config)
        self.temb_dim = config['temb_dim']
        self.T = config['T']
        self.emb = nn.Embedding(self.T, self.temb_dim)
        #self.gaussian = gaussian_reparameterize

    def compute_kld_loss(self, log_var, mu):
        log_var = log_var.contiguous().view(-1, 1)
        mu = mu.contiguous().view(-1, 1)
        kld_loss = tr.mean(-0.5 * tr.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        return kld_loss

    def forward(self, x, t):
        x = self.encoder(x)
        #mean, log_var = tr.chunk(x, 2, dim=1)
        #x = self.gaussian(mean, log_var)
        noise = self.emb(t).view(-1, 1, 14, 14)
        x += noise
        x = self.decoder(x)
        return x

class DownSampling(nn.Module):
    def __init__(
            self, 
            in_channels: int=0,
            type: str='conv',
        ) -> None:
        super().__init__()
        self.pad = False
        if type == 'conv':
            assert in_channels != 0, 'in_channels must be specified while using the Convolutional DownSampling'
            self.layer = nn.Conv2d(in_channels, in_channels, 3, 2, 1)
            if in_channels % 2 == 1:
                self.pad = True
        elif type == 'maxpool':
            self.layer = nn.AvgPool2d()
        else:
            raise ValueError('type must be \'conv\' or \'maxpool\'')
    
    def forward(self, x):
        x = self.layer(x)
        if self.pad:
            x = F.pad(x, [0, 1, 0, 1], mode='constant', value=0)
        return x
    
class Encoder(nn.Module):
    """
    The Encoder of the AutoEncoder

    ---------resblock--------
        b * c * h * w 
    ->  b * c_1 * h/2 * w/2
    ->  b * c_2 * h/4 * w/4
    ...
    ...
    ->  b * c_n * h/n * w/n
    -------------------------

    -------bottleneck--------
        b * c_n * h/n * w/n
    ->  b * c_n * h/n * w/n
    -------------------------

    -------embedding---------
        b * c_n * h/n * w/n
    ->  b * 2z * h/n * w/n     # 2z for mean and variance
    -------------------------

    """
    def __init__(
            self,
            config: Dict,
        ) -> None:
        super().__init__()
        self.config = config
        self.n_res_blocks = config['n_res_blocks']
        self.in_channels = config['in_channels']
        self.out_channels = config['out_channels']
        self.downsampling_type = config['downsampling_type']
        self.embedding_dim = config['embedding_dim']

        self.res_blocks = nn.ModuleList()

        for i in range(self.n_res_blocks):
            block = nn.Sequential(
                ResCNNBlock(self.in_channels[i], self.out_channels[i]),
                DownSampling(self.out_channels[i], self.downsampling_type) if i != self.n_res_blocks - 1 else nn.Identity(),
            )
            self.res_blocks.append(block)
        
        self.bottleneck = nn.Sequential(
            ResCNNBlock(self.out_channels[-1], self.out_channels[-1]),
            # todo : add attention
            Attention(self.out_channels[-1]),
            ResCNNBlock(self.out_channels[-1], self.out_channels[-1]),
        )

        self.norm = nn.GroupNorm(num_groups=32, num_channels=self.out_channels[-1])
        self.embeding = nn.Conv2d(self.out_channels[-1], self.embedding_dim, 3, 1, 1)

    def forward(self, x):
        for block in self.res_blocks:
            x = block(x)
        x = self.bottleneck(x)
        x = self.norm(x)
        # todo : figure out why we use swish activation
        x = swish_act(x)
        x = self.embeding(x)
        return x
    
class Decoder(nn.Module):
    """
    The Decoder of the AutoEncoder

    ------de_embedding-------
        b * z * h/n * w/n
    ->  b * c_n * h/n * w/n
    -------------------------

    --------resblock---------
        b * c_n * h/n * w/n
    ->  b * c_n-1 * h/(n//2) * w/(n//2)
    ->  b * c_n-2 * h/(n//4) * w/(n//4)
    ...
    ...
    ->  b * c * h * w
    -------------------------

    -------bottleneck--------
        b * c_n * h/n * w/n
    ->  b * c_n * h/n * w/n
    -------------------------

    """
    def __init__(
            self,
            config: Dict,
        ) -> None:
        super().__init__()
        self.config = config
        self.n_res_blocks = config['n_res_blocks']
        self.in_channels = config['in_channels']
        self.out_channels = config['out_channels']
        self.upsampling_type = config['upsampling_type']
        self.embedding_dim = config['embedding_dim']

        self.de_embeding = nn.Conv2d(self.embedding_dim, self.in_channels[0], 3, 1, 1)
        self.norm = nn.GroupNorm(num_groups=32, num_channels=self.in_channels[0])
        
        self.bottleneck = nn.Sequential(
            ResCNNBlock(self.out_channels[0], self.out_channels[0]),
            # todo : add attention
            Attention(self.out_channels[0]),
            ResCNNBlock(self.out_channels[0], self.out_channels[0]),
        )

        self.res_blocks = nn.ModuleList()

        for i in range(self.n_res_blocks):
            block = nn.Sequential(
                ResCNNBlock(self.in_channels[i], self.out_channels[i]),
                UpSampling(self.out_channels[i], self.upsampling_type) if i != self.n_res_blocks - 1 else nn.Identity(),
            )
            self.res_blocks.append(block)

        self.project = nn.Conv2d(self.out_channels[-1], self.config['final_channels'], 1, 1, 0)

    def forward(self, x):
        x = self.de_embeding(x)
        x = self.bottleneck(x)
        x = self.norm(x)
        x = swish_act(x)
        for block in self.res_blocks:
            x = block(x)
        x = self.project(x)
        return x
    
def gaussian_reparameterize(mean, log_var):
    # x is shape of (b, 2z, h, w)
    std = tr.exp(0.5 * log_var)
    eps = tr.randn_like(std)
    return mean + eps * std

class Attention(nn.Module):
    def __init__(
            self,
            channels: int, # we take the channels as the features's dimension in the attention
        ) -> None:
        super().__init__()
        self.channels = channels
        self.query = nn.Conv2d(channels, channels, 1)
        self.key = nn.Conv2d(channels, channels, 1)
        self.value = nn.Conv2d(channels, channels, 1)

        # see the code of stable diffusion model
        self.proj = nn.Conv2d(channels, channels, 1, 1)

    # todo : imporve the way of reshaping the tensor (don't overuse the size() method)
    def compute_qkv(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        q = q.view(q.size(0), q.size(1), -1)
        k = k.view(k.size(0), k.size(1), -1)
        v = v.view(v.size(0), v.size(1), -1)
        return q, k, v
    
    def compute_att_score(self, q, k):
        att_score = tr.bmm(q.transpose(1, 2), k)
        att_score = F.softmax(att_score, dim=-1)
        return att_score
    
    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.compute_qkv(x)
        att_score = self.compute_att_score(q, k)
        x = tr.bmm(v, att_score.transpose(1, 2))
        x = x.view(b, c, h, w)
        x = self.proj(x)
        return x


class UpSampling(nn.Module):
    def __init__(
            self,
            in_channels: int=0,
            type: Literal['interpolate', 'deconv', 'up'] = 'interpolate',
        ) -> None:
        super().__init__()
        self.type = type
        if type == 'deconv':
            assert in_channels != 0, 'in_channels must be specified while using the Tranposed Convolutional UpSampling'
            self.layer = nn.ConvTranspose2d(in_channels, in_channels, 3, 2, 1, 1)
        elif type == 'interpolate':
            assert in_channels != 0, 'in_channels must be specified while using the Interpolation UpSampling'
            self.layer = nn.Conv2d(in_channels,in_channels, kernel_size=3, stride=1, padding=1)
        elif type == 'up':
            self.layer = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
    def forward(self, x):
        x = self.layer(x)
        if self.type == 'interpolate':
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        return x

class ResCNNBlock(nn.Module):
    def __init__(
            self, 
            in_channels: int, 
            out_channels: int, 
            kernel_size: int=3, 
            stride: int=1, 
            padding: int=1, 
            ) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding)
        self.activation = swish_act
        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=out_channels)
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=out_channels)
    
    def forward(self, x):
        x = self.activation(x)
        x_ = self.conv1(x)
        x = self.norm1(x_)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = x + x_
        return x


def swish_act(x):
    return x * tr.sigmoid(x)


if __name__ == "__main__":
    import yaml
    def load_config(yaml_file):
        f = open(yaml_file, 'r')
        config = yaml.safe_load(f)
        return config
    config = load_config(r'D:\projects\diffusion\config\ddpm_vae.yaml')
    model = AutoEncoder(config['vae'])
    x = tr.randn(10, 3, 256, 256)
    y, mean, log_var = model(x)
    print(y.shape)
    print(model.compute_kld_loss(mean, log_var))