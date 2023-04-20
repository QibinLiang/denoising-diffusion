import torch as tr
from typing import Dict
from models.unet import UNet


class DDPM(tr.nn.Module):
    """ Denoising Diffusion Probabilistic Model a generative model for images
        (https://arxiv.org/abs/2006.11239)

    Args:
        config (Dict): configuration dictionary

    Usage:
        >>> ddpm = DDPM(config)
        >>> x = torch.randn(1, 3, 256, 256)
        >>> x = ddpm(x)
        >>> x.shape
        torch.Size([1, 3, 256, 256])
    """
    def __init__(self, config: Dict, device: tr.device = None) -> None:
        super().__init__()
        ddpm_config = config['ddpm']
        unet_config = config['unet']
        self.T = ddpm_config['T']
        self.feat_dim = ddpm_config['feat_dim']
        self.betas = self._linear_beta_scheduler(self.T)
        self.alphas = 1-self.betas
        self.cum_alphas = tr.cumprod(self.alphas, dim=0)
        self.sqrt_cum_alphas = tr.sqrt(self.cum_alphas)
        self.sqrt_one_minus_cum_alphas = tr.sqrt(1 - self.cum_alphas)
        if device is None:
            device = tr.device("cpu")
        self.set_device(device)

        self.emb1 = tr.nn.Embedding(self.T, ddpm_config['emb_dim'])
        self.unet = UNet(unet_config)

    def _linear_beta_scheduler(self, T):
        beta_start = 1000 / T * 0.0001
        beta_end = 1000/ T * 0.02
        betas = tr.linspace(beta_start, beta_end, T)
        return betas
    
    def _sigmoid_beta_scheduler(self, T, beta_start=-6, beta_end=6):
        betas = tr.linspace(beta_start, beta_end, T)
        betas = tr.sigmoid(betas)
        return betas

    def compute_epsilon_theta(self, x_t, t):
        noise = self.emb1(t)
        noise = noise.view(x_t.shape)
        out = self.unet(x_t)
        out = out + noise
        return out
    
    def set_device(self, device):
        self.device=device
        self.betas = self.betas.to(self.device)
        self.alphas = self.alphas.to(self.device)
        self.cum_alphas = self.cum_alphas.to(self.device)
        self.sqrt_cum_alphas = self.sqrt_cum_alphas.to(self.device)
        self.sqrt_one_minus_cum_alphas = self.sqrt_one_minus_cum_alphas.to(self.device)


    def forward(self, x_t, t):
        return self.compute_epsilon_theta(x_t, t)

    def diffusion(self, x_0, t):
        epsilon = tr.randn(x_0.shape)
        epsilon = epsilon.to(self.device)
        x_t = tr.sqrt(self.cum_alphas[t]).view(-1,1,1,1) * x_0 +\
                self.sqrt_one_minus_cum_alphas[t].view(-1,1,1,1)  * epsilon
        return x_t, epsilon

    def loss_fn(self, epsilon_theta, epsilon):
        return (epsilon_theta - epsilon).square().mean()
    
    def sample_x_t_1(self, x_t, t):
        if t > 1:
            z = tr.randn(x_t.shape).to(self.device)
        else:
            z = 0
        std = tr.sqrt(self.betas[t])
        t = tr.tensor([t] * x_t.shape[0]).to(self.device)
        epsilon_theta = self.compute_epsilon_theta(x_t, t)
        coeff = self.betas[t] / self.sqrt_one_minus_cum_alphas[t]
        coeff = coeff.view(x_t.shape[0],1,1,1)
        x_t_1 = 1 / tr.sqrt(self.alphas[t]).view(x_t.shape[0],1,1,1) *\
                (x_t - coeff *epsilon_theta) +\
                std * z
        return x_t_1
    
    def sample(self, T, bs=200):
        x_t = tr.randn(bs, *self.feat_dim).to(self.device)
        x_s = [x_t]
        for t in range(T-1, 1, -1):
            x_s.append(self.sample_x_t_1(x_s[-1], t))
        return x_s