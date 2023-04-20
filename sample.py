import torch as tr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from torchvision.utils import make_grid

# todo : refector the sampling to an exectuable script
# todo : add argparser
T = 1000
device = tr.device("cuda:0")
model = tr.load("ckpt/ddpm_mnist/model.pt")
model.set_device(device)
model.to(device)

model.eval()
with tr.no_grad():
    # todo : support modifying the image grid
    samples = model.sample(T, 25)
    fig= plt.figure(figsize=(12,12))
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    plt.axis(False)
    sample = samples[0]
    sample = make_grid(sample, 5)
    sample = (sample.permute(1,2,0).detach().cpu().numpy() * 255).astype(np.int32)
    im = plt.imshow(sample,cmap='gray')
    def update(frame):            
        if frame >399:
            sample = samples[-1]
            sample = make_grid(sample, 5)
            img = sample.permute(1,2,0).detach().cpu().numpy()
            img = (img * 255).astype(np.int32)
            img = np.clip(img, a_min=0, a_max=255)
            if frame == 459:
                plt.savefig('asset/final.png')
            return [im.set_array(img)]
        else:
            sample = samples[-400+frame]
            sample = make_grid(sample, 5)
            img = sample.permute(1,2,0).detach().cpu().numpy()
            img = (img * 255).astype(np.int32)
            img = np.clip(img, a_min=0, a_max=255)
            return [im.set_array(img)]
    # todo : support customizing the number of frame and the start frame
    anim = animation.FuncAnimation(fig, update, frames=460,interval=20)
    anim.save("asset/anime.gif")