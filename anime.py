import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig, ax = plt.subplots()
xdata, ydata = [], []
ln, = plt.plot([], [], 'bo',animated=True)

def init():
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    return ln,

def update(imgs):
    ln.set_data(imgs[:, 0], imgs[:, 1])
    return ln,

def get_anime(fig, update, frames):
    anim = animation.FuncAnimation(fig, update, frames,interval=10,
                    init_func=init,blit=True)
    plt.show()
