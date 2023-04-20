import torch as tr
import numpy as np
import sklearn.datasets as skd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, Compose, Normalize
from torchvision.datasets import MNIST
from models.unet import UNet

def get_data():
    n_samples = 10000
    noisy_curve = skd.make_s_curve(n_samples=n_samples, noise=0.1)
    noisy_circles = skd.make_circles(n_samples=n_samples, factor=.5, noise=.05)
    noisy_moons = skd.make_moons(n_samples=n_samples, noise=.05)
    blobs = skd.make_blobs(n_samples=n_samples, random_state=8)
    no_structure = np.random.rand(n_samples, 2), None
    return noisy_curve,noisy_circles, noisy_moons, blobs, no_structure

class MockDataset(Dataset):
    def __init__(self, task):
        if task == 'moons':
            task_id = 2
        elif task == 'curve':
            task_id = 0
        self.data, _ = get_data()[task_id]
    
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        return tr.from_numpy(self.data[idx].astype(np.float32))

def get_dataloader(batch_size, task):
    dataset = MockDataset(task)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader