import os
import pdb
import torch as tr
from typing import Dict
from utils import utils 
from functools import partial
from models.diffusion_vae import DDPM
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from utils import mnist_datautils, mock_datautils

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


# todo : support distributed training
# todo : support pixel-CNN discribed in DDPM paper
# todo : support perceptual compression as introduced in Stable Diffusion
# todo : support more dataset such as church
# todo : support text-to-image model for generating 256 * 256 image

# To get arguments from commandline
def get_args():
    parser = ArgumentParser(description='This is a python script for training the diffusion model. This script currently implement\
                       3 datasets, MNIST, moons and curve. The MNIST is a handwriting number dataset with one channel. Both\
                       moons and curve are the mock dataset that generate a bunch of 2-D data on the fly')
    parser.add_argument('--config', type=str, help='The path of config file', required=True)
    parser.add_argument('--task', type=str, default='mnist',
                        help='The dataset used to train the diffusion model'
                             '(option: \'mnist\', \'moons\', \'curve\')')
    parser.add_argument('--ddp', action='store_true', help='Whether to use distributed training')
    parser.add_argument('--init_method', type=str, default='tcp://')
    parser.add_argument('--rank', type=int, default=0, help='The rank of the process')
    parser.add_argument('--world_size', type=int, default=1, help='The number of processes')
    args = parser.parse_args()
    return args

def train(dataloader:DataLoader, model:DDPM, optimizer:tr.optim.Optimizer, config:Dict, logger=None,device=None, ddp=False):
    utils.check_create_path(config['ckpt'])
    if device == None:
        device = tr.device("cpu")
    steps = 0
    cum_loss = 0
    model.set_device(device)
    for e in range(config['epochs']):
        for _, item in enumerate(dataloader):
            data, label = item
            B = data.shape[0]
            t = tr.randint(0, config['ddpm']['T'], (B, )).to(device)
            data = data.to(device)
            t = t.to(device)
            if ddp:
                data, epsilon = model.module.diffusion(data, t) 
            else:
                data, epsilon = model.diffusion(data, t)   
            epsilon_theta = model(data, t)
            if ddp:
                loss = model.module.loss_fn(epsilon_theta, epsilon)
            else:
                loss = model.loss_fn(epsilon_theta, epsilon)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            cum_loss += loss
            steps += 1
            if steps % config['log_interval'] ==0:
                if logger == None:
                    print("epochs:{}, steps:{}, loss:{:.4f}".format(e, steps, cum_loss.item()/ steps))
                else:
                    logger.info(f"epochs: {e} - steps: {steps} - loss: {cum_loss.item()/ steps:.5f}")
            if steps % config['save_interval'] ==0:
                if ddp:
                    tr.save(model.module, os.path.join(config['ckpt'],"{}.pt".format(steps)))
                else:
                    tr.save(model, os.path.join(config['ckpt'],"{}.pt".format(steps)))
    if ddp:
        tr.save(model.module, os.path.join(config['ckpt'],"final.pt"))
    else:
        tr.save(model, os.path.join(config['ckpt'],"final.pt"))

def select_dataset(task, is_ddp=False):
    assert task in ['mnist', 'moons', 'curve'], '{} is not supported'.format(task)
    if task == 'mnist':
        return partial(mnist_datautils.get_dataloader, path='data', ddp=is_ddp)
    elif task == 'moons':
        return partial(mock_datautils.get_dataloader, task='moons')
    elif task == 'curve':
        return partial(mock_datautils.get_dataloader, task='curve')

def main():
    # init logger
    logger = utils.Logger("log")
    args = get_args()
    config = utils.load_config(args.config)
    if args.task != config['task']:
        logger.warning("The task is incompatible with the configuration, which means it might cause an unexpected error")
    lr = config['lr']
    is_ddp = args.ddp
    if is_ddp:
        logger.info("using distributed data parallel......")
        device = tr.device("cuda:" + str(args.rank))
        dist.init_process_group(backend='nccl', init_method=args.init_method, rank=args.rank, world_size=args.world_size)
        model = DDPM(config)
        model = model.to(device)
        model = DDP(model, find_unused_parameters=True)
        dataloader = select_dataset(args.task)(batch_size=config['batch_size'], )
    else:
        device = tr.device("cuda:0")
        model = DDPM(config)
        model = model.to(device)
        dataloader = select_dataset(args.task)(batch_size=config['batch_size'])
    optimizer = tr.optim.Adam(model.parameters(), lr=lr)
    logger.info("start training")
    train(dataloader, model, optimizer,config=config, logger=logger, device=device)
    logger.info("finish training")

if __name__ == "__main__":
    main()