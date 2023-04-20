import torch as tr
import torchvision as tv
import models.unet as unet
from utils.fish_datautils import FishDataset
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid

def valid(model, data, grid_n_row, steps,device):
    model.eval()
    with tr.no_grad():
        model = model.to(device)
        data = data.to(device)
        pred = model(data)
        pred_ = tr.softmax(pred, dim=1)
        pred_ = tr.argmax(pred_, dim=1, keepdim=True)
        pred_ = make_grid(pred_, grid_n_row).float()
        imgs = make_grid(data, grid_n_row).float()
        save_image(pred_, "pred_{}.png".format(steps))
        save_image(imgs, "imgs_{}.png".format(steps))

def train(
        model, dataloader, 
        optimizer, criterion,
        epochs,device=tr.device("cuda:0"), 
        verbose_interval=5,
        valid_img=True
    ):

    model = model.to(device)
    steps = 0
    total_loss = 0.0
    for e in range(epochs):
        for i, item in enumerate(dataloader):
            orig, gt = item

            orig = orig.to(device)
            gt = gt.to(device)
            gt = gt.squeeze(1).long()
            pred = model(orig)
            pred = tr.nn.functional.pad(pred, (7, 7, 6, 7), mode='replicate')
            pred = tr.softmax(pred, dim=1)
            loss = criterion(pred, gt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            steps += 1
            total_loss += loss.cpu().detach().item()
            if steps % verbose_interval == 0:
                print("step:{}, loss: {:.4f}".\
                      format(steps, total_loss/steps))
            if valid_img and steps % 400 == 0:
                valid(model, orig, 1, steps, device)

def main():
    import utils.utils
    lr = 1e-5
    batch_size = 3
    epochs=20
    device = tr.device("cuda:0")
    config = utils.utils.load_config(r'D:\projects\diffusion\config\unet_fish_seg.yaml')
    model = unet.UNet(config['model']) 
    dataloader = DataLoader(FishDataset(r"data\data.json"), batch_size=batch_size, shuffle=True)
    optimizer = tr.optim.Adam(model.parameters(), lr=lr)
    criterion = tr.nn.CrossEntropyLoss() 
    train(model, dataloader, optimizer, criterion, epochs, device)

if __name__ == "__main__":
    main()
