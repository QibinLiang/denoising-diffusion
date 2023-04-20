import json
import PIL.Image as image
import torchvision as tv
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Pad, Compose


class FishDataset(Dataset):
    def __init__(self, json_path, transform=None) -> None:
        super().__init__()
        json_file = open(json_path, 'r')
        self.data = json.load(json_file)
        self.key_list = list(self.data.keys())
        self.transform = transform
        if transform is None:
            self.transform = Compose([ToTensor()])

        else:
            self.transform = transform
    
    def __len__(self):
        return len(self.key_list)
    
    def __getitem__(self, index):
        key = self.key_list[index]
        orig_path = self.data[key]['orig']
        gt_path = self.data[key]['gt']
        
        #load data
        orig = image.open(orig_path, mode='r')
        gt = image.open(gt_path, mode='r')
        orig = self.transform(orig)
        gt = self.transform(gt)
        return orig, gt