import os
import json
import torch as tr
import numpy as np
import torchvision as tv

def preoprocess(folder_path, output="data.json"):
    data = {}
    folders = [os.path.join(folder_path,path) for path in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path,path))]
    for category in folders:
        category_folders = os.listdir(category)
        class_name = os.path.basename(category)
        
        for item in category_folders:
            if item.endswith("GT"):
                gt = item
            else:
                img_folder = item

        img_folder = os.path.join(category, img_folder)
        gt_folder = os.path.join(category, gt)

        for img_file in os.listdir(img_folder):
            img_path = os.path.join(img_folder, img_file)
            gt_path = os.path.join(gt_folder, img_file)
            key = class_name + "_" + img_file.split(".")[0]
            data[key] = {"orig":img_path, "gt":gt_path}
    jdata = json.dumps(data, indent=4)
    with open(output, "w") as f:
        f.write(jdata)

if __name__ == "__main__":
    preoprocess(r"D:\download\finshdata\Fish_Dataset\Fish_Dataset")