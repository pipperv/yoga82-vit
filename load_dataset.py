import os
import numpy as np

from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision import transforms

def list_images(directory_path):
    dir_list = os.listdir(directory_path)
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']  # Add more extensions if needed
    classes = dir_list
    image_files = []
    for directory in dir_list:
        img_dir = directory_path+"/"+directory
        dir_files = [img_dir+"/"+file for file in os.listdir(img_dir) if any(file.lower().endswith(ext) for ext in image_extensions)]
        image_files.append(dir_files)
    return image_files, classes

def verify_img(img_dir):
    try:
        img = Image.open(img_dir)
        img.verify()
        img.close()
        return True
    except:
        return False

def verify_dataset(directory_path='Images'):
    directory_path = directory_path
    image_list, classes = list_images(directory_path)

    total_imgs = 0
    total_valid_imgs = 0
    valid_dataset = []

    for class_imgs in tqdm(image_list):
        total_class_imgs = len(class_imgs)
        total_valid_class_imgs = 0
        for img_name in class_imgs:
            total_imgs+=1
            if verify_img(img_name):
                total_valid_class_imgs+=1
                total_valid_imgs+=1
                valid_dataset.append(img_name)
            else:
                pass
        #print(f"Class {i} ({class_name}): {total_valid_class_imgs}/{total_class_imgs}")

    print(f"Dataset Total: {total_valid_imgs}/{total_imgs}")
    return valid_dataset, classes

class Yoga82(Dataset):

    def __init__(self, dataset_path='Images', transform=None, device='cuda', resize=180) -> None:
        super().__init__()
        self.dataset_path = dataset_path
        self.dataset, self.label_names = verify_dataset(self.dataset_path)
        self.size = len(self.dataset)
        self.transform = transform
        self.device = device
        if resize:
            self.resize = True
            self.reziser = transforms.Resize((resize, resize)).to(self.device)
        else:
            self.resize = False

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        img_path = self.dataset[idx]
        class_name = img_path.split('/')[1]
        img = read_image(img_path).to(self.device)
        label = self.label_names.index(class_name)
        if self.resize:
            img = self.reziser(img)
        if self.transform:
            img = self.transform(img)
        return img, label

    def get_class_name(self, idx):
        return self.label_names[idx]