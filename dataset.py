import torch

from PIL import Image
from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode

class Yoga82(Dataset):

    def __init__(self, img_paths, labels, transform=None, device='cuda') -> None:
        super().__init__()
        self.dataset, self.labels = img_paths, labels
        self.size = len(self.dataset)
        self.transform = transform
        self.device = device

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        img_path = self.dataset[idx]
        label = self.labels[idx]

        #img = Image.open(img_path).convert("RGB")
        img = read_image(img_path,ImageReadMode.RGB)
        img = img.to(float) / 255
        if self.transform:
            img = self.transform(img)

        return img.to(torch.float).to(self.device), label