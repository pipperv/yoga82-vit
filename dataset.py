import torch

from PIL import Image
from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode
from torchvision import transforms

class Yoga82(Dataset):

    def __init__(self, img_paths, labels, transform=None, resize=144, num_channels=3 ,device='cuda') -> None:
        super().__init__()
        self.paths, self.labels = img_paths, labels
        self.size = len(self.labels)
        self.transform = transform
        self.device = device
        self.resizer = transforms.Resize((resize, resize))
        self.images = torch.zeros((self.size,num_channels,resize,resize), dtype=torch.float)
        for i, img_path in enumerate(self.paths):
            img = read_image(img_path,ImageReadMode.RGB).to(torch.float) / 255.0
            img = self.resizer(img)
            self.images[i] = img


    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]

        #img = Image.open(img_path).convert("RGB")
        #img = read_image(img_path,ImageReadMode.RGB)
        #img = img.to(float) / 255
        if self.transform:
            img = self.transform(img)

        return img, label