from PIL import Image
from torch.utils.data import Dataset

class Yoga82(Dataset):

    def __init__(self, img_paths, labels, transform=None) -> None:
        super().__init__()
        self.paths, self.labels = img_paths, labels
        self.size = len(self.labels)
        self.transform = transform


    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        img_path = self.paths[idx]
        label = self.labels[idx]

        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        return img, label