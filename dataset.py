import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np


class RGBDdataset(Dataset):
    def __init__(self, image_dir, mask_dir, depth_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.depth_dir = depth_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index])
        depth_path = os.path.join(self.depth_dir, self.images[index])
        image = np.array(Image.open(img_path).convert("RGB"))
        depth = np.array(Image.open(depth_path).convert("L"), dtype=np.float32)

        image = np.dstack((image, depth))
        mask = np.array(Image.open(mask_path).convert("RGB"))

        # mask[mask == 255.0] = 1.0

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask
