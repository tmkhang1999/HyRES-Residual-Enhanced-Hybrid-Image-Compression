from pathlib import Path

from PIL import Image
import torch.nn as nn
from torch.utils.data import Dataset


class ImageFolder(Dataset):
    """Load an image folder database. Training and testing image samples
    are respectively stored in separate directories:

    code-block::
        - rootdir/
            - train/
                - img000.png
                - img001.png
            - test/
                - img000.png
                - img001.png

    Args:
        root (string): root directory of the dataset
        transform (callable, optional): a function or transform that takes in a
            PIL image and returns a transformed version
        split (string): split mode ('train' or 'val')
    """

    def __init__(self, root, transform=None, split="train"):
        splitdir = Path(root) / split

        if not splitdir.is_dir():
            raise RuntimeError(f'Invalid directory "{root}"')

        self.samples = [f for f in splitdir.iterdir() if f.is_file()]

        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            img: `PIL.Image.Image` or transformed `PIL.Image.Image`.
        """
        img = Image.open(self.samples[index]).convert("RGB")

        # Handle images smaller than crop size by resizing them
        if self.transform and hasattr(self.transform, 'transforms'):
            # Find crop transform and its size
            crop_size = None
            for t in self.transform.transforms:
                if hasattr(t, 'size') and (
                        t.__class__.__name__ == 'RandomCrop' or
                        t.__class__.__name__ == 'CenterCrop'
                ):
                    crop_size = t.size if isinstance(t.size, tuple) else (t.size, t.size)
                    break

            # Resize if image is smaller than crop size
            if crop_size and (img.width < crop_size[0] or img.height < crop_size[1]):
                # Add a small buffer (1 pixel) to avoid rounding issues
                scale = max(crop_size[0] / img.width, crop_size[1] / img.height) * 1.01
                new_width = max(int(img.width * scale), crop_size[0])
                new_height = max(int(img.height * scale), crop_size[1])
                img = img.resize((new_width, new_height), Image.BILINEAR)

        if self.transform:
            return self.transform(img)
        return img

    def __len__(self):
        return len(self.samples)


class CustomDataParallel(nn.DataParallel):
    """Custom DataParallel to access the module methods."""
    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)
