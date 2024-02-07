import torch
from torch.utils.data import Dataset
import json
from PIL import Image
import os


class FishnetDataset(Dataset):
    def __init__(self, labels_file, img_dir, transform=None):
        self.label_dict = json.load(open(labels_file))
        self.label_list = list(self.label_dict.keys())
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.label_dict)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.)
        image = Image.open(img_path).convert("RGB")
        # Your annotations file will dictate how you retrieve the following information:
        box_list = self.img_labels.iloc[idx, 1]  # This should be adjusted based on your annotation format
        boxes = torch.as_tensor(box_list, dtype=torch.float32)
        labels = self.img_labels.iloc[idx, 2]  # Adjust based on your needs
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        if self.transform:
            image = self.transform(image)
        return image, target
