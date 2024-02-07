import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import json
from PIL import Image
import os
import io
from google.cloud import storage


class FishnetDataset(Dataset):
    def __init__(self, labels_file, bucket_name, gcp_cred_path, transform=None):
        self.label_dict = json.load(open(labels_file))
        self.label_list = list(self.label_dict.keys())
        self.bucket_name = bucket_name
        self.transform = transform
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = gcp_cred_path
        self.storage_client = storage.Client()
        self.bucket = self.storage_client.bucket(bucket_name)

    def __len__(self):
        return len(self.label_dict)

    def __getitem__(self, idx):
        blob = self.bucket.blob(self.label_list[idx] + ".jpg")
        img_bytes = blob.download_as_bytes()

        # Convert bytes to an image
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')

        # Transform, if any
        if self.transform:
            img = self.transform(img)

        # Get the label
        target = self.label_dict[self.label_list[idx]][1]

        return img, target



if __name__ == "__main__": 
    # Create the dataset
    dataset = FishnetDataset(labels_file="../data/labels.json", bucket_name="fishnet3-object-detection", gcp_cred_path="../fishnet3-56e06381ff35.json")

    # Get an image from the dataset
    img, target = dataset[0]
    print(target)
    # Display the image
    print(img)