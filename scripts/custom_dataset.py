from torch.utils.data import Dataset
import json
from PIL import Image
import os
import io
from google.cloud import storage


class FishnetDataset(Dataset):
    def __init__(self, labels_file, bucket_name, gcp_cred_path, transform=None, resize_shape: tuple = (400, 800), download_data: bool = False):
        self.label_dict = json.load(open(labels_file))
        self.label_list = list(self.label_dict.keys())
        self.bucket_name = bucket_name
        self.transform = transform
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = gcp_cred_path
        self.storage_client = storage.Client()
        self.bucket = self.storage_client.bucket(bucket_name)
        self.resize_shape = resize_shape
        self.download_data = download_data
        self.data_dir = "../data/dataset"

        if self.download_data:
            os.makedirs(self.data_dir, exist_ok=True)
            # Download and save data locally
            total_files = len(self.label_list)
            c = 0
            for label in self.label_list:
                img_path = os.path.join(self.data_dir, label + ".jpg")
                blob = self.bucket.blob(label + ".jpg")
                blob.download_to_filename(img_path)
                c += 1
                if c % 100 == 0:
                    print(f"Downloaded {c}/{total_files} images")
            

    def __len__(self):
        return len(self.label_dict)

    def __getitem__(self, idx):
        if self.download_data:
            img_path = os.path.join(self.data_dir, self.label_list[idx] + ".jpg")
            img = Image.open(img_path).convert('RGB')
        else:
            blob = self.bucket.blob(self.label_list[idx] + ".jpg")
            img_bytes = blob.download_as_bytes()
            img = Image.open(io.BytesIO(img_bytes)).convert('RGB')

        # Find the resizing scales for x and y
        x_scale = self.resize_shape[1] / img.size[0]
        y_scale = self.resize_shape[0] / img.size[1]

        # Transform, if any
        if self.transform:
            img = self.transform(img)

        # Get the label
        target = self.label_dict[self.label_list[idx]][1]

        # Scale the bounding boxes
        for i in range(len(target)):
            for j in range(2):
                target[i][0][j] = str(int(int(target[i][0][j]) * x_scale))
                target[i][1][j] = str(int(int(target[i][1][j]) * y_scale))

        # Pad the target with missing values
        while len(target) < 20:
            target.append([["-1","-1"], ["-1","-1"], "Missing"])

        return img, target




if __name__ == "__main__": 
    # Create the dataset
    dataset = FishnetDataset(labels_file="../data/labels.json", bucket_name="fishnet3-object-detection", gcp_cred_path="../fishnet3-56e06381ff35.json")

    # Get an image from the dataset
    img, target = dataset[0]
    print(target)
    # Display the image
    print(img)