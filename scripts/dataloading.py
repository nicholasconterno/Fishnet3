import torch
import json
from google.cloud import storage
import os

def load_data_json(file_path):
    """
    Load the JSON file from the given file path.
    TODO: Upadate to fit the actual data format
    Args:
        file_path: str, path to the JSON file

    Returns:
        dict: data from the JSON file
    """
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def create_dataloaders(data):
    """
    Create the dataloaders for the training, validation, and test sets.

    Args:
        data: dict, data from the JSON file

    Returns:
        dict: dataloaders for the training, validation, and test sets
    """
    # split data in to train, val, and test using scikit-learn
    # TODO: Implement the split

    # Create the dataloaders
    dataloaders = {
        'train': torch.utils.data.DataLoader(data['train'], batch_size=32, shuffle=True, num_workers=2),
        'val': torch.utils.data.DataLoader(data['val'], batch_size=32, shuffle=False, num_workers=2),
        'test': torch.utils.data.DataLoader(data['test'], batch_size=32, shuffle=False, num_workers=2)
    }
    return dataloaders

def get_file_from_gcp(image_name, destination_location, bucket_name='fishnet3-object-detection'):
    """
    Get the specified file from the GCP bucket.

    Args:
        image_name: str, name of the image file
        bucket_name: str, name of the GCP bucket

    Returns:
        str: path to the image file
    """

    # Download the file from a GCP bucket
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    full_name = f"{image_name}.jpg"
    blob = bucket.blob(full_name)
    blob.download_to_filename(os.path.join(destination_location, full_name))
    return os.path.join(destination_location, full_name)

if __name__ == '__main__':
    # Get the files from the GCP bucket
    image_name = 'da4bce02-db28-11ea-b26e-1f17bea1cdba'
    destination_location = 'data/test_images/'
    new_file = get_file_from_gcp(image_name, destination_location)
    print(f"File downloaded to {new_file}.")