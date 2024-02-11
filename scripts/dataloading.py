import json
from google.cloud import storage
import os
import pandas as pd

def load_data_json(file_path):
    """
    Load the JSON file from the given file path.
    Args:
        file_path: str, path to the JSON file

    Returns:
        dict: data from the JSON file
        df: DataFrame of the data
    """
    label_dict = json.load(open(file_path))
    file_list = list(label_dict.keys())

    # print(f"Loaded {len(file_list)} labels from {file_path}")
    # print(f"Example label: {file_list[0]}")
    # print(f"Example label data: {label_dict[file_list[0]]}")
    # print(f"Example label data no is_fish: {label_dict[file_list[0]][1:][0]}")

    # Create dataframe from the label_dict data (without the is_fish column)
    label_list = []
    for file in file_list:
        label_list.append(label_dict[file][1:][0])

    df = pd.DataFrame({'image': file_list, 'label': label_list})
    print(df.head(2))

    return label_dict, df

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
    # os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "C:/Users/nickj/Documents/Duke/Masters/AIPI540/Fishnet3/fishnet3-56e06381ff35.json"
    # # Get the files from the GCP bucket
    # image_name = 'da4bce02-db28-11ea-b26e-1f17bea1cdba'
    # destination_location = 'data/test_images/'
    # new_file = get_file_from_gcp(image_name, destination_location)
    # print(f"File downloaded to {new_file}.")

    # Load the JSON file
    file_path = 'data/labels.json'
    label_dict, label_df = load_data_json(file_path)