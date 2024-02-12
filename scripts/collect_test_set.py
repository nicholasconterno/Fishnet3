import json
from PIL import Image
import os
import io
import pandas as pd
from google.cloud import storage


def collect_test_set(img_output_dir: str = "../data/test_imgs", 
                     label_output_dir: str = "../data",
                     size: int = 2000) -> None:
    """
    Collect the test set from the GCP bucket and save the labels to a JSON file.

    Args:
    - img_output_dir: str, the directory to save the test images.
    - label_output_dir: str, the directory to save the test labels.
    - size: int, the number of images to download.
    """
    # Set up GCP credentials
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "../fishnet3-56e06381ff35.json"
    storage_client = storage.Client()
    bucket = storage_client.bucket("fishnet3-object-detection")

    # Open the fishnet_labels_in_bucket.csv to get all labels
    labels_df = pd.read_csv("../data/fishnet_labels_in_bucket.csv")

    # Load the labels used for training and validation
    with open("../data/labels.json", "r") as f:
        train_labels_dict = json.load(f)
    train_imgid_list = list(train_labels_dict.keys())

    # Select "size" number of labels for the test set. No overlap in img_id with train and validation sets.
    # Equal number of images from each cam_id
    test_imgid_list = []
    cam_ids = labels_df["cam_id"].unique()
    for cam_id in cam_ids:
        cam_imgids = labels_df[labels_df["cam_id"] == cam_id]["img_id"].tolist()
        cam_imgids = [imgid for imgid in cam_imgids if imgid not in train_imgid_list]
        cam_imgids = list(set(cam_imgids))
        test_imgid_list.extend(cam_imgids[:size // len(cam_ids)])
    test_imgid_list = test_imgid_list[:size]

    # Create a dict to save the test labels
    test_labels_dict = {}
    for imgid in test_imgid_list:
        label_data = labels_df[labels_df["img_id"] == imgid]
        formatted_data = []
        for i in range(len(label_data)):
            x_data = [str(label_data.iloc[i]["x_min"]), str(label_data.iloc[i]["x_max"])]
            y_data = [str(label_data.iloc[i]["y_min"]), str(label_data.iloc[i]["y_max"])]
            box_label = label_data.iloc[i]["label_l1"]
            curr_box = [x_data, y_data, box_label]
            formatted_data.append(curr_box)
        test_labels_dict[imgid] = formatted_data
    
    # Save the test labels to a JSON file
    with open(os.path.join(label_output_dir, "test_labels.json"), "w") as f:
        json.dump(test_labels_dict, f)

    # Download the test images
    if not os.path.exists(img_output_dir):
        os.makedirs(img_output_dir)
    c = 0
    for imgid in test_imgid_list:
        img_path = os.path.join(img_output_dir, imgid + ".jpg")
        blob = bucket.blob(imgid + ".jpg")
        blob.download_to_filename(img_path)
        c += 1
        if c % 100 == 0:
            print(f"Downloaded {c}/{size} images")


if __name__ == "__main__":
    collect_test_set(img_output_dir="../data/test_imgs",
                     label_output_dir="../data",
                     size=2000)
    

    