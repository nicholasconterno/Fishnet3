# %pip install google-cloud-storage

from google.cloud import storage
import os

def upload_images_to_gcs(bucket_name, source_folder):
    '''
    Uploads images from the source folder to the GCS bucket.
    '''
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    
    for filename in os.listdir(source_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            local_path = os.path.join(source_folder, filename)
            blob = bucket.blob(filename)
            
            blob.upload_from_filename(local_path)
            print(f"Uploaded {filename} to '{bucket_name}' bucket.")


if __name__ == '__main__':
    bucket_name = 'fishnet3-object-detection'  # Replace with your GCS bucket name
    source_folder = "data/foid_images_v100/images"
    # Replace with the path to your folder containing images

    upload_images_to_gcs(bucket_name, source_folder)


