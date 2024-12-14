from google.cloud import storage
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

def download_blob(blob, source_directory, destination_directory):
    if blob.name.endswith('/'):
        return

    destination_path = os.path.join(destination_directory, blob.name[len(source_directory)+1:])
    
    # Check if the file already exists
    if os.path.exists(destination_path):
        print(f"Skipping {blob.name}, already exists at {destination_path}")
        return
    
    os.makedirs(os.path.dirname(destination_path), exist_ok=True)
    
    blob.download_to_filename(destination_path)
    print(f"Downloaded {blob.name} to {destination_path}")

def download_bucket_directory(bucket_name, source_directory, destination_directory):
    storage_client = storage.Client(project="crystalball-444623")
    bucket = storage_client.bucket(bucket_name)
    blobs = list(bucket.list_blobs(prefix=source_directory))

    with ThreadPoolExecutor() as executor:
        list(tqdm(executor.map(lambda blob: download_blob(blob, source_directory, destination_directory), blobs), 
                  total=len(blobs), desc="Downloading files", unit="file"))

# Example usage
bucket_name = "waymo_open_dataset_v_1_4_3"
source_directory = "individual_files/training"
destination_directory = "../data/waymo_open_dataset_v_1_4_3/individual_files/training"

download_bucket_directory(bucket_name, source_directory, destination_directory)
