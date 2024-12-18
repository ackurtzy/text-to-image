"""
Script to download all NDJSON images from the Quick, Draw! Dataset.

This script automates the process of:
1. Fetching the list of NDJSON files from the Google Cloud Storage bucket for the Quick, Draw! Dataset.
2. Downloading each NDJSON file to a local directory.
3. Converting the downloaded vector images into rasterized images using `convert_and_save`.
4. Saving the rasterized images as NumPy `.npy` files.
5. Removing the original NDJSON files after conversion to save space.

Dependencies:
- `gsutil` command-line tool (Google Cloud SDK).

Output:
    Rasterized images will be saved in `.npy` format under the specified destination directory.

Configuration:
    - `file_list_path`: File to store the list of NDJSON URLs.
    - `destination_dir`: Directory where files are downloaded and processed.
    - `max_images`: Maximum number of images to process per type.
"""

import subprocess
import os

from vector_to_raster_lib import convert_and_save

# Path to the file containing URLs
file_list_path = "simplified_file_list.txt"

# Destination directory to store the downloaded files
destination_dir = "./guessed_quickdraw_data"

# Max images per type
max_images = 56000

# Ensure the destination directory exists
os.makedirs(destination_dir, exist_ok=True)

subprocess.run(
    f"gsutil ls gs://quickdraw_dataset/full/simplified/* > {file_list_path}",
    shell=True,
    check=True,
)

# Read the file line by line and download each file
with open(file_list_path, "r") as file:
    for line in file:
        # Remove any extra whitespace or newlines
        gcs_url = line.strip()

        if gcs_url:  # Ensure the line isn't empty
            print(f"Downloading {gcs_url}...")
            try:
                # Execute the gsutil cp command to download the file
                subprocess.run(["gsutil", "cp", gcs_url, destination_dir], check=True)
                input_path = destination_dir + "/" + gcs_url.split("/")[-1]
                output_path = (
                    destination_dir
                    + "/"
                    + gcs_url.split("/")[-1].replace("ndjson", "npy")
                )
                convert_and_save(input_path, output_path, max_images)
                if os.path.exists(input_path):
                    os.remove(input_path)
            except subprocess.CalledProcessError as e:
                print(f"Failed to download {gcs_url}: {e}")

print("Download complete.")
