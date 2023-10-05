import os
import pandas as pd
import numpy as np
import re


def parse_filename(file_name):
    pattern = r'cam(\d)_f(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z)\.jpg_(\d+)\.png'

    match = re.match(pattern, file_name)

    if match:
        c = int(match.group(1))
        t = match.group(2)
        j = int(match.group(3))
        return c, t, j
    else:
        return None


def split_task1(clean_data_path):

    # Load image names from the CSV file
    df = pd.read_csv(label_path)

    for index, row in df.iterrows():
        image_name = row['image_name']
        camera_id = row['c']
        camera = 'cam_0'+ str(camera_id)
        label = row['label']
        split = row['split']

        # Construct source and destination paths
        source_path = os.path.join(data_folder_path, camera, image_name)
        destination_path = os.path.join(clean_data_path, split, label)

        # Create the destination directory if it doesn't exist
        os.makedirs(destination_path, exist_ok=True)

        # Move the file to the appropriate directory
        os.rename(source_path, os.path.join(destination_path, image_name))

def split_task2(clean_data_path):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(label_path)

    # Iterate through the DataFrame and move files accordingly
    for index, row in df.iterrows():
        image_name = row['image_name']
        camera_id = row['c']
        camera = 'cam_0'+ str(camera_id)
        instance = str(row['cj'])
        split = str(row['split'])

        # Construct the source and destination paths
        source_path = os.path.join(data_folder_path, camera, image_name)
        destination_path = os.path.join(clean_data_path_2, split, instance)

        # Move the file to the destination directory
        # Create the destination directory if it doesn't exist
        os.makedirs(destination_path, exist_ok=True)

        # Move the file to the appropriate directory
        os.rename(source_path, os.path.join(destination_path, image_name))


if __name__ == '__main__':
    label_path = '../../data/processed_data/clean_labels.csv'
    data_folder_path = '../../data/raw data/foundation_images'
    clean_data_path_1 = '../../data/processed_data/Task1'
    clean_data_path_2 = '../../data/processed_data/Task2'

   # split_task1(clean_data_path_1)
    split_task2(clean_data_path_2)

