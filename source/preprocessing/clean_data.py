from os import walk
import pandas as pd


def missing_data(self):
    label_path = '../../data/raw data/stage_labels.csv'
    data_folder_path = '../../data/raw data/foundation_images'

    # Load image names from the CSV file
    df = pd.read_csv(label_path, sep=';')

    csv_image_names = set(df['image_name'].tolist())

    folder_image_names = set()
    for (dir_path, dir_names, file_names) in walk(data_folder_path):
        for file in file_names:
            if file.lower().endswith('.png'):
                folder_image_names.add(file)

    # Find missing images
    missing_in_csv = folder_image_names.difference(csv_image_names)
    missing_in_folders = csv_image_names.difference(folder_image_names)

    print("Images missing in CSV file:")
    print(missing_in_csv)

    with open("missing_in_csv.txt", "w") as output:
        for item in missing_in_csv:
            output.write(item + '\n')

    print("\nImages missing in folders:")
    print(missing_in_folders)

    with open("missing_in_folders.txt", "w") as output:
        for item in missing_in_folders:
            output.write(item + '\n')


