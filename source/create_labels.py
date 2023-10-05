import pandas as pd
import os
import pandas as pd

def label_number(stages, max_length):

    stage_names = stages.split()
    stage_numbers = [int(stage[6]) for stage in stage_names if len(stage) >= 6]

    stage_numbers = sorted(stage_numbers)

    stage_numbers.extend([-1] * (max_length - len(stage_numbers)))

    return stage_numbers


label_path = '../data/processed_data/clean_labels.csv'
df = pd.read_csv(label_path)

grouped = df.groupby('cj')['label'].apply(lambda x: ' '.join(x)).reset_index()
max_label_length = max(len(row['label'].split()) for _, row in grouped.iterrows())

# Iterate over groups and create text files
for _, row in grouped.iterrows():
    cj_value = row['cj']
    labels_text = row['label']
    label_numbers = label_number(labels_text, max_label_length)
    my_str_list = [str(num) for num in label_numbers]

    # Create a directory if it doesn't exist
    os.makedirs('train_labels', exist_ok=True)  # Adjust 'output_folder' as needed

    # Write the labels to a text file
    file_path = os.path.join('train_labels', f'{cj_value}.txt')
    with open(file_path, 'w') as f:
        f.write(' '.join(my_str_list))

    print(f'Text file created for {cj_value} at: {file_path}')
