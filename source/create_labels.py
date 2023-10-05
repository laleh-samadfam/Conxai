import pandas as pd
import os
import pandas as pd

def label_number(stages):

    stage_names = stages.split()
    stage_numbers = [stage[6] for stage in stage_names if len(stage) >= 6]
    result_string = ' '.join(stage_numbers)
    return result_string


label_path = '../data/processed_data/clean_labels.csv'
df = pd.read_csv(label_path)

grouped = df.groupby('cj')['label'].apply(lambda x: ' '.join(x)).reset_index()

# Iterate over groups and create text files
for _, row in grouped.iterrows():
    cj_value = row['cj']
    labels_text = row['label']
    print(label_number(labels_text))

    # Create a directory if it doesn't exist
    os.makedirs('train_labels', exist_ok=True)  # Adjust 'output_folder' as needed

    # Write the labels to a text file
    file_path = os.path.join('train_labels', f'{cj_value}.txt')
    with open(file_path, 'w') as f:
        f.write(label_number(labels_text))

    print(f'Text file created for {cj_value} at: {file_path}')
