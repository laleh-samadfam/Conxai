import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision import models
from PIL import Image
import numpy as np
from torch.nn.utils.rnn import pad_sequence


# Custom dataset to load image sequences and extract features
class ImageSequenceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.sequences = [os.path.join(root_dir, seq) for seq in os.listdir(root_dir)]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence_folder = self.sequences[idx]
        images = []
        for img_name in sorted(os.listdir(sequence_folder)):
            img_path = os.path.join(sequence_folder, img_name)
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            images.append(image)
        #image_tensor = torch.stack(images, dim=0)
        instance = sequence_folder.split(os.path.sep)[-1]
        return images, instance


# Function to extract ResNet features for an image
def extract_resnet_features(image):
    with torch.no_grad():
        features = resnet(image)
    return features.squeeze()


# Load a pre-trained ResNet model for feature extraction
resnet = models.resnet18(pretrained=True)
# Remove the final classification layer
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])
resnet.eval()


root_dir = '../data/processed_data/Task2/train/images'

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

dataset = ImageSequenceDataset(root_dir, transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# Iterate through the DataLoader to extract features for each sequence
for images, instance_number in dataloader:
    for i in range(len(images)):
        features_tensor = extract_resnet_features(images[i])
        # Convert the features tensor to a numpy array
        features = features_tensor.cpu().numpy()

        directory_path = 'train_features/{}'.format(instance_number[0])
        os.makedirs(directory_path, exist_ok=True)

        np.save(os.path.join(directory_path, '{}.npy'.format(i)), features)



