import torch.nn as nn
import os
import torch
from models import ResnetConxai
from train import train_model
from torchvision import datasets, models, transforms



if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_dir = '../data/processed_data/Task1'
    target_size = (224, 224)

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    train_dataset = datasets.ImageFolder(os.path.join(data_dir, "train"), data_transforms["train"])
    valid_dataset = datasets.ImageFolder(os.path.join(data_dir, "val"), data_transforms["val"])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=4, shuffle=True, num_workers=2)

    model = ResnetConxai(7)

    criterion = nn.CrossEntropyLoss()         # classification loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)       # Decay LR by a factor of 0.1 every 7 epochs

    model.to(device)
    train_loss, val_loss, loss_iters, valid_acc = train_model(
        model=model, optimizer=optimizer, scheduler=scheduler, criterion=criterion,
        train_loader=train_loader, valid_loader=valid_loader, num_epochs=30, device=device
    )