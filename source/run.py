import torch.nn as nn
import os
from torchvision import datasets, models
import torch
from models import ResnetConxai
from train import train_model


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_dir = '../data/processed data/'

    train_dataset = datasets.ImageFolder(os.path.join(data_dir, "train"))
    valid_dataset = datasets.ImageFolder(os.path.join(data_dir, "val"))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=32, shuffle=True, num_workers=2)

    model = ResnetConxai(7)

    # classification loss function
    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    # Decay LR by a factor of 0.1 every 7 epochs
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    model.to(device)

    train_loss, val_loss, loss_iters, valid_acc = train_model(
        model=model, optimizer=optimizer, scheduler=scheduler, criterion=criterion,
        train_loader=train_loader, valid_loader=valid_loader, num_epochs=30, device=device
    )