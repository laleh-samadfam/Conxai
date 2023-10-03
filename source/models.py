from torch import nn
import torch
from torchvision import models


class ResnetConxai(torch.nn.Module):

    def __init__(self, num_classes=7):
        super(ResnetConxai, self).__init__()
        resnet = models.resnet18(pretrained=True)  # https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

        # Freeze the pre-trained layers (optional)
        for param in resnet.parameters():
            param.requires_grad = False

        # replacing classification head
        num_nuerons = resnet.fc.in_features
        resnet.fc = nn.Linear(num_nuerons, num_classes)
        self.resnet = resnet

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

