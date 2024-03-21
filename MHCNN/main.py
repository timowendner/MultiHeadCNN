import numpy as np
import torch
from torch import nn
# from torch.utils.data import Dataset, DataLoader
import pickle as pkl

from dataloader import get_dataloaders, ImageDataset
from model import CNN
from training_testing import train_network


def main():
    datapath = '/Users/timowendner/Programming/MultiHeadCNN/data'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainloader, testloader = get_dataloaders(
        datapath, classes=10, device=device, batch_size=16
    )

    model = CNN([0, 1])
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()
    model, optimizer = train_network(
        model, optimizer, criterion,
        trainloader, testloader,
        num_epoch=100
    )


if __name__ == '__main__':
    main()
