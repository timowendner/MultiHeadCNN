import torch
from torch import nn

from .dataloader import get_dataloaders, ImageDataset
from .model import CNN
from .training_testing import train_network
from .utils import Result


def main():
    datapath = '/Users/timowendner/Programming/MultiHeadCNN/data'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    classes = 10
    batch_size = 16
    lr = 0.0001
    num_epochs = 100
    layers = [32, 64, 64, 64]
    trainloader, testloader = get_dataloaders(
        datapath, classes=classes,
        device=device, batch_size=batch_size
    )

    model = CNN(layers)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    result = Result(classes)
    model, optimizer = train_network(
        model, optimizer, criterion,
        trainloader, testloader, result,
        num_epoch=num_epochs
    )


if __name__ == '__main__':
    main()
