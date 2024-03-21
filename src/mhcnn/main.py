import torch
from torch import nn
import toml

from .dataloader import get_dataloaders, ImageDataset
from .model import CNN
from .training_testing import train_network
from .utils import Result


def run(config_path: str):
    try:
        with open(config_path, "r") as file:
            config = toml.load(file)
    except FileNotFoundError:
        print("The file does not exist or the path is incorrect.")
    except Exception as e:
        print("An error occurred while loading the config file:", e)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainloader, testloader = get_dataloaders(
        config['datapath'], classes=config['classes'],
        device=device, batch_size=config['batch_size']
    )

    model = CNN(
        config['conv_layers'],
        config['linear_layers'],
        in_channels=config['in_channels'],
        out_channels=config['classes']
    )
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    criterion = nn.CrossEntropyLoss()
    result = Result(config['classes'])
    model, optimizer = train_network(
        model, optimizer, criterion,
        trainloader, testloader, result,
        num_epoch=config['num_epochs']
    )


def main():
    datapath = '/Users/timowendner/Programming/MultiHeadCNN/data'
    classes = 10
    batch_size = 16
    lr = 0.0001
    num_epochs = 100
    layers = [32, 64, 64, 64]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
