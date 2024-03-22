import torch
from torch import nn
import toml

from .dataloader import get_dataloaders, ImageDataset
from .models.resnet import ResNet
from .models.cnn import SimpleCNN
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
        device=device, batch_size=config['batch_size'],
        data_on_device=config['data_on_device']
    )

    if config['model_type'] == 'cnn':
        Model_type: nn.Module = SimpleCNN
    elif config['model_type'] == 'resnet':
        Model_type: nn.Module = ResNet
    else:
        raise AssertionError('Model type has not been implemented')

    model = Model_type(
        config['conv_layers'],
        config['linear_layers'],
        config['input_shape'],
        classes=config['classes'],
        kernel=config['kernel'],
        block_size=config['block_size'],
        block_count=config['block_count'],
        dropout=config['dropout'],
        padding=config['padding'],
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
    run('config.toml')


if __name__ == '__main__':
    main()
