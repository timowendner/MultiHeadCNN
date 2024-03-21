import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision
import os


def download_dataset(datapath: str):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_path = os.path.join(datapath, 'train')
    trainset = torchvision.datasets.CIFAR10(
        root=train_path, train=True, download=True, transform=transform
    )
    test_path = os.path.join(datapath, 'test')
    testset = torchvision.datasets.CIFAR10(
        root=test_path, train=False, download=True, transform=transform
    )
    return trainset, testset


def get_dataloaders(
    datapath: str, classes: int, device: torch.device,
    batch_size: int = 16, data_on_device: bool = False
):
    trainset, testset = download_dataset(datapath)

    trainset = ImageDataset(trainset, classes, device, data_on_device)
    testset = ImageDataset(testset, classes, device, data_on_device)

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=True)

    return trainloader, testloader


class ImageDataset(Dataset):
    def __init__(
        self, dataset: Dataset, classes: int, device: torch.device,
        data_on_device: bool = False
    ) -> None:
        self.dataset = []
        for image, label in dataset:
            label = torch.eye(classes)[label]
            if data_on_device:
                label = label.to(device)
                image = image.to(device)
            self.dataset.append((image, label))
        self.device = device

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        return image.to(self.device), label.to(self.device)
