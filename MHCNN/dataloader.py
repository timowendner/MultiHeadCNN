# import torch
# import numpy as np
# from IPython.display import Image, Audio
# from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision
# import os
# import glob


def download_dataset(train_path: str, test_path: str):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    trainset = torchvision.datasets.CIFAR10(
        root=train_path, train=True, download=True, transform=transform
    )
    testset = torchvision.datasets.CIFAR10(
        root=test_path, train=False, download=True, transform=transform
    )


# class ImageDataset(Dataset):
#     def __init__(self) -> None:
#         super().__init__()
