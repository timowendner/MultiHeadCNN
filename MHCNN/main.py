import numpy as np
# import torch
# from torch import nn
# from torch.utils.data import Dataset, DataLoader
import pickle as pkl

from dataloader import download_dataset


def main():
    train_path = '/Users/timowendner/Programming/MultiHeadCNN/data/train'
    test_path = '/Users/timowendner/Programming/MultiHeadCNN/data/test'
    download_dataset(train_path, test_path)


if __name__ == '__main__':
    main()
