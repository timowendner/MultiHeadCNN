import torch
import time
import datetime
import numpy as np
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import Optimizer


def train_network(
    model: nn.Module,
    optimizer: Optimizer,
    criterion: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    num_epoch: int = 100,
) -> tuple[nn.Module, Optimizer]:

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params:,}")

    model.train()

    for epoch in range(1, num_epoch+1):
        time_now = datetime.datetime.now()
        time_now = time_now.strftime("%H:%M")
        desc = f'{time_now} Starting Epoch {epoch:>3}'
        for images, targets in tqdm(
            train_loader, desc=f'{desc:<25}', ncols=80
        ):
            outputs = model(images)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        lr = optimizer.param_groups[0]['lr']
        error = test_network(
            model, test_loader, desc='      Testing Network'
        )
        print(f'      current accuracy: {error*100:.2f}%, lr: {lr}\n')
    return model, optimizer


def test_network(
    model: nn.Module,
    dataloader: DataLoader,
    desc: str = None
) -> float:
    n = 0
    correct = 0
    model.eval()
    for images, targets in tqdm(
        dataloader, desc=f'{desc:<25}', ncols=80
    ):
        outputs: torch.tensor = model(images)

        n += images.shape[0]
        outputs = torch.argmax(outputs, dim=1)
        targets = torch.argmax(targets, dim=1)
        correct += sum(outputs == targets)

    model.train()
    return correct / n
