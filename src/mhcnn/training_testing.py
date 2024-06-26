import torch
import time
import datetime
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import Optimizer

from .utils import Result


def train_network(
    model: nn.Module,
    optimizer: Optimizer,
    criterion: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    result: Result,
    num_epoch: int = 100,
) -> tuple[nn.Module, Optimizer]:

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params:,}")

    model.train()

    for epoch in range(1, num_epoch+1):
        result.register_train()
        time_now = datetime.datetime.now()
        time_now = time_now.strftime("%H:%M")
        desc = f'{time_now} Starting Epoch {epoch:>3}'
        for images, targets in tqdm(
            train_loader, desc=f'{desc:<25}', ncols=80
        ):
            outputs = model(images)
            loss = criterion(outputs, targets)
            result.add(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        lr = optimizer.param_groups[0]['lr']
        result = test_network(
            model, test_loader, result, desc='      Testing Network'
        )
        print(
            f'      train accuracy: {result.acc_train()*100:.2f}%,',
            f'test accuracy: {result.acc_test()*100:.2f}%, lr: {lr}\n'
        )
    return model, optimizer


def test_network(
    model: nn.Module,
    dataloader: DataLoader,
    result: Result,
    desc: str = None,
) -> float:
    result.register_test()
    model.eval()
    for images, targets in tqdm(
        dataloader, desc=f'{desc:<25}', ncols=80
    ):
        outputs: torch.tensor = model(images)
        result.add(outputs, targets)
    model.train()
    return result
