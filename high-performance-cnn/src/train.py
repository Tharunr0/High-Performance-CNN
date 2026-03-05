import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast # KEY for Performance
from . import config

def train_one_epoch(model, loader, optimizer, criterion, scaler):
    model.train()
    loop = tqdm(loader, leave=True)
    total_loss = 0
    correct = 0

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(config.DEVICE, non_blocking=True)
        targets = targets.to(config.DEVICE, non_blocking=True)

        with autocast():
            predictions = model(data)
            loss = criterion(predictions, targets)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

      
        total_loss += loss.item()
        _, predicted = predictions.max(1)
        correct += predicted.eq(targets).sum().item()
        
        loop.set_description(f"Train Loss: {loss.item():.4f}")

    acc = 100. * correct / len(loader.dataset)
    return total_loss / len(loader), acc

def validate(model, loader, criterion):
    model.eval()
    loop = tqdm(loader, leave=True)
    correct = 0
    total_loss = 0

    with torch.no_grad():
        for data, targets in loop:
            data = data.to(config.DEVICE)
            targets = targets.to(config.DEVICE)

            predictions = model(data)
            loss = criterion(predictions, targets)
            
            total_loss += loss.item()
            _, predicted = predictions.max(1)
            correct += predicted.eq(targets).sum().item()
            
            loop.set_description(f"Val Loss: {loss.item():.4f}")

    acc = 100. * correct / len(loader.dataset)
    return total_loss / len(loader), acc