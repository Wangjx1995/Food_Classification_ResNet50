import numpy as np
import torch,math
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import os
from torchvision import models
from tqdm import tqdm
from torch import amp
import pandas as pd

def run_epoch(model, loader, train=True, desc=""):
    if train:
        model.train()
        torch.set_grad_enabled(True)
    else:
        model.eval()
        torch.set_grad_enabled(False)

    epoch_loss, correct, total = 0.0, 0, 0
    pbar = tqdm(loader, desc=desc, leave=False)

    for images, targets in pbar:
        images  = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with amp.autocast(device_type=CUDA, enabled=(device.type==CUDA)):
            outputs = model(images)
            loss = criterion(outputs, targets)

        if train:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        bs = images.size(0)
        epoch_loss += loss.item() * bs
        correct    += (outputs.argmax(1) == targets).sum().item()
        total      += bs

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = epoch_loss / max(1, total)
    acc = correct / max(1, total)
    return avg_loss, acc