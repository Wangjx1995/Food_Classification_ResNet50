# Package
import data_loader as dl
import net as nt
import result_export as re
import torch,math
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch import amp
import pandas as pd


# Parameter
CUDA = "cuda"
CPU = "cpu"
NUM_CLASSES = 5
FREEZE_UPTO = 3
BATCH_SIZE = 8
NUM_WORKERS = 0
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
T_MAX = 20
EPOCHS = 20
PATIENCE = 5
NO_IMPROVE = 0
BEST_PATH = "./logs/best_resnet50_food.pth"
BEST_VAL_ACC = -math.inf
RESULT_EXPORT_DIR = "./result"

# Data Load
train_dataset = dl.FoodDataset(
    image_paths=dl.train_paths,
    labels=dl.train_labels,
    class_to_idx=dl.class_to_idx,
    augmenters_dict=dl.augmenters_by_class,
    default_transform=dl.val_transform
)

val_dataset = dl.FoodDataset(
    image_paths=dl.val_paths,
    labels=dl.val_labels,
    class_to_idx=dl.class_to_idx,
    augmenters_dict=None,
    default_transform=dl.val_transform
)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS, pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

# model
device = torch.device(CUDA if torch.cuda.is_available() else CPU)
model = nt.resnet50(num_classes=NUM_CLASSES, pretrained=True, freeze_upto=FREEZE_UPTO)
model = model.to(device)

# .....
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_MAX)
scaler = torch.cuda.amp.GradScaler(enabled=(device.type == CUDA))


def run_epoch(mod, loader, train=True, desc=""):
    if train:
        mod.train()
        torch.set_grad_enabled(True)
    else:
        mod.eval()
        torch.set_grad_enabled(False)

    epoch_loss, correct, total = 0.0, 0, 0
    pbar = tqdm(loader, desc=desc, leave=False)

    for images, targets in pbar:
        images  = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with amp.autocast(device_type=CUDA, enabled=(device.type==CUDA)):
            outputs = mod(images)
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

for epoch in range(1, EPOCHS+1):
    print(f"\nEpoch {epoch}/{EPOCHS}  lr={optimizer.param_groups[0]['lr']:.2e}")

    train_loss, train_acc = run_epoch(model, train_loader, train=True,  desc="Train")
    val_loss,   val_acc   = run_epoch(model, val_loader,   train=False, desc="Val")

    scheduler.step()

    print(f"Train  loss:{train_loss:.4f}  acc:{train_acc:.4f}")
    print(f"Val    loss:{val_loss:.4f}    acc:{val_acc:.4f}")

    if val_acc > BEST_VAL_ACC:
        BEST_VAL_ACC = val_acc
        torch.save(model.state_dict(), BEST_PATH)
        print(f"Saved best to {BEST_PATH} (acc={BEST_VAL_ACC:.4f})")
        NO_IMPROVE = 0
    else:
        NO_IMPROVE += 1
        if NO_IMPROVE >= PATIENCE:
            print("Early stopping.")
            break

model.load_state_dict(torch.load(BEST_PATH, map_location=device))
model.eval(); torch.set_grad_enabled(False)
print("Loaded best weights.")

idx_to_class = {v: k for k, v in dl.class_to_idx.items()}
class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

re.evaluate_and_report(model, val_loader, class_names, device, out_dir=RESULT_EXPORT_DIR)




