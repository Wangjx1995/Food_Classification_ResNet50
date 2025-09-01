# Package
from PIL.Image import EXTENSION

import data_augmentation as aug
import cv2
import torch
from torch.utils.data import Dataset
import os
from glob import glob
from sklearn.model_selection import train_test_split

# Parameter
CLASS1 = 'riceball'
CLASS2 = 'bread'
CLASS3 = 'bento'
CLASS4 = 'instant_noodle'
CLASS5 = 'drink'
IMAGE = 'image'
DATA_DIR = "./data"
EXTENSION = "*.jpg"

# FoodDataset define
class FoodDataset(Dataset):
    def __init__(self, image_paths, labels, class_to_idx, augmenters_dict=None, default_transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.class_to_idx = class_to_idx
        self.augmenters_dict = augmenters_dict
        self.default_transform = default_transform

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label_str = self.labels[idx]

        if self.augmenters_dict and label_str in self.augmenters_dict:
            image = self.augmenters_dict[label_str](image=image)[IMAGE]

        else:
            image = aug.default_transform(image=image)[IMAGE]

        image = image.float()
        label = self.class_to_idx[label_str]
        return image, torch.tensor(label, dtype=torch.long)
    def __len__(self):
        return len(self.image_paths)

# Augmenters set up
classes = [CLASS1, CLASS2, CLASS3, CLASS4, CLASS5]
class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
augmenters_by_class = {
    CLASS1: aug.riceball_transform,
    CLASS2: aug.bread_transform,
    CLASS3: aug.bento_transform,
    CLASS4: aug.instant_noodle_transform,
    CLASS5: aug.drink_transform
}
val_transform = aug.default_transform

image_paths = []
labels = []

# Data Load
for cls_name in classes:
    cls_folder = os.path.join(DATA_DIR, cls_name)
    for img_path in glob(os.path.join(cls_folder, EXTENSION)):
        image_paths.append(img_path)
        labels.append(cls_name)

# Data split
train_paths, val_paths, train_labels, val_labels = train_test_split(
    image_paths, labels,
    test_size=0.2,
    stratify=labels,
    random_state=42
)