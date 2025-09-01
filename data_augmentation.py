
# Package
from albumentations import (
    Compose, Resize, HorizontalFlip, Rotate, RandomGamma,
    RandomBrightnessContrast, GaussianBlur, GaussNoise,
    CoarseDropout, MultiplicativeNoise, HueSaturationValue,
    Normalize
)
from albumentations.pytorch import ToTensorV2

# Transformer for riceball

riceball_transform = Compose([
    Resize(224, 224),
    HorizontalFlip(p=0.5),
    RandomGamma(gamma_limit=(80, 120), p=0.5),
    Rotate(limit=10, p=0.7),
    Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
    ToTensorV2()
])

# Transformer for bread

bread_transform = Compose([
    Resize(224, 224),
    HorizontalFlip(p=0.5),
    MultiplicativeNoise(multiplier=(0.9, 1.1), p=0.5),
    GaussianBlur(blur_limit=(3, 7), p=0.5),
    Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
    ToTensorV2()
])

# Transformer for bento

bento_transform = Compose([
    Resize(224, 224),
    MultiplicativeNoise(multiplier=(0.85, 1.15), p=0.5),
    CoarseDropout(num_holes_range=(3,5), hole_height_range=(15,20), hole_width_range=(15,20), p=0.5),
    GaussianBlur(blur_limit=(3, 5), p=0.5),
    Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
    ToTensorV2()
])

# Transformer for instant_noodle

instant_noodle_transform = Compose([
    Resize(224, 224),
    GaussNoise(std_range=(0.3, 0.7), p=0.5),
    Rotate(limit=8, p=0.5),
    Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
    ToTensorV2()
])

#Transformer for drink

drink_transform = Compose([
    Resize(224, 224),
    HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
    GaussianBlur(blur_limit=(3, 5), p=0.5),
    CoarseDropout(num_holes_range=(3,5), hole_height_range=(15,20), hole_width_range=(15,20), p=0.5),
    Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
    ToTensorV2()
])

#default_transformer

default_transform = Compose([
    Resize(224, 224),
    Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
    ToTensorV2()
])