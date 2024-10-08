
import albumentations as alb
from albumentations.pytorch.transforms import ToTensorV2

from PIL import Image
from torchvision.transforms import transforms
try:
    from .augmentations import *
except Exception:
    from augmentations import *
import sys
sys.path.append('../')
import os

def create_data_transforms_alb(args, split='train'):
    if split == 'train':
        return alb.Compose([
            # alb.HorizontalFlip(),
            alb.Resize(args.image_size, args.image_size),
            alb.Normalize(mean=args.mean, std=args.std),
            ToTensorV2(),
        ])
    elif split == 'val':
        return alb.Compose([
            alb.Resize(args.image_size, args.image_size),
            alb.Normalize(mean=args.mean, std=args.std),
            ToTensorV2(),
        ])
    elif split == 'test':
        return alb.Compose([
            alb.Resize(args.image_size, args.image_size),
            alb.Normalize(mean=args.mean, std=args.std),
            ToTensorV2(),
        ])
def create_data_transforms_alb2(args, split='train'):
    if split == 'train':

        return alb.Compose([
            alb.HorizontalFlip(),
            alb.Rotate(limit=30),
            alb.RandomResizedCrop(256, 256, scale=(0.5, 1.0), p=0.5),
            alb.Resize(args.image_size, args.image_size),
            alb.HorizontalFlip(),
            alb.ToGray(p=0.1),
            alb.GaussNoise(p=0.1),
            alb.Resize(args.image_size, args.image_size),
            alb.Normalize(mean=args.mean, std=args.std),
        ])
    elif split == 'val':
        return alb.Compose([
            alb.Resize(args.image_size, args.image_size),
            alb.Normalize(mean=args.mean, std=args.std),
        ])
    elif split == 'test':
        return alb.Compose([
            alb.Resize(args.image_size, args.image_size),
            alb.Normalize(mean=args.mean, std=args.std),
            ToTensorV2(),
        ])


create_data_transforms = create_data_transforms_alb2

