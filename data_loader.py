from torchvision import datasets, transforms, torch
from torch.utils.data import DataLoader
import torch.nn
import random

TRAIN_DIR = "data/train"
VAL_DIR = "data/valid"
TEST_DIR = "data/test"


mean = [0.47889522, 0.47227842, 0.43047404]
std = [0.24205776, 0.23828046, 0.25874835]

class Cutout(torch.nn.Module):
    def __init__(self, num_holes=1, size=16):
        super().__init__()
        self.num_holes = num_holes
        self.size = size
    
    def __call__(self, img):
        h, w = img.shape[1:]
        mask = torch.ones_like(img)
        
        for _ in range(self.num_holes):
            y = random.randint(0, h - self.size)
            x = random.randint(0, w - self.size)
            mask[:, y:y+self.size, x:x+self.size] = 0
        return img * mask

class CombinedDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform_original, transform_augmented):
        self.dataset = datasets.ImageFolder(root)
        self.transform_original = transform_original
        self.transform_augmented = transform_augmented
    
    def __len__(self):
        return len(self.dataset) * 2 
    
    def __getitem__(self, idx):
        img, label = self.dataset[idx % len(self.dataset)]
        if idx < len(self.dataset):
            return self.transform_original(img), label 
        else:
            return self.transform_augmented(img), label

def load_datasets(size, apply_rotation=False, apply_blur=False, apply_brightness=False, apply_cutout=False,
                  rotation = 30, brightness = 0.3, blur_size = 3, num_holes=1, hole_size = 6):
    original_transforms = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    augmented_transforms = [transforms.Resize(size)]
    
    if apply_rotation:
        augmented_transforms.append(transforms.RandomRotation(degrees=rotation))
    if apply_brightness:
        augmented_transforms.append(transforms.ColorJitter(brightness=brightness))
    if apply_blur:
        augmented_transforms.append(transforms.GaussianBlur(kernel_size=blur_size))
    augmented_transforms.extend([
        transforms.ToTensor(),
    ])

    if apply_cutout:
        augmented_transforms.append(Cutout(num_holes=num_holes, size=hole_size))
    
    augmented_transforms.extend([
        transforms.Normalize(mean=mean, std=std)
    ])

    
    train_dataset = CombinedDataset(TRAIN_DIR, original_transforms, transforms.Compose(augmented_transforms))
    
    val_test_transforms = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    val_dataset = datasets.ImageFolder(VAL_DIR, transform=val_test_transforms)
    test_dataset = datasets.ImageFolder(TEST_DIR, transform=val_test_transforms)
    
    return train_dataset, val_dataset, test_dataset

def create_data_loaders(train_dataset, val_dataset, test_dataset, batch_size):
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_dataloader, val_dataloader, test_dataloader

