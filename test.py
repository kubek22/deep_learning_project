from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import torch
import random
import numpy as np

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

BATCH_SIZE = 64

TRAIN_DIR = "data/train"
VAL_DIR = "data/valid"
TEST_DIR = "data/test"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=transform)
val_dataset = datasets.ImageFolder(VAL_DIR, transform=transform)
test_dataset = datasets.ImageFolder(TEST_DIR, transform=transform)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

print(train_dataset.classes)

for x, y in train_dataloader:
    x, y = x.to(device), y.to(device)
    print(x)
    print(y)
    break

