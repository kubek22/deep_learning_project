from torchvision import datasets, transforms
import torchvision.models as models
from torch.utils.data import DataLoader
import time
import torch
import random
import numpy as np
from training_pipeline import repeat_training

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

BATCH_SIZE = 256

TRAIN_DIR = "data/train"
VAL_DIR = "data/valid"
TEST_DIR = "data/test"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cinic_mean = [0.47889522, 0.47227842, 0.43047404]
cinic_std = [0.24205776, 0.23828046, 0.25874835]

transform = transforms.Compose([
    transforms.Resize((224, 224)), # original 32x32
    transforms.ToTensor(),
    transforms.Normalize(mean=cinic_mean, std=cinic_std)
])

train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=transform)
val_dataset = datasets.ImageFolder(VAL_DIR, transform=transform)
test_dataset = datasets.ImageFolder(TEST_DIR, transform=transform)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# ResNet
N_CLASSES = 10
def init_resnet():
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Linear(model.fc.in_features, N_CLASSES)
    model = model.to(device)
    return model

n = 5
LR = 0.001
model_path = "output/models/resnet.pth"
history_path = "output/history/resnet.pkl"
epochs = 20

start_time = time.time()
repeat_training(n, init_resnet, LR, model_path, history_path, epochs, train_dataloader, val_dataloader, test_dataloader, device)
end_time = time.time()
print(f"total time: {end_time - start_time}\n")
