from torchvision import datasets, transforms
import torchvision.models as models
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import random
import numpy as np

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# make it bigger
BATCH_SIZE = 64 # 128 / 256

TRAIN_DIR = "data/train"
VAL_DIR = "data/valid"
TEST_DIR = "data/test"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cinic_mean = [0.47889522, 0.47227842, 0.43047404]
cinic_std = [0.24205776, 0.23828046, 0.25874835]

transform = transforms.Compose([
    transforms.Resize((224, 224)), # original 32x32
    transforms.ToTensor(),
    transforms.Normalize(mean=cinic_mean,std=cinic_std)
])

train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=transform)
val_dataset = datasets.ImageFolder(VAL_DIR, transform=transform)
test_dataset = datasets.ImageFolder(TEST_DIR, transform=transform)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

print(train_dataset.classes)

# ResNet
# possibly remove some first layers
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
N_CLASSES = 10
model.fc = torch.nn.Linear(model.fc.in_features, N_CLASSES)
model = model.to(device)

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
# maybe set params for Adam
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 2
# measure epoch time
# TODO pack in functions

for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    correct = 0
    n = 0
    for x, y in train_dataloader:
        x, y = x.to(device), y.to(device)

        output = model(x)
        loss = criterion(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        _, predicted = output.max(1)
        correct += predicted.eq(y).sum().item()
        n += y.size(0)

    avg_loss = epoch_loss / n
    accuracy = 100 * correct / n
    print(f"epoch: {epoch + 1}, loss: {avg_loss}, accuracy: {accuracy}")

print("Finished Training")

PATH = "output/resnet.pth"
torch.save(model.state_dict(), PATH)

correct = 0
n = 0
model.eval()
with torch.no_grad():
    for x, y in test_dataloader:
        x, y = x.to(device), y.to(device)
        output = model(x)
        _, predicted = torch.max(output, 1)

        loss = criterion(output, y)
        n += y.size(0)
        correct += predicted.eq(y).sum().item()

        batch_loss = loss.item()

avg_loss = batch_loss / n
accuracy = 100 * correct / n
print(f"loss: {avg_loss}, accuracy: {accuracy}")
