from torchvision import datasets, transforms
import torchvision.models as models
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import time
import torch
import random
import numpy as np
from serialization import load, save
from training_functions import train, evaluate

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

# make it bigger
BATCH_SIZE = 256 # 128 / 256

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

print(train_dataset.classes)

# ResNet
# possibly remove some first layers
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
N_CLASSES = 10
model.fc = torch.nn.Linear(model.fc.in_features, N_CLASSES)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
# maybe set params for Adam
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 4
model_path = "output/models/resnet.pth"

start_time = time.time()
print("starting training...")
training_history = train(epochs, model, train_dataloader, val_dataloader, optimizer, criterion, device, model_path)
print("training finished")
print(training_history)
end_time = time.time()
print(f"training time: {end_time - start_time}\n")

print("evaluating model...")
test_accuracy, test_avg_loss = evaluate(model, test_dataloader, criterion, device)
print(f"test loss: {test_avg_loss}, test accuracy: {test_accuracy}")

training_history["accuracy_test"] = test_accuracy
training_history["loss_test"] = test_avg_loss

history_path = "output/history/resnet.pkl"
save(training_history, history_path)
training_history = load(history_path)
