import torchvision.models as models
import time
import torch
import random
import numpy as np
from training_pipeline import repeat_training
from data_loader import load_datasets, create_data_loaders
import os

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

size = (224, 224)
BATCH_SIZE = 256

train_dataset, val_dataset, test_dataset = load_datasets(size)
train_dataloader, val_dataloader, test_dataloader = create_data_loaders(train_dataset, val_dataset, test_dataset, BATCH_SIZE)

N_CLASSES = 10
def init_resnet():
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    # freezing layers
    # for param in model.parameters():
    #     param.requires_grad = False
    model.fc = torch.nn.Linear(model.fc.in_features, N_CLASSES)
    # for param in model.fc.parameters():
    #     param.requires_grad = True
    model = model.to(device)
    return model

model_dir = "output/models/resnet"
history_dir = "output/history/resnet"

os.makedirs(model_dir, exist_ok=True)
os.makedirs(history_dir, exist_ok=True)

model_path = model_dir + "/resnet.pth"
history_path = history_dir + "/resnet.pkl"

epochs = 10
n = 5
lr = 0.001

start_time = time.time()
repeat_training(n, init_resnet, lr, model_path, history_path, epochs, train_dataloader, val_dataloader, test_dataloader, device)
end_time = time.time()
print(f"total time: {end_time - start_time}\n")
