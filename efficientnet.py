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
# smaller batch size maybe
# BATCH_SIZE = 256
BATCH_SIZE = 128

train_dataset, val_dataset, test_dataset = load_datasets(size)
train_dataloader, val_dataloader, test_dataloader = create_data_loaders(train_dataset, val_dataset, test_dataset, BATCH_SIZE)

N_CLASSES = 10
def init_efficientnet():
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    # freezing layers
    # for param in model.parameters():
    #     param.requires_grad = False
    num_features = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(num_features, N_CLASSES)
    # for param in model.classifier.parameters():
    #     param.requires_grad = True
    model = model.to(device)
    return model


model_dir = "output/models/efficientnet"
history_dir = "output/history/efficientnet"

os.makedirs(model_dir, exist_ok=True)
os.makedirs(history_dir, exist_ok=True)

model_path = model_dir + "/efficientnet.pth"
history_path = history_dir + "/efficientnet.pkl"

n = 5
lr = 0.001
epochs = 10
# epochs may be decreased to 5

start_time = time.time()
repeat_training(n, init_efficientnet, lr, model_path, history_path, epochs, train_dataloader, val_dataloader, test_dataloader, device)
end_time = time.time()
print(f"total time: {end_time - start_time}\n")
