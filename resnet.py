import torchvision.models as models
import time
import torch
import random
import numpy as np
from training_pipeline import repeat_training
from data_loader import load_datasets, create_data_loaders

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset, val_dataset, test_dataset = load_datasets()
train_dataloader, val_dataloader, test_dataloader = create_data_loaders(train_dataset, val_dataset, test_dataset)

N_CLASSES = 10
def init_resnet():
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Linear(model.fc.in_features, N_CLASSES)
    model = model.to(device)
    return model

n = 5
lr = 0.001
model_path = "output/models/resnet/resnet.pth"
history_path = "output/history/resnet/resnet.pkl"
epochs = 20

start_time = time.time()
repeat_training(n, init_resnet, lr, model_path, history_path, epochs, train_dataloader, val_dataloader, test_dataloader, device)
end_time = time.time()
print(f"total time: {end_time - start_time}\n")
