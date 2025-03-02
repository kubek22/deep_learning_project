from torchvision import datasets, transforms
from torch.utils.data import DataLoader

TRAIN_DIR = "data/train"
VAL_DIR = "data/valid"
TEST_DIR = "data/test"
BATCH_SIZE = 256

mean = [0.47889522, 0.47227842, 0.43047404]
std = [0.24205776, 0.23828046, 0.25874835]

transform = transforms.Compose([
    transforms.Resize((224, 224)), # original 32x32
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

def load_datasets():
    train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=transform)
    val_dataset = datasets.ImageFolder(VAL_DIR, transform=transform)
    test_dataset = datasets.ImageFolder(TEST_DIR, transform=transform)
    return train_dataset, val_dataset, test_dataset

def create_data_loaders(train_dataset, val_dataset, test_dataset):
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
    return train_dataloader, val_dataloader, test_dataloader
