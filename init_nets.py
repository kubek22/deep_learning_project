import torchvision.models as models
import torch

N_CLASSES = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def init_resnet(device=device):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    # freezing layers
    # for param in model.parameters():
    #     param.requires_grad = False
    model.fc = torch.nn.Linear(model.fc.in_features, N_CLASSES)
    # for param in model.fc.parameters():
    #     param.requires_grad = True
    model = model.to(device)
    return model

def init_efficientnet(device=device):
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
