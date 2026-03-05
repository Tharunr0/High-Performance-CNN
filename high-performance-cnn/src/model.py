import torch.nn as nn
from torchvision import models
from . import config

def build_model(pretrained=True):
    # Load EfficientNet (Modern & Fast) or ResNet50
    model = models.efficientnet_b0(weights='DEFAULT' if pretrained else None)
    

    for param in model.features.parameters():
        param.requires_grad = False

    in_features = model.classifier[1].in_features
    
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, config.NUM_CLASSES)
    )
    
    return model.to(config.DEVICE)