import torch
import torch.nn as nn
from torchvision.models import vgg11, mobilenet_v2

from federated_utils import ResNet18_CIFAR


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


class VGG11_CIFAR(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.backbone = vgg11(weights=None)
        self.backbone.classifier[-1] = nn.Linear(4096, num_classes)

    def forward(self, x):
        return self.backbone(x)


class MobileNetV2_CIFAR(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.backbone = mobilenet_v2(weights=None)
        in_feat = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Linear(in_feat, num_classes)

    def forward(self, x):
        return self.backbone(x)


ARCH_REGISTRY = {
    "resnet18": ResNet18_CIFAR,
    "simplecnn": SimpleCNN,
    "vgg11": VGG11_CIFAR,
    "mobilenetv2": MobileNetV2_CIFAR,
}