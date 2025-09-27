import torch.nn as nn
import torchvision.models as models

class FaceNetResNet(nn.Module):
    def __init__(self, embedding_size=512):
        super().__init__()
        self.backbone = models.resnet18(weights='IMAGENET1K_V1')
        self.backbone.fc = nn.Identity()
        self.embedding = nn.Linear(512, embedding_size)

    def forward(self, x):
        x = self.backbone(x)
        return self.embedding(x)
