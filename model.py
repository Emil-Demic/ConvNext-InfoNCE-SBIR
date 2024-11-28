import torch.nn.functional as F

from torch import nn
from torch.nn import AdaptiveAvgPool2d, AdaptiveMaxPool2d
from torchvision.models import vgg16


class SbirModel(nn.Module):
    def __init__(self, pretrained=True):
        super(SbirModel, self).__init__()
        if pretrained:
            from torchvision.models import VGG16_Weights
            self.embedding_net = vgg16(weights=VGG16_Weights.DEFAULT).features
        else:
            self.embedding_net = vgg16().features
        self.num_features = 512
        self.pool = AdaptiveMaxPool2d(1)

    def forward(self, data):
        res1 = self.embedding_net(data[0])
        res2 = self.embedding_net(data[1])
        res1 = self.pool(res1).view(-1, self.num_features)
        res2 = self.pool(res2).view(-1, self.num_features)
        res1 = F.normalize(res1)
        res2 = F.normalize(res2)
        return res1, res2

    def get_embedding(self, data):
        res = self.embedding_net(data)
        res = self.pool(res).view(-1, self.num_features)
        res = F.normalize(res)
        return res
