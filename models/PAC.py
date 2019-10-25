# import torch.nn as nn
import torchvision.models as models
# import sys
import torch
import torch.nn as nn
# from torch.autograd import Function, Variable
from models.PAC_pooling import WildcatPool2d, ClassWisePool
from models.base_model import BaseModel


class PAC_Classifier(BaseModel):

    def __init__(self, model, num_classes, pooling=WildcatPool2d(), dense=False):
        super(PAC_Classifier, self).__init__()

        self.dense = dense

        self.features = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4)

        # classification layer
        num_features = model.layer4[1].conv1.in_channels
        self.classifier = nn.Sequential(
            nn.Conv2d(num_features, num_classes, kernel_size=1, stride=1, padding=0, bias=True))

        self.spatial_pooling = pooling

        # image normalization
        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        if not self.dense:
            x = self.spatial_pooling(x)
        return x

    def get_config_optim(self, lr, lrp):
        return [{'params': self.features.parameters(), 'lr': lr * lrp},
                {'params': self.classifier.parameters()},
                {'params': self.spatial_pooling.parameters()}]


def resnet101_PAC(num_classes, pretrained=True, kmax=1, kmin=None, alpha=1, num_maps=1):
    model = models.resnet101(pretrained)
    pooling = nn.Sequential()
    pooling.add_module('class_wise', ClassWisePool(num_maps))
    pooling.add_module('spatial', WildcatPool2d(kmax, kmin, alpha))
    return PAC_Classifier(model, num_classes * num_maps, pooling=pooling)


class PAC_Counter(BaseModel):

    def __init__(self, PAC_model,  # supervisor_load_path
                 ):
        super(PAC_Counter, self).__init__()

        # image classificator for weak supervision
        # self.supervisor = resnet101_PAC(num_classes, pretrained=True, kmax=kmax, alpha=alpha, num_maps=num_maps)
        # self.supervisor.load_state_dict(torch.load(supervisor_load_path)['state_dict'])

        self.supervisor = PAC_model
        self.supervisor.eval()
        self.supervisor.cuda()
        for params in self.supervisor.parameters():
            params.requires_grad=False

#     def supervision(self, x):
    def forward(self, x):
        x = self.supervisor(x)
        x = torch.nn.functional.sigmoid(x)
        return x

    def forward_and_activations(self, x):
        cl_maps = self.supervisor.features(x)
        cl_maps = self.supervisor.classifier(cl_maps)

        cl_count = self.supervisor.spatial_pooling(cl_maps)
        cl_count = torch.nn.functional.sigmoid(cl_count)
        return cl_count, cl_maps
