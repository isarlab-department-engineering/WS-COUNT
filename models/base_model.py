# import torch
# from torch.autograd import Variable
import torch.nn as nn
# import torch.nn.functional as F
# import torchvision.models as models


class BaseModel(nn.Module):

    def __init__(self):
        super(BaseModel, self).__init__()

    def forward(self, x):
        pass

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def get_config_optim(self, lr, lrp):
        pass
