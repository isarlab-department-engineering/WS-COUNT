import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from models.PAC import resnet101_PAC
from util import extract_tiles
from models.base_model import BaseModel
# import random
# from scipy.ndimage import zoom
#
# import torchvision.transforms as transforms
# from tqdm import tqdm
# from tensorboardX import SummaryWriter
# import numpy as np
# from matplotlib import pyplot as plt
# from torch.nn.modules.batchnorm import BatchNorm1d
# from util import conditioned_rmse, interval_rmse, init_dataset, from_tensor_to_ndarray_img, \
#     four_crops, extract_tiles, count_from_tiles, extract_loss_weights, class_from_tiles, set_seeds
# from scipy.misc import imresize


class WSCOUNT(BaseModel):

    def __init__(self,subsampled_dim1, subsampled_dim2, subsampled_t4_dim1, subsampled_t4_dim2,
                 subsampled_t16_dim1, subsampled_t16_dim2, supervisor_model=None,
                 # kmax, alpha, num_maps_classifier, supervisor_load_path,
                 num_classes=1, num_maps=8, on_gpu=True):

        super(WSCOUNT, self).__init__()
        self.on_GPU = on_gpu
        model = models.resnet101(True)

        # features net
        self.features = nn.Sequential(
                                      model.conv1,
                                      model.bn1,
                                      model.relu,
                                      model.maxpool,
                                      model.layer1,
                                      model.layer2,
                                      model.layer3,
                                      model.layer4
                                      )

        # classification layer
        num_features = model.layer4[1].conv1.in_channels
        self.conv_maps = nn.Sequential(
                                        nn.Conv2d(num_features, num_classes*num_maps, kernel_size=1, stride=1,
                                                  padding=0, bias=True),
                                        nn.BatchNorm2d(num_classes*num_maps),
                                        nn.ReLU(inplace=True)
                                       )

        self.fcs_input = subsampled_dim1 * subsampled_dim2 * num_maps*num_classes
        self.fcs_input_ts4 = subsampled_t4_dim1 * subsampled_t4_dim2 * num_maps*num_classes
        self.fcs_input_ts16 = subsampled_t16_dim1 * subsampled_t16_dim2 * num_maps*num_classes
        self.fcs_output = 1

        self.regressor = nn.Sequential(
                                       nn.Linear(self.fcs_input, 1000),
                                       nn.BatchNorm1d(1000),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(1000, 1000),
                                       nn.BatchNorm1d(1000),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(1000, self.fcs_output)
                                       )

        self.regressor_ts4 = nn.Sequential(
                                       nn.Linear(self.fcs_input_ts4, 1000),
                                       nn.BatchNorm1d(1000),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(1000, 1000),
                                       nn.BatchNorm1d(1000),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(1000, self.fcs_output)
                                       )

        self.regressor_ts16 = nn.Sequential(
                                       nn.Linear(self.fcs_input_ts16, 1000),
                                       nn.BatchNorm1d(1000),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(1000, 1000),
                                       nn.BatchNorm1d(1000),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(1000, self.fcs_output)
                                       )

        # image classificator for weak supervision
        # self.supervisor = resnet101_PAC(num_classes, pretrained=True, kmax=kmax, alpha=alpha, num_maps=num_maps_classifier)
        # self.supervisor.load_state_dict(torch.load(supervisor_load_path)['state_dict'])

        self.supervisor = supervisor_model

        if self.on_GPU:
            self.supervisor.cuda()

        self.supervisor.eval()
        for params in self.supervisor.parameters():
            params.requires_grad = False

    def supervision(self, img):
        with torch.no_grad():

            # ------scale 3
            x = self.supervisor(img)
            sup = torch.nn.functional.sigmoid(x)

            # ------scale 2
            tiles_4t = extract_tiles(img)
            if self.on_GPU:
                tiles_4t = Variable(tiles_4t.cuda())
            else:
                tiles_4t = Variable(tiles_4t)
            x_4 = self.supervisor(tiles_4t)
            sup_4 = torch.nn.functional.sigmoid(x_4)

            # ------scale 1
            tiles_16t = extract_tiles(tiles_4t)
            if self.on_GPU:
                tiles_16t = Variable(tiles_16t.cuda())
            else:
                tiles_16t = Variable(tiles_16t)
            x_16 = self.supervisor(tiles_16t)
            sup_16 = torch.nn.functional.sigmoid(x_16)

        return sup, sup_4, sup_16

    def forward(self, img):

        # ------scale 3
        x = self.features(img)
        x = self.conv_maps(x)
        x = x.view(-1, self.num_flat_features(x))
        # if(x.shape[-1]==self.fcs_input):
        count = self.regressor(x).clamp(min=0.0)

        # ---- scale 2
        tiles_4t = extract_tiles(img)
        if self.on_GPU:
            tiles_4t = Variable(tiles_4t.cuda())
        else:
            tiles_4t = Variable(tiles_4t)
        x_4 = self.features(tiles_4t)
        x_4 = self.conv_maps(x_4)
        x_4 = x_4.view(-1, self.num_flat_features(x_4))
        count_4t = self.regressor_ts4(x_4).clamp(min=0.0)

        # ------- scale 1
        tiles_16t = extract_tiles(tiles_4t)
        if self.on_GPU:
            tiles_16t = Variable(tiles_16t.cuda())
        else:
            tiles_16t = Variable(tiles_16t)

        x_16 = self.features(tiles_16t)
        x_16 = self.conv_maps(x_16)
        x_16 = x_16.view(-1, self.num_flat_features(x_16))
        count_16t = self.regressor_ts16(x_16).clamp(min=0.0)

        return count, count_4t, count_16t

    # def forward_and_activations(self, x):
    #     x = self.features(x)
    #     # print(self.classifier[0])
    #
    #     ac = self.classifier(x)
    #     #ac = self.classifier[0](x)
    #     #ac = self.classifier[1](ac)
    #     #ac = self.classifier[2](ac)
    #
    #     x = self.classifier(x)
    #     x = x.view(-1, self.num_flat_features(x))
    #     if(x.shape[-1]==self.fcs_input):
    #         x = self.regressor(x)
    #     if(x.shape[-1]==self.fcs_input_ts4):
    #         x = self.regressor_ts4(x)
    #     if(x.shape[-1]==self.fcs_input_ts16):
    #         x = self.regressor_ts16(x)
    #     return ac, x

    # def num_flat_features(self, x):
    #     size = x.size()[1:]  # all dimensions except the batch dimension
    #     num_features = 1
    #     for s in size:
    #         num_features *= s
    #     return num_features

    def get_config_optim(self, lr, lrp):
        return [{'params': self.features.parameters(), 'lr': lr * lrp},
                {'params': self.conv_maps.parameters(), 'lr': lr},
                {'params': self.regressor.parameters(), 'lr': lr},
                {'params': self.regressor_ts4.parameters(), 'lr': lr},
                {'params': self.regressor_ts16.parameters(), 'lr': lr}]
