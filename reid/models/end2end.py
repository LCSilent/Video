from __future__ import absolute_import

from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn import init
import torch
import torchvision
import math

from .resnet import *


__all__ = ["End2End_AvgPooling"]


class AvgPooling(nn.Module):
    def __init__(self, input_feature_size, num_classes, mode, embeding_fea_size=1024, dropout=0.5):
        super(self.__class__, self).__init__()

        is_output_feature = {"Dissimilarity":True, "Classification":False}
        self.is_output_feature = is_output_feature[mode]

        # embeding
        self.embeding = nn.Linear(input_feature_size, embeding_fea_size)
        self.embeding_bn = nn.BatchNorm1d(embeding_fea_size)
        init.kaiming_normal_(self.embeding.weight, mode='fan_out')
        init.constant_(self.embeding.bias, 0)
        init.constant_(self.embeding_bn.weight, 1)
        init.constant_(self.embeding_bn.bias, 0)
        self.drop = nn.Dropout(dropout)

        # classifier
        self.classify_fc = nn.Linear(embeding_fea_size, num_classes)
        init.normal_(self.classify_fc.weight, std = 0.001)
        init.constant_(self.classify_fc.bias, 0)

    def forward(self, inputs):
        avg_pool_feat = inputs.mean(dim = 1)
        # inputs = inputs.permute(0,2,1)
        # avg_pool_feat = F.avg_pool1d(inputs,16)
        # avg_pool_feat = avg_pool_feat.view(16,2048)
        # print(avg_pool_feat.shape)
        if (not self.training) and self.is_output_feature:
            return F.normalize(avg_pool_feat, p=2, dim=1)
        # print(avg_pool_feat.shape)
        # embeding
        net = self.drop(avg_pool_feat)
        net = self.embeding(net)
        net = self.embeding_bn(net)
        net = F.relu(net)

        net = self.drop(net)

        # classifier
        predict = self.classify_fc(net)
        return predict




class End2End_AvgPooling(nn.Module):

    def __init__(self, pretrained=True, dropout=0, num_classes=0, mode="retrieval"):
        super(self.__class__, self).__init__()

        self.CNN = resnet50(dropout=dropout)
        self.avg_pooling = AvgPooling(input_feature_size=2048, num_classes=num_classes, dropout=dropout, mode=mode)

    def forward(self, x):
        assert len(x.data.shape) == 5
        # reshape (batch, samples, ...) ==> (batch * samples, ...)
        oriShape = x.data.shape
        x = x.view(-1, oriShape[2], oriShape[3], oriShape[4])

        # resnet encoding
        x = self.CNN(x)
        x = F.avg_pool2d(x, x.size()[2:])
        resnet_feature = x.view(x.size(0), -1)
        # print(resnet_feature.shape)

        # reshape back into (batch, samples, ...)
        resnet_feature = resnet_feature.view(oriShape[0], oriShape[1], -1)
        # print(resnet_feature.shape)

        # avg pooling
        # if eval and cut_off_before_logits, return predict;  else return avg pooling feature
        predict = self.avg_pooling(resnet_feature)
        return predict

