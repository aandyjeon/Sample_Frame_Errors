from torch import nn
from torch.utils import model_zoo
from torchvision.models.resnet import BasicBlock, model_urls, Bottleneck
import torch
import math
from torch import nn as nn
from utils.util import *

# [ANDY]
import numpy as np


class ResNet(nn.Module):
    def __init__(self, block, layers, args = None, classes = 7):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.args   = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.conv1  = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)

        # [ANDY] various feature extractors
        if self.args.ours:
            self.dic_extractors = nn.ModuleDict({})
            for i in range(self.args.num_extractors):
                self.inplanes               = 256
                self.layer4                 = self._make_layer(block, 512, layers[3], stride=2)
                self.dic_extractors[str(i)] = self.layer4

            self.p_logvar = nn.Sequential(nn.Linear(self.args.num_extractors * 512 * block.expansion, self.args.num_extractors * 512),
                                          nn.ReLU())
            self.p_mu = nn.Sequential(nn.Linear(self.args.num_extractors * 512 * block.expansion, self.args.num_extractors * 512),
                                      nn.LeakyReLU())

        else:
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

            self.p_logvar = nn.Sequential(nn.Linear(512 * block.expansion, 512),
                                          nn.ReLU())
            self.p_mu = nn.Sequential(nn.Linear(512 * block.expansion, 512),
                                      nn.LeakyReLU())

        self.avgpool = nn.AvgPool2d(7, stride=1)
        if self.args.ours:
            self.class_classifier = nn.Linear(self.args.num_extractors * 512, classes)
        else:
            self.class_classifier = nn.Linear(512, classes)
            
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, gt=None, train=True, classifiy=False, **kwargs):
        end_points = {}
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        # [ANDY] various feature extractors
        if self.args.ours:
            dic_features = {}
            for i in range(self.args.num_extractors):
                dic_features[i] = self.dic_extractors[str(i)](x)
                end_points[f'features_{i}'] = dic_features[i]

            # [ANDY] regularize feature maps to generate different features
            if self.args.regularizer == 'cosine_sim':
                cos  = nn.CosineSimilarity()
                loss_regularizer = 0.
                for i in range(self.args.num_extractors):
                    for j in list(dic_features.values())[i + 1:]:
                        loss_regularizer += torch.sum(cos(dic_features[i], j))

            # [ANDY] concatenate all the features
            x = dic_features[0]
            for i in range(1, self.args.num_extractors):
                x  = torch.cat((x, dic_features[i]), axis = 1)

        else:
            x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        logvar = self.p_logvar(x)
        mu = self.p_mu(x)

        end_points['logvar'] = logvar
        end_points['mu'] = mu

        if train:
            x = reparametrize(mu, logvar)
        else:
            x = mu
        end_points['Embedding'] = x
        x = self.class_classifier(x)
        end_points['Predictions'] = nn.functional.softmax(input=x, dim=-1)
        
        if self.args.ours:
            return x, end_points, loss_regularizer
        else:
            return x, end_points


def resnet18(pretrained=True, args = None, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], args = args, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']), strict=False)
    return model

def resnet50(pretrained=True, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']), strict=False)
    return model
