import os, sys, pdb
import kornia
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torch.nn import functional as F
from torchvision import transforms
from pytorch_wavelets import DWTForward
from typing import Any, cast, Dict, List, Optional, Union
import numpy as np

__all__ = ['DinoResNet', 'dino_resnet50']


model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class DinoResNet(nn.Module):

    def __init__(self, block, layers, num_classes=2, zero_init_residual=False):
        super(DinoResNet, self).__init__()

        self.unfoldSize = 2
        self.unfoldIndex = 0
        assert self.unfoldSize > 1
        assert -1 < self.unfoldIndex and self.unfoldIndex < self.unfoldSize*self.unfoldSize
        self.inplanes = 64

        # --- DWT Layer ---
        self.dwt = DWTForward(J=1, mode='symmetric', wave='bior1.3')

        # --- ResNet Branch Layers ---
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64 , layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # --- DINOv2 Branch ---
        self.dinov2_branch = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        self.dino_norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # --- Learnable fusion weight ---
        self.fusion_weight = nn.Parameter(torch.ones(1))

        # --- Fusion and Final Classifier ---
        resnet_out_features = 128 * block.expansion  # 128 * 4 = 512 for Bottleneck
        dino_out_features = self.dinov2_branch.embed_dim  # For vits14 this is 384
        fused_features_dim = resnet_out_features + dino_out_features # 512 + 384 = 896

        self.classifier = nn.Sequential(
            nn.Linear(fused_features_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

        # --- Weight Initialization ---
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        # --- DWT Decomposition ---
        Yl, Yh = self.dwt(x) # Yl: low-freq, Yh: high-freq

        # --- Branch 1: ResNet with High-Frequency component ---
        # Using the diagonal high-frequency components and resizing
        x_resnet = Yh[0][:, :, 2, :, :]
        x_resnet = transforms.Resize([x.shape[-2], x.shape[-1]], antialias=True)(x_resnet)
        
        # Pass through ResNet layers
        feat_resnet = self.conv1(x_resnet)
        feat_resnet = self.bn1(feat_resnet)
        feat_resnet = self.relu(feat_resnet)
        feat_resnet = self.maxpool(feat_resnet)
        feat_resnet = self.layer1(feat_resnet)
        feat_resnet = self.layer2(feat_resnet)
        feat_resnet = self.avgpool(feat_resnet)
        feat_resnet = feat_resnet.view(feat_resnet.size(0), -1)

        # --- Branch 2: DINOv2 with Low-Frequency component ---
        # Resizing the low-frequency component back to original size
        x_dino_unnormalized = transforms.Resize([x.shape[-2], x.shape[-1]], antialias=True)(Yl)
        # Normalize the input specifically for the DINOv2 branch
        x_dino = self.dino_norm(x_dino_unnormalized)
        feat_dino = self.dinov2_branch(x_dino)

        # --- Feature Fusion ---
        # Use the learnable weight to scale the DINOv2 features before concatenation
        feat_fused = torch.cat((feat_resnet, feat_dino * self.fusion_weight), dim=1)

        # --- Final Classification ---
        output = self.classifier(feat_fused)
        
        return output


def dino_resnet50(pretrained=False, **kwargs):
    """
    Constructs a DinoResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model with ImageNet pre-trained weights on the ResNet part.
                           Note: DINOv2 is always pre-trained. This flag is for the ResNet branch.
    """
    model = DinoResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        # This part loads weights for a full ResNet50 and might not match the custom ResNet branch.
        # It's kept here for reference but might need adjustment for partial loading.
        # A more robust solution would be to load state_dict and ignore mismatched keys.
        print("Warning: 'pretrained=True' is not fully implemented for the custom ResNet branch.")
        # state_dict = model_zoo.load_url(model_urls['resnet50'])
        # model.load_state_dict(state_dict, strict=False)
    return model 