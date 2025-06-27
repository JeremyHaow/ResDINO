import os, sys, pdb
import kornia
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torch.nn import functional as F
from pytorch_wavelets import DWTForward
from typing import Any, cast, Dict, List, Optional, Union


__all__ = ['DinoResNet', 'dino_resnet50']


# --- EMA Attention Module ---
# GitHub: https://github.com/YOLOonMe/EMA-attention-module
# Paper: https://arxiv.org/abs/2305.13563v2
class EMA(nn.Module):
    def __init__(self, channels, factor=8):
        super(EMA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


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

        self.inplanes = 64

        # --- DWT Layer ---
        self.dwt = DWTForward(J=1, mode='symmetric', wave='bior1.3')

        # --- ResNet Branch Layers ---
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64 , layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        
        resnet_feat_dim = 128 * block.expansion
        
        # --- EMA Attention on ResNet Features ---
        self.ema = EMA(channels=resnet_feat_dim)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # --- DINOv2 Branch ---
        self.dinov2_branch = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
        dino_feat_dim = self.dinov2_branch.embed_dim # 1024 for vitl14
        
        # Freeze DINOv2 parameters
        for param in self.dinov2_branch.parameters():
            param.requires_grad = False
            
        # --- Dynamic Gating for DINOv2 Features ---
        self.gate = nn.Sequential(
            nn.Linear(resnet_feat_dim, 1),
            nn.Sigmoid()
        )

        # --- Fusion and Final Classifier ---
        fused_features_dim = resnet_feat_dim + dino_feat_dim

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
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

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
        # Unpack the dual inputs
        image_aug, image_dino = x
        
        # --- DWT Decomposition on Augmented Image ---
        Yl, Yh = self.dwt(image_aug)

        # --- Branch 1: ResNet with High-Frequency component ---
        # Using the diagonal high-frequency components
        x_resnet = Yh[0][:, :, 2, :, :]
        
        feat_map_resnet = self.conv1(x_resnet)
        feat_map_resnet = self.bn1(feat_map_resnet)
        feat_map_resnet = self.relu(feat_map_resnet)
        feat_map_resnet = self.maxpool(feat_map_resnet)
        feat_map_resnet = self.layer1(feat_map_resnet)
        feat_map_resnet = self.layer2(feat_map_resnet)
        
        # Apply EMA attention
        feat_map_refined = self.ema(feat_map_resnet)

        # Get ResNet feature vector
        feat_vector_resnet = self.avgpool(feat_map_refined).flatten(1)

        # --- Branch 2: DINOv2 with Clean Image ---
        # DINOv2 expects a pre-normalized image
        feat_dino = self.dinov2_branch(image_dino)

        # --- Feature Fusion with Dynamic Gating ---
        gate_weight = self.gate(feat_vector_resnet)
        
        # Apply gate to DINO features
        feat_dino_gated = feat_dino * gate_weight

        # Concatenate features
        feat_fused = torch.cat((feat_vector_resnet, feat_dino_gated), dim=1)

        # --- Final Classification ---
        output = self.classifier(feat_fused)
        
        return output


def dino_resnet50(pretrained=False, **kwargs):
    """
    Constructs a DinoResNet-50 model.
    The `pretrained` flag is unused as DINOv2 is always pre-trained, 
    and the ResNet branch is trained from scratch.
    """
    model = DinoResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        print("Warning: 'pretrained=True' is ignored. DINOv2 is loaded with its own pretrained weights, and the ResNet branch is trained from scratch.")
    return model 