import os, sys, pdb  # 导入基础模块：操作系统接口(os)、系统参数和函数(sys)、Python调试器(pdb)
import kornia  # 导入计算机视觉库Kornia，提供了图像处理的各种工具，比如边缘检测
import torch  # 导入PyTorch库，这是一个深度学习框架，提供张量计算和神经网络搭建功能
import torch.nn as nn  # 导入神经网络模块，包含了各种网络层的定义，如卷积层、全连接层等
import torch.utils.model_zoo as model_zoo  # 导入模型下载工具，用于获取预训练模型
from torch.nn import functional as F  # 导入函数式API，提供激活函数、池化等操作
from torchvision import transforms  # 导入图像变换工具，用于调整图像大小、裁剪等处理
from typing import Any, cast, Dict, List, Optional, Union  # 导入类型注解工具，帮助代码更清晰
import numpy as np  # 导入NumPy库，提供多维数组处理和数学运算功能
from pytorch_wavelets import DWTForward, DWTInverse

# 定义可以从此模块导入的类和函数名称列表，外部可以通过 from resnet import * 获取这些内容
__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


# 预训练模型的URL字典，用于下载已经在ImageNet数据集上训练好的模型权重
# ImageNet是一个包含数百万张图像的大型数据集，这些预训练模型可以识别上千种物体
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',  # ResNet-18模型的权重文件链接
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',  # ResNet-34模型的权重文件链接
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',  # ResNet-50模型的权重文件链接
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',  # ResNet-101模型的权重文件链接
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',  # ResNet-152模型的权重文件链接
}


def conv3x3(in_planes, out_planes, stride=1):
    """
    创建一个3x3卷积层，带有填充
    
    卷积层是神经网络处理图像的基本单元，可以提取图像特征。
    例如：边缘、纹理、形状等特征。
    
    参数说明:
        in_planes: 输入的特征通道数（比如RGB图像是3通道）
        out_planes: 输出的特征通道数（由网络设计者决定）
        stride: 卷积滑动步长，控制输出特征图的大小
        
    举例：
        如果输入是一张224x224的RGB图像(3通道)，使用64个3x3卷积核，步长为1：
        conv3x3(3, 64, 1)会创建一个卷积层，输出是64通道的224x224特征图
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)  # padding=1保证输出特征图大小不变（当stride=1时）


def conv1x1(in_planes, out_planes, stride=1):
    """
    创建一个1x1卷积层，没有填充
    
    1x1卷积主要用于调整通道数量，降维或升维，不改变特征的空间分布。
    相当于对每个像素位置做一个全连接层。
    
    参数说明:
        in_planes: 输入的特征通道数
        out_planes: 输出的特征通道数
        stride: 卷积滑动步长，通常为1
        
    举例：
        如果有一个512通道的特征图，想减少到64通道：
        conv1x1(512, 64, 1)会创建一个降维卷积层
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


# class Down_wt(nn.Module):
#     def __init__(self, in_ch, out_ch):
#         super(Down_wt, self).__init__()
#         self.wt = DWTForward(J=1, mode='zero', wave='haar')
#         self.conv_bn_relu = nn.Sequential(
#             nn.Conv2d(in_ch * 4, out_ch, kernel_size=1, stride=1),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU(inplace=True),
#         )

#     def forward(self, x):
#         yL, yH = self.wt(x)
#         y_HL = yH[0][:, :, 0, ::]
#         y_LH = yH[0][:, :, 1, ::]
#         y_HH = yH[0][:, :, 2, ::]
#         x = torch.cat([yL, y_HL, y_LH, y_HH], dim=1)
#         x = self.conv_bn_relu(x)
#         return x


class EMA(nn.Module):
    def __init__(self, channels, factor=8, stride=1):
        super(EMA, self).__init__()
        self.groups = factor
        self.stride = stride
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
        out = (group_x * weights.sigmoid()).reshape(b, c, h, w)
        if self.stride > 1:
            out = F.avg_pool2d(out, kernel_size=self.stride, stride=self.stride)
        return out


class BasicBlock(nn.Module):
    # 基本残差块的通道扩展系数，表示输出通道数与输入通道数的比例
    # 对于基本块，输入和输出通道数相同，所以扩展系数为1
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        """
        初始化基本残差块
        
        残差块是ResNet的核心组件，通过"捷径连接"让深层网络更容易训练。
        基本思想是：除了学习输入到输出的转换，还提供一条"捷径"让输入可以直接加到输出上。
        
        参数详解:
            inplanes: 输入特征图的通道数
            planes: 输出特征图的通道数（实际输出通道数是planes*expansion）
            stride: 第一个卷积层的步长，控制是否降采样（减小特征图大小）
            downsample: 用于调整输入特征（捷径分支），使其能够与主路径的输出相加
        """
        super(BasicBlock, self).__init__()  # 调用父类初始化方法
        # 第一个卷积层：可能改变通道数和特征图大小(当stride>1时)
        self.conv1 = conv3x3(inplanes, planes, stride)  # 3x3卷积，输入通道数inplanes，输出通道数planes
        # 批量归一化层：使网络训练更稳定，加速收敛
        # 对每个通道的数据进行归一化处理，使其均值为0，方差为1
        self.bn1 = nn.BatchNorm2d(planes)  # 对planes个通道的特征图分别做归一化
        # ReLU激活函数：给网络引入非线性能力
        # 公式：f(x) = max(0, x)，即小于0的值变为0，大于0的值保持不变
        self.relu = nn.ReLU(inplace=True)  # inplace=True表示直接修改输入数据，节省内存
        # 第二个卷积层：保持通道数和特征图大小不变
        self.conv2 = EMA(planes, stride=stride)  # 用EMA替换第二个3x3卷积，传入stride参数
        self.bn2 = nn.BatchNorm2d(planes)  # 第二个批归一化层
        # 下采样层：当特征图大小或通道数需要变化时，对捷径分支进行调整
        self.downsample = downsample  # 可能是None，或者是一个下采样模块
        self.stride = stride  # 保存步长值，以便其他方法使用

    def forward(self, x):
        """
        前向传播函数 - 定义数据如何通过这个残差块
        
        残差块的数据流向：
        1. 保存原始输入用于后面的捷径连接
        2. 输入通过两个卷积层和批归一化层进行变换
        3. 如果需要，对捷径分支进行下采样调整
        4. 将变换后的特征与捷径分支相加，再通过激活函数输出
        
        参数:
            x: 输入特征图，形状为[批量大小, 通道数, 高度, 宽度]
        
        返回:
            out: 处理后的特征图
        """
        identity = x  # 保存输入，用于残差连接（捷径分支）

        # 主路径处理过程
        out = self.conv1(x)  # 第一个卷积操作
        out = self.bn1(out)  # 第一个批归一化操作，增加稳定性
        out = self.relu(out)  # 使用ReLU激活函数引入非线性

        out = self.conv2(out)  # 第二个卷积操作
        out = self.bn2(out)  # 第二个批归一化操作

        # 捷径分支处理：如果需要对输入进行调整（改变通道数或特征图大小）
        if self.downsample is not None:
            # 使用下采样模块处理输入，使其形状与主路径输出匹配
            identity = self.downsample(x)

        # 残差连接：将主路径输出与捷径分支相加
        # 这是残差学习的核心，允许网络只学习输入与输出的差异(残差)
        out += identity  
        out = self.relu(out)  # 最后的ReLU激活

        return out  # 返回处理后的特征图


class Bottleneck(nn.Module):
    # 瓶颈残差块的通道扩展系数
    # 输出通道数是planes*expansion，这里是4倍
    # 这种设计可以减少参数和计算量，适合更深的网络
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        """
        初始化瓶颈残差块
        
        瓶颈块是更高效的残差块版本，使用三个卷积层：
        1. 1x1卷积降维（减少通道数）
        2. 3x3卷积提取特征
        3. 1x1卷积升维（恢复或增加通道数）
        
        这种"瓶颈"设计减少了计算量，适合构建更深的网络（如ResNet-50及以上）。
        
        参数详解:
            inplanes: 输入通道数
            planes: 中间卷积层的通道数（最终输出是planes*expansion）
            stride: 中间卷积层的步长，控制特征图大小变化
            downsample: 用于调整捷径分支，使其能与主路径输出匹配
        """
        super(Bottleneck, self).__init__()  # 调用父类初始化方法
        
        # 第一个1x1卷积，用于降维（减少通道数）
        self.conv1 = conv1x1(inplanes, planes)  # 输入inplanes通道，输出planes通道
        self.bn1 = nn.BatchNorm2d(planes)  # 第一个批归一化层
        
        # 第二个3x3卷积，提取特征
        # 通道数保持不变，但可能改变特征图大小（当stride>1时）
        self.conv2 = EMA(planes, stride=stride)  # 用EMA替换第二个3x3卷积，传入stride参数
        self.bn2 = nn.BatchNorm2d(planes)  # 第二个批归一化层
        
        # 第三个1x1卷积，用于升维（增加通道数）
        # 输出通道数是planes*expansion（通常是planes的4倍）
        self.conv3 = conv1x1(planes, planes * self.expansion)  # 输出变为4倍通道数
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)  # 第三个批归一化层
        
        # ReLU激活函数
        self.relu = nn.ReLU(inplace=True)  # 使用inplace操作节省内存
        
        # 下采样模块和步长
        self.downsample = downsample  # 用于捷径分支的调整
        self.stride = stride  # 保存步长值

    def forward(self, x):
        """
        瓶颈残差块的前向传播逻辑
        
        数据流向：
        1. 保存原始输入用于捷径连接
        2. 输入依次通过三个卷积层和批归一化层
        3. 如果需要，对捷径分支进行调整
        4. 主路径输出与捷径分支相加，再通过激活函数
        
        参数:
            x: 输入特征图
        
        返回:
            out: 处理后的特征图
        """
        identity = x  # 保存输入，用于残差连接

        # 主路径：三个卷积层的处理序列
        out = self.conv1(x)  # 第一个1x1卷积（降维）
        out = self.bn1(out)  # 批归一化
        out = self.relu(out)  # ReLU激活

        out = self.conv2(out)  # 第二个卷积操作
        out = self.bn2(out)  # 批归一化
        out = self.relu(out)  # ReLU激活

        out = self.conv3(out)  # 第三个1x1卷积（升维）
        out = self.bn3(out)  # 批归一化
        # 注意：这里没有激活，因为需要先与捷径分支相加

        # 处理捷径分支
        if self.downsample is not None:
            # 如果需要调整捷径分支（通道数或特征图大小变化）
            identity = self.downsample(x)

        # 残差连接：将主路径与捷径分支相加
        out += identity  # 这是残差学习的关键
        out = self.relu(out)  # 最后的ReLU激活

        return out  # 返回处理后的特征图


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=2, zero_init_residual=False):
        """
        初始化ResNet（残差网络）模型
        
        ResNet是一种深度卷积神经网络，通过残差连接解决了深层网络训练难的问题。
        它已经成为计算机视觉领域最常用的基础网络之一。
        
        简单来说，ResNet就像是一个复杂的图像分析器，能够自动学习识别图像中的特征。
        
        参数详解:
            block: 残差块类型 - 决定使用哪种构建块（BasicBlock或Bottleneck）
            layers: 每层中残差块的数量列表 - 例如[2,2,2,2]表示每层各有2个块
            num_classes: 分类类别数 - 模型最终要区分多少种类别
            zero_init_residual: 是否将残差分支的最后BN层初始化为0 - 一种优化技巧
        """
        super(ResNet, self).__init__()  # 调用父类初始化

        # 以下是图像预处理相关的参数
        self.unfoldSize = 2  # 展开大小，用于图像处理算法
        self.unfoldIndex = 0  # 展开索引，指定处理哪部分
        assert self.unfoldSize > 1  # 确保展开大小大于1（检查参数有效性）
        assert -1 < self.unfoldIndex and self.unfoldIndex < self.unfoldSize*self.unfoldSize  # 确保索引在有效范围内
        
        # 网络初始参数设置
        self.inplanes = 64  # 初始通道数，后续层会基于此值计算通道数
        
        # 第一个卷积层：输入图像通常是3通道(RGB)，转换为64通道的特征图
        # kernel_size=3表示卷积核大小为3x3，stride=2表示特征图尺寸减半
        # padding=1表示在输入周围填充一圈0，保持卷积后的特征图形状合适
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        
        # 批归一化层：对卷积输出的64个通道分别做归一化处理
        # 让网络训练更稳定、更快收敛
        self.bn1 = nn.BatchNorm2d(64)
        
        # ReLU激活函数：引入非线性变换，增强网络表达能力
        # inplace=True表示直接在原内存上修改数据，节省内存
        self.relu = nn.ReLU(inplace=True)
        
        # 最大池化层：在3x3的区域内取最大值，stride=2表示特征图再次减半
        # 这一步会丢弃一些信息，但保留最显著的特征，同时减少计算量
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # 构建残差网络的主体部分 - 由多个残差层组成
        # 每层包含多个相同类型的残差块
        self.layer1 = self._make_layer(block, 64, layers[0])  # 第一层，输出64通道
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)  # 第二层，输出128通道，特征图尺寸减半
        
        # 全局平均池化：将每个通道的特征图平均成一个值
        # 输出大小为(1,1)表示每个通道只保留一个值，大幅减少参数量
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 全连接层：将特征映射到具体的类别得分
        # 输入512通道（取决于前面的block和通道数），输出num_classes个得分
        self.fc1 = nn.Linear(512, num_classes)

        # 初始化网络参数 - 合适的初始化对训练很重要
        for m in self.modules():  # 遍历网络中的所有模块
            if isinstance(m, nn.Conv2d):  # 如果是卷积层
                # 使用He初始化方法：根据输入特征的方差调整权重
                # 这种初始化适合ReLU激活函数，能使网络更容易训练
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):  # 如果是批归一化层
                # 将权重初始化为1，偏置初始化为0
                # 这是批归一化层的标准初始化方法
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # 零初始化残差块最后的BN层（一种优化技巧）
        # 论文表明这样做可以提高模型性能0.2~0.3%
        # 原理：使每个残差块初始时像一个恒等映射，随着训练逐渐学习残差
        if zero_init_residual:  # 如果启用这个选项
            for m in self.modules():  # 再次遍历所有模块
                if isinstance(m, Bottleneck):  # 如果是瓶颈残差块
                    nn.init.constant_(m.bn3.weight, 0)  # 将最后BN层权重置0
                elif isinstance(m, BasicBlock):  # 如果是基本残差块
                    nn.init.constant_(m.bn2.weight, 0)  # 将最后BN层权重置0

    def _make_layer(self, block, planes, blocks, stride=1):
        """
        创建一个残差层，包含多个残差块
        
        一个残差层由多个相同类型的残差块组成，第一个块可能会改变通道数和特征图大小，
        后续的块保持输入输出一致。
        
        参数详解:
            block: 残差块类型（BasicBlock或Bottleneck）
            planes: 残差块中间层的通道数
            blocks: 该层包含的残差块数量
            stride: 第一个残差块的步长，控制特征图大小是否减半
            
        返回:
            nn.Sequential: 由多个残差块组成的层
        """
        downsample = None  # 初始化下采样为None
        
        # 判断是否需要对捷径分支进行处理
        # 当步长不为1（特征图大小需要变化）或输入输出通道数不同时，需要调整捷径分支
        if stride != 1 or self.inplanes != planes * block.expansion:
            # 创建一个下采样模块，包含1x1卷积和批归一化
            # 用于调整捷径分支的通道数和特征图大小，使其与主路径匹配
            downsample = nn.Sequential(
                # 1x1卷积调整通道数和特征图大小
                conv1x1(self.inplanes, planes * block.expansion, stride),
                # 批归一化层
                nn.BatchNorm2d(planes * block.expansion),
            )
            
        layers = []  # 创建一个空列表存放残差块
        
        # 添加第一个残差块（可能包含下采样）
        # 这个块可能会改变特征图大小和通道数
        layers.append(block(self.inplanes, planes, stride, downsample))
        
        # 更新当前通道数为block的输出通道数
        self.inplanes = planes * block.expansion
        
        # 添加剩余的残差块，这些块保持特征图大小和通道数不变
        for _ in range(1, blocks):  # 从1开始，因为第0个已经添加
            layers.append(block(self.inplanes, planes))
            
        # 返回一个Sequential容器，包含所有残差块
        # *layers将列表展开为多个参数传入Sequential
        return nn.Sequential(*layers)

    def _preprocess_dwt(self, x, mode='symmetric', wave='bior1.3'):
        '''
        使用离散小波变换（DWT）进行图像预处理
        
        小波变换类似于高级版的傅里叶变换，能够捕捉图像中的频率信息和空间信息。
        它把图像分解成多个子图像，包含不同方向和尺度的细节。
        
        这种预处理可以帮助网络更好地关注图像中的边缘和纹理特征。
        
        使用前需安装: pip install pywavelets pytorch_wavelets
        
        参数详解:
            x: 输入图像张量，形状为[批量大小,通道数,高度,宽度]
            mode: 边界处理模式，'symmetric'表示对称延拓边界
            wave: 小波类型，'bior1.3'是一种双正交小波
            
        返回:
            经过小波变换后的特征图，只保留了某个方向的高频分量
        '''
        from pytorch_wavelets import DWTForward, DWTInverse  # 导入小波变换工具
        
        # 创建小波变换滤波器，J=1表示只分解一层
        DWT_filter = DWTForward(J=1, mode=mode, wave=wave).to(x.device)
        
        # 进行小波变换，获得低频成分Yl和高频成分Yh
        # Yl包含图像的大致形状，Yh包含细节和边缘信息
        Yl, Yh = DWT_filter(x)
        
        # 只取高频成分中的一个方向(索引2)，并调整大小与原图像一致
        # 这相当于提取了图像的某个方向的边缘信息
        return transforms.Resize([x.shape[-2], x.shape[-1]])(Yh[0][:, :, 2, :, :])

    def _preprocess_fft(self, x, scale=8):
        """
        使用快速傅里叶变换（FFT）进行图像预处理
        
        傅里叶变换将图像从空间域转换到频率域，可以分析图像的频率成分。
        低频部分对应图像的大体结构和平滑区域，高频部分对应边缘和细节。
        
        此函数通过过滤掉图像中心的低频成分，保留边缘和细节信息。
        
        参数详解:
            x: 输入图像张量
            scale: 缩放因子，控制过滤低频成分的范围
            
        返回:
            经过频域处理后的图像
        """
        from torch.nn import functional as F  # 导入函数式API
        
        # 二维快速傅里叶变换，将图像转换到频率域
        # norm="ortho"表示使用正交化处理，保持能量一致
        x = torch.fft.fft2(x, norm="ortho")
        
        # 将频谱中心移动到中央，便于处理
        # 原来低频在四角，高频在中心，移动后低频在中心，高频在四周
        x = torch.fft.fftshift(x, dim=[-2, -1])
        
        # 获取图像的形状信息
        b, c, h, w = x.shape
        
        # 将中心区域（低频部分）置为0，相当于高通滤波
        # 这样只保留高频信息，对应图像中的边缘和细节
        x[:, :, h//2-h//scale:h//2+h//scale, w//2-w//scale:w//2+w//scale] = 0.0
        
        # 将频谱中心移回原位
        x = torch.fft.ifftshift(x, dim=[-2, -1])
        
        # 反变换回空间域
        x = torch.fft.ifft2(x, norm="ortho")
        
        # 取实部，因为理论上应该没有虚部，但计算误差可能导致有微小虚部
        x = torch.real(x)
        
        # 用ReLU激活，将负值设为0
        x = F.relu(x, inplace=True)
        
        return x

    def _preprocess_dct(self, x, diagonal=64):
        """
        使用离散余弦变换（DCT）进行图像预处理
        
        DCT类似于傅里叶变换，但只使用余弦函数，所有值都是实数。
        JPEG图像压缩就使用了DCT。
        
        此函数对图像进行DCT变换，保留一部分频率成分，然后反变换回来。
        
        参数详解:
            x: 输入图像张量
            diagonal: 对角线参数，控制保留多少频率成分
            
        返回:
            处理后的图像
        """
        from scipy.fftpack import dct, idct  # 导入DCT和逆DCT函数
        
        device = x.device  # 记录当前设备（CPU或GPU）
        x = x.cpu()  # 将数据移到CPU，因为scipy不支持GPU
        
        b, c, h, w = x.shape  # 获取形状信息
        
        # 进行二维DCT变换，先在最后一个维度上变换，再在倒数第二个维度上变换
        # norm='ortho'表示正交化处理
        x = torch.tensor(dct(dct(x.numpy(), axis=-1, norm='ortho'), axis=-2, norm='ortho'))
        
        # 创建一个掩码，只保留部分频率成分
        # tril创建下三角矩阵，rot90旋转90度，用于选择特定频率区域
        mask = torch.rot90(torch.tril(torch.ones(h, w), diagonal=diagonal).bool(), k=-1, dims=(0, 1))
        
        # 应用掩码，将不保留的频率成分置为0
        x[:, :, mask] = 0
        
        # 进行二维逆DCT变换，将频域数据转回空间域
        x = torch.tensor(idct(idct(x.numpy(), axis=-1, norm='ortho'), axis=-2, norm='ortho'))
        
        return x.to(device)  # 将数据移回原设备并返回

    def _preprocess_edge(self, x):
        """
        使用Sobel算子进行边缘检测预处理
        
        Sobel算子是一种经典的边缘检测算法，能够提取图像中物体的轮廓。
        它通过计算图像的梯度（亮度变化）来找出边缘。
        
        这种预处理让网络更关注物体的形状而不是颜色或纹理。
        
        参数:
            x: 输入图像张量
            
        返回:
            检测到的边缘图像
        """
        # kornia.filters.sobel是一个实现Sobel边缘检测的函数
        # 它计算图像在水平和垂直方向的梯度，并返回梯度幅值
        return kornia.filters.sobel(x)

    def _preprocess_NPR(self, x, factor=0.5):
        """
        使用最近邻像素重采样（NPR）进行预处理
        
        这个函数先将图像缩小，然后再放大回原来的大小。
        由于使用了最近邻插值，会产生像素化（方块状）效果，丢失一些细节。
        
        这种预处理可以帮助网络关注图像的大体结构而不是细节。
        
        参数:
            x: 输入图像张量
            factor: 缩放因子，控制中间过程缩小的程度
            
        返回:
            重采样后的图像
        """
        return F.interpolate(
            # 第一步：将图像缩小到原来的factor倍（例如0.5倍）
            # mode='nearest'表示使用最近邻插值方法
            F.interpolate(x, scale_factor=factor, mode='nearest', recompute_scale_factor=True), 
            # 第二步：将缩小后的图像放大回原来大小
            # 1/factor是为了回到原始大小（例如1/0.5=2倍）
            scale_factor=1/factor, 
            mode='nearest', recompute_scale_factor=True
        )

    def forward(self, x):
        """
        模型的前向传播函数 - 定义数据如何从输入到输出的完整路径
        
        前向传播是神经网络处理数据的过程，类似于流水线作业：
        1. 首先对输入图像进行预处理
        2. 然后依次通过各个网络层（卷积、批归一化、激活等）
        3. 最后得到分类结果
        
        参数:
            x: 输入图像张量，形状为[批量大小,通道数,高度,宽度]
            
        返回:
            模型的输出（分类得分），每个类别一个数值
        """
        # 使用离散小波变换预处理输入图像
        # 这一步提取图像中的边缘和纹理信息，帮助网络更好地识别特征
        x = 1 * self._preprocess_dwt(x)  # 乘以1是保持数值不变

        # 网络的主体部分
        x = self.conv1(x)  # 第一个卷积层，提取基本特征
        x = self.bn1(x)    # 批归一化，稳定训练过程
        x = self.relu(x)   # ReLU激活，引入非线性
        x = self.maxpool(x)  # 最大池化，减小特征图尺寸，提取显著特征

        # 通过残差层，逐步提取更高级的特征
        x = self.layer1(x)  # 第一个残差层
        x = self.layer2(x)  # 第二个残差层

        # 全局平均池化：将每个通道的特征图平均为一个值
        # 这步大幅减少了参数数量，相当于对每个通道做全局特征总结
        x = self.avgpool(x)  
        
        # 展平特征图：从[批量大小,通道数,1,1]变为[批量大小,通道数]
        # 准备输入到全连接层
        x = x.view(x.size(0), -1)  
        
        # 全连接层：将特征映射到具体的类别得分
        # 输出形状为[批量大小,类别数]
        x = self.fc1(x)  

        return x  # 返回模型输出


def resnet18(pretrained=False, **kwargs):
    """
    构建ResNet-18模型
    
    ResNet-18是ResNet家族中最小的标准模型，有18层卷积层。
    它使用BasicBlock构建，适合中小型任务，计算量小但性能不错。
    
    参数:
        pretrained: 是否使用ImageNet预训练权重 - True表示使用，False表示从头训练
        **kwargs: 其他参数，如num_classes（类别数量）等
        
    返回:
        构建好的ResNet-18模型实例
        
    用法示例:
        model = resnet18(pretrained=True, num_classes=10) # 创建一个10分类的预训练模型
    """
    # 创建ResNet-18模型
    # BasicBlock：使用基本残差块
    # [2,2,2,2]：四个残差层分别有2个BasicBlock
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    
    if pretrained:
        # 如果使用预训练权重，从预设的URL下载模型权重
        # 这些权重是在ImageNet（一个大型图像数据集）上训练得到的
        # 使用预训练权重可以加速模型训练，提高性能
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
        
    return model  # 返回构建好的模型


def resnet34(pretrained=False, **kwargs):
    """
    构建ResNet-34模型
    
    ResNet-34比ResNet-18更深，有34层卷积层。
    它仍使用BasicBlock构建，但每层包含更多的块，因此特征提取能力更强。
    
    参数:
        pretrained: 是否使用ImageNet预训练权重
        **kwargs: 其他参数
        
    返回:
        构建好的ResNet-34模型实例
    """
    # 创建ResNet-34模型
    # [3,4,6,3]表示四个残差层分别有3、4、6、3个BasicBlock
    # 总计：(3+4+6+3)*2+2 = 34层（每个Block有2层卷积，加上首尾各1层）
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    
    if pretrained:
        # 加载预训练权重
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
        
    return model


def resnet50(pretrained=False, **kwargs):
    """
    构建ResNet-50模型
    
    ResNet-50使用Bottleneck残差块（每块3层卷积），而不是BasicBlock（每块2层）。
    虽然残差层数量与ResNet-34相同，但由于使用了Bottleneck，总共有50层卷积。
    它计算效率高，同时有强大的特征提取能力，是实际应用中最常用的版本之一。
    
    参数:
        pretrained: 是否使用ImageNet预训练权重
        **kwargs: 其他参数
        
    返回:
        构建好的ResNet-50模型实例
    """
    # 创建ResNet-50模型
    # 使用Bottleneck残差块（3层卷积）
    # [3,4,6,3]表示四个残差层分别有3、4、6、3个Bottleneck
    # 总计：(3+4+6+3)*3+2 = 50层
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    
    if pretrained:
        # 加载预训练权重
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
        
    return model


def resnet101(pretrained=False, **kwargs):
    """
    构建ResNet-101模型
    
    ResNet-101是一个更深的网络，有101层卷积。
    它在第三个残差层有23个Bottleneck块，提供了强大的特征提取能力。
    适合复杂的计算机视觉任务，但需要更多计算资源。
    
    参数:
        pretrained: 是否使用ImageNet预训练权重
        **kwargs: 其他参数
        
    返回:
        构建好的ResNet-101模型实例
    """
    # 创建ResNet-101模型
    # [3,4,23,3]表示四个残差层分别有3、4、23、3个Bottleneck
    # 注意第三层有23个块，与ResNet-50的6个相比大幅增加
    # 总计：(3+4+23+3)*3+2 = 101层
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    
    if pretrained:
        # 加载预训练权重
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
        
    return model


def resnet152(pretrained=False, **kwargs):
    """
    构建ResNet-152模型
    
    ResNet-152是标准ResNet家族中最深的网络，有152层卷积。
    它在第三个残差层有36个Bottleneck块，提供了极其强大的特征提取能力。
    适合最复杂的计算机视觉任务，但需要大量计算资源。
    
    参数:
        pretrained: 是否使用ImageNet预训练权重
        **kwargs: 其他参数
        
    返回:
        构建好的ResNet-152模型实例
    """
    # 创建ResNet-152模型
    # [3,8,36,3]表示四个残差层分别有3、8、36、3个Bottleneck
    # 第二层和第三层的块数量大幅增加
    # 总计：(3+8+36+3)*3+2 = 152层
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    
    if pretrained:
        # 加载预训练权重
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
        
    return model