# 架构重构日志：动态双分支融合模型的实现

本文档旨在详细记录模型架构的一次重大重构。本次重构的核心目标是实现一个更先进、更强大的动态双分支融合网络。该网络巧妙地结合了一个用于处理高频细节的**自定义ResNet分支**和一个用于提取强大语义特征的**预训练DINOv2模型**，并通过一个复杂的动态机制来智能地融合它们的输出。

---

## 概览：设计思路与目标

本次重构由三个核心设计目标驱动，旨在全面提升模型的性能、效率和智能化水平：

1.  **差异化数据处理 (Differential Data Handling)**
    *   **目标**: 为两个功能不同的模型分支提供各自最适宜的数据输入。ResNet分支通过数据增强（如翻转、旋转）来学习对纹理和伪影特征的鲁棒性；而DINOv2分支则需要接近原始、无失真的"干净"图像来提取最准确的全局语义信息。
    *   **价值**: 避免数据增强对DINOv2语义提取能力的干扰，同时最大化ResNet分支从增强数据中学习的能力。

2.  **旗舰级特征提取 (Enhanced Feature Extraction)**
    *   **目标**: 将DINOv2主干网络从`ViT-Small`升级为更强大的`ViT-Large`，并将其完全**冻结**，作为一个纯粹的、高性能的特征提取器。
    *   **价值**: 在不增加训练成本（甚至降低）的前提下，直接利用`ViT-Large`（3亿参数）在海量数据上预训练好的强大知识，极大地提升模型的特征表征能力，同时有效防止过拟合。

3.  **智能化特征融合 (Intelligent Feature Fusion)**
    *   **目标**: 抛弃原有的静态加权融合方式，引入一个由**EMA (高效多尺度注意力) 模块**引导的**动态门控机制 (Dynamic Gating)**。
    *   **价值**: 让模型学会"思考"。它能根据从高频信息中看到的内容（ResNet分支），来动态地判断当前图像的语义特征（DINOv2分支）有多重要，从而为每一张图片量身定制最优的特征融合策略。

---

## 第一步：数据管道重构 (`data/datasets.py`)

**目标**: 使数据加载器（DataLoader）能够为每个样本同时生成"增强版"和"干净版"两张图像。

**实现细节**:
我们将所有的数据变换逻辑集中到了 `Get_Transforms` 一个函数中。该函数现在会创建并返回三个独立的变换流程：`transform_train`, `transform_eval`, 和 `transform_dino`。`TrainDataset`类的 `__getitem__` 方法被相应修改，以对同一张PIL图像应用两种不同的变换，最终返回一个包含两个图像张量的元组。

**关键代码变更 (`data/datasets.py`)**:

```python
# Get_Transforms 函数现在返回三个独立的变换流程
def Get_Transforms(args):
    # transform_train 和 transform_eval 的定义与之前类似，包含随机翻转、旋转等增强
    # ...

    # --- 新增的 DINOv2 专用变换流程 (仅缩放和归一化) ---
    dino_norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform_dino = transforms.Compose([
        transforms.Resize([args.input_size, args.input_size], interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        dino_norm,
    ])

    return transforms.Compose(transform_train), transforms.Compose(transform_eval), transform_dino


class TrainDataset(Dataset):
    def __init__(self, is_train, args):
        # 一次性接收所有变换流程
        transform_train, transform_eval, self.transform_dino = Get_Transforms(args)
        self.transform = transform_train if is_train else transform_eval
        # ...

    def __getitem__(self, index):
        # ...
        image = Image.open(image_path).convert('RGB')
        
        # 对同一张图应用两种变换
        image_aug = self.transform(image)      # 用于 ResNet 分支 (带增强)
        image_dino = self.transform_dino(image) # 用于 DINOv2 分支 (干净)

        # 返回一个包含两个图像张量的元组
        return (image_aug, image_dino), torch.tensor(int(targets))
```

---

## 第二步：训练引擎适配 (`engine_finetune.py`)

**目标**: 使训练和评估循环能够识别并正确处理新的双图像数据格式。

**实现细节**:
在 `train_one_epoch` 和 `evaluate` 函数的循环体内，增加了 `isinstance(samples, (list, tuple))` 的检查。如果输入是元组，代码会分别处理元组内的每个张量。同时，由于Mixup的混合逻辑与双输入架构不兼容，我们增加了逻辑来阻止其在当前模式下被错误地使用。

**关键代码变更 (`engine_finetune.py`)**:

```python
# 在 train_one_epoch 函数中:
    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # ...
        # 处理双图像输入元组
        if isinstance(samples, (list, tuple)):
            # Mixup 与双输入架构不兼容，需要明确禁止
            if mixup_fn is not None:
                raise ValueError("Mixup is not supported for models with dual-image inputs.")
            # 分别将每个图像张量移动到计算设备
            samples = [s.to(device, non_blocking=True) for s in samples]
        else:
            samples = samples.to(device, non_blocking=True)
        # ...

# 在 evaluate 函数中 (逻辑类似):
    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        # ...
        # 处理双图像输入元组
        if isinstance(images, (list, tuple)):
            images = [img.to(device, non_blocking=True) for img in images]
        else:
            images = images.to(device, non_blocking=True)
        # ...
```

---

## 第三步：核心模型重构 (`models/dino_resnet.py`)

**目标**: 彻底重构模型，集成 `ViT-Large` 主干、`EMA` 注意力模块，并实现动态门控融合机制。

**实现细节**:
这是最大刀阔斧的改动，`DinoResNet` 类被完全重写。

1.  **集成EMA注意力模块**: 您提供的`EMA`模块代码被完整地加入到文件顶部。
2.  **升级并冻结DINOv2**: 模型加载代码更新为`torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')`。紧接着，通过一个循环将其所有参数的`requires_grad`属性设置为`False`，将其彻底冻结。
3.  **实现动态门控融合**:
    *   一个`self.ema`实例被创建，用于处理ResNet分支提取出的特征图。
    *   一个全新的`self.gate`模块被创建，它是一个由`nn.Linear`和`nn.Sigmoid`组成的微型网络，负责生成动态权重。
4.  **重写前向传播 (`forward`)**: `forward`方法被重写，以清晰地执行我们设计的全新数据流。

**全新模型架构代码 (`models/dino_resnet.py`)**:

```python
# ... (EMA 类的完整定义被添加在文件顶部) ...

class DinoResNet(nn.Module):
    def __init__(self, block, layers, num_classes=2, zero_init_residual=False):
        super(DinoResNet, self).__init__()
        # ... (ResNet分支的相关层，如conv1, bn1, layer1, layer2被定义) ...
        resnet_feat_dim = 128 * block.expansion # ResNet分支输出特征图的通道数
        
        # --- 步骤3a: 在ResNet特征图上应用EMA注意力 ---
        self.ema = EMA(channels=resnet_feat_dim)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # --- 步骤2: 加载、升级并冻结DINOv2 ---
        self.dinov2_branch = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
        dino_feat_dim = self.dinov2_branch.embed_dim # ViT-Large 的特征维度是 1024
        for param in self.dinov2_branch.parameters():
            param.requires_grad = False # 冻结参数，不参与训练
            
        # --- 步骤3b: 定义动态门控模块 ---
        self.gate = nn.Sequential(
            nn.Linear(resnet_feat_dim, 1), # 输入为ResNet特征向量，输出为一个标量
            nn.Sigmoid()                   # 将输出压缩到 0-1 之间，作为权重
        )

        # --- 分类器 ---
        fused_features_dim = resnet_feat_dim + dino_feat_dim
        self.classifier = nn.Sequential(...)
        # ... (权重初始化等) ...

    def forward(self, x):
        # 1. 解包来自数据加载器的双输入
        image_aug, image_dino = x
        
        # 2. ResNet分支: 处理带数据增强的图像的高频分量
        _, Yh = self.dwt(image_aug)
        x_resnet = Yh[0][:, :, 2, :, :] # 提取高频信息
        feat_map_resnet = self.layer2(self.layer1(self.maxpool(self.relu(self.bn1(self.conv1(x_resnet))))))
        
        # 3. EMA注意力: 优化高频特征图
        feat_map_refined = self.ema(feat_map_resnet)
        feat_vector_resnet = self.avgpool(feat_map_refined).flatten(1)

        # 4. DINOv2分支: 处理"干净"图像以提取语义特征
        feat_dino = self.dinov2_branch(image_dino)

        # 5. 动态融合: 使用高频特征的上下文来决定语义特征的重要性
        gate_weight = self.gate(feat_vector_resnet) # 计算动态权重
        feat_dino_gated = feat_dino * gate_weight # 应用权重
        feat_fused = torch.cat((feat_vector_resnet, feat_dino_gated), dim=1) # 拼接

        # 6. 分类
        output = self.classifier(feat_fused)
        return output
```
---

## 总结：全新的模型数据流

1.  **输入**: `DataLoader` 提供一个数据元组: `(增强图像, 干净图像)`。
2.  **分支一 (高频)**: `增强图像` -> `DWT` -> **高频分量** -> `ResNet分支` -> `特征图` -> `EMA注意力` -> **优化的ResNet特征向量**。
3.  **动态门控**: `优化的ResNet特征向量` -> `Gate模块` -> **动态权重** (0-1之间)。
4.  **分支二 (语义)**: `干净图像` -> **冻结的DINOv2-Large** -> **语义特征向量**。
5.  **融合**: `动态权重` **乘以** `语义特征向量`，得到加权后的语义特征。
6.  **拼接**: 将`优化的ResNet特征向量`与`加权后的语义特征`沿特征维度拼接。
7.  **输出**: `拼接后的融合向量` -> `MLP分类器` -> **最终预测结果**。 