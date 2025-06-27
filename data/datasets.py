# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import io, os, pdb  # 导入io用于内存流操作，os用于文件操作，pdb用于调试
import cv2, math, random  # 导入cv2用于图像处理，math用于数学运算，random用于随机数生成
import numpy as np  # 导入numpy用于数值计算

import torch  # 导入PyTorch主库
from PIL import Image, ImageFile  # 导入PIL用于图像处理
from torch.utils.data import Dataset  # 导入PyTorch的数据集基类
from torchvision import transforms  # 导入torchvision的变换模块
from torchvision.transforms import functional as F  # 导入变换的函数式接口
from torchvision.transforms import InterpolationMode  # 导入插值模式

ImageFile.LOAD_TRUNCATED_IMAGES = True  # 允许加载截断的图片，防止因图片损坏报错


class RandomJPEG():
    # 随机JPEG压缩变换
    def __init__(self, quality=95, interval=1, p=0.1):
        # quality可以是整数或区间，interval为步长，p为应用概率
        if isinstance(quality, tuple):
            self.quality = [i for i in range(quality[0], quality[1]) if i % interval == 0]
        else:
            self.quality = quality
        self.p = p

    def __call__(self, img):
        # 以概率p对图片进行JPEG压缩
        if random.random() < self.p:
            if isinstance(self.quality, list):
                quality = random.choice(self.quality)
            else:
                quality = self.quality
            buffer = io.BytesIO()  # 创建内存缓冲区
            img.save(buffer, format='JPEG', quality=quality)  # 以指定质量保存为JPEG
            buffer.seek(0)  # 指针回到开头
            img = Image.open(buffer)  # 重新读取图片
        return img  # 返回处理后的图片


class RandomGaussianBlur():
    # 随机高斯模糊变换
    def __init__(self, kernel_size, sigma=(0.1, 2.0), p=1.0):
        self.blur = transforms.GaussianBlur(kernel_size=kernel_size, sigma=sigma)  # 创建高斯模糊变换
        self.p = p  # 应用概率

    def __call__(self, img):
        # 以概率p对图片进行高斯模糊
        if random.random() < self.p:
            return self.blur(img)
        return img


class RandomMask(object):
    # 随机遮挡（mask）变换
    def __init__(self, ratio=0.5, patch_size=16, p=0.5):
        """
        Args:
            ratio (float or tuple of float): 如果为float，表示遮挡比例；如果为tuple，则在区间内随机采样
            patch_size (int): 遮挡块的尺寸（正方形）
            p (float): 应用概率
        """
        if isinstance(ratio, float):
            self.fixed_ratio = True  # 是否为固定比例
            self.ratio = (ratio, ratio)
        elif isinstance(ratio, tuple) and len(ratio) == 2 and all(isinstance(r, float) for r in ratio):
            self.fixed_ratio = False
            self.ratio = ratio
        else:
            raise ValueError("Ratio must be a float or a tuple of two floats.")

        self.patch_size = patch_size
        self.p = p

    def __call__(self, tensor):
        # 以概率p对输入张量进行遮挡
        if random.random() > self.p: return tensor  # 按概率跳过

        _, h, w = tensor.shape  # 获取张量的高和宽
        mask = torch.ones((h, w), dtype=torch.float32)  # 创建全1的mask

        if self.fixed_ratio:
            ratio = self.ratio[0]
        else:
            ratio = random.uniform(self.ratio[0], self.ratio[1])  # 随机采样遮挡比例

        # 计算需要遮挡的块数
        num_masks = int((h * w * ratio) / (self.patch_size ** 2))

        # 生成不重叠的随机位置
        selected_positions = set()
        while len(selected_positions) < num_masks:
            top = random.randint(0, (h // self.patch_size) - 1) * self.patch_size
            left = random.randint(0, (w // self.patch_size) - 1) * self.patch_size
            selected_positions.add((top, left))

        for (top, left) in selected_positions:
            mask[top:top+self.patch_size, left:left+self.patch_size] = 0  # 对应区域置0

        return tensor * mask.expand_as(tensor)  # 应用mask并返回


def Get_Transforms(args):
    # 根据参数生成训练和评估的变换流程

    size = args.input_size  # 输入图片的目标尺寸

    TRANSFORM_DICT = {
        'resize_BILINEAR': {
            'train': [
                transforms.RandomResizedCrop([size, size], interpolation=InterpolationMode.BILINEAR),  # 随机裁剪并缩放
            ],
            'eval': [
                transforms.Resize([size, size], interpolation=InterpolationMode.BILINEAR),  # 直接缩放
            ],
        },

        'resize_NEAREST': {
            'train': [
                transforms.RandomResizedCrop([size, size], interpolation=InterpolationMode.NEAREST),
            ],
            'eval': [
                transforms.Resize([size, size], interpolation=InterpolationMode.NEAREST),
            ],
        },

        'crop': {
            'train': [
                transforms.RandomCrop([size, size], pad_if_needed=True),  # 随机裁剪，必要时填充
            ],
            'eval': [
                transforms.CenterCrop([size, size]),  # 居中裁剪
            ],
        },

        'source': {
            'train': [
                transforms.RandomCrop([size, size], pad_if_needed=True),
            ],
            'eval': [
            ],
        },
    }

    # region [Augmentations]
    transform_train, transform_eval = TRANSFORM_DICT[args.transform_mode]['train'], TRANSFORM_DICT[args.transform_mode]['eval']

    # 训练增强流程
    transform_train.extend([
        transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
        transforms.RandomRotation(180),  # 随机旋转
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),  # 颜色扰动
        transforms.ToTensor(),  # 转为张量
        RandomMask(ratio=(0.00, 0.75), patch_size=16, p=0.5),  # 随机遮挡
    ])

    transform_eval.append(transforms.ToTensor())  # 评估流程转为张量
    # endregion

    # region [Perturbatiocns in Testing]
    if args.jpeg_factor is not None:
        transform_eval.insert(0, RandomJPEG(quality=args.jpeg_factor, p=1.0))  # 测试时可加JPEG扰动
    if args.blur_sigma is not None:
        transform_eval.insert(0, transforms.GaussianBlur(kernel_size=5, sigma=args.blur_sigma))  # 测试时可加高斯模糊
    if args.mask_ratio is not None and args.mask_patch_size is not None:
        transform_eval.append(RandomMask(ratio=args.mask_ratio, patch_size=args.mask_patch_size, p=1.0))  # 测试时可加遮挡
    # endregion

    # --- DINO Transform (minimal augmentation) ---
    dino_norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform_dino = transforms.Compose([
        transforms.Resize([size, size], interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        dino_norm,
    ])

    return transforms.Compose(transform_train), transforms.Compose(transform_eval), transform_dino


class TrainDataset(Dataset):
    # 自定义训练/评估数据集

    def __init__(self, is_train, args):
        # is_train: 是否为训练集
        # args: 参数对象

        transform_train, transform_eval, self.transform_dino = Get_Transforms(args)
        self.transform = transform_train if is_train else transform_eval
        root = args.data_path if is_train else args.eval_data_path  # 数据路径

        dataset_list = root.replace(' ', '').split(',')  # 支持多个数据集路径
        num_datasets = len(dataset_list)

        if num_datasets == 1:
            real_list, fake_list = self.get_real_and_fake_lists(dataset_list[0])  # 获取真实和伪造图片列表
            if is_train and args.num_train is not None:
                # 训练时可指定样本数，真实和伪造各取一半
                self.data_list = real_list[:args.num_train//2] + fake_list[:args.num_train//2]
            else:
                self.data_list = real_list + fake_list
        else:
            assert args.num_train is not None
            self.data_list = []
            for dataset in dataset_list:
                real_list, fake_list = self.get_real_and_fake_lists(dataset)
                # 多数据集时，平均分配样本数
                self.data_list.extend(real_list[:args.num_train//(2 * num_datasets)] + fake_list[:args.num_train//(2 * num_datasets)])

    def get_image_paths(self, dir_path):
        # 获取指定目录下所有图片文件路径
        image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp')
        image_paths = []
        for root, dirs, files in sorted(os.walk(dir_path)):
            for file in sorted(files):
                if file.lower().endswith(image_extensions):
                    image_paths.append(os.path.join(root, file))
        return image_paths

    def get_real_and_fake_lists(self, folder_path):
        # 遍历文件夹，分别获取真实和伪造图片的路径及标签
        real_list, fake_list = [], []
        for root, dirs, files in sorted(os.walk(folder_path, followlinks=True)):
            for dir_name in sorted(dirs):
                if dir_name == "0_real":
                    real_dir_path = os.path.join(root, dir_name)
                    real_list.extend([{"image_path": image_path, "label" : 0} for image_path in self.get_image_paths(real_dir_path)])
                elif dir_name == "1_fake":                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 
                    fake_dir_path = os.path.join(root, dir_name)
                    fake_list.extend([{"image_path": image_path, "label" : 1} for image_path in self.get_image_paths(fake_dir_path)])
        return real_list, fake_list

    def __len__(self):
        # 返回数据集大小
        return len(self.data_list)

    def __getitem__(self, index):
        # 根据索引获取样本
        sample = self.data_list[index]
        image_path, targets = sample['image_path'], sample['label']
        image = Image.open(image_path).convert('RGB')  # 打开图片并转为RGB
        
        image_aug = self.transform(image)      # For ResNet branch
        image_dino = self.transform_dino(image) # For DINO branch

        return (image_aug, image_dino), torch.tensor(int(targets))  # 返回图片张量和标签
