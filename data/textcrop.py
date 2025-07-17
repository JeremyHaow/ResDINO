import numpy as np
from PIL import Image
import random
from scipy import stats
from skimage import feature, filters
from skimage.filters.rank import entropy
from skimage.morphology import disk
import torch
from torchvision.transforms import CenterCrop
from torchvision import transforms

############################################################### TEXTURE CROP ###############################################################

def texture_crop(image, stride=224, window_size=224, metric='he', position='top', n=10, drop = False):
    """
    根据纹理复杂度对图像进行分块、排序和筛选。

    该函数首先使用滑动窗口将输入图像裁剪成多个重叠或不重叠的图块，然后计算每个图块的纹理复杂度，
    并根据复杂度排序，最后返回纹理最丰富或最平滑的n个图块。

    Args:
        image (PIL.Image.Image): 输入的PIL图像对象。
        stride (int, optional): 滑动窗口的步长。默认为224。
        window_size (int, optional): 裁剪窗口的大小。默认为224。
        metric (str, optional): 用于评估纹理复杂度的指标。可选值:
                                'sd': 标准差 (Standard Deviation)
                                'ghe': 全局直方图熵 (Global Histogram Entropy)
                                'le': 局部熵 (Local Entropy)
                                'ac': 自相关性 (Autocorrelation)
                                'td': 纹理多样性 (Texture Diversity)
                                默认为 'he' (实际上会使用'ghe')。
        position (str, optional): 'top'表示选择纹理最复杂的n个图块，'bottom'表示选择最平滑的n个。默认为 'top'。
        n (int, optional): 需要返回的图块数量。默认为10。
        drop (bool, optional): 是否丢弃图像边缘无法完整覆盖的区域。
                               False表示会额外处理边缘，保证整个图像都被覆盖。默认为 False。

    Returns:
        list[PIL.Image.Image]: 包含n个已筛选图块的列表。
    """
    cropped_images = []
    images = []

    # 1. 使用滑动窗口裁剪图像
    x, y = 0, 0
    for y in range(0, image.height - window_size + 1, stride):
        for x in range(0, image.width - window_size + 1, stride):
            cropped_images.append(image.crop((x, y, x + window_size, y + window_size)))
    
    # 2. 如果不丢弃边缘区域，则手动处理右侧和底部的边缘，确保全图覆盖
    if not drop:
        # 保存最后一次的x,y坐标，用于判断是否已达边缘
        last_x = x + stride
        last_y = y + stride

        # 处理右侧边缘
        if last_x + window_size > image.width:
            for y in range(0, image.height - window_size + 1, stride):
                cropped_images.append(image.crop((image.width - window_size, y, image.width, y + window_size)))
        # 处理底部边缘
        if last_y + window_size > image.height:
            for x in range(0, image.width - window_size + 1, stride):
                cropped_images.append(image.crop((x, image.height - window_size, x + window_size, image.height)))
        # 处理右下角
        if last_x + window_size > image.width and last_y + window_size > image.height:
            cropped_images.append(image.crop((image.width - window_size, image.height - window_size, image.width, image.height)))

    # 3. 计算每个图块的纹理度量值
    for crop in cropped_images:
        crop_gray = crop.convert('L')  # 转换为灰度图以计算纹理
        crop_gray_np = np.array(crop_gray)
        
        # 根据指定的metric计算度量值
        if metric == 'sd':
            # 标准差：衡量像素值离散程度，值越大，对比度越高，纹理可能越丰富
            m = np.std(crop_gray_np / 255.0)
        elif metric == 'ghe':
            # 全局直方图熵：衡量像素值分布的随机性，熵越大，纹理越复杂
            m = histogram_entropy_response(crop_gray_np / 255.0)
        elif metric == 'le':
            # 局部熵：在邻域内计算熵，更能反映空间纹理信息
            m = local_entropy_response(crop_gray_np)
        elif metric == 'ac':
            # 自相关性：衡量像素间的重复性，值越小，纹理越不规则
            m = autocorrelation_response(crop_gray_np / 255.0)
        elif metric == 'td':
            # 纹理多样性：基于相邻像素差异的总和
            m = texture_diversity_response(crop_gray_np / 255.0)
        
        images.append((crop, m))

    # 4. 根据度量值对图块进行降序排序 (复杂度高的在前)
    images.sort(key=lambda x: x[1], reverse=True)
    
    # 5. 根据'position'参数选择顶部或底部的n个图块
    if position == 'top':
        texture_images = [img for img, _ in images[:n]]
    elif position == 'bottom':
        texture_images = [img for img, _ in images[-n:]]

    # 6. 如果筛选出的图块数量不足n个，则重复填充直至满足n个
    if len(texture_images) > 0:
        repeat_images = texture_images.copy()
        while len(texture_images) < n:
            texture_images.append(repeat_images[len(texture_images) % len(repeat_images)])

    return texture_images


def autocorrelation_response(image_array):
    """
    计算图像的平均自相关性响应。
    自相关性描述了信号与其自身在不同时间（或空间）滞后下的相似度。
    纹理平滑的区域自相关性高，纹理复杂的区域自相关性低。
    """
    # 1. 傅里叶变换到频域
    f = np.fft.fft2(image_array, norm='ortho')
    # 2. 计算功率谱（幅度值的平方）
    power_spectrum = np.abs(f) ** 2
    # 3. 傅里叶逆变换得到自相关函数
    acf = np.fft.ifft2(power_spectrum, norm='ortho').real
    # 4. 将零频分量移到中心
    acf = np.fft.fftshift(acf)
    # 5. 归一化并取平均值作为最终响应
    acf /= acf.max()
    acf = np.mean(acf)

    return acf

def histogram_entropy_response(image):
    """
    计算图像的全局直方图熵（香农熵）。
    熵衡量了图像像素强度分布的不确定性。熵越高，说明像素值分布越均匀，纹理可能越复杂。
    """
    # 1. 计算归一化的直方图（像素值出现的概率分布）
    histogram, _ = np.histogram(image.flatten(), bins=256, range=(0, 1), density=True) 
    # 2. 确保概率和为1
    prob_dist = histogram / histogram.sum()
    # 3. 使用scipy计算香农熵。加上一个极小值(1e-7)避免log(0)导致NaN
    entr = stats.entropy(prob_dist + 1e-7, base=2)

    return entr

def local_entropy_response(image):
    """
    计算图像的平均局部熵。
    与全局熵不同，它在每个像素的邻域内计算熵，更能反映空间上的纹理信息。
    """
    # 使用一个半径为10的圆形核计算每个像素点的局部熵
    entropy_image = entropy(image, disk(10))  
    # 返回整张图的平均局部熵
    mean_entropy = np.mean(entropy_image)

    return mean_entropy

def texture_diversity_response(image):
    """
    计算纹理多样性，即图像中所有相邻像素（水平、垂直、对角线）差异的绝对值总和。
    这个值越大，说明局部像素变化越剧烈，纹理越丰富。
    """
    M = image.shape[0]  
    l_div = 0

    # 水平方向差异
    for i in range(M):
        for j in range(M - 1):
            l_div += abs(image[i, j] - image[i, j + 1])

    # 垂直方向差异
    for i in range(M - 1):
        for j in range(M):
            l_div += abs(image[i, j] - image[i + 1, j])

    # 主对角线方向差异
    for i in range(M - 1):
        for j in range(M - 1):
            l_div += abs(image[i, j] - image[i + 1, j + 1])

    # 副对角线方向差异
    for i in range(M - 1):
        for j in range(M - 1):
            l_div += abs(image[i + 1, j] - image[i, j + 1])

    return l_div


############################################################## THRESHOLDTEXTURECROP ##############################################################

def threshold_texture_crop(image, stride=224, window_size=224, threshold=5, drop = False):
    """
    根据熵值阈值筛选高纹理区域的图块。

    Args:
        image (PIL.Image.Image): 输入的PIL图像对象。
        stride (int, optional): 滑动窗口的步长。默认为224。
        window_size (int, optional): 裁剪窗口的大小。默认为224。
        threshold (float, optional): 熵值的阈值，只有大于该值的图块才会被选中。默认为5。
        drop (bool, optional): 是否丢弃图像边缘无法完整覆盖的区域。默认为False。

    Returns:
        list[PIL.Image.Image]: 包含所有满足阈值条件的图块的列表。
    """
    cropped_images = []
    texture_images = []

    # 1. 使用滑动窗口裁剪图像 (逻辑同 texture_crop)
    for y in range(0, image.height - window_size + 1, stride):
        for x in range(0, image.width - window_size + 1, stride):
            cropped_images.append(image.crop((x, y, x + window_size, y + window_size)))

    # 2. 如果不丢弃，处理边缘区域 (逻辑同 texture_crop)
    if not drop:
        last_x = x + stride
        last_y = y + stride

        if last_x + window_size > image.width:
            for y in range(0, image.height - window_size + 1, stride):
                cropped_images.append(image.crop((image.width - window_size, y, image.width, y + window_size)))
        if last_y + window_size > image.height:
            for x in range(0, image.width - window_size + 1, stride):
                cropped_images.append(image.crop((x, image.height - window_size, x + window_size, image.height)))
        if last_x + window_size > image.width and last_y + window_size > image.height:
            cropped_images.append(image.crop((image.width - window_size, image.height - window_size, image.width, image.height)))

    # 3. 遍历所有图块，计算熵并与阈值比较
    for crop in cropped_images:
        crop_gray = crop.convert('L')
        crop_gray_np = np.array(crop_gray) / 255.0
        
        # 计算全局直方图熵
        histogram, _ = np.histogram(crop_gray_np.flatten(), bins=256, range=(0, 1), density=True) 
        prob_dist = histogram / histogram.sum()
        m = stats.entropy(prob_dist + 1e-7, base=2)
        
        # 如果熵值大于阈值，则保留该图块
        if m > threshold: 
            texture_images.append(crop)

    # 4. 如果没有一个图块满足条件，则返回原始图像的中心裁剪作为一个备选方案
    if len(texture_images) == 0:
        texture_images = [CenterCrop(image.size)(image)]

    return texture_images

class TextureCrop:
    """
    一个封装了texture_crop功能的变换类，可用于torchvision.transforms.Compose。
    它会从texture_crop返回的多个图块中选择一个。
    """
    def __init__(self, window_size, stride=32, metric='ghe', position='top', n=10, random_choice=True):
        self.window_size = window_size
        self.stride = stride if stride is not None else window_size
        self.metric = metric
        self.position = position
        self.n = n
        self.random_choice = random_choice

    def __call__(self, image):
        """
        Args:
            image (PIL Image): 输入图像。
        Returns:
            PIL Image: 经过纹理裁剪后选择的图块。
        """
        # 调用核心函数获取一组高纹理图块
        texture_images = texture_crop(
            image,
            stride=self.stride,
            window_size=self.window_size,
            metric=self.metric,
            position=self.position,
            n=self.n
        )
        
        # 如果没有找到符合条件的图块，则使用中心裁剪作为备选方案
        if not texture_images:
            return transforms.CenterCrop(self.window_size)(image)
        
        # 根据设置，随机选择一个或选择第一个（最符合条件的）
        if self.random_choice:
            return random.choice(texture_images)
        else:
            return texture_images[0]