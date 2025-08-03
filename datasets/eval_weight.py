import os
import numpy as np
from PIL import Image
import math

# 颜色映射表
COLORMAP = {
    0: (0, 0, 0),        # 背景，黑色
    51: (255, 0, 0),     # 红色
    102: (0, 255, 0),    # 绿色
    153: (0, 0, 255),    # 蓝色
    204: (255, 255, 0),  # 黄色
    255: (255, 0, 255)   # 紫色
}

def count_pixels_in_class(annotation_dir, categories, snr_levels, colormap):
    """
    统计每个类别的像素数量
    :param annotation_dir: 标注文件根目录
    :param categories: 类别列表
    :param snr_levels: SNR 列表
    :param colormap: 颜色映射字典
    :return: 每个类别的像素计数数组
    """
    pixel_counts = np.zeros(len(colormap))

    for category in categories:
        cat_dir = os.path.join(annotation_dir, category)
        for class_dir in os.listdir(cat_dir):
            class_path = os.path.join(cat_dir, class_dir)
            if os.path.isdir(class_path):
                for snr in snr_levels:
                    snr_path = os.path.join(class_path, snr)
                    if not os.path.exists(snr_path):
                        continue
                    for file in os.listdir(snr_path):
                        if file.endswith('.png'):
                            png_path = os.path.join(snr_path, file)
                            image = np.array(Image.open(png_path).convert('RGB'))  # 转换为 RGB
                            for i, (value, color) in enumerate(colormap.items()):
                                pixel_counts[i] += np.sum(np.all(image == color, axis=-1))

    return pixel_counts

def compute_class_balancing_weights(pixel_counts):
    """
    计算类别平衡权重
    :param pixel_counts: 每个类别的像素计数
    :return: 每个类别的权重
    """
    weights = np.array([1 / math.log(1 + count + 1e-6) for count in pixel_counts])  # 加 1e-6 防止 log(0)
    return weights

# 配置参数
annotation_dir = '../data/signal/annotations/train/'  # 标注文件根目录
categories = ['1', '2', '3', '4', '5']
snr_levels = ['-5', '0', '5', '10', '15', '20']

# 统计像素数量并计算权重
pixel_counts = count_pixels_in_class(annotation_dir, categories, snr_levels, COLORMAP)
class_weights = compute_class_balancing_weights(pixel_counts)

# 输出结果
print("类别像素计数:", pixel_counts)
print("类别平衡权重:", class_weights)
