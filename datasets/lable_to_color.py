"""
此代码将灰度图标签文件转化为彩色标注
"""
import os
import shutil
import scipy.io
import numpy as np
import re
from PIL import Image

# 定义灰度值到颜色的映射表
COLORMAP = {
    0: (0, 0, 0),  # 背景，黑色
    51: (255, 0, 0),  # 灰度值 0.2 -> 红色
    102: (0, 255, 0),  # 灰度值 0.4 -> 绿色
    153: (0, 0, 255),  # 灰度值 0.6 -> 蓝色
    204: (255, 255, 0),  # 灰度值 0.8 -> 黄色
    255: (255, 0, 255)  # 灰度值 1.0 -> 紫色
}


def convert_gray_to_color(image_path, save_path):
    # 打开灰度图像
    image = Image.open(image_path).convert('L')
    image = np.array(image)

    # 创建彩色图像
    color_image = np.zeros((*image.shape, 3), dtype=np.uint8)

    # 遍历灰度值并应用颜色映射
    for gray_val, color in COLORMAP.items():
        color_image[image == gray_val] = color

    # 保存彩色图像
    color_image = Image.fromarray(color_image)
    color_image.save(save_path)
    print(f"Converted {image_path} to color map and saved as {save_path}")


def reorganize_dataset_and_convert_mat(src_root, dst_root):
    for main_dir in ['annotations', 'images']:
        for split in ['train', 'validation']:
            src_dir = os.path.join(src_root, main_dir, split)
            for class_dir in os.listdir(src_dir):
                class_path = os.path.join(src_dir, class_dir)
                if os.path.isdir(class_path):
                    for subtype_dir in os.listdir(class_path):
                        subtype_path = os.path.join(class_path, subtype_dir)
                        if os.path.isdir(subtype_path):
                            mat_dir = os.path.join(subtype_path, 'mat')
                            if os.path.exists(mat_dir):
                                for file in os.listdir(mat_dir):
                                    if file.endswith('.mat'):
                                        parts = file.split('_')
                                        snr = parts[-2]

                                        mat_path = os.path.join(mat_dir, file)
                                        data = scipy.io.loadmat(mat_path)
                                        matched_keys = [key for key in data.keys() if re.match(r'iq_signal\d+', key)]
                                        for key in matched_keys:
                                            iq_signal = data[key]
                                            if iq_signal.shape[1] >= 1000:
                                                iq_signal_trimmed = iq_signal[:, -1000:]
                                                dst_path = os.path.join(dst_root, main_dir, split, class_dir,
                                                                        subtype_dir, snr)
                                                os.makedirs(dst_path, exist_ok=True)
                                                npy_filename = file.replace('.mat', f'_{key}.npy')
                                                npy_path = os.path.join(dst_path, npy_filename)
                                                np.save(npy_path, iq_signal_trimmed)
                                                print(f"Converted {mat_path} ({key}) to {npy_path}")
                                            else:
                                                print(f"Skipping {mat_path} ({key}): Data has less than 1000 columns.")

                            for file in os.listdir(subtype_path):
                                if file.endswith('.png'):
                                    parts = file.split('_')
                                    snr = parts[-2]
                                    dst_path = os.path.join(dst_root, main_dir, split, class_dir, subtype_dir, snr)
                                    os.makedirs(dst_path, exist_ok=True)
                                    src_file_path = os.path.join(subtype_path, file)
                                    dst_file_path = os.path.join(dst_path, file)

                                    # 转换灰度图为彩色图
                                    convert_gray_to_color(src_file_path, dst_file_path)


# 源目录和目标目录
src_dataset = 'dataset'
dst_dataset = 'dataset2'

# 重新组织数据集并转换 mat 文件和标注图
reorganize_dataset_and_convert_mat(src_dataset, dst_dataset)
