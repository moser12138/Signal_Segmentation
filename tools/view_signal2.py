# 不同类别分别存储
import glob
import argparse
import cv2
import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from models.signal_segnet import SemanticSegmentationNet

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

color_map = [
    (0, 0, 0),
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255)]


def parse_args():
    parser = argparse.ArgumentParser(description='Custom Input')
    parser.add_argument('--a', help='siganl', default='signal', type=str)
    parser.add_argument('--c', help='cityscapes pretrained or not', type=bool, default=True)
    # parser.add_argument('--p', help='dir for pretrained model', default='../output/signal/1_signal/best_1_0_81_88.pt', type=str)
    # parser.add_argument('--p', help='dir for pretrained model', default='../output/signal/2_signal/best_2_0_83_35.pt', type=str)
    # parser.add_argument('--p', help='dir for pretrained model', default='../output/signal/3_signal//best_3_15_87_23.pt', type=str)
    parser.add_argument('--p', help='dir for pretrained model', default='../output/signal/4_signal/best_4_20_87_44.pt', type=str)
    # parser.add_argument('--p', help='dir for pretrained model', default='../output/signal/5_signal/experiment2/best_5_20_84_49.pt', type=str)

    # parser.add_argument('--r', help='root or dir for input images', default='./fig_signal/1_signal/', type=str)
    # parser.add_argument('--r', help='root or dir for input images', default='./fig_signal/2_signal/', type=str)
    # parser.add_argument('--r', help='root or dir for input images', default='./fig_signal/3_signal/', type=str)
    parser.add_argument('--r', help='root or dir for input images', default='./fig_signal/4_signal/', type=str)
    # parser.add_argument('--r', help='root or dir for input images', default='./fig_signal/5_signal/', type=str)
    parser.add_argument('--t', help='the format of input images (.jpg, .png, ...)', default='.png', type=str)
    args = parser.parse_args()
    return args


def input_transform(image):
    image = image.astype(np.float32)[:, :, ::-1]
    image = image / 255.0
    image -= mean
    image /= std
    return image


def load_pretrained(model, pretrained):
    pretrained_dict = torch.load(pretrained, map_location='cpu')
    if 'state_dict' in pretrained_dict:
        pretrained_dict = pretrained_dict['state_dict']
    model_dict = model.state_dict()
    pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items() if
                       (k[6:] in model_dict and v.shape == model_dict[k[6:]].shape)}
    msg = 'Loaded {} parameters!'.format(len(pretrained_dict))
    print('Attention!!!')
    print(msg)
    print('Over!!!')
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict, strict=False)
    return model


if __name__ == '__main__':
    args = parse_args()
    images_list = glob.glob(args.r + '*' + args.t)
    sv_path = args.r + 'outputs/'

    model = SemanticSegmentationNet(num_classes=6, augment=False)
    model = load_pretrained(model, args.p).cuda()
    model.eval()

    with torch.no_grad():
        for img_path in images_list:
            img_name = os.path.basename(img_path)
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            original_img = img.copy()

            img = input_transform(img)
            img = img.transpose((2, 0, 1)).copy()
            img = torch.from_numpy(img).unsqueeze(0).cuda()
            pred = model(img)
            pred = torch.argmax(pred, dim=1).squeeze(0).cpu().numpy()

            # 存储不同类别的分割结果
            for i, color in enumerate(color_map):
                mask = (pred == i).astype(np.uint8) * 255
                category_img = np.zeros_like(original_img).astype(np.uint8)
                for j in range(3):
                    category_img[:, :, j] = mask * (color[j] / 255.0)

                category_img = Image.fromarray(category_img)
                category_path = os.path.join(sv_path, f"category_{i}")
                os.makedirs(category_path, exist_ok=True)
                category_img.save(os.path.join(category_path, img_name))

            # 叠加分割结果到原始图像
            seg_img = np.zeros_like(original_img, dtype=np.uint8)
            for i, color in enumerate(color_map):
                seg_img[pred == i] = color

            overlay = cv2.addWeighted(original_img, 0.2, seg_img, 0.8, 0)
            overlay_img = Image.fromarray(overlay)

            os.makedirs(sv_path, exist_ok=True)
            overlay_img.save(os.path.join(sv_path, img_name))
