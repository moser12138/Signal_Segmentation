import argparse
import os
import pprint
import logging
import timeit
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from collections import defaultdict

import _init_paths
import models
import datasets
from configs import config
from configs import update_config
from utils.utils import create_logger
from models.signal_segnet import SemanticSegmentationNet

# 颜色标签映射
RGB_LABEL_MAP = {
    (0, 0, 0): 0,           # background
    (255, 0, 0): 1,         # LFM
    (0, 255, 0): 2,         # NLFM
    (0, 0, 255): 3,         # SFM
    (255, 255, 0): 4,       # FSK
    (255, 0, 255): 5        # QAM
}

IDX2CLASS = ['BG', 'LFM', 'NLFM', 'SFM', 'FSK', 'QAM']
NUM_CLASSES = len(IDX2CLASS)
# DETECTION_THRESHOLD = 0.0003  # 默认像素比例阈值

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate multi-label classification from segmentation results')
    parser.add_argument('--cfg', default="configs/signal/signal.yaml", type=str, help='experiment configure file name')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    update_config(config, args)
    return args

def extract_gt_from_filename(name):
    """从文件名中提取真实多标签（one-hot）"""
    label_vec = np.zeros(NUM_CLASSES, dtype=int)
    label_vec[0] = 1  # 背景始终为1
    for i, cls in enumerate(IDX2CLASS[1:], start=1):  # 跳过BG
        if cls in name:
            label_vec[i] = 1
    return label_vec

def extract_pred_from_mask(pred_mask_rgb, threshold, verbose=True):
    """从RGB预测图中提取标签向量（超过阈值像素占比则认为存在）"""
    pred_vec = np.zeros(NUM_CLASSES, dtype=int)
    pred_vec[0] = 1  # BG始终存在
    h, w, _ = pred_mask_rgb.shape
    total_pixels = h * w
    pred_flat = pred_mask_rgb.reshape(-1, 3)
    color_count = defaultdict(int)

    for rgb in map(tuple, pred_flat):
        color_count[rgb] += 1

    for rgb, idx in RGB_LABEL_MAP.items():
        if idx == 0:
            continue
        pixel_count = color_count.get(rgb, 0)
        pixel_ratio = pixel_count / total_pixels
        if pixel_ratio >= threshold:
            pred_vec[idx] = 1
        if verbose:
            print(f"{IDX2CLASS[idx]}: {pixel_ratio:.2%}")

    return pred_vec

def evaluate(model, dataloader, logger,DETECTION_THRESHOLD):
    model.eval()
    gt_list = []
    pred_list = []

    with torch.no_grad():
        for i, (image, label, edge, size, name) in enumerate(dataloader):
            image = image.cuda()
            pred = model(image)[0]
            pred = torch.argmax(pred, dim=1).squeeze(0).cpu().numpy()

            # 转换为RGB mask
            rgb_mask = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
            for rgb, idx in RGB_LABEL_MAP.items():
                rgb_mask[pred == idx] = rgb

            pred_label = extract_pred_from_mask(rgb_mask, threshold=DETECTION_THRESHOLD)
            gt_label = extract_gt_from_filename(name[0])

            gt_list.append(gt_label)
            pred_list.append(pred_label)

    gt_array = np.stack(gt_list)
    pred_array = np.stack(pred_list)

    # Sample-level overall metrics
    acc = accuracy_score(gt_array, pred_array)
    precision = precision_score(gt_array, pred_array, average='samples', zero_division=0)
    recall = recall_score(gt_array, pred_array, average='samples', zero_division=0)
    f1 = f1_score(gt_array, pred_array, average='samples', zero_division=0)

    logger.info("===> Multi-label Classification Evaluation (Sample-wise Average):")
    logger.info(f"  Accuracy  : {acc:.4f}")
    logger.info(f"  Precision : {precision:.4f}")
    logger.info(f"  Recall    : {recall:.4f}")
    logger.info(f"  F1 Score  : {f1:.4f}")
    print("===> Multi-label Classification Evaluation (Sample-wise Average):")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  F1 Score  : {f1:.4f}")

    # Per-class metrics
    logger.info("\n===> Per-Class Metrics:")
    print("\n===> Per-Class Metrics:")
    for i, class_name in enumerate(IDX2CLASS):
        class_gt = gt_array[:, i]
        class_pred = pred_array[:, i]

        class_acc = accuracy_score(class_gt, class_pred)
        class_prec = precision_score(class_gt, class_pred, zero_division=0)
        class_rec = recall_score(class_gt, class_pred, zero_division=0)
        class_f1 = f1_score(class_gt, class_pred, zero_division=0)

        logger.info(f"  [{class_name}] Acc: {class_acc:.4f}, Prec: {class_prec:.4f}, Rec: {class_rec:.4f}, F1: {class_f1:.4f}")
        print(f"  [{class_name}] Acc: {class_acc:.4f}, Prec: {class_prec:.4f}, Rec: {class_rec:.4f}, F1: {class_f1:.4f}")

    return acc, precision, recall, f1


def main(category, snr, dth):
    args = parse_args()
    logger, final_output_dir, _ = create_logger(config, args.cfg, f'multilabel_{category}_{snr}')
    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    model = models.model_factory.get_seg_model(config, imgnet_pretrained=True)
    model_state_file = os.path.join(final_output_dir, f'best_{category}_{snr}.pt')
    logger.info('=> loading model from {}'.format(model_state_file))

    pretrained_dict = torch.load(model_state_file)
    if 'state_dict' in pretrained_dict:
        pretrained_dict = pretrained_dict['state_dict']
    model_dict = model.state_dict()
    pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items() if k[6:] in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    model = model.cuda()

    test_list_path = os.path.join(config.DATASET.TEST_SET, f"validation_{category}_{snr}.lst")
    test_dataset = eval('datasets.' + config.DATASET.DATASET)(
        root=config.DATASET.ROOT,
        list_path=test_list_path,
        num_classes=config.DATASET.NUM_CLASSES,
        multi_scale=False,
        flip=False,
        ignore_label=config.TRAIN.IGNORE_LABEL,
        base_size=config.TEST.BASE_SIZE,
        crop_size=(config.TEST.IMAGE_SIZE[1], config.TEST.IMAGE_SIZE[0])
    )

    testloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    start = timeit.default_timer()
    evaluate(model, testloader, logger,DETECTION_THRESHOLD=dth)
    end = timeit.default_timer()
    logger.info('Mins: %d' % np.int32((end - start) / 60))
    logger.info('Done')

if __name__ == '__main__':
    # main('1', '-5', 0.0003)
    # main('1', '0', 0.0003)
    # main('1', '5', 0.0004)
    # main('1', '10', 0.0004)
    # main('1', '15', 0.0004)
    # main('1', '20', 0.0004)

    # main('2', '-5', 0.0003)
    main('2', '0', 0.003)
    main('2', '5', 0.003)
    main('2', '10', 0.003)
    main('2', '15', 0.003)
    main('2', '20', 0.003)
    #
    # main('3', '-5', 0.0003)
    # main('3', '0', 0.003)
    # main('3', '5', 0.004)
    # main('3', '10', 0.004)
    # main('3', '15', 0.004)
    # main('3', '20', 0.004)
    #
    # main('5', '-5', 0.0003)
    # main('5', '0', 0.003)
    # main('5', '5', 0.004)
    # main('5', '10', 0.004)
    # main('5', '15', 0.004)
    # main('5', '20', 0.004)
    # main('4', '-5', 0.0003)
    # main('4', '0', 0.003)
    # main('4', '5', 0.004)
    # main('4', '10', 0.004)
    # main('4', '15', 0.004)
    # main('4', '20', 0.004)


