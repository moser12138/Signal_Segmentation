import argparse
import os
import pprint

import logging
import timeit

import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
from tensorboardX import SummaryWriter

import _init_paths
import models
import datasets
from configs import config
from configs import update_config
from utils.criterion import CrossEntropy, OhemCrossEntropy, BondaryLoss
from utils.function import train, validate
from utils.utils import create_logger, FullModel

def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default="configs/signal/signal.yaml",
                        type=str)
    parser.add_argument('--seed', type=int, default=304)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    update_config(config, args)

    return args


def main(category, snr):
    args = parse_args()

    # 设置随机数种子，以便在所有的随机操作（如数据集的打乱、权重初始化等）中都使用相同的初始状态，从而使实验结果可重现性
    if args.seed > 0:
        import random
        print('Seeding with', args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    logger, final_output_dir, tb_log_dir = create_logger(config, args.cfg, f'train_{category}_{snr}')

    logger.info(pprint.pformat(args))  # 将命令行参数的值记录在日志中
    logger.info(config)  # 函数将配置文件的内容记录在日志中

    # 创建了一个字典writer_dict，其中包含了TensorBoard摘要写入器需要的信息。使用SummaryWriter()函数创建一个写入器，tb_log_dir是TensorBoard摘要写入的目录。train_global_steps和valid_global_steps初始化为0，用于记录训练和验证的全局步数
    writer_dict = {
        'writer': SummaryWriter(tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    # cudnn related setting  大部分为读取config的配置值
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED
    gpus = list(config.GPUS)

    if torch.cuda.device_count() != len(gpus):
        print("The gpu numbers do not match!")
        return 0

    # 判断配置文件config中是否设置了模型的预训练
    imgnet = 'imagenet' in config.MODEL.PRETRAINED

    # 创建模型。该函数接受两个参数，一个是配置文件config，另一个是imgnet_pretrained，用于指示是否使用预训练的ImageNet模型。根据imgnet的值，该函数将会返回一个对应的模型
    model = models.model_factory.get_seg_model(config, imgnet_pretrained=imgnet)
    # model = models.model_factory.get_pred_model(config)


    # 每个GPU的个数乘以GPU数
    batch_size = config.TRAIN.BATCH_SIZE_PER_GPU * len(gpus)

    # prepare data
    crop_size = (config.TRAIN.IMAGE_SIZE[1], config.TRAIN.IMAGE_SIZE[0])
    # output_file = os.path.join(output_dir, key)
    train_list_path = os.path.join(config.DATASET.TRAIN_SET, f"train_{category}_{snr}.lst")
    train_dataset = eval('datasets.' + config.DATASET.DATASET)(
        root=config.DATASET.ROOT,
        list_path=train_list_path,
        num_classes=config.DATASET.NUM_CLASSES,
        multi_scale=config.TRAIN.MULTI_SCALE,
        flip=config.TRAIN.FLIP,
        ignore_label=config.TRAIN.IGNORE_LABEL,  # 忽略标签的像素值
        base_size=config.TRAIN.BASE_SIZE,  # 基础尺寸，用于进行尺度变换
        crop_size=crop_size,  # 需要裁剪的尺寸
        scale_factor=config.TRAIN.SCALE_FACTOR)  # 尺度因子，用于进行尺度变换。

    trainloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=config.TRAIN.SHUFFLE,
        num_workers=config.WORKERS,
        pin_memory=False,
        drop_last=True)

    test_size = (config.TEST.IMAGE_SIZE[1], config.TEST.IMAGE_SIZE[0])
    test_list_path = os.path.join(config.DATASET.TEST_SET, f"validation_{category}_{snr}.lst")

    test_dataset = eval('datasets.' + config.DATASET.DATASET)(
        root=config.DATASET.ROOT,
        list_path=test_list_path,
        num_classes=config.DATASET.NUM_CLASSES,
        multi_scale=False,
        flip=False,
        ignore_label=config.TRAIN.IGNORE_LABEL,
        base_size=config.TEST.BASE_SIZE,
        crop_size=test_size)

    testloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.TEST.BATCH_SIZE_PER_GPU * len(gpus),
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=False)

    # criterion
    if config.LOSS.USE_OHEM:
        sem_criterion = OhemCrossEntropy(ignore_label=config.TRAIN.IGNORE_LABEL,
                                         thres=config.LOSS.OHEMTHRES,
                                         min_kept=config.LOSS.OHEMKEEP,
                                         weight=train_dataset.class_weights)
    else:
        sem_criterion = CrossEntropy(ignore_label=config.TRAIN.IGNORE_LABEL,
                                     weight=train_dataset.class_weights)

    model = FullModel(model, sem_criterion)
    model = nn.DataParallel(model, device_ids=gpus).cuda()

    # optimizer
    if config.TRAIN.OPTIMIZER == 'sgd':
        params_dict = dict(model.named_parameters())
        params = [{'params': list(params_dict.values()), 'lr': config.TRAIN.LR}]

        optimizer = torch.optim.SGD(params,
                                    lr=config.TRAIN.LR,
                                    momentum=config.TRAIN.MOMENTUM,
                                    weight_decay=config.TRAIN.WD,
                                    nesterov=config.TRAIN.NESTEROV,
                                    )
    else:
        raise ValueError('Only Support SGD optimizer')

    epoch_iters = int(train_dataset.__len__() / config.TRAIN.BATCH_SIZE_PER_GPU / len(gpus))

    best_mIoU = 0
    last_epoch = 0

    flag_rm = config.TRAIN.RESUME
    if config.TRAIN.RESUME:
        model_state_file = os.path.join(final_output_dir, f'checkpoint_{category}_{snr}.pth.tar')
        if os.path.isfile(model_state_file):
            checkpoint = torch.load(model_state_file, map_location={'cuda:0': 'cpu'})
            best_mIoU = checkpoint['best_mIoU']
            last_epoch = checkpoint['epoch']
            dct = checkpoint['state_dict']

            model.module.model.load_state_dict(
                {k.replace('model.', ''): v for k, v in dct.items() if k.startswith('model.')})
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))

    start = timeit.default_timer()
    end_epoch = config.TRAIN.END_EPOCH
    num_iters = config.TRAIN.END_EPOCH * epoch_iters
    # real_end = 120 + 1 if 'camvid' in config.DATASET.TRAIN_SET else end_epoch
    real_end = end_epoch

    for epoch in range(last_epoch, real_end):

        current_trainloader = trainloader
        if current_trainloader.sampler is not None and hasattr(current_trainloader.sampler, 'set_epoch'):
            current_trainloader.sampler.set_epoch(epoch)

        train(config, epoch, config.TRAIN.END_EPOCH,
              epoch_iters, config.TRAIN.LR, num_iters,
              trainloader, optimizer, model, writer_dict)

        if flag_rm == 1 or (epoch % 5 == 0 and epoch < real_end - 100) or (epoch >= real_end - 100):
            valid_loss, mean_IoU, IoU_array = validate(config, testloader, model, writer_dict)
        if flag_rm == 1:
            flag_rm = 0

        logger.info('=> saving checkpoint to {}'.format(
            final_output_dir + f'checkpoint_{category}_{snr}.pth.tar'))
        torch.save({
            'epoch': epoch + 1,
            'best_mIoU': best_mIoU,
            'state_dict': model.module.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, os.path.join(final_output_dir, f'checkpoint_{category}_{snr}.pth.tar'))
        if mean_IoU > best_mIoU:
            best_mIoU = mean_IoU
            torch.save(model.module.state_dict(), os.path.join(final_output_dir, f'best_{category}_{snr}.pt'))
        msg = 'Loss: {:.3f}, MeanIU: {: 4.4f}, Best_mIoU: {: 4.4f}'.format(
            valid_loss, mean_IoU, best_mIoU)
        logging.info(msg)
        logging.info(IoU_array)

    torch.save(model.module.state_dict(),os.path.join(final_output_dir, 'final_state.pt'))

    writer_dict['writer'].close()
    end = timeit.default_timer()
    logger.info('Hours: %d' % int((end - start) / 3600))
    logger.info('Done')


if __name__ == '__main__':
    categories = ['1', '2', '3', '4', '5']
    snr_levels = ['-5', '0', '5', '10', '15', '20']

    # for category in categories:
    #     for snr in snr_levels:
    #         print(f"Training for category {category}, SNR {snr}")
    #         train_model(model, data_dir, category, snr, resume=True)

    # main('1', '-5') #完成
    # main('1', '0') #完成
    # main('1', '5') #完成
    # main('1', '10') #完成
    # main('1', '15') #完成
    # main('1', '20') #完成

    # main('2', '-5') #完成
    # main('2', '0') #完成
    # main('2', '5') #完成
    # main('2', '10') #完成
    # main('2', '15') #完成
    # main('2', '20') #完成

    # main('3', '-5') #完成
    # main('3', '0') #完成
    # main('3', '5') #完成
    # main('3', '10') #完成
    # main('3', '15') #完成
    # main('3', '20') #完成

    # main('4', '-5') #完成
    # main('4', '0') #完成
    # main('4', '5') #完成
    # main('4', '10') #完成
    # main('4', '15') #完成
    # main('4', '20') #完成

    # main('5', '-5') #完成 （使用了像素权重）
    # main('5', '0') #完成 （使用了像素权重）
    # main('5', '5') #完成 （使用了像素权重）
    # main('5', '10') #完成 （使用了像素权重）
    # main('5', '15') #完成 （使用了像素权重）
    # main('5', '20') #完成 （使用了像素权重）

    # main('5', '-5') #完成 （未使用了像素权重）
    # main('5', '0') #完成 （未使用了像素权重）
    # main('5', '5') #完成 （未使用了像素权重）
    # main('5', '10') #完成 （未使用了像素权重）
    # main('5', '15') #完成 （未使用了像素权重）
    # main('5', '20') #完成 （未使用了像素权重）

    # main('5', '-5') #完成 resnet对比试验
    # main('5', '0') #完成 resnet对比试验
    # main('5', '5') #完成 resnet对比试验
    # main('5', '10') #完成 resnet对比试验
    # main('5', '15') #完成 resnet对比试验
    # main('5', '20') #完成 resnet对比试验

    # main('5', '-5') #完成 bisenet对比试验
    # main('5', '0') #完成 bisenet对比试验
    # main('5', '5') #完成 bisenet对比试验
    # main('5', '10') #完成 bisenet对比试验
    # main('5', '15') #完成 bisenet对比试验
    # main('5', '20') #完成 bisenet对比试验

    # main('5', '-5') #完成 ddrnet对比试验
    main('5', '0') #完成 ddrnet对比试验
    main('5', '5') #完成 ddrnet对比试验
    main('5', '10') #完成 ddrnet对比试验
    main('5', '15') #完成 ddrnet对比试验
    main('5', '20') #完成 ddrnet对比试验

    # main('5', '-5') #消融实验
    # main('5', '0') #消融实验
    # main('5', '5') #消融实验
    # main('5', '10') #消融实验
    # main('5', '15') #消融实验
    # main('5', '20') #消融实验
