import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from .model_utils import BasicBlock, Bottleneck, segmenthead, DAPPM, PAPPM, PagFM, Bag, Light_Bag
import logging

from .signal_segnet import SemanticSegmentationNet
# from .MFFNet import SemanticSegmentationNet
from .compare.bisenetv2 import BiSeNetV2
from .compare.resnet import Resnet18
from .compare.DDRNet_23_slim import DualResNet

BatchNorm2d = nn.BatchNorm2d
bn_mom = 0.1
algc = False

def get_seg_model(cfg, imgnet_pretrained):

    model = SemanticSegmentationNet(num_classes=cfg.DATASET.NUM_CLASSES, augment=True)
    model = BiSeNetV2(num_classes=cfg.DATASET.NUM_CLASSES, augment=True)
    model = Resnet18(num_classes=cfg.DATASET.NUM_CLASSES, augment=True)
    model = DualResNet(BasicBlock, [2, 2, 2, 2], num_classes=cfg.DATASET.NUM_CLASSES, augment=True)

    if imgnet_pretrained:
        pretrained_state = torch.load(cfg.MODEL.PRETRAINED, map_location='cpu')
        model_dict = model.state_dict()
        pretrained_state = {k: v for k, v in pretrained_state.items() if (k in model_dict and v.shape == model_dict[k].shape)}
        model_dict.update(pretrained_state)
        msg = 'Loaded {} parameters!'.format(len(pretrained_state))

        logging.info('Attention!!!')
        logging.info(msg)
        logging.info('Over!!!')

        model.load_state_dict(model_dict, strict = False)
    else:
        pretrained_dict = torch.load(cfg.MODEL.PRETRAINED, map_location='cpu')
        if 'state_dict' in pretrained_dict:
            pretrained_dict = pretrained_dict['state_dict']
        model_dict = model.state_dict()
        pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items() if (k[6:] in model_dict and v.shape == model_dict[k[6:]].shape)}
        msg = 'Loaded {} parameters!'.format(len(pretrained_dict))

        logging.info('Attention!!!')
        logging.info(msg)
        logging.info('Over!!!')

        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict, strict = False)
    
    return model

# def get_pred_model(name, num_classes):
def get_pred_model(num_classes):
    model = SemanticSegmentationNet(num_classes=num_classes, augment=False)
    return model

# 测试模型的推理速度（FPS）和延迟（latency），并打印出结果
if __name__ == '__main__':
    device = torch.device('cuda')
    model = get_pred_model(num_classes=19)
    model.eval()
    model.to(device)
    iterations = None
    
    input = torch.randn(1, 3, 1024, 2048).cuda()

    with torch.no_grad():#上下文管理器来禁用梯度计算，从而减少内存消耗，并且不记录计算历史
        for _ in range(10):
            model(input)
    
        if iterations is None:
            elapsed_time = 0
            iterations = 100
            while elapsed_time < 1:
                torch.cuda.synchronize()
                torch.cuda.synchronize()
                t_start = time.time()
                for _ in range(iterations):
                    model(input)
                torch.cuda.synchronize()
                torch.cuda.synchronize()
                elapsed_time = time.time() - t_start
                iterations *= 2
            FPS = iterations / elapsed_time
            iterations = int(FPS * 6)
    
        print('=========Speed Testing=========')
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        t_start = time.time()
        for _ in range(iterations):
            model(input)
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        elapsed_time = time.time() - t_start
        latency = elapsed_time / iterations * 1000
    torch.cuda.empty_cache()
    FPS = 1000 / latency
    print(FPS)