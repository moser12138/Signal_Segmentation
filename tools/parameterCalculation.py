import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

from models.model_utils import BasicBlock

from models.compare.DDRNet_23_slim import DualResNet
from models.compare.resnet import Resnet18
from models.compare.bisenetv2 import BiSeNetV2
from models.MFFNet import SemanticSegmentationNet


def compute_model_complexity(model, input_size):
    """
    计算模型的参数量（M）和 FLOPs（GFlops）
    :param model: PyTorch 模型
    :param input_size: 输入数据的尺寸，例如 (1, 3, 224, 224)
    """
    summary_info = summary(model, input_size=input_size, verbose=0)

    # 以 M（百万）为单位的参数量，以 G（十亿）为单位的 FLOPs
    params = summary_info.total_params / 1e6  # M
    flops = summary_info.total_mult_adds / 1e9  # GFlops

    return params, flops

# 初始化模型和输入尺寸
model1 = SemanticSegmentationNet(num_classes=6, augment=True)
model2 = BiSeNetV2(num_classes=6, augment=True)
model3 = Resnet18(num_classes=6, augment=True)
model4 = DualResNet(BasicBlock, [2, 2, 2, 2], num_classes=6, augment=True)


input_size = (1, 3, 800, 800)  # batch_size=1, 1通道, 224x224 分辨率

# 计算参数量和 FLOPs
params1, flops1 = compute_model_complexity(model1, input_size)
params2, flops2 = compute_model_complexity(model2, input_size)
params3, flops3 = compute_model_complexity(model3, input_size)
params4, flops4 = compute_model_complexity(model4, input_size)

print(f"Model Params: {params1:.3f}M")
print(f"Model FLOPs: {flops1:.3f}G")

print(f"Model Params: {params2:.3f}M")
print(f"Model FLOPs: {flops2:.3f}G")

print(f"Model Params: {params3:.3f}M")
print(f"Model FLOPs: {flops3:.3f}G")

print(f"Model Params: {params4:.3f}M")
print(f"Model FLOPs: {flops4:.3f}G")
