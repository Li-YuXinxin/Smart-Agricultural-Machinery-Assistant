import math
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.common_utils import default_logger
from pathlib import Path
from config.config import settings
from safetensors.torch import load_file
import torchvision.models as models

class ConsineClassifier:

    # 余弦分类器(课程中使用，实际开发中可以直接使用线性分类器)
    def __init__(self, in_features: int, num_classes: int, scale: float = 1.0):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.scale = nn.Parameter(torch.tensor(scale,dtype=torch.float))
        self.weight = nn.Parameter(torch.empty(in_features, num_classes))
        # 让参数生效
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5.0))

    def forward(self, x):
        # 归一化权重
        weight_norm = F.normalize(self.weight, dim=1, p=2)
        x_norm = F.normalize(x, dim=1, p=2)
        # 计算余弦相似度
        cos_sim = F.linear(x_norm, weight_norm)
        return cos_sim * self.scale
    
def adapt_timm_resnet_state_dict(state_dict):
    # 将微软的模型，映射为原始模型状态
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("resnet.embedder.embedder.convolution"):
            # 卷积层
            suffix = k.split(".")[-1]
            new_state_dict[f"conv1.{suffix}"] = v
            continue
        elif k.startswith("resnet.embedder.embedder.normalization"):
            # 归一化层
            suffix = k.split(".")[-1]
            new_state_dict[f"bn1.{suffix}"] = v
            continue
        elif k.startswith("classifier."):
            # 全连接层
            parts = k.split(".")
            if len(parts) >= 3 and parts[1] == '1':
                new_state_dict[f"fc.{parts[-1]}"] = v
            continue
        
        # (主分支)提取编码器中，卷积层和归一化层的权重、平移、滑动均值、滑动方差
        pattern = re.compile(r"^resnet\.encoder\.stages\.(\d+)\
                             \.layers\.(\d+)\.layer\.(\d+)\
                             \.(convolution|normalization)\
                             \.(weight|bias|running_mean|running_var)$")
        
        m = pattern.match(k)
        if m:
            stage = int(m.group(1))
            block = int(m.group(2))
            layer_idx = int(m.group(3))
            kind = m.group(4)
            suffix = m.group(5)
            layer_name = f"layer{stage+1}"
            if kind == 'convolution':
                new_key = f"{layer_name}.{block}.conv{layer_idx+1}.{suffix}"
            else:
                new_key = f"{layer_name}.{block}.bn{layer_idx+1}.{suffix}"
            new_state_dict[new_key] = v
            continue
        
        # 提取(捷径分支)编码器中，卷积层和归一化层的权重、平移、滑动均值、滑动方差
        shortcut_pattern = re.compile(r"^resnet\.encoder\.stages\.(\d+)\
                                    \.layers\.(\d+)\.shortcut\
                                    \.(convolution|normalization)\
                                    \.(weight|bias|running_mean|running_var)$")
        
        m = shortcut_pattern.match(k)
        if m:
            stage = int(m.group(1))
            block = int(m.group(2))
            layer_idx = int(m.group(3))
            kind = m.group(4)
            suffix = m.group(5)
            layer_name = f"layer{stage+1}"
            if kind == 'convolution':
                new_key = f"{layer_name}.{block}.downsample.0.{suffix}"
            else:
                new_key = f"{layer_name}.{block}.downsample.1.{suffix}"
            new_state_dict[new_key] = v
            continue
        
    default_logger.info(f"适配后的STATE_DICT的键的数量：{len(new_state_dict)}")
    return new_state_dict
    
def load_resnet_50_from_local_safetensors():
    # 从本地加载ResNet-50模型
    # model_path = Path(settings.RESNET50_MODEL_PATH)
    try: 
        raw_state_dict = load_file(f"{settings.RESNET50_MODEL_PATH}/model.safetensors")
        default_logger.info(f"原始STATE_DICT的键的数量：{len(raw_state_dict)}")
    except Exception as e:
        raise RuntimeError(f"加载原始ResNet-50模型失败：{e}")
    
    sample_keys = list(raw_state_dict.keys())  # 获取所有键名
    # if any(k.startswith("resnet.encoder.stages") for k in sample_keys):
    #     default_logger.info(f"非标准key")
    # else:
    #     default_logger.info(f"标准key")
        
    # 转换为标准key
    adapted_state_dict = adapt_timm_resnet_state_dict(raw_state_dict)

    # 构建标准模型
    model = models.resnet50(weight=None)

    # 处理旧的fc权重和偏置项
    old_fc_weight = None
    if 'fc.weight' in adapted_state_dict:
        old_fc_weight = adapted_state_dict['fc.weight']
        del adapted_state_dict['fc.weight']
    if 'fc.bias' in adapted_state_dict:
        del adapted_state_dict['fc.bias']    
        
    # 处理骨架权重
    missing_keys, unexpected_keys = model.load_state_dict(adapted_state_dict, strict=False)

    # 标准模型必有的key
    essential_keys = ['conv1.weight', 'bn1.weight', 'layer1.0.conv1.weight']

    # 查找必备但却丢失的key
    missing_essential = [k for k in essential_keys if k in missing_keys]

    if missing_essential:
        default_logger.error(f"缺失的必备key: {missing_essential}")
        raise RuntimeError(f"缺失的必备key: {missing_essential}")
    
    # 替换fc层
    in_features = model.fc.in_features
    num_classes = 1000
    model.fc = ConsineClassifier(in_features, num_classes)
    if old_fc_weight is not None:
        model.fc.weight = old_fc_weight
        default_logger.info(f"旧的fc层权重已替换为新的权重")
    else:
        default_logger.info(f"没有旧的fc层权重")
    return model