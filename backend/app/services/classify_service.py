import json
import math
import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.common_utils import default_logger
from pathlib import Path
from config.config import settings
from safetensors.torch import load_file
import torchvision.models as models
import torchvision.transforms as transforms
import threading
import gc
from torchvision.models import ResNet50_Weights
import transformers
from torchvision import datasets
from torch.utils.data import WeightedRandomSampler

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

class ClassifyService:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized'):
            # 如果已经初始化，则直接返回
            return
        # 如果没有初始化，做初始化标记
        self._initialized = True
        self.device = torch.device(settings.DEVICE)
        self.model = None
        # 当前模型支持的识别列表
        self.class_names = []
        # 标识模型是否载入成功
        self.model_ready = False
        # 图像归一化处理
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            # 归一化：0-1
            transforms.ToTensor(),
            # 归一化参数平均差/标准差
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
        ])
        self._load_model()
        
    def unload_model(self):
        # 卸载模型
        if self.model is not None:
            # 清空内存
            del self.model
            self.model = None
            gc.collect()
            if torch.cuda.is_available():
                # 清空显存
                torch.cuda.empty_cache()
            default_logger.info(f"模型已卸载")
        else:
            default_logger.info(f"模型未加载")

    def _load_model(self):
        #加载模型
        self.unload_model()
        pth_path = Path(settings.RESNET50_FINETUNED_PTH_PATH)
        class_path = Path(settings.RESNET50_FINETUNED_CLASSNAMES_PATH)
        
        #尝试加载微调模型
        if pth_path.exists() and class_path.exists():
            try:
                with open(class_path, 'r',encoding='utf-8') as f:
                    class_names = json.load(f)
                num_classes = len(class_names)
                model = models.resnet50(weights=None)
                
                #提取分类的特征
                in_features = model.fc.in_features
                #替换分类器
                model.fc = ConsineClassifier(in_features, num_classes)
                #提取字典
                state_dict = torch.load(pth_path,map_location=self.device)
                #保存字典到标准模型
                model.load_state_dict(state_dict)
                #设置模型放入的位置
                model.to(self.device)
                #执行以上配置
                model.eval()
                
                self.model = model
                self.class_names = class_names
                #加载完毕
                self.model_ready = True
                default_logger.info(f"模型已加载,支持{num_classes}")
                return
                
            except Exception as e:
                default_logger.error(f"模型加载失败:{e}")
                
        default_logger.error(f"缺失微调模型,开始预训练模型")
        try:
            model = load_resnet_50_from_local_safetensors()
            self.model = model
            self.class_names = self._get_imagenet_classes()
            # 加载完毕
            self.model_ready = True
            default_logger.info(f"预训练模型已加载,支持{len(self.class_names)}个分类")
        except Exception as e:
            default_logger.error(f"加载预训练模型失败: {e}")
            raise RuntimeError(f"无法加载模型: {e}")
        
    def _get_imagenet_classes(self):
        # 获取ImageNet分类列表
        try:
            weights = ResNet50_Weights.IMAGENET1K_V1()
            return weights.meta['categories']
        except Exception as e:
            default_logger.warning(f"获取ImageNet分类列表失败：{e}")
            return [f"class_{i}" for i in range(1000)]
        
    # def _load_class_names_from_data(self):
    # 从数据加载分类列表
    
    def _expand_fc_layer(self, model, old_num_classes, new_num_classes):
        # 扩展fc层
        fc = model.fc
        if not isinstance(fc, ConsineClassifier):
            raise TypeError("模型fc层必须是ConsineClassifier类型")
        in_features = fc.in_features
        old_scale = fc.scale.data.item()
        new_scale = math.sqrt(new_num_classes)

        if old_num_classes == 0:
            # 如果旧分类数为0，则直接创建新分类器
            new_fc = ConsineClassifier(in_features,
            new_num_classes, scale=new_scale)
            # 调整新分类的参数
            # 调整新分类的参数
            nn.init.kaiming_uniform_(new_fc.weight,
                a=math.sqrt(5))

            # 保存新分类器到标准模型
            model.fc = new_fc
            default_logger.info(f"fc层已扩展为{new_num_classes}个分类")
            return model
        
        # 继续微调
        # 保存旧权重
        old_weight = fc.weight.data
        new_fc = ConsineClassifier(in_features, new_num_classes, scale=new_scale)

        if new_num_classes > old_num_classes:
            # 如果新分类数大于旧分类数，则只复制旧分类数的权重
            new_fc.weight.data[:old_num_classes] = old_weight[:old_num_classes]
            # 填充新分类的权重
            nn.init.kaiming_uniform_(new_fc.weight.data[old_num_classes:,:], a=math.sqrt(5))
            default_logger.info(f"fc层已扩展为{new_num_classes}个分类")
        else:
            # 截断
            new_fc.weight.data[:,:] = old_weight[:new_num_classes, :]
            default_logger.info(f"fc层已截断为{new_num_classes}个分类")
            
        model.fc = new_fc
        return model
    
    def fintune(self, data_dir: Path, epoch = 10,
        stop_check=None, reset_model=False):
        # 微调模型
        default_logger.info(f"开始微调模型,支持{epoch}个epoch")
        try:
            # 定义微调参数
            # 对应的训练集
            train_transform = transformers.Compose([
                transformers.RandomResizedCrop(224, scale=(0.7, 1.0)),
                transformers.RandomHorizontalFlip(p=0.5),
                transformers.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                transformers.RandomRotation(20),
                transformers.ToTensor(),
                transformers.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            # 对应的验证集
            val_transform = transformers.Compose([
                transformers.Resize(224, 224),
                transformers.ToTensor(),
                transformers.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            full_dataset = datasets.ImageFolder(root=str(data_dir), transform=train_transform)
            
            # 从数据集中提取分类
            class_names = full_dataset.classes

            # 分类的个数
            num_class = len(class_names)
            if num_class == 0:
                default_logger.error(f"数据集{data_dir}中没有分类")
                return False

            # 以2:8的比例划分验证集和训练集
            total_len = len(full_dataset)
            indices = list(range(total_len))
            # 固定随机种子，防止不同计算框架的随机性不一致
            nn.random.seed(42)
            # 打乱列表中元素次序
            nn.random.shuffle(indices)
            # 验证集的长度
            val_len = int(0.2 * total_len)
            # 固定训练集(后百分之八十)
            train_indices = indices[val_len:]
            # 固定验证集(前百分之二十)
            val_indices = indices[:val_len]
            # 从数据集中抽取验证集
            val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
            # 设置验证集的变换参数
            val_dataset.dataset.transform = val_transform

            # 从数据集中抽取训练集
            train_dataset = torch.utils.data.Subset(full_dataset, train_indices)

            # 类别均衡采样
            train_labels = [full_dataset.targets[i] for i in train_indices]
            class_counts = np.bincount(train_labels, minlength=num_class)
            class_counts = np.maximum(class_counts, 1)
            
            # 检查个数为0个样本,将其改为1
            class_counts = np.maximum(class_counts, 1)

            # 类型权重
            class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
            sample_weights = class_weights[train_labels]

            # 随机样本
            sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

            batch_size = settings.BATCH_SIZE
            
            '''创建 DataLoader
            Windows下num_workers参数需要设置为0(避免多进程错误), 否则容易引起错误导致失败
            即使使用了WeightedRandomSampler也只能提高低数量样本的抽出概率          
            并不能完全解决某样本0抽出的问题''' 
            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=batch_size,
                sampler=sampler, shuffle=True)

            val_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=batch_size, 
                shuffle=False, num_workers=0)       # 验证集不需要打乱（shuffle=False）
            # TODO-END: 后期可用 sklearn 的方式替换

        except Exception as e:
            default_logger.error(f"数据集加载出错: {e}")
            return False

        '''加载模型'''        
        pth_path = Path(settings.RESNET50_FINETUNED_PTH_PATH)
        class_path = Path(settings.RESNET50_FINETUNED_CLASSNAMES_PATH)
        old_num_class = 0
        model = None
        
        '''初始化模型'''
        # 读取上次微调后的模型，会引发灾难遗忘，可以省略
        # if not reset_model and pth_path.exists() and class_path.exists():
        #     # 继续微调
        #     with open(class_path, 'r', encoding='utf-8') as f:
        #         old_class_names = json.load(f)
        #     old_num_class = len(old_class_names)

        # try:
        #     model = models.resnet50(weight=None)
        #     in_feature = model.fc.in_features
        #     model.fc = ConsineClassifier(in_feature, old_num_class)
        #     state_dict = torch.load(pth_path, map_location=self.device)
        #     model.load_state_dict(state_dict)
        #     default_logger.info(f"模型加载成功，分类数为{old_num_class}")
        # except Exception as e:
        #     default_logger.error(f"模型加载出错：{e}")
        #     model = None

        # 从本地加载预训练模型
        if model is None:
            try:
                # 如果获取微调模型失败，则从本地加载模型
                model = load_resnet_50_from_local_safetensors()
                old_num_class = 0
                default_logger.info(f"原始模型加载成功,分类数为{old_num_class}")
            except Exception as e:
                default_logger.error(f"原始模型加载出错: {e}")
                raise RuntimeError(f"原始模型加载出错: {e}")

        # 如果当前数据集的类别数与模型原来的类别数不同, 则调整fc
        if old_num_class != num_class:
            # 调整fc层(增加/缩减)
            model = self._expand_fc_layer(model, old_num_class, num_class)
            default_logger.info(f"fc层调整成功, 分类数为{num_class}")
        else:
            # 调整学习的缩放倍数
            current_scale = model.fc.scale.data.item()
            target_scale = math.sqrt(num_class)
            if abs(current_scale - target_scale) > 1e-3:
                model.fc.scale.data.fill_(target_scale)
                default_logger.info(f"fc层缩放倍数调整成功, 分类数为{num_class}")
            default_logger.info(f"fc层分类不变")

            model = model.to(self.device)

            # 冻结骨干
            for param in model.parameters():
                param.requires_grad = False

            # 可改变fc
            for param in model.fc.parameters():
                param.requires_grad = True

            # 使用自定义损失函数(标签平滑)
            criterion = LabelSmoothingCrossEntropy(smoothing=0.1)

            # 优化器
            lr_fc = 5e-4
            optimizer = optim.AdamW(model.fc.parameters(),
                lr=lr_fc, weight_decay=0.01)

            # 调度器
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                T_max=epoch, eta_min=1e-6)

            # 跑圈训练
            best_acc = 0.0
            best_model_sate = None

            for e in range(1, epoch+1):
                if stop_check and stop_check(e):
                    default_logger.info(f"微调训练提前结束,当前epoch为{e}")
                    return False
                
            # 微调
            model.train()
            # 调整
            running_loss, correct, total = 0.0, 0, 
            
            ''''''
            for images, labels in train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                # 优化参数清零
                optimizer.zero_grad()
                # 导出结果
                outputs = model(images)
                # 计算损失量(向前传播)
                loss = criterion(outputs, labels)
                # 向后传播
                loss.backward()
                # 调整梯度(确保梯度的合理范围, 不能超过 1.0 )
                clip_grad_norm_(model.fc.parameters(), 1.0)
                # 落实更新
                optimizer.step()
                running_loss += loss.item() * images.size(0)
                # 计算准确率，不能超过 1
                _, pred = torch.max(outputs, 1) # 前一次
                total += labels.size(0)         # 总体
                correct += (pred == labels).sum().item()    # 当前
                
            train_loss = running_loss / total   # 整体微调的损失
            train_acc = correct / total         # 整体微调的正确率
            
            # 验证
            model.eval()
            val_loss, val_correct, val_total = 0.0, 0, 0
            with torch.no_grad():    # with: 临时解冻
                for images, labels in val_loader:
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    # 导出结果
                    outputs = model(images)
                    # 计算损失量(向前传播)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * images.size(0)
                    # 计算准确率，不能超过 1
                    _, pred = torch.max(outputs, 1) # 前一次
                    val_total += labels.size(0)         # 总体
                    val_correct += (pred == labels).sum().item()    # 当前
            val_loss /= val_total
            val_acc = val_correct / val_total
            
            # 调整学习率
            scheduler.step()
            # 获得当前的学习率
            current_lr = optimizer.param_groups[0]['lr']
            
            if val_acc > best_acc:
                # 保存模型
                best_acc = val_acc
                best_model_sate = copy.deepcopy(model.state_dict())
                default_logger.info(f" 验证准确率提升, 当前epoch为{e}, 准确率{val_acc}")
        
        '''保存模型'''
        # 微调失败
        if best_model_sate is None:
            default_logger.info(f"微调失败")
            return False
        
        try:
            # 保存微调后的模型权重
            torch.save(best_model_sate, settings.RESNET50_FINETUNED_PTH_PATH)
        except Exception as e:
            default_logger.error(f"保存微调后的模型失败: {e}")