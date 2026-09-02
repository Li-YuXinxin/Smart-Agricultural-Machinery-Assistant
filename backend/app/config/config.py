from pathlib import Path
from pydantic_settings import BaseSettings
import torch
from utils.common_utils import default_logger

# 定位到backend目录
BASE_DIR = Path(__file__).resolve().parent.parent.parent

class Settings(BaseSettings):
    DATA_DIR: Path = f"{BASE_DIR}/data"
    MODELS_DIR: Path = f"{DATA_DIR}/models"

    MODELS_PTH_DIR: str = f"{MODELS_DIR}/pth"
    MODELS_GGUF_DIR: str = f"{MODELS_DIR}/gguf"

    # 微调后的图像识别模型目录
    RESNET50_MODEL_ID: str = "microsoft/resnet-50"
    RESNET50_MODEL_PATH: str = f"{MODELS_PTH_DIR}/{RESNET50_MODEL_ID}"
    
    # 微调后的图像识别模型目录
    RESNET50_FINETUNED_PTH_DIR: str = f"{MODELS_PTH_DIR}/resnet_50_finetuned"
    RESNET50_FINETUNED_PTH_PATH: str = f"{RESNET50_FINETUNED_PTH_DIR}/resnet_50_finetuned.pth"

    # 微调后的模型所能够识别的类型列表
    JSON_DIR: str = f"{DATA_DIR}/json"
    RESNET50_FINETUNED_JSON_DIR: str = f"{JSON_DIR}/resnet_50_finetuned"
    RESNET50_FINETUNED_CLASSESNAMES: str = f"{JSON_DIR}/resnet_50_finetuned/class_name.json"
    
    # 显存检测
    DEVICE: str = "cpu"
    LLM_GPU_LAYERS: int = 0 # 模型层数，默认0层，不使用GPU加速
    GPU_MEMORY_THRESHOLD_GB: float = 5.9    # 显存阈值，默认5.9GB，超过阈值则使用CPU
    
    # 微调参数
    BATCH_SIZE: int = 16    # 批次大小，默认16
    CONFIDENCE_THRESHOLD: float = 0.25  # 置信度阈值，默认0.25
    FULL_EPOCHS: int = 20   # 全部训练轮数，默认20(实际中最少40)

    def _detect_and_configure_device(self):
        # 重置运算设备
        self.DEVICE = 'cpu'
        self.LLM_GPU_LAYERS = 0
        # 检查GPU是否可用
        if torch.cuda.is_available():
            # 检查计算设备是否满足条件
            device_idx = torch.cuda.current_device()
            total_memory_bytes = torch.cuda.get_device_properties(device_idx).total_memory
            # 检查显存是否足够
            total_memory_gb = total_memory_bytes / (1024**3)
            if total_memory_gb >= self.GPU_MEMORY_THRESHOLD_GB:
                self.DEVICE = 'cuda'
                self.LLM_GPU_LAYERS = -1
                default_logger.info(f"使用GPU,显存检测：{total_memory_gb:.2f}GB")
            else:
                default_logger.warning(f"显存不足，使用CPU")
        else:
            default_logger.warning(f"CUDA不可用, 使用CPU")
        
        # 保存参数到配置文件
        torch.set_default_device(self.DEVICE)
        default_logger.info(f"默认设备: {self.DEVICE}")
        
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self._detect_and_configure_device()

settings = Settings()

# 创建数据目录
for d in [
    settings.MODELS_PTH_DIR,
    settings.RESNET50_FINETUNED_PTH_DIR,
    settings.RESNET50_FINETUNED_JSON_DIR
]:
    Path(d).mkdir(parents=True, exist_ok=True)
    
default_logger.info("配置文件加载完毕")