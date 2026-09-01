from pathlib import Path
from pydantic_settings import BaseSettings

# 定位到backend目录
BASE_DIR = Path(__file__).resolve().parent.parent.parent

class Settings(BaseSettings):
    DATA_DIR: Path = f"{BASE_DIR}/data"
    MODELS_DIR: Path = f"{DATA_DIR}/models"

    MODELS_PTH_DIR: str = f"{MODELS_DIR}/pth"
    MODELS_GGUF_DIR: str = f"{MODELS_DIR}/gguf"

    RESNET50_MODEL_ID: str = "microsoft/resnet-50"
    RESNET50_MODEL_PATH: str = f"{MODELS_PTH_DIR}/{RESNET50_MODEL_ID}"
    
    RESNET50_FINETUNED_PTH_DIR: str = f"{MODELS_PTH_DIR}/resnet_50_finetuned"
    RESNET50_FINETUNED_PTH_PATH: str = f"{RESNET50_FINETUNED_PTH_DIR}/resnet_50_finetuned.pth"

    JSON_DIR: str = f"{DATA_DIR}/json"
    RESNET50_FINETUNED_CLASSESNAMES: str = f"{JSON_DIR}/resnet_50_finetuned/class_name.json"