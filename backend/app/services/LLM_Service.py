from pathlib import Path
import threading
import torch
from llama_cpp import Llama
from utils.common_utils import default_logger   # 自建模块: 日志
from config.config import settings              # 自建模块: 配置

class LLMService:
    '''线程安全的单例模式'''
    _instance = None            # 类属性，用来保存唯一的那个实例。
    _lock = threading.Lock()    # 线程锁，防止多线程同时创建出多个对象。
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    
    '''初始化方法'''
    def __init__(self):
        if hasattr(self, '_initialized'):
            return
        self._initialized = True
        
        self.model_path = Path(settings.LLM_MODEL_PATH)
        self.llm = None
        self._loaded = False
        
    def generate():
        return "hello world"
        
    def _load_model(self):
        
        if self._loaded:
            return
        
        # 确定模型加载的设备
        n_gpu_layers = settings.LLM_GPU_LAYERS
        try:
            # 加载压缩后的模型
            self.llm = Llama(
                model_path=self.model_path,
                n_ctx=settings.LLM_CONTEXT_SIZE,
                n_gpu_layers=n_gpu_layers,
                n_threads=settings.LLM_THREADS,
                # 是否打印详细信息
                verbose=False
            )
        except Exception as e:
            default_logger.error(f"加载模型失败:{e}") 
        
llm_service = LLMService()