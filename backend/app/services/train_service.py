import threading
from typing import Optional, Callable
import asyncio
from utils.common_utils import default_logger
from pathlib import Path
from config.config import settings
from services.classify_service import classify_service

'''
    训练服务类: 采用单例模式，主要用于管理后台训练任务的生命周期。
'''
class TrainService:
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
            return
        self._initialized = True
        self.status = "idle"    # 默认状态为闲状态
        self.logs = []
        self.result = None
        self.current_task = None    # 后台线程
        self.stop_requested = False # 是否终止训练
        self._broadcat_callback: Optional[Callable] = None  # 广播回调函数
        self._main_loop: Optional[asyncio.AbstractEventLoop] = None
        self._epochs = settings.FULL_EPOCHS
        self.data_dir = Path(settings.UPLOAD_DATASET_UNZIPED_DIR)   # 默认训练目录

    def set_main_loop(self, main_loop: asyncio.AbstractEventLoop):
        self._main_loop = main_loop
            
    def set_broadcast_callback(self, callback: Callable):
        self._broadcast_callback = callback

    def _broadcast(self):
        if self._broadcast_callback:
            try:
                self._broadcast_callback(self.get_status())
            except Exception as e:
                default_logger.error(f"广播训练状态失败:{e}")
        
    def get_status(self):
        # 返回值是字典,前端转换为json字符串
        return {
            "status": self.status,
            "result": self.result
        }
        
    def start_training(self, data_dir: Path, epochs: Optional[int] = 10):
        if self.status == settings.TRAIN_STATUS_RUNNING:
            default_logger.warning("训练已在运行中")
            return False

        # 如果正在训练，则所有状态重置
        self.status = settings.TRAIN_STATUS_RUNNING
        self.result = None
        self.stop_requested = False
        # 处理用户传入的参数
        if epochs is None:
            self._epochs = settings.FULL_EPOCHS
        else:
            self._epochs = epochs
        
        if data_dir is None:
            self._data_dir = Path(settings.UPLOAD_DATASET_UNZIPED_DIR)
        else:
            self._data_dir = data_dir
            
        self._epochs = epochs
        
        self._broadcast()
        
        # 启动后台进程
        self.current_task = threading.Thread(target=self._run, args=(data_dir), daemon=True)
        self.current_task.start()
        return True    
            
    def request_stop(self):
        if self.status == settings.TRAIN_STATUS_RUNNING:
            self.stop_requested = True
            default_logger.info("请求终止训练")
            return True
        return False
    
    def _run(self):
        try:
            success=classify_service.fintune(data_dir=self._data_dir, epoch=self._epochs, stop_check=self.stop_requested)
            if self.stop_requested:
                # 用户中途结束训练
                self.status = settings.TRAIN_STATUS_FAILED
                self.result = {"success": False, "error": "用户中途结束训练"}
                default_logger.info("用户中途结束训练")
            elif success:
                self.status = settings.TRAIN_STATUS_DONE
                self.result = {"success": True, "message": "训练完成"}
                default_logger.info("训练完成")
            else:
                self.status = settings.TRAIN_STATUS_FAILED
                self.result = {"success": False, "error": "训练失败"}
                default_logger.info("训练失败")
        except Exception as e:
            self.status = settings.TRAIN_STATUS_FAILED
            self.result = {"success": False, "error": str(e)}
            default_logger.warning(f"训练模型失败:{e}")
        finally:
            # 无论结果如何，都需要把结果广播给前端
            self._broadcast()
            
train_service = TrainService()