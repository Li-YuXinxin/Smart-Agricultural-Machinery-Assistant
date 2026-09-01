from pathlib import Path
import logging
import sys
from logging.handlers import RotatingFileHandler

# 定位到backend目录
BASE_DIR = Path(__file__).resolve().parent.parent.parent
# 日志位置
_LOG_FILE = f"{BASE_DIR}/data/logs/app.log"
_LOGGER_CACHE = {}

def setup_logger(
    name: str = "智慧农技助手",     # 日志名称
    log_file: str = None,          # 日志文件
    level: str = "INFO",           # 日志级别
    max_bytes: int = 1024 * 1024 * 10,  # 10MB, 日志上限
    backup_count: int = 5,         # 备份数量
) -> logging.Logger:
    # 获得日志对象
    logger = logging.getLogger(name)
    # 容错处理
    logger.setLevel(getattr(logging, level.upper()))
    if logger.handlers:
        return logger   # 如果已经存在logger，直接返回

    # 控制台输出相关
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(console_formatter)
    # 将控制台处理器添加到日志器logger
    logger.addHandler(console_handler)
    
    # 文件输出相关
    if log_file:
        try:
            # 将字符串封装为路径
            log_path = Path(log_file)
            # 强制创建该文件的文件夹
            log_path.parent.mkdir(parents=True, exist_ok=True)
            # 创建文件处理器
            file_handler = RotatingFileHandler(
                log_path,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding="utf-8"
            )
            file_handler.setLevel(logging.INFO)
            file_formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
            file_handler.setFormatter(file_formatter)
            # 将文件处理器添加到日志器logger
            logger.addHandler(file_handler)

        except Exception as e:
            logger.warning(f"无法创建日志文件: {e}")

    return logger

def get_logger(name: str = "智慧农技助手") -> logging.Logger:
    # 从缓存中获取日志器logger，保证每个name只有一个logger
    if name in _LOGGER_CACHE:
        return _LOGGER_CACHE[name]
    # 如果缓存中不存在，则创建新的logger并返回
    _LOGGER_CACHE[name] = setup_logger(name, log_file=_LOG_FILE,level="INFO")
    return _LOGGER_CACHE[name]

# 默认的日志对象
default_logger = get_logger("智慧农技助手")