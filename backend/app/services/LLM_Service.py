from pathlib import Path
import threading
from typing import Dict, List, Optional
from llama_cpp import Llama
from utils.common_utils import default_logger   # 自建模块: 日志
from config.config import settings              # 自建模块: 配置

class LLM_Service:
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

        # self.model_path = Path(settings.LLM_MODEL_PATH)
        self.model_path = settings.LLM_MODEL_PATH
        self.llm = None
        self._loaded = False
        self._load_lock = threading.Lock()

    def _load_model(self):

        if self._loaded:
            return

        # 确定模型加载的设备
        n_gpu_layers = settings.LLM_GPU_LAYERS
        with self._load_lock:                 # ← 加锁
            if self._loaded:                  # 双重检查
                return

        try:
            # 加载压缩后的模型DEVICE
            self.llm = Llama(
                model_path=self.model_path,
                n_ctx=settings.LLM_CONTEXT_SIZE,
                n_gpu_layers=n_gpu_layers,
                n_threads=settings.LLM_THREADS,
                # 是否打印详细信息
                verbose=False
            )
            # 标志模型已加载
            self._loaded = True
            default_logger.info(f"模型加载完成,设备是:{settings.DEVICE}")
            # self.llm = None

        except Exception as e:
            default_logger.error(f"加载模型失败:{e}")
            self._loaded = False
            self.llm = None
        
    '''检查模型是否加载完成'''    
    def is_ready(self)->bool:
        return self._loaded and self.llm is not None
    
    def chat(self, messages:List[Dict[str, str]], temperature:float=0.7, max_tokens:int = 512, top_p:float=0.9, stream:bool=False)->str:
        if not self.is_ready():
            default_logger.error("模型未加载完成,重新载入")
            self._load_model()
            
        if not self.is_ready():
            error_msg = "模型未加载完成,请检查模型路径是否正确"
            default_logger.error(error_msg)
            if stream:
                def error_gen():
                    # 发送错误信息，异步流式返回
                    yield f"data:{error_msg}"
                return error_gen()
            return error_msg
                
        try:
            response = self.llm.create_chat_completion(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                stream=stream,
                # 每一次对话的结束标记，整个会话的结束标记
                stop=["<|im_end|>","<|endoftext|>"]
            )
            
            # 异步流式返回
            if stream:
                try:
                    # 遍历模型返回每个块
                    for chunk in response:
                        if 'choices' in chunk and len(chunk['choices']) > 0:
                            delta = chunk['choices'][0]['delta'].get('delta', {})
                            content = delta.get('content', '')
                            if content:
                                # 发送内容，异步流式返回
                                yield content
                except Exception as e:
                    default_logger.error(f"异步流式返回失败: {e}")
                    yield f"错误: {str(e)}"
            else:
                if 'choices' in response and len(response['choices']) > 0:
                    # 含有答案, 则返回答案内容
                    default_logger.info(f"模型返回: {response['choices'][0]['message']['content']}")
                    return response['choices'][0]['message']['content']
                else:
                    default_logger.error("模型返回为空。")
                    return "模型返回为空, 请重试!"
                    
        except Exception as e:
            default_logger.error(f"LLM模型调用失败，{e}")
            return f"LLM模型调用失败:{str(e)}, 请重试!"
        
    def generate(self, user_prompt:str, 
                 system_prompt:Optional[str]=None, 
                 temperature:float=0.7, 
                 max_tokens:int = 512)->str:
        messages = []
        # 系统提示
        if system_prompt:
            messages.append({"role":"system","content":system_prompt})
        # 用户提示（包含用户问题）
        messages.append({"role":"user","content":user_prompt})
        result = self.chat(messages, temperature, max_tokens)
        # 处理模型为空的情况
        return result if isinstance(result, str) else ""
        
llm_service = LLM_Service()