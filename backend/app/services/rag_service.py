import threading
import asyncio

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.common_utils import default_logger   # 自建模块: 日志
from config.config import settings              # 自建模块: 配置
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, DocxLoader, TextLoader

'''
    RAG服务类: 采用单例模式，主要用于管理向量数据库和检索器。
'''
class RagService:
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
        self.embeddings = None              # 向量化模型
        self.vectorstore = None             # 向量数据库
        self.retriever = None               # 检索器
        self._embedding_loaded = False      # 向量化模型是否已加载
        self._embedding_model_dir = None    # 向量化模型目录
        
        # 检查向量化模型是否存在
        self._check_embedding_model()
        # 初始化向量数据库
        self._init_chromadb()
        
        
    '''初始化向量数据库'''  
    def _init_chromadb(self):
        try:
            # 初始化向量数据库
            self.vectorstore = Chroma(
                # 相当于数据库的名字
                collection_name="saa_knowledge",
                embedding_function=None,
                persist_directory=settings.CHROMA_PERSIST_DIR
            )
            # 初始化检索器
            self.retriever = None
            default_logger.info("向量数据库初始化完成")
        except Exception as e:
            default_logger.error(f"向量数据库初始化失败: {e}")
            self.vectorstore = None
            
            
    '''加载嵌入(向量化)模型'''
    def _load_embeddings(self):
        # 如果已经载入了,则不再载入
        if self._embedding_loaded: 
            return
        
        try:
            # 加载嵌入模型
            self.embeddings = HuggingFaceEmbeddings(
                model_name=settings.EMBEDDING_MODEL_PATH,
                # 向量化模型体积过小,直接放入内存,速度快,节省显存
                model_kwargs={"device": 'cpu'},
                # 向量化模型输出的向量归一化,使向量长度为1
                encode_kwargs={"normalize_embeddings": True}
            )
            self._embedding_loaded = True
            default_logger.info("嵌入模型加载完成")
        except Exception as e:
            default_logger.error(f"嵌入模型加载失败: {e}")
            self.embeddings = None
            self._embedding_loaded = False
            
        
    '''# 确保所有组件(向量模型,向量数据库,检索器)都已初始化'''
    def _ensure_components(self):
        # 向量数据库未初始化直接返回失败
        if self.vectorstore is None:
            default_logger.info("向量数据库未初始化, 开始初始化")
            return False

        # 如果向量化模型未加载,则加载
        if not self._embedding_loaded:
            self._load_embeddings()

        # 如果向量化模型再次加载失败,则返回False
        if not self._embedding_loaded:
            default_logger.error("嵌入模型加载失败")
            return False
        
        # 如果向量化模型已加载,则设置向量数据库的向量化函数
        if self.vectorstore._embedding_function is None and self.embeddings is not None:
            self.vectorstore._embedding_function = self.embeddings

        # 如果检索器未加载,则加载
        if self.retriever is None and self.embeddings is not None:
            self.retriever = self.vectorstore.as_retriever()

        default_logger.info("所有组件初始化完成")
        return True
    
    '''添加文档到向量数据库'''
    def add_document(self, file_path: str)->dict:
        # 组件未就绪则直接返回失败信息
        if not self._ensure_components():
            default_logger.error("向量化模型未加载或者向量数据库未初始化,无法添加文档")
            return {
                "success": False,
                "trunk_count":0, 
                "message": "向量化模型未加载"
            }
        
        _file_path = Path(file_path)
        # 获得文件的小写后缀
        suffix = _file_path.suffix.lower()
        loader = None
        
        try:
            if suffix == ".pdf":
                loader = PyPDFLoader(file_path)
            elif suffix == ".docx":
                loader = DocxLoader(file_path)
            elif suffix == ".txt":
                encodings = ["utf-8", "gbk", "gb2312", "latin-1"]
                
                for enc in encodings:
                    try:
                        loader = TextLoader(file_path, encoding=enc)
                        # 如果格式不正确, 调用load()才会引发一场
                        loader.load()
                        break
                    except Exception as e:
                        default_logger.error(f"编码{enc}加载失败：{e}")
                        continue
                    
                if loader is None:
                    default_logger.error(f"所有编码都加载失败：{file_path}")
                    return {
                        "success": False,
                        "trunk_count":0,
                        "message": f"无法加载文件：{file_path}"
                    }
                    
            elif suffix == ".doc":
                try:
                    loader = TextLoader(file_path, encoding="latin-1")
                    loader.load()
                except Exception as e:
                    default_logger.error(f"doc文件加载失败：{e}")
                    return {
                        "success": False,
                        "trunk_count":0,
                        "message": "doc文件加载失败"
                    }

            else:
                default_logger.error(f"不支持的文件格式: {suffix}")
                return {
                    "success": False,
                    "trunk_count":0,
                    "message": f"不支持的文件格式: {suffix}"
                }
                
            # 获得文档的内容
            documents = loader.load()
            
            if not documents:
                default_logger.error(f"文档加载失败：{file_path}")
                return {
                    "success": False,"trunk_count":0,
                    "message": "文档加载失败"
                }
                
                
            # 文本分割器,将文档分割成多个段落,每个段落500字,重叠50字
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)    
            
            # 分割后的段落列表
            chunks = text_splitter.split_documents(documents)

            if not chunks:
                default_logger.error(f"文档分割失败：{file_path}")
                return {
                    "success": False,
                    "trunk_count":0,
                    "message": "文档分割失败"
                }
                
            for chunk in chunks:
                # 为每个段落添加元数据,记录来源文件路径,便于后期删除
                chunk.metadata['source'] = file_path
            # 添加段落到向量数据库
            self.vectorstore.add_documents(chunks)
            # 刷新检索器,确保新的段落被检索 ,默认返回4个段落(可以适当加大k值)
            self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 4})
            
            default_logger.info(f"文档{file_path}添加到向量数据库,共{len(chunks)}个段落")
            return {
                "success": True,
                "trunk_count":len(chunks),
                "message": "文档添加成功"
            }
                
        except Exception as e:
            default_logger.error(f"添加文档{file_path}失败：{e}")
            return {
                "success": False,
                "trunk_count":0,
                "message": f"添加文档{file_path}失败"
            }