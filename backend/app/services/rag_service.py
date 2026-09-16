import threading
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.common_utils import default_logger   # 自建模块: 日志
from config.config import settings              # 自建模块: 配置
from services.LLM_Service import llm_service
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader

'''提示词模板'''
class PromptTemplate: 
    SAA_EXPERT: str = """
你是一个专业的农业技术专家, 拥有丰富的农作物种植, 病虫害防治, 土壤管理的经验, 
请遵循以下的规则, 来回答用户的问题:
1.回答专业，准确，基于科学知识
2.使用通俗易懂的语言，避免过于学术化
3.提供具体的操作的建议和步骤
4.如果超出你的知识范围，请如实告知用户
5.涉及农药使用，务必提醒用户注意事项
请以专业，耐心的态度，回答用户的问题。
"""
    
    
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
            default_logger.info("嵌入模型加载失败")
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
                loader = Docx2txtLoader(file_path)
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
    
    '''查询向量数据库'''
    def query(self, question: str)->dict:
        if not self._ensure_components():
            default_logger.error("向量化模型未加载或者向量数据库未初始化，无法查询")
            return {
                "answer": "RAG系统为就绪", 
                "source":[]
            }

        try:
            if self.retriever is None:
                default_logger.error("检索器未初始化,无法查询")
                return {
                    "answer": "检索器未初始化", 
                    "source":[]
                }
                
            # 在向量数据库中查询
            docs = self.retriever.get_relevant_documents(question)
            default_logger.info(f"查询到{len(docs)}个相关文档落")
            
            for i, doc in enumerate(docs):
                default_logger.info(f"文档{i}: {doc.page_content}")

            if not docs:
                return {
                    "answer": "查询到0个相关文档",
                    "source": []
                }

            # 只把前四个获得片段合成完整内容
            context = "\n\n".join([doc.page_content for doc in docs[:settings.K]])
            if len(context) > 2000 :
                # 防止内容过长浪费流量
                context = context[:2000] + "..."
            
            system_prompt = PromptTemplate.SAA_EXPERT
            user_prompt = f"""
请根据以下知识库内容回答用户问题,如果知识库中没有相关信息,请如实告知.

知识库内容:
{context}

用户问题:
{question}

回答:
"""

            answer = llm_service.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.5,
                max_tokens=512,
            )

            # 从检索到的文档中提取来源文件路径
            source = [doc.metadata.get("source", "未知来源") for doc in docs]
            # 对向量片段的来源进行去重(保持顺序)
            source = list(dict.fromkeys(source))

            if "SAA-LLM模型未加载" in answer:
                default_logger.error("LLM模型未加载,无法回答问题")
                return { 
                    "answer": f"SAA-LLM模型未加载,以下是检索到的相关内容:{context}",
                    "source":source
                }

            # 返回经过llm处理后的回答和来源文件路径
            default_logger.info(f"查询到{len(source)}个相关文档,经过LLM处理,回答为: {answer}")
            return { 
                "answer": answer,
                "source":source
            }


        except Exception as e:
            default_logger.error(f"查询失败：{e}")
            return { 
                "answer": f"RAG系统查询失败:{e}",
                "source":[]
            }
    
    def delete_document(self,file_path: str)->dict:
        if not self._ensure_components():
            default_logger.error("数据库未加载或未就绪")
            return {"succcess":False,"trunk_count":0,"message": "数据库组件未就绪"}
            
# 全局单例 
rag_service = RagService()