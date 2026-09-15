import json
from fastapi import APIRouter
from services.LLM_Service import llm_service
from services.rag_service import rag_service
from pydantic import BaseModel
from utils.common_utils import default_logger   # 自建模块: 日志
from fastapi.responses import StreamingResponse

router = APIRouter(prefix="/api/chat")

class ChatRequest(BaseModel):
    # 会话ID
    session_id: str
    # 用户问题
    message: str
    #温度
    temperature: float = 0.7
    # 最大令牌数
    max_tokens: int = 512
    
async def generate(request: ChatRequest):
    full_response = ""
    try:
        # 调用RAG服务查询(非流式)
        result = rag_service.query(request.message)
        # 从结果中提取来源文档
        sources = result.get("source", [])
        # 从结果中提取答案
        answer = result.get("answer", "")
        if answer:
            full_response = answer
            if sources:
                source_names = sources[:10]
                full_response += f"\n来源文档: {', '.join(source_names)}"
            else:
                full_response += "\n未使用本地知识库"
        else:
            full_response = "未找到相关答案"

        #为了保证以后可以修改为流式输出，所以这里模仿流式输出的格式
        yield f"data:{json.dumps({'chunk':full_response,'done':False})}"
        default_logger.info(f"生成完成：{full_response}")
        yield f"data:{json.dumps({'chunk':'', 'done':True})}"


    except Exception as e:
        default_logger.error(f"问答失败：{e}")
        yield f"data:{json.dumps({'error':'', 'done':True})}"

@router.post("/")
async def chat_stream(request: ChatRequest):
    default_logger.info(f"开始处理: {request.message}")
    return StreamingResponse(generate(request), media_type="text/event-stream")