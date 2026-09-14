import uuid
from fastapi import APIRouter, File, UploadFile, Depends
from sqlalchemy.orm import Session
from pathlib import Path
from config.config import settings
from fastapi import HTTPException
from utils.common_utils import default_logger
from services.rag_service import rag_service

router = APIRouter(prefix="/api/knowledge") 

@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...)
):
    file_name = file.filename
    # 如果没有则"",反之".xxx"
    ext = Path(file_name).suffix.lower()
    if ext not in settings.ALLOWED_EXTENSIONS:
        default_logger.error(f"文件扩展名错误：{ext}")
        raise HTTPException(status_code=400, detail="文件扩展名不支持")
    
    temp_dir = Path(settings.UPLOAD_FILE_DIR)
    unique_name = f"{uuid.uuid4().hex}_{file_name}"
    # 保存文件的路径
    save_path = temp_dir / unique_name
    
    try:
        # 读取文件内容
        content = await file.read()
        if len(content) > settings.MAX_UPLOAD_FILE_SIZE:
            default_logger.error(f"文件大小超过最大限制：{settings.MAX_UPLOAD_FILE_SIZE}")
            raise HTTPException(status_code=400, detail="文件大小超过最大限制")
        # 保存文件
        default_logger.info(f"保存文件: {save_path}")
        with open(save_path, "wb") as f:
            f.write(content)
        default_logger.info(f"保存文件：{save_path}")
        # 将文件添加到RAG服务(向量数据库)
        result = rag_service.add_document(str(save_path))
        
        if not result["success"]:
            default_logger.error(f"添加文件到RAG服务失败：{result['message']}")
            raise HTTPException(status_code=500,detail="添加文件到RAG服务失败")
        
        return {
            "message": "文件上传成功",
            "success": True,
            "document":str(save_path)
        }
    
    except HTTPException:
        raise
    except Exception as e:
        default_logger.error(f"文件上传失败:{e}")
        raise HTTPException(status_code=500, detail="文件上传失败")
    