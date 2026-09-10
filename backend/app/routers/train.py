import asyncio
import base64
from pathlib import Path
import shutil
from fastapi import APIRouter, HTTPException, Request, WebSocket, WebSocketDisconnect
from services.train_service import train_service
from utils.common_utils import default_logger
from config.config import settings
from PIL import Image
import io

'''
    websocket管理类: 管理所有WebSocket连接，实现连接建立、断开和消息广播
'''
class ConnectioConnnManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []
        
    async def connect(self, webstock: WebSocket):
        await webstock.accept()
        # 如果连接成功，则把 ws 添加到 active_connections 
        self.active_connections.append(webstock)
        
    def disconnect(self, websocket: WebSocket): 
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            
    async def broadcast(self, message: dict):
        # 广播消息到所有连接的客户端
        if not self.active_connections:
            return
        # 并播消息到所有连接的客户端
        tasks = [conn.send_json(message) for conn in self.active_connections]
        # 等待所有任务完成
        await asyncio.gather(*tasks, return_exceptions=True)
        
manager = ConnectioConnnManager()

'''
    状态广播函数: 将训练状态从后台线程广播到主事件循环，推送到前端
'''
def broadcast_status(status: dict):
    # 广播状态消息到所有连接的客户端
    # default_logger.info(f"broadcast_status 被调用, 状态: {status}")
    loop = train_service._main_loop
    if loop is None or not loop.is_running():
        # default_logger.warning("loop 不可用，广播失败") 
        return
    asyncio.run_coroutine_threadsafe(manager.broadcast(status), loop)
    # default_logger.info("广播已提交到事件循环")

train_service.set_broadcast_callback(broadcast_status)


'''
    WebSocket端点: 建立WebSocket连接，接收前端连接并保持通信
'''
router = APIRouter(prefix="/api/train")
# ws 端点
@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    # 建立连接
    await manager.connect(websocket)
    try:
        # 发送当前状态
        await websocket.send_json(train_service.get_status())
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        # 如果客户要断链，则从active_connections中移除WS
        manager.disconnect(websocket)

    except Exception as e:
        default_logger.error(f"处理ws消息失败:{e}")
        manager.disconnect(websocket)

'''
    获取训练状态端点
'''
@router.get("/status")
async def get_train_status():
    return train_service.get_status()

'''
    停止训练端点
'''
@router.post("/stop")
async def stop_training():
    if train_service.status != settings.TRAIN_STATUS_RUNNING:
        raise HTTPException(status_code=400, detail="训练未运行中")
    success = train_service.request_stop()
    if not success:
        raise HTTPException(status_code=400, detail="终止训练失败")
    return {"message": "训练已终止"}

'''
    微调ResNet50端点
'''
@router.post("/finetune")
async def finetune_resnet50(request: Request):
    # 从请求头中获取content-type
    content_type = request.headers.get("content-type","")
    if content_type == "application/json":
        # 从请求体中获取json字符串
        body = await request.json()
        
        label = body.get("label")
        #TODO-BUG：20260910 clear_old -> clearOld
        clear_old=body.get("clearOld",False)
        images_base64 = body.get("images", [])

        if not images_base64:
            raise HTTPException(status_code=400, detail="请上传图片")

        if not label:
            raise HTTPException(status_code=400, detail="请上传标签")


        default_logger.info(f"开始微调ResNet50, 标签:{label}\n, 是否清除旧模型: {clear_old}")

        train_dir = Path(settings.UPLOAD_DATASET_UNZIPED_DIR)
        
        if clear_old:
            default_logger.info("清除旧数据")
            for item in train_dir.iterdir():
                if item.is_dir():
                    # 删除非空文件夹
                    shutil.rmtree(item)
                else:
                    item.unlink()


        # 新加入的分类目录
        class_dir = train_dir / label
        
        # 创建分类目录（如果不存在）
        class_dir.mkdir(parents=True, exist_ok=True)

        # 实际保存的图片的数量
        saved_count = 0

        for idx, b64_data in enumerate(images_base64):
            try:
                if "," in b64_data:
                    b64_data = b64_data.split("," , 1)[-1]
                    
                    # base64字符串的长度必须是4的倍数
                    b64_data += "=" * (4 - len(b64_data) % 4) if len(b64_data) % 4 else ""
                    img_bytes = base64.b64decode(b64_data)                   
                    img = Image.open(io.BytesIO(img_bytes))            
                    img.verify()
                    img = Image.open(io.BytesIO(img_bytes))
                    ext = ".jpg"
                    if img.format == "PNG":
                        ext = ".png"
                    elif img.format == "GIF":
                        ext = ".gif"
                    elif img.format == "WEBP":
                        ext = ".webp"

                    safe_name = f"{label}_{idx+1}{ext}"
                    # 保存图片的路径
                    file_path = class_dir / safe_name
                    with open(file_path, "wb") as f:
                        f.write(img_bytes)
                        saved_count += 1
            except Exception as e:
                default_logger.error(f"保存图片失败:{e}")

        if saved_count == 0:
            raise HTTPException(status_code=400, detail="保存的图片数量为0")

        default_logger.info(f"成功保存{saved_count}张图片到目录:{class_dir}")

        success = train_service.start_training(train_dir, settings.FULL_EPOCHS)

        if not success:
            raise HTTPException(status_code=409, detail="正在运行中或者启动训练失败")


        return {
            "message": "训练已启动",
            "status": settings.TRAIN_STATUS_RUNNING
        }
    else:
        raise HTTPException(status_code=415, detail="请上传json格式的请求体")