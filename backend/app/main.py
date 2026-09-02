import multiprocessing
# 多线程支持
multiprocessing.freeze_support()
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import gc
import uvicorn
from routers import test, test2, test3, test4
from utils.common_utils import default_logger
from config.config import settings
from services.classify_service import load_resnet_50_from_local_safetensors


# 创建 FastAPI 应用实例
app = FastAPI(
    title = "智慧农技助手",
    version = "1.0.0",
    description = "智慧农机助手的API接口",
)

# 跨域配置 CORS
app.add_middleware(
                    CORSMiddleware,
                    allow_origins = ["*"],
                    allow_credentials = True,
                    allow_methods = ["*"],
                    allow_headers = ["*"]
                    )

# 生命周期事件——启动钩子
@app.on_event("startup")
async def startup_event():
    # 启动时的初始化操作
    load_resnet_50_from_local_safetensors()
    default_logger.info("应用启动完成")

# 生命周期事件——关闭钩子
@app.on_event("shutdown")
async def shutdown_event():
    # 关闭时的清理操作
    gc.collect()    # 强制清除当前程序占用的内存和显存
    # print("应用关闭完成")
    default_logger.info("应用关闭完成")

# 注册路由
app.include_router(test.router)
app.include_router(test2.router)
app.include_router(test3.router)
app.include_router(test4.router)
    
if __name__=="__main__":
    # 启动服务器
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
    
    
    