from fastapi import APIRouter

# 实例化路由器
router=APIRouter(prefix="/api/test")
@router.get("/")
@router.post("/")
async def test():
    # 返回json格式的数据
    return {"message": "测试成功", "code": 200}