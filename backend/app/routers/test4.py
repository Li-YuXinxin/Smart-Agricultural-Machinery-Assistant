from fastapi import APIRouter
router = APIRouter(prefix="/api/test4")

@router.get("/status")
async def test4():
    return {"module": "测试模块4", "version": "1.0.0", "code": 200}