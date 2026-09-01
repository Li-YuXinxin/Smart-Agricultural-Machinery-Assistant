from fastapi import APIRouter
router = APIRouter(prefix="/api/test3")

@router.get("/")
async def test3():
    return {"service": "智慧农技助手", "status": "running", "code": 200}