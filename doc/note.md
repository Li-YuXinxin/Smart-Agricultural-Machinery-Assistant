
## 环境
1. 算框架
    - n卡并且显存>=6GB:
    pip install torch==2.1.0+cu121 torchvision==0.16.0+cu121 torchaudio==2.1.0+cu121 -f https://mirrors.aliyun.com/pytorch-wheels/cu121/
    - 其他:
    pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2





## 编码阶段

**1. 使用日志**
```python
logger = logging.getLogger(name)
```

**2. 路由**
```python
router = APIRouter(prefix="/api/test")
app.include_router(test.router)
```

**3. 严格类型检查**
```
class Settings(BaseSettings):
```

**4. 自定义分类器**
```
class ConsineClassifier:
```

**5. 自定义标准模型到非标准模型的key字典**

**6. 读取模型结构**
```
load_file
```

**7. 构建标准模型**
```
model = models.resnet50(weights=None)
```

**8. 卸载模型**
```python
del self.model
self.model = None
gc.collect()
torch.cuda.empty_cache()
```

**9. 加载模型**
```python
# 提取字典
state_dict = torch.load(pth_path, map_location=self.device)
# 保存字典到标准模型
model.load_state_dict(state_dict)
# 设置模型放入的位置
model.to(self.device)
# 执行以上配置
model.eval()
```

**10. 扩展分类器**
```python
nn.init.kaiming_uniform_(new_fc.weight, a=math.sqrt(5))
```

**11. 划分训练集和验证集**
```python
val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
```

**12. 类别均衡采样**
```python
sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
```
**13. 标签平滑损失函数**
```python
class LabelSmoothingLoss(nn.Module):
```

**14. 扩展fc层**
如果旧分类数为0，则直接创建新分类器
如果新分类数大于旧分类数，则只复制旧分类数的权重
反之则截断

**15. 保存模型**
```python
torch.save(...)
```

**16. ws的管理**
```python
await websocket.accept()
active_connections.append(websocket)
active_connections.remove(websocket)
```

**17. 广播消息**
```python
await asyncio.gather(*tasks, return_exceptions=True)
```

**18. 定义ws端点**
```python
@router.websocket("/ws")
```

**19. 传递错误信息到前端**
```python
raise HTTPException(status_code=400, detail="训练未运行中")
```

**20. 从请求中获取参数**
```python
await request.json()
```

**21. base64转换到图片**
```python
base64.b64decode(b64_data)
```

**22. 开始线程**
```python
threading.Thread(target=classify, args=(img_bytes,)).start()
```

**23. 将变量包装为函数**
lambda:变量

**24. 循环广播**
```python
asyncio.get_running_loop()
```

**25. 获得小程序的文件系统管理器**
```python
wx.getFileSystemManager()
```

**26. 获得当前页面**
```python
getApp()
```

**27. 连接ws端点**
```python
wx.connectSocket
```

**28. 发送http请求**
```python
wx.request
```

**29. 将图片转换为张量**
```python
img = Image.open(io.BytesIO(img_bytes))
img = self.transform(image).unsqueeze(0).to(self.device)
```