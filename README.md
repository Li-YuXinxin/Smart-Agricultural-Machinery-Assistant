# 🌿 智能植物养护助手

一个集 **花卉识别**、**模型训练**、**知识库管理** 和 **AI 智能问答** 于一体的植物养护系统，包含 Python 后端服务和微信小程序前端。

---

## ✨ 核心功能

| 功能模块 | 说明 |
|---------|------|
| 🌸 **花卉识别** | 基于 ResNet-50 的花卉图像分类，支持 Top-5 预测及置信度展示 |
| 🧠 **模型训练** | 用户上传花卉图片，在线微调 ResNet-50 模型，扩展可识别品种 |
| 📖 **知识库管理** | 上传养护文档（PDF/DOCX/TXT），构建 RAG 向量知识库 |
| 💬 **智能问答** | 基于 RAG + LLM（Qwen2.5）的植物养护 AI 顾问，流式输出 |

---

## 🛠️ 技术栈

### 后端

| 技术 | 用途 |
|------|------|
| Python 3 + **FastAPI** | Web 框架 |
| **PyTorch** + torchvision | 深度学习推理与微调 |
| **LangChain** + ChromaDB | RAG 向量检索 |
| **llama-cpp-python** | 本地 LLM 推理（Qwen2.5-1.5B GGUF） |
| **sentence-transformers** | 文本向量化（all-MiniLM-L6-v2） |

### 前端

| 技术 | 用途 |
|------|------|
| **微信小程序**原生开发 | Skyline 渲染引擎 + glass-easel 组件框架 |
| **WebSocket** | 实时训练状态推送 |
| **SSE**（Server-Sent Events） | 流式 AI 对话 |

---

## 📁 项目结构

```
lyx_stu001/
├── backend/                          # 后端项目
│   └── app/
│       ├── main.py                   # FastAPI 应用入口
│       ├── config/
│       │   └── config.py             # 全局配置（路径、模型、设备检测）
│       ├── routers/
│       │   ├── chat.py               # 聊天问答路由
│       │   ├── classify.py           # 图像分类路由
│       │   ├── knowledge.py          # 知识库管理路由
│       │   └── train.py              # 模型训练路由（含 WebSocket）
│       ├── services/
│       │   ├── classify_service.py   # 图像分类服务（ResNet-50 推理 + 微调）
│       │   ├── train_service.py      # 训练任务生命周期管理
│       │   ├── LLM_Service.py        # LLM 大语言模型服务
│       │   └── rag_service.py        # RAG 检索增强生成服务
│       └── utils/
│           └── common_utils.py       # 日志工具
│
├── miniprogram/                      # 微信小程序前端
│   ├── app.js / app.json / app.wxss  # 全局入口与配置
│   ├── components/navigation-bar/    # 自定义导航栏组件
│   ├── pages/
│   │   ├── home/                     # 首页
│   │   ├── classify/                 # 拍照识花
│   │   ├── train/                    # 训练新品种
│   │   ├── knowledge/                # 养护笔记（知识库）
│   │   └── chat/                     # AI 养护顾问
│   └── styles/utilities.wxss         # 全局工具类样式
│
└── doc/                              # 项目文档
```

---

## 🚀 快速开始

### 环境要求

- Python 3.10+
- Node.js（微信开发者工具依赖）
- 微信开发者工具

### 1. 克隆项目

```bash
git clone <仓库地址>
cd lyx_stu001
```

### 2. 后端环境配置

```bash
cd backend

# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 安装依赖 — 有 NVIDIA 显卡且显存 ≥ 6GB
pip install torch==2.1.0+cu121 torchvision==0.16.0+cu121 torchaudio==2.1.0+cu121 \
    -f https://mirrors.aliyun.com/pytorch-wheels/cu121/

# 安装依赖 — 无独显或显存不足
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2

# 安装其他依赖
pip install -r requirements.txt
```

### 3. 模型文件准备

将以下模型文件放置到对应目录：

| 模型 | 说明 | 路径 |
|------|------|------|
| ResNet-50 预训练权重 | 微软 safetensors 格式 | `config.py` 中 `RESNET50_MODEL_PATH` |
| Qwen2.5-1.5B GGUF | 本地 LLM 量化模型 | `config.py` 中 `LLM_MODEL_PATH` |
| all-MiniLM-L6-v2 | 文本向量化模型 | `config.py` 中 `EMBEDDING_MODEL_PATH` |

### 4. 启动后端服务

```bash
cd backend/app
uvicorn main:app --host 0.0.0.0 --port 8001 --reload
```

### 5. 前端小程序

1. 打开**微信开发者工具**
2. 导入 `miniprogram/` 目录
3. 在 [app.js](miniprogram/app.js) 中修改 `apiBase` 为后端实际地址
4. 编译运行

---

## 📡 API 接口

### 图像分类

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/classify/stats` | 获取模型统计（品种数、是否微调、品种列表） |
| POST | `/api/classify/` | 图像分类识别（Base64 图片 → Top-5 结果） |

### 模型训练

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/train/status` | 获取训练状态（status、epoch 进度） |
| POST | `/api/train/finetune` | 提交微调任务（图片 + 标签） |
| POST | `/api/train/stop` | 终止当前训练 |
| WS | `/api/train/ws` | WebSocket 训练状态实时推送 |

### 知识库管理

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/knowledge/list` | 列出已上传文档 |
| POST | `/api/knowledge/upload` | 上传文档（自动向量化） |
| GET | `/api/knowledge/download` | 下载知识库文档 |
| POST | `/api/knowledge/delete` | 删除文档及向量数据 |

### AI 问答

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/chat/stream` | AI 问答（SSE 流式输出） |

---

## 🏗️ 系统架构

```
┌──────────────────────────────────────────────────────────────┐
│                    微信小程序前端                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐    │
│  │  首页     │  │ 拍照识花  │  │ 训练品种  │  │ 养护顾问  │    │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘    │
└───────┼──────────────┼──────────────┼──────────────┼──────────┘
        │      HTTP/WS │              │      SSE     │
        ▼              ▼              ▼              ▼
┌──────────────────────────────────────────────────────────────┐
│                    FastAPI 后端                               │
│  ┌──────────────────────────────────────────────────────┐   │
│  │                   Services 服务层                     │   │
│  │  ClassifyService │ TrainService │ RagService │ LLM   │   │
│  └──────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │                    基础设施                           │   │
│  │  PyTorch/ResNet-50 │ ChromaDB │ Qwen2.5 GGUF        │   │
│  └──────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────┘
```

---

## 🎨 设计系统

项目采用 **Organic Biophilic（自然有机）** 设计风格，以森林、阳光、土壤为灵感。

- **主色**：森林绿 `#2D7A4F` / 琥珀黄 `#D4943A`
- **背景**：淡叶绿 `#F4F9F0`
- **圆角**：统一使用 8rpx 网格系统
- **组件**：Design Tokens + 工具类样式库

详细设计规范见 [doc/design-system.md](doc/design-system.md)。

---

## 📚 项目文档

- [开发文档](doc/开发文档.md) — 后端模块详解、前端页面说明、数据流架构
- [接口文档](doc/接口文档.md) — 完整 API 接口参数与响应示例
- [设计系统](doc/design-system.md) — 色彩、排版、间距、组件规范
- [开发笔记](doc/note.md) — 关键技术实现要点

---

## 📄 License

本项目仅用于学习与实训目的。
