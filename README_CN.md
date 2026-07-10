# Local AI Chat Room

一个基于 Flask 的 AI 聊天应用，集成了 Ollama、Bing 搜索和 GPU 监控功能。

## 功能特性

- **动态模型列表**: 自动从 Ollama API 检测可用模型
- **自动思考格式检测**: 识别模型特定的思考格式（DeepSeek、Qwen、Llama、Gemini 等）
- **双语言支持**: 中英文双语界面，一键切换
- **Bing 网络搜索**: 内置网络搜索功能
- **GPU 监控**: 实时 NVIDIA GPU 显存和利用率监控
- **Markdown 渲染**: 完整的 Markdown 支持和代码高亮
- **思考过程展示**: 可切换的推理过程显示
- **多模型切换**: 实时切换模型
- **响应式设计**: 支持桌面和移动设备
- **暗色模式**: 根据系统偏好自动适配暗色主题

## 环境要求

- Python 3.8+
- Ollama（已安装模型）
- NVIDIA GPU（可选，用于 GPU 监控）

## 安装步骤

1. 克隆仓库：
```bash
git clone https://github.com/yourusername/local-ai-chat-room.git
cd local-ai-chat-room
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

3. 安装 Ollama 模型（例如：DeepSeek-R1、Qwen3）：
```bash
ollama pull DeepSeek-R1
ollama pull Qwen3
```

## 使用方法

1. 启动 Ollama 服务：
```bash
ollama serve
```

2. 运行应用：
```bash
python app.py
```

3. 打开浏览器访问：
```
http://localhost:5000
```

## 环境变量

| 变量 | 默认值 | 描述 |
|------|--------|------|
| `OLLAMA_URL` | `http://localhost:11434/api/chat` | Ollama 聊天 API 端点 |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama 主机地址（用于模型列表） |
| `HOST` | `0.0.0.0` | 绑定的主机地址 |
| `PORT` | `5000` | 监听端口 |
| `DEBUG` | `false` | 启用调试模式 |
| `SECRET_KEY` | 自动生成 | 会话密钥 |

## 配置说明

您可以使用 `.env` 文件配置应用：

```env
OLLAMA_URL=http://localhost:11434/api/chat
OLLAMA_HOST=http://localhost:11434
HOST=0.0.0.0
PORT=5000
DEBUG=false
SECRET_KEY=your-secret-key-here
```

## 支持的模型

应用会自动检测 Ollama 中已安装的所有模型。支持以下模型类型的自动思考格式检测：

- **DeepSeek-R1**: 使用 `**思考**` / `**答案**` 格式
- **Qwen3**: 使用 `[思考]` / `[答案]` 格式
- **Llama 3**: 使用 `<think>` / `<answer>` 格式
- **Gemini**: 使用 `<thinking>` / `<response>` 格式
- **Phi-3**: 使用 `<|think|>` / `<|endthink|>` 格式
- **Mistral**: 使用 `<s>` / `</s>` 格式
- **Yi**: 使用 `<|im_start|>` / `<|im_end|>` 格式
- **Zephyr**: 使用 `<system>` / `<user>` 格式

其他模型将使用通用思考格式，适用于大多数开源模型。

## 项目结构

```
local-ai-chat-room/
├── app.py              # 主 Flask 应用
├── requirements.txt    # Python 依赖
├── .gitignore          # Git 忽略规则
├── README.md           # 项目文档（英文）
├── README_CN.md        # 项目文档（中文）
└── templates/
    └── index.html      # 前端聊天界面
```

## 许可证

MIT License

## 贡献

欢迎贡献！请提交 Pull Request。

## 致谢

- [Ollama](https://ollama.com/) - AI 模型服务
- [Flask](https://flask.palletsprojects.com/) - Web 框架
- [Beautiful Soup](https://www.crummy.com/software/BeautifulSoup/) - HTML 解析
- [Highlight.js](https://highlightjs.org/) - 代码高亮
- [Marked](https://marked.js.org/) - Markdown 解析