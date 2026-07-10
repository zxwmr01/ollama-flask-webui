# Local AI Chat Room

[English](README.md) | [中文](README_CN.md)

A Flask-based AI chat application with Ollama integration, Bing search capabilities, and GPU monitoring.

## Features

- **Dynamic Model Listing**: Automatically detects available models from Ollama API
- **Automatic Thinking Format Detection**: Recognizes model-specific thinking formats (DeepSeek, Qwen, Llama, Gemini, etc.)
- **Dual-Language Support**: Chinese/English bilingual interface with one-click switching
- **Bing Web Search**: Built-in web search functionality
- **GPU Monitoring**: Real-time NVIDIA GPU memory and utilization monitoring
- **Markdown Rendering**: Full Markdown support with code highlighting
- **Thinking Process**: Toggleable reasoning process display
- **Multi-Model Switching**: Switch between models on the fly
- **Responsive Design**: Works on desktop and mobile devices
- **Dark Mode**: Automatic dark mode support based on system preferences

## Requirements

- Python 3.8+
- Ollama (with models installed)
- NVIDIA GPU (optional, for GPU monitoring)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/zxwmr01/local-ai-chat-room.git
cd local-ai-chat-room
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install Ollama models（Example: DeepSeek-R1 and Qwen3）:
```bash
ollama pull deepseek-r1:1.5b
ollama pull qwen3:4b
```

## Usage

1. Start the Ollama service:
```bash
ollama serve
```

2. Run the application:
```bash
python app.py
```

3. Open your browser and navigate to:
```
http://localhost:5000
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_URL` | `http://localhost:11434/api/chat` | Ollama chat API endpoint |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama host for model listing |
| `HOST` | `0.0.0.0` | Host to bind |
| `PORT` | `5000` | Port to listen |
| `DEBUG` | `false` | Enable debug mode |
| `SECRET_KEY` | auto-generated | Session secret key |

## Configuration

You can configure the application using a `.env` file:

```env
OLLAMA_URL=http://localhost:11434/api/chat
OLLAMA_HOST=http://localhost:11434
HOST=0.0.0.0
PORT=5000
DEBUG=false
SECRET_KEY=your-secret-key-here
```

## Supported Models

The application automatically detects all models available in your Ollama installation. It supports automatic thinking format detection for the following model types:

- **DeepSeek-R1**: Uses `**思考**` / `**答案**` format
- **Qwen3**: Uses `[思考]` / `[答案]` format  
- **Llama 3**: Uses `<think>` / `<answer>` format
- **Gemini**: Uses `<thinking>` / `<response>` format
- **Phi-3**: Uses `<|think|>` / `<|endthink|>` format
- **Mistral**: Uses `<s>` / `</s>` format
- **Yi**: Uses `<|im_start|>` / `<|im_end|>` format
- **Zephyr**: Uses `<system>` / `<user>` format

Other models will use a generic thinking format that works with most open-source models.

## Project Structure

```
local-ai-chat-room/
├── app.py              # Main Flask application
├── requirements.txt    # Python dependencies
├── .gitignore          # Git ignore rules
├── README.md           # Project documentation (English)
├── README_CN.md        # Project documentation (Chinese)
└── templates/
    └── index.html      # Frontend chat interface
```

## License

[MIT License](LICENSE)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgements

- [Ollama](https://ollama.com/) - AI model serving
- [Flask](https://flask.palletsprojects.com/) - Web framework
- [Beautiful Soup](https://www.crummy.com/software/BeautifulSoup/) - HTML parsing
- [Highlight.js](https://highlightjs.org/) - Code highlighting
- [Marked](https://marked.js.org/) - Markdown parsing
