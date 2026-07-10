#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Local AI Chat Room - Open Source Edition
A Flask-based AI chat application with Ollama integration,
Bing search capabilities, and GPU monitoring.

License: MIT
"""

import os
from flask import Flask, render_template, request, Response, jsonify, session
import requests
import json
import secrets
import re
import logging
from collections import defaultdict
from datetime import datetime, timedelta

from bs4 import BeautifulSoup
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', secrets.token_hex(16))

app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=7)
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'

server_sessions = defaultdict(dict)
session_last_access = {}

OLLAMA_URL = os.environ.get('OLLAMA_URL', "http://localhost:11434/api/chat")
OLLAMA_HOST = os.environ.get('OLLAMA_HOST', "http://localhost:11434")

DEFAULT_MODEL = "deepseek-r1:1.5b"

DEEPSEEK_THINKING_PATTERN = re.compile(r'<(think|thinking)>(.*?)</(think|thinking)>', re.DOTALL)
QWEN_THINKING_PATTERN = re.compile(r'<(thought|thinking)>(.*?)</(thought|thinking)>', re.DOTALL)
GENERIC_THINKING_PATTERNS = [
    re.compile(r'<(think|thinking|thought)>(.*?)</(think|thinking|thought)>', re.DOTALL),
    re.compile(r'\*\*思考\*\*:(.*?)(?=\*\*答案\*\*|$)', re.DOTALL),
    re.compile(r'Thinking:(.*?)(?=Answer:|$)', re.DOTALL),
    re.compile(r'思考：(.*?)(?=答案：|$)', re.DOTALL),
]


def detect_thinking_format(model_name: str) -> str:
    """Detect thinking format based on model name"""
    model_lower = model_name.lower()
    if 'deepseek' in model_lower:
        return "deepseek"
    elif 'qwen' in model_lower or 'tongyi' in model_lower:
        return "qwen"
    elif 'llama' in model_lower:
        return "llama"
    elif 'mistral' in model_lower:
        return "mistral"
    else:
        return "generic"


def get_ollama_models() -> dict:
    """Fetch available models from Ollama API"""
    try:
        response = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
        if response.status_code == 200:
            data = response.json()
            models = {}
            for model_info in data.get('models', []):
                name = model_info.get('name', '')
                model_name = name.split(':')[0] if ':' in name else name
                
                thinking_format = detect_thinking_format(name)
                
                models[name] = {
                    "name": model_name,
                    "description": model_info.get('details', {}).get('model', name),
                    "thinking_format": thinking_format,
                    "size": model_info.get('details', {}).get('size', 0),
                    "modified_at": model_info.get('modified_at', '')
                }
            
            if models:
                return models
    except Exception as e:
        logger.warning(f"Failed to fetch Ollama models: {e}")
    
    return {
        "deepseek-r1:1.5b": {
            "name": "DeepSeek-R1 1.5B",
            "description": "Lightweight reasoning model, excellent at logical reasoning and code generation",
            "thinking_format": "deepseek"
        },
        "qwen3:4b": {
            "name": "Qwen3 4B",
            "description": "Tongyi Qianwen 3rd generation, balanced performance and speed, Chinese optimized",
            "thinking_format": "qwen"
        }
    }


AVAILABLE_MODELS = get_ollama_models()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


LANG_EN = {
    "welcome_title": "Welcome to Local AI Chat Room",
    "welcome_markdown": "I support full **Markdown** and **code highlighting**!",
    "features_title": "Features",
    "feature_markdown": "Headings, lists, tables, etc.",
    "feature_code": "Automatic language detection",
    "feature_thinking": "Toggle reasoning display",
    "feature_models": "Multi-Model Switching",
    "feature_gpu": "Real-time memory and utilization",
    "feature_search": "Bing search integration",
    "markdown_rendering": "Markdown Rendering",
    "code_highlighting": "Code Highlighting",
    "gpu_monitoring": "GPU Monitoring",
    "welcome_footer": "Click the top-right button to switch models. Click the top-left menu for configuration.<br>Toggle thinking process and web search on the left side of the input area.",
    "welcome_start": "Start your first conversation!",
    "thinking": "Thinking",
    "web_search": "Web Search",
    "placeholder": "Type your message... (Enter to send, Shift+Enter for new line)",
    "send": "Send",
    "typing": "Generating response...",
    "no_response": "No response received",
    "settings": "Settings",
    "customize": "Customize your AI assistant",
    "system_prompt": "System Prompt",
    "prompt_placeholder": "Enter system prompt to define AI behavior...",
    "save_clear": "Save & Clear",
    "reset": "Reset",
    "conversation": "Conversation",
    "clear_history": "Clear History",
    "conv_continuous": "Supports continuous conversation",
    "conv_switch_clear": "Switching model clears history",
    "gpu_monitor": "GPU Monitor",
    "loading_gpu": "Loading GPU info...",
    "no_gpu": "No NVIDIA GPU detected",
    "memory": "Memory",
    "usage": "Usage",
    "gpu_core": "GPU Core",
    "switched_model": "Switched to **{model}**\n\nConversation history cleared, start a new conversation!",
    "prompt_updated": "System prompt updated, conversation history cleared.",
    "history_cleared": "Conversation history cleared",
    "thinking_enabled": "Thinking process enabled",
    "thinking_disabled": "Thinking process disabled",
    "search_enabled": "Web search enabled",
    "search_disabled": "Web search disabled",
    "prompt_saved": "System prompt saved",
    "error_connect": "Failed to connect to Ollama service",
    "error_timeout": "Request timed out, please retry",
    "error_server": "Server error: {error}",
    "error_model": "Invalid model",
    "error_update": "Failed to update: {error}",
    "error_clear": "Failed to clear: {error}",
    "error_switch": "Failed to switch model: {error}",
    "error_search": "Failed to toggle search: {error}",
    "thinking_process": "Thinking Process",
    "ai_assistant": "AI Assistant",
    "you": "You"
}

LANG_ZH = {
    "welcome_title": "欢迎使用本地 AI 聊天室",
    "welcome_markdown": "我支持完整的 **Markdown** 和 **代码高亮**！",
    "features_title": "功能特性",
    "feature_markdown": "标题、列表、表格等",
    "feature_code": "自动语言检测",
    "feature_thinking": "切换推理过程显示",
    "feature_models": "多模型切换",
    "feature_gpu": "实时显存和利用率",
    "feature_search": "必应搜索集成",
    "markdown_rendering": "Markdown 渲染",
    "code_highlighting": "代码高亮",
    "gpu_monitoring": "GPU 监控",
    "welcome_footer": "点击右上角按钮切换模型，点击左上角菜单进行配置。<br>在输入区域左侧切换思考过程和网络搜索。",
    "welcome_start": "开始您的第一次对话！",
    "thinking": "思考过程",
    "web_search": "网络搜索",
    "placeholder": "输入您的消息...（回车发送，Shift+回车换行）",
    "send": "发送",
    "typing": "正在生成回复...",
    "no_response": "未收到回复",
    "settings": "设置",
    "customize": "自定义您的 AI 助手",
    "system_prompt": "系统提示词",
    "prompt_placeholder": "输入系统提示词来定义 AI 行为...",
    "save_clear": "保存并清空",
    "reset": "重置",
    "conversation": "对话",
    "clear_history": "清空历史",
    "conv_continuous": "支持连续对话",
    "conv_switch_clear": "切换模型会清空历史",
    "gpu_monitor": "GPU 监控",
    "loading_gpu": "加载 GPU 信息...",
    "no_gpu": "未检测到 NVIDIA GPU",
    "memory": "显存",
    "usage": "使用率",
    "gpu_core": "GPU 核心",
    "switched_model": "已切换到 **{model}**\n\n对话历史已清空，开始新的对话！",
    "prompt_updated": "系统提示词已更新，对话历史已清空。",
    "history_cleared": "对话历史已清空",
    "thinking_enabled": "思考过程已启用",
    "thinking_disabled": "思考过程已禁用",
    "search_enabled": "网络搜索已启用",
    "search_disabled": "网络搜索已禁用",
    "prompt_saved": "系统提示词已保存",
    "error_connect": "无法连接到 Ollama 服务",
    "error_timeout": "请求超时，请重试",
    "error_server": "服务器错误: {error}",
    "error_model": "无效的模型",
    "error_update": "更新失败: {error}",
    "error_clear": "清空失败: {error}",
    "error_switch": "切换模型失败: {error}",
    "error_search": "切换搜索失败: {error}",
    "thinking_process": "思考过程",
    "ai_assistant": "AI 助手",
    "you": "你"
}


def get_translation(lang='en'):
    """Get translation dictionary for specified language"""
    return LANG_ZH if lang == 'zh' else LANG_EN

# ============================================================
# Bing Search Engine Module - Maximum results, no filtering
# ============================================================

class SearchResult:
    """Search result data structure"""
    def __init__(self, title: str, url: str, snippet: str = "", date: str = "", source: str = ""):
        self.title = title
        self.url = url
        self.snippet = snippet
        self.date = date
        self.source = source
        self.content = ""
    
    def to_dict(self):
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet[:500] if self.snippet else "",
            "date": self.date,
            "source": self.source,
            "content": self.content[:1000] if self.content else ""
        }


class BingSearchEngine:
    """Bing search engine - maximum results, no filtering"""
    
    def __init__(self):
        self.search_url = "https://www.bing.com/search"
        self.news_url = "https://www.bing.com/news/search"
        self.user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        self.timeout = 30
    
    def search(self, query: str, max_results: int = 25, search_type: str = "web") -> list:
        """Perform Bing search and return as many results as possible"""
        results = []
        
        try:
            headers = {
                "User-Agent": self.user_agent,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9,zh-CN;q=0.8",
                "Accept-Encoding": "gzip, deflate",
                "Connection": "keep-alive",
            }
            
            url = self.news_url if search_type == "news" else self.search_url
            params = {
                "q": query,
                "count": max_results,
                "setlang": "en-US",
                "first": 1
            }
            
            logger.info(f"Bing {search_type} searching for: {query}")
            response = requests.get(url, params=params, headers=headers, timeout=self.timeout)
            
            if response.status_code != 200:
                logger.error(f"Bing search failed with status: {response.status_code}")
                return results
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            result_selectors = [
                '#b_results > li.b_algo',
                '.b_algo',
                'li.b_algo',
                '.b_ans',
                '.b_slidebar'
            ]
            
            if search_type == "news":
                result_selectors = [
                    '.newsresult',
                    '.news-card',
                    '.tablenews',
                    '.news-item'
                ]
            
            result_blocks = []
            for selector in result_selectors:
                result_blocks = soup.select(selector)
                if result_blocks:
                    logger.info(f"Found {len(result_blocks)} raw results")
                    break
            
            for block in result_blocks[:max_results]:
                try:
                    title_elem = None
                    title_selectors = ['h2 a', '.b_title a', 'a.title', '.news_title a']
                    for ts in title_selectors:
                        title_elem = block.select_one(ts)
                        if title_elem:
                            break
                    
                    if not title_elem:
                        continue
                    
                    title = title_elem.get_text(strip=True)
                    url = title_elem.get('href', '')
                    
                    if not title or not url or len(title) < 3:
                        continue
                    
                    snippet = ""
                    snippet_selectors = [
                        '.b_caption p',
                        '.b_snippet',
                        '.b_attribution',
                        '.snippet',
                        '.news-desc',
                        '.description'
                    ]
                    for ss in snippet_selectors:
                        snippet_elem = block.select_one(ss)
                        if snippet_elem:
                            snippet = snippet_elem.get_text(strip=True)
                            break
                    
                    date = ""
                    date_selectors = ['.news_dt', '.date', '.time', '.news-time']
                    for ds in date_selectors:
                        date_elem = block.select_one(ds)
                        if date_elem:
                            date = date_elem.get_text(strip=True)
                            break
                    
                    source = ""
                    source_selectors = ['.b_attribution', '.source', '.news-source']
                    for ss in source_selectors:
                        source_elem = block.select_one(ss)
                        if source_elem:
                            source = source_elem.get_text(strip=True)
                            break
                    
                    results.append(SearchResult(title, url, snippet[:800], date, source))
                        
                except Exception as e:
                    logger.warning(f"Error parsing result: {e}")
                    continue
            
            logger.info(f"Bing returned {len(results)} results for: {query}")
            return results
            
        except requests.exceptions.Timeout:
            logger.error("Bing search request timeout")
            return []
        except Exception as e:
            logger.error(f"Bing search failed: {e}")
            return []
    
    def fetch_page_content(self, url: str, max_length: int = 6000) -> str:
        """Fetch page content"""
        try:
            headers = {
                "User-Agent": self.user_agent,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
            }
            response = requests.get(url, headers=headers, timeout=15, allow_redirects=True)
            
            if response.status_code != 200:
                return ""
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript"]):
                tag.decompose()
            
            content_selectors = [
                'main', 'article', 
                '.post-content', '.entry-content', '.article-content',
                '.content', '.main-content', '#content',
                '.post-body', '.markdown-body', '.article-body',
                '.news-content', '.detail-content', '.story-content',
                '.blog-post', '.post', '.entry'
            ]
            
            content_text = ""
            for selector in content_selectors:
                element = soup.select_one(selector)
                if element:
                    content_text = element.get_text(separator='\n', strip=True)
                    break
            
            if not content_text and soup.body:
                paragraphs = soup.body.find_all('p')
                if len(paragraphs) > 3:
                    content_text = '\n'.join([p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 30])
                else:
                    content_text = soup.body.get_text(separator='\n', strip=True)
            elif not content_text:
                content_text = soup.get_text(separator='\n', strip=True)
            
            lines = [line.strip() for line in content_text.split('\n') if line.strip() and len(line.strip()) > 20]
            content_text = '\n\n'.join(lines[:60])
            
            if len(content_text) > max_length:
                content_text = content_text[:max_length] + "..."
            
            return content_text
            
        except Exception as e:
            logger.warning(f"Failed to fetch page: {e}")
            return ""


class WebSearchTool:
    """Web search tool - maximum results, no filtering"""
    
    def __init__(self):
        self.engine = BingSearchEngine()
        self.executor = ThreadPoolExecutor(max_workers=8)
    
    def search_and_summarize(self, query: str, max_results: int = 20, fetch_content: bool = True, search_type: str = "web") -> str:
        """Perform search and return formatted results summary"""
        raw_results = self.engine.search(query, max_results=25, search_type=search_type)
        
        if not raw_results:
            return f"No search results found for '{query}'. Please try different keywords."
        
        final_results = raw_results[:max_results]
        
        logger.info(f"Returning {len(final_results)} results for: {query}")
        
        if fetch_content:
            self._fetch_contents_concurrent(final_results[:12])
        
        formatted = self._format_search_results(final_results, query, search_type)
        
        return formatted
    
    def _fetch_contents_concurrent(self, results: list):
        """Fetch page contents concurrently"""
        def fetch_single(result):
            if result.url and not result.content:
                result.content = self.engine.fetch_page_content(result.url)
            return result
        
        futures = []
        for result in results:
            future = self.executor.submit(fetch_single, result)
            futures.append(future)
        
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                logger.warning(f"Fetch failed: {e}")
    
    def _format_search_results(self, results: list, query: str, search_type: str) -> str:
        """Format search results output"""
        type_label = "News" if search_type == "news" else "Web"
        
        formatted = f"## {type_label} Search Results: '{query}'\n\n"
        formatted += f"Found {len(results)} related results:\n\n"
        
        for i, result in enumerate(results, 1):
            formatted += f"### {i}. {result.title}\n"
            formatted += f"URL: {result.url}\n"
            
            if result.source:
                formatted += f"Source: {result.source}\n"
            if result.date:
                formatted += f"Date: {result.date}\n"
            if result.snippet:
                formatted += f"Snippet: {result.snippet}\n"
            if result.content:
                content_preview = result.content[:600]
                formatted += f"Details: {content_preview}"
                if len(result.content) > 600:
                    formatted += "..."
                formatted += "\n"
            
            formatted += "\n"
        
        formatted += "---\n"
        formatted += "Please answer the user's question based on the above search results.\n"
        formatted += "Requirements:\n"
        formatted += "1. Cite sources when referencing information from search results\n"
        formatted += "2. Provide comprehensive answers by synthesizing multiple sources\n"
        formatted += "3. If search results are insufficient, use your knowledge to supplement\n"
        
        return formatted
    
    def search_structured(self, query: str, max_results: int = 20, search_type: str = "web") -> dict:
        """Return structured search results"""
        raw_results = self.engine.search(query, max_results=25, search_type=search_type)
        final_results = raw_results[:max_results]
        
        search_results = []
        for result in final_results:
            search_results.append({
                "title": result.title,
                "url": result.url,
                "snippet": result.snippet,
                "date": result.date,
                "source": result.source
            })
        
        return {
            "query": query,
            "search_type": search_type,
            "results": search_results,
            "count": len(search_results),
            "raw_count": len(raw_results)
        }


web_search = WebSearchTool()


def extract_search_keyword(query: str) -> str:
    """Extract search keywords from user query"""
    remove_words = [
        "help me", "please", "search", "find", "look up", "check",
        "I want to know", "tell me", "what", "why", "how", "can you",
        "could you", "would you", "search for", "look for"
    ]
    
    keyword = query
    for word in remove_words:
        keyword = keyword.replace(word, "")
    
    keyword = re.sub(r'[,.;:?!\"\'()\[\]{}<>]', '', keyword)
    
    if len(keyword) > 60:
        keyword = keyword[:60]
    
    return keyword.strip()


# ============================================================
# GPU Monitoring Functions
# ============================================================

try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False
    print("Warning: pynvml not installed. GPU monitoring disabled.")


def get_gpu_info():
    """Get GPU memory and utilization information"""
    if not NVML_AVAILABLE:
        return []
    
    gpu_info_list = []
    
    try:
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            
            try:
                name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
            except:
                name = f"GPU {i}"
            
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            total_mb = memory_info.total / 1024 / 1024
            used_mb = memory_info.used / 1024 / 1024
            memory_util = (used_mb / total_mb) * 100 if total_mb > 0 else 0
            
            try:
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_util = utilization.gpu
            except:
                gpu_util = 0
            
            try:
                temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            except:
                temperature = 0
            
            gpu_info_list.append({
                "index": i,
                "name": name,
                "total_memory": round(total_mb),
                "used_memory": round(used_mb),
                "free_memory": round(total_mb - used_mb),
                "memory_util": round(memory_util, 1),
                "gpu_util": gpu_util,
                "temperature": temperature
            })
        
        pynvml.nvmlShutdown()
        
    except Exception as e:
        logger.error(f"GPU info error: {e}")
        return []
    
    return gpu_info_list


# ============================================================
# Content Processing Functions
# ============================================================

def process_deepseek_content(content):
    """Process DeepSeek-R1 content"""
    if not content:
        return "", ""
    
    think_match = DEEPSEEK_THINKING_PATTERN.search(content)
    if think_match:
        thinking = think_match.group(2).strip()
        answer = DEEPSEEK_THINKING_PATTERN.sub('', content).strip()
        return thinking, answer
    
    return "", content


def process_qwen_content(content):
    """Process Qwen3 content"""
    if not content:
        return "", ""
    
    think_match = QWEN_THINKING_PATTERN.search(content)
    if think_match:
        thinking = think_match.group(2).strip()
        answer = QWEN_THINKING_PATTERN.sub('', content).strip()
        return thinking, answer
    
    thinking_markers = ['Thinking:', 'Thought:', 'Analysis:', 'Reasoning:', '思考：', '思考过程：', '分析：', '推理：']
    answer_markers = ['Answer:', 'Response:', 'Final Answer:', 'Conclusion:', '答案：', '回答：', '最终答案：', '结论：']
    
    for marker in thinking_markers:
        if marker in content:
            parts = content.split(marker, 1)
            if len(parts) == 2:
                remaining = parts[1]
                for ans_marker in answer_markers:
                    if ans_marker in remaining:
                        thinking_part, answer_part = remaining.split(ans_marker, 1)
                        return thinking_part.strip(), answer_part.strip()
                return remaining.strip(), ""
    
    return "", content


def process_generic_content(content):
    """Process content with generic thinking format detection"""
    if not content:
        return "", ""
    
    for pattern in GENERIC_THINKING_PATTERNS:
        think_match = pattern.search(content)
        if think_match:
            thinking = think_match.group(2).strip()
            answer = pattern.sub('', content).strip()
            return thinking, answer
    
    thinking_markers = ['Thinking:', 'Thought:', 'Analysis:', 'Reasoning:', '思考：', '思考过程：', '分析：', '推理：']
    answer_markers = ['Answer:', 'Response:', 'Final Answer:', 'Conclusion:', '答案：', '回答：', '最终答案：', '结论：']
    
    for marker in thinking_markers:
        if marker in content:
            parts = content.split(marker, 1)
            if len(parts) == 2:
                remaining = parts[1]
                for ans_marker in answer_markers:
                    if ans_marker in remaining:
                        thinking_part, answer_part = remaining.split(ans_marker, 1)
                        return thinking_part.strip(), answer_part.strip()
                return remaining.strip(), ""
    
    return "", content


def process_content_by_model(content, model_name):
    """Process content by model type"""
    model_config = AVAILABLE_MODELS.get(model_name, {})
    thinking_format = model_config.get("thinking_format", "generic")
    
    if thinking_format == "deepseek":
        return process_deepseek_content(content)
    elif thinking_format == "qwen":
        return process_qwen_content(content)
    elif thinking_format == "llama" or thinking_format == "mistral":
        return process_generic_content(content)
    else:
        return process_generic_content(content)


def get_server_session():
    """Get server-side session data"""
    session_id = session.get('server_session_id')
    if not session_id:
        session_id = secrets.token_hex(32)
        session['server_session_id'] = session_id
        session.permanent = True
    
    session_last_access[session_id] = datetime.now()
    return server_sessions[session_id]


def save_server_session(data):
    """Save server-side session data"""
    session_id = session.get('server_session_id')
    if session_id:
        server_sessions[session_id] = data
        session_last_access[session_id] = datetime.now()


def cleanup_expired_sessions():
    """Clean up expired sessions"""
    now = datetime.now()
    expired = []
    for sid, last_access in session_last_access.items():
        if now - last_access > timedelta(days=7):
            expired.append(sid)
    for sid in expired:
        if sid in server_sessions:
            del server_sessions[sid]
        del session_last_access[sid]


# ============================================================
# Flask Routes
# ============================================================

@app.route('/')
def index():
    """Serve the frontend chat page"""
    cleanup_expired_sessions()
    server_data = get_server_session()
    
    if 'messages' not in server_data:
        server_data['messages'] = []
    if 'system_prompt' not in server_data:
        server_data['system_prompt'] = """You are a helpful AI assistant. Please answer the user's questions.

When web search is enabled, you will receive search results injected by the system. Please answer based on the search results.

Requirements:
1. Cite sources when referencing information from search results
2. Provide comprehensive answers by synthesizing multiple sources
3. If search results are insufficient, use your knowledge to supplement"""
    if 'current_model' not in server_data:
        server_data['current_model'] = DEFAULT_MODEL
    if 'enable_search' not in server_data:
        server_data['enable_search'] = False
    
    save_server_session(server_data)
    return render_template('index.html')


@app.route('/get_config', methods=['GET'])
def get_config():
    """Get current configuration"""
    server_data = get_server_session()
    lang = server_data.get('language', 'en')
    return jsonify({
        'system_prompt': server_data.get('system_prompt', ''),
        'messages': server_data.get('messages', []),
        'current_model': server_data.get('current_model', DEFAULT_MODEL),
        'enable_search': server_data.get('enable_search', False),
        'available_models': AVAILABLE_MODELS,
        'language': lang,
        'translations': get_translation(lang)
    })


@app.route('/update_system_prompt', methods=['POST'])
def update_system_prompt():
    """Update system prompt"""
    data = request.json
    server_data = get_server_session()
    server_data['system_prompt'] = data.get('system_prompt', '')
    server_data['messages'] = []
    save_server_session(server_data)
    return jsonify({'status': 'success'})


@app.route('/toggle_search', methods=['POST'])
def toggle_search():
    """Toggle search feature"""
    data = request.json
    server_data = get_server_session()
    server_data['enable_search'] = data.get('enable_search', False)
    save_server_session(server_data)
    logger.info(f"Search enabled: {server_data['enable_search']}")
    return jsonify({'status': 'success', 'enable_search': server_data['enable_search']})


@app.route('/switch_language', methods=['POST'])
def switch_language():
    """Switch interface language"""
    data = request.json
    new_lang = data.get('language', 'en')
    if new_lang not in ['en', 'zh']:
        new_lang = 'en'
    
    server_data = get_server_session()
    server_data['language'] = new_lang
    save_server_session(server_data)
    
    return jsonify({
        'status': 'success',
        'language': new_lang,
        'translations': get_translation(new_lang)
    })


@app.route('/switch_model', methods=['POST'])
def switch_model():
    """Switch model"""
    data = request.json
    new_model = data.get('model')
    if new_model in AVAILABLE_MODELS:
        server_data = get_server_session()
        server_data['current_model'] = new_model
        server_data['messages'] = []
        save_server_session(server_data)
        logger.info(f"Switched to model: {new_model}")
        return jsonify({'status': 'success', 'model': new_model})
    return jsonify({'status': 'error', 'message': 'Invalid model'}), 400


@app.route('/clear_history', methods=['POST'])
def clear_history():
    """Clear conversation history"""
    server_data = get_server_session()
    server_data['messages'] = []
    save_server_session(server_data)
    return jsonify({'status': 'success'})


@app.route('/api/gpu_info', methods=['GET'])
def api_gpu_info():
    """Get GPU information"""
    info = get_gpu_info()
    return jsonify({
        'success': True,
        'gpus': info,
        'has_gpu': NVML_AVAILABLE and len(info) > 0
    })


@app.route('/api/search', methods=['POST'])
def api_search():
    """API endpoint for web search"""
    data = request.json
    query = data.get('query', '')
    max_results = data.get('max_results', 20)
    fetch_content = data.get('fetch_content', True)
    search_type = data.get('search_type', 'web')
    
    if not query:
        return jsonify({'error': 'No query provided'}), 400
    
    result_text = web_search.search_and_summarize(query, max_results, fetch_content, search_type)
    
    return jsonify({
        'success': True,
        'query': query,
        'search_type': search_type,
        'result': result_text
    })


@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat requests"""
    user_message = request.json.get('message', '')
    server_data = get_server_session()
    current_model = server_data.get('current_model', DEFAULT_MODEL)
    system_prompt = server_data.get('system_prompt', '')
    enable_search = server_data.get('enable_search', False)
    
    search_context = ""
    if enable_search:
        search_keyword = extract_search_keyword(user_message)
        logger.info(f"Search enabled for: {search_keyword}")
        
        # Get up to 20 search results
        search_context = web_search.search_and_summarize(
            search_keyword, 
            max_results=18,
            fetch_content=True,
            search_type="web"
        )
        
        enhanced_prompt = system_prompt + f"\n\nWeb search is enabled, here are the search results:\n{search_context}"
    else:
        enhanced_prompt = system_prompt
    
    history = server_data.get('messages', [])[-20:]
    
    messages_list = server_data.get('messages', [])
    messages_list.append({"role": "user", "content": user_message})
    if len(messages_list) > 30:
        messages_list = messages_list[-30:]
    server_data['messages'] = messages_list
    save_server_session(server_data)
    
    messages = []
    if enhanced_prompt:
        messages.append({"role": "system", "content": enhanced_prompt})
    messages.extend(history)
    messages.append({"role": "user", "content": user_message})
    
    ollama_payload = {
        "model": current_model,
        "messages": messages,
        "stream": True,
        "options": {
            "temperature": 0.7,
            "top_p": 0.9,
            "num_predict": 4096,
            "repeat_penalty": 1.1,
        }
    }
    
    def generate():
        full_content = ""
        try:
            response = requests.post(OLLAMA_URL, json=ollama_payload, stream=True, timeout=180)
            response.raise_for_status()
            
            for line in response.iter_lines(decode_unicode=True):
                if line:
                    try:
                        chunk_data = json.loads(line)
                        
                        if 'message' in chunk_data and 'content' in chunk_data['message']:
                            content_chunk = chunk_data['message']['content']
                            full_content += content_chunk
                            yield f"data: {json.dumps({'type': 'raw', 'content': content_chunk})}\n\n"
                        
                        if chunk_data.get('done', False):
                            thinking, answer = process_content_by_model(full_content, current_model)
                            yield f"data: {json.dumps({'type': 'final', 'thinking': thinking, 'answer': answer})}\n\n"
                    
                    except json.JSONDecodeError:
                        continue
                        
        except requests.exceptions.ConnectionError:
            yield f"data: {json.dumps({'type': 'error', 'content': 'Failed to connect to Ollama service'})}\n\n"
        except requests.exceptions.Timeout:
            yield f"data: {json.dumps({'type': 'error', 'content': 'Request timed out, please retry'})}\n\n"
        except Exception as e:
            logger.error(f"Ollama error: {e}")
            yield f"data: {json.dumps({'type': 'error', 'content': f'Server error: {str(e)}'})}\n\n"
        
        yield f"data: {json.dumps({'type': 'done'})}\n\n"
    
    return Response(generate(), mimetype='text/event-stream')


@app.route('/save_assistant_reply', methods=['POST'])
def save_assistant_reply():
    """Save assistant reply to session"""
    data = request.json
    answer = data.get('answer', '')
    if answer:
        server_data = get_server_session()
        messages_list = server_data.get('messages', [])
        messages_list.append({"role": "assistant", "content": answer})
        if len(messages_list) > 30:
            messages_list = messages_list[-30:]
        server_data['messages'] = messages_list
        save_server_session(server_data)
    return jsonify({'status': 'success'})


if __name__ == '__main__':
    HOST = os.environ.get('HOST', '0.0.0.0')
    PORT = int(os.environ.get('PORT', 5000))
    DEBUG = os.environ.get('DEBUG', 'false').lower() == 'true'

    print("=" * 60)
    print("Local AI Chat Room - Open Source Edition")
    print("=" * 60)
    print(f"Host: {HOST}")
    print(f"Port: {PORT}")
    print(f"Debug: {DEBUG}")
    print(f"Ollama URL: {OLLAMA_URL}")
    print("")
    print("Features:")
    print("- Ollama integration (DeepSeek-R1, Qwen3)")
    print("- Bing web search")
    print("- GPU monitoring (NVIDIA)")
    print("- Markdown rendering with code highlighting")
    print("- Thinking process display")
    print("- Multi-model switching")
    print("")
    print("Environment Variables:")
    print("  - OLLAMA_URL: Ollama API endpoint (default: http://localhost:11434/api/chat)")
    print("  - HOST: Host to bind (default: 0.0.0.0)")
    print("  - PORT: Port to listen (default: 5000)")
    print("  - DEBUG: Enable debug mode (default: false)")
    print("  - SECRET_KEY: Session secret key (auto-generated if not set)")
    print("")
    print("Press Ctrl+C to stop the server")
    print("=" * 60)
    app.run(host=HOST, port=PORT, debug=DEBUG, threaded=True)
