#!/usr/bin/env python3
"""
Simple LLM HTTP server using FastAPI.
- Loads a local GGUF model once on startup (configurable via MODEL_PATH env var)
- POST /ask accepts JSON { prompt, system_prompt?, generation_params?, conversation? }
- Supports streaming SSE responses and a small test UI with conversation support
"""

import os
import logging
import threading
from pathlib import Path
from typing import Optional, Dict, Any, List
import asyncio
import re
import time
import json

from contextlib import asynccontextmanager
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

try:
    from llama_cpp import Llama
except Exception:
    Llama = None

# GPU detection and safe Llama factory
import inspect
import subprocess

def _detect_cuda_hardware() -> bool:
    # Check torch if available
    try:
        import torch
        if torch.cuda.is_available():
            return True
    except Exception:
        pass
    # Check nvidia-smi
    try:
        p = subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True, timeout=2)
        if p.returncode == 0 and p.stdout.strip():
            return True
    except Exception:
        pass
    # Check device file or env
    if os.path.exists('/dev/nvidia0'):
        return True
    cvd = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if cvd and cvd != '-1':
        return True
    return False

def detect_cuda_available() -> bool:
    # Allow forcing via env var LLM_USE_CUDA: '1'/'true' to enable or '0'/'false' to disable
    env = os.environ.get('LLM_USE_CUDA', '').strip().lower()
    if env in ('0', 'false', 'no'):
        return False
    if env in ('1', 'true', 'yes'):
        return _detect_cuda_hardware()
    # Auto-detect
    return _detect_cuda_hardware()

def _smi_memory_used():
    """Return list of memory.used values (MB) for each GPU, or None if not available."""
    try:
        p = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=3)
        if p.returncode == 0 and p.stdout.strip():
            return [int(x.strip()) for x in p.stdout.strip().splitlines()]
    except Exception:
        pass
    return None


def create_llama_instance(model_path: str, verify_gpu: bool = False):
    """Create a Llama instance, attempting to use GPU if available and supported, falling back to CPU.

    Tries multiple possible GPU parameter names (`gpu_layers`, `n_gpu_layers`) to support different
    versions of `llama-cpp-python`/bindings. The number of GPU layers can be set via
    `LLM_GPU_LAYERS` or `LLM_N_GPU_LAYERS` (fallback to 8).

    If verify_gpu=True and a GPU instantiation succeeds, we perform a small generation and check
    `nvidia-smi` memory before/after to ensure the instance actually used the GPU. If verification
    fails we fall back to CPU.
    """
    base_kwargs = dict(model_path=model_path,
                       n_ctx=int(os.environ.get('LLM_N_CTX', '8192')),
                       n_threads=int(os.environ.get('LLM_N_THREADS', '4')),
                       verbose=False)
    # Try GPU if flagged on app.state (set during startup)
    use_gpu = getattr(app.state, 'llm_use_gpu', False)
    # If we've explicitly verified that GPU does NOT work, avoid attempting GPU unless verify_gpu=True
    if use_gpu and (getattr(app.state, 'llm_gpu_verified', None) is False) and not verify_gpu:
        use_gpu = False
    if use_gpu:
        # Determine desired gpu_layers from env or default
        try:
            gpu_layers = int(os.environ.get('LLM_GPU_LAYERS', os.environ.get('LLM_N_GPU_LAYERS', '8')))
        except Exception:
            gpu_layers = 8

        tried = []
        # Try parameter names in order of preference
        for param_name in ('gpu_layers', 'n_gpu_layers'):
            gpu_kwargs = base_kwargs.copy()
            gpu_kwargs[param_name] = gpu_layers
            tried.append(param_name)
            try:
                inst = Llama(**gpu_kwargs)

                # If requested, verify that the instance actually uses the GPU by measuring
                # nvidia-smi memory before/after a very small generation.
                verified = False
                if verify_gpu:
                    sm_before = _smi_memory_used()
                    try:
                        # perform a tiny generation (1 token) to trigger GPU allocation
                        _ = inst.create_chat_completion(messages=[{'role':'user', 'content': 'probe'}], max_tokens=1, temperature=0.0)
                    except Exception:
                        # generation may fail in some models; ignore and rely on smi check
                        pass
                    time.sleep(0.15)
                    sm_after = _smi_memory_used()
                    if sm_before is not None and sm_after is not None:
                        # consider GPU used if any GPU memory increased by >= 10 MB
                        deltas = [a - b for a, b in zip(sm_after, sm_before)]
                        if any(d >= 10 for d in deltas):
                            verified = True
                    else:
                        # Could not read nvidia-smi; fall back to attribute inspection
                        for name in dir(inst):
                            if any(k in name.lower() for k in ('gpu', 'cuda', 'device')):
                                verified = True
                                break

                if verify_gpu and not verified:
                    # Close and drop this instance and try next option or CPU fallback
                    try:
                        if hasattr(inst, 'close'):
                            inst.close()
                    except Exception:
                        pass
                    logging.warning('GPU instantiation for %s=%s did not exhibit GPU usage; falling back', param_name, gpu_layers)
                    continue

                # Record the successful GPU config on app.state for UI/status visibility
                try:
                    app.state.llm_last_gpu_param = param_name
                    app.state.llm_last_gpu_layers = gpu_layers
                    app.state.llm_has_gpu = True
                    app.state.llm_gpu_verified = verified
                except Exception:
                    pass
                logging.info('Instantiated Llama with GPU param %s=%s (verified=%s)', param_name, gpu_layers, bool(verified))
                return inst
            except TypeError:
                # Parameter not accepted by this Llama build; try next
                continue
            except Exception as e:
                # Instantiation failed with this param; log and try next
                logging.warning('Llama init with %s=%s failed: %s', param_name, gpu_layers, str(e))
                continue
        # Mark that we attempted GPU params but none succeeded
        try:
            app.state.llm_last_gpu_param = None
            app.state.llm_last_gpu_layers = None
            app.state.llm_has_gpu = False
            app.state.llm_gpu_verified = False
        except Exception:
            pass
        logging.warning('Tried GPU params %s but could not instantiate GPU-enabled Llama. Falling back to CPU.', tried)
    # Fallback to CPU-only instantiation
    try:
        app.state.llm_last_gpu_param = None
        app.state.llm_last_gpu_layers = None
        app.state.llm_has_gpu = False
        app.state.llm_gpu_verified = False
    except Exception:
        pass
    return Llama(**base_kwargs)

try:
    import google as genai
except Exception as e:
    logging.error(f"Failed to import google genai: {e}")
    genai = None

APP_DIR = Path(__file__).resolve().parent
PID_FILE = APP_DIR / '.llm_server_pid'

# Explicitly load .env from the app directory
load_dotenv(APP_DIR / '.env')

#DEFAULT_MODEL = os.environ.get('MODEL_PATH', '/home/asher/.lmstudio/models/lmstudio-community/gemma-3-1b-it-GGUF/gemma-3-1b-it-Q4_K_M.gguf')
DEFAULT_MODEL = os.environ.get('MODEL_PATH', '/home/asher/.lmstudio/models/lmstudio-community/gemma-3n-E4B-it-text-GGUF/gemma-3n-E4B-it-Q4_K_M.gguf')
SHUTDOWN_TOKEN = os.environ.get('LLM_SHUTDOWN_TOKEN')
HOST = os.environ.get('LLM_HOST', '0.0.0.0')
PORT = int(os.environ.get('LLM_PORT', '5005'))
GEMINI_MODEL_NAME = os.environ.get('GEMINI_MODEL_NAME', 'gemini-2.5-flash-lite')

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize local llama-cpp if possible. Use a pool of instances to allow concurrent requests.
    if Llama is None:
        logging.warning('llama_cpp not available; local generation will not be possible.')
        app.state.llm = None
        app.state.llm_model_path = None
    else:
        model_path = os.environ.get('MODEL_PATH', DEFAULT_MODEL)
        if not Path(model_path).exists():
            logging.error('Model path not found: %s', model_path)
            app.state.llm = None
            app.state.llm_model_path = None
        else:
            # Store model path and initialize a simple pool manager
            app.state.llm_model_path = model_path
            # Detect CUDA GPU availability for llama-cpp usage
            app.state.llm_use_gpu = detect_cuda_available()
            logging.info('CUDA available for local LLM: %s (LLM_USE_CUDA env overrides detection)', app.state.llm_use_gpu)
            # Pool configuration
            app.state.llm_pool = asyncio.Queue()
            app.state.llm_pool_max = int(os.environ.get('LLM_POOL_MAX', '2'))
            app.state.llm_pool_count = 0
            preload = int(os.environ.get('LLM_POOL_PRELOAD', '1'))
            # Optionally preload a single instance (default: yes)
            if preload and app.state.llm_pool_max > 0:
                try:
                    inst = create_llama_instance(model_path, verify_gpu=True)
                    app.state.llm_pool.put_nowait(inst)
                    app.state.llm_pool_count = 1
                    # Keep this instance in app.state.llm for backwards compatibility
                    app.state.llm = inst
                    logging.info('Preloaded 1 Llama model from %s (pool max=%s, gpu=%s, verified=%s)', model_path, app.state.llm_pool_max, app.state.llm_use_gpu, getattr(app.state, 'llm_gpu_verified', False))
                except Exception as e:
                    logging.exception('Failed to preload Llama model: %s', e)
                    app.state.llm_model_path = None
                    app.state.llm = None
            else:
                app.state.llm = None

    # Initialize Gemini if API key is available
    gemini_key = os.environ.get('GEMINI_API_KEY')
    logging.info('Checking Gemini initialization... Key present: %s, genai module present: %s',
                 bool(gemini_key), bool(genai))
    
    if genai and gemini_key:
        try:
            genai.configure(api_key=gemini_key.strip())
            # Default to gemini-2.5-flash-lite
            app.state.gemini_model_name = GEMINI_MODEL_NAME
            app.state.gemini_model = genai.GenerativeModel(app.state.gemini_model_name)
            logging.info('Gemini API initialized successfully with %s', app.state.gemini_model_name)
        except Exception as e:
            logging.exception('Failed to initialize Gemini: %s', e)
            app.state.gemini_model = None
    else:
        app.state.gemini_model = None
        if not gemini_key:
            logging.warning('Gemini API key NOT found in environment. Check your .env file at %s', APP_DIR / '.env')
        if not genai:
            logging.error('google-generativeai package is not installed or failed to import.')

    try:
        app.state.llm_lock = asyncio.Lock()
    except Exception:
        app.state.llm_lock = None

    try:
        PID_FILE.write_text(str(os.getpid()))
    except Exception:
        pass
    yield
    try:
        if PID_FILE.exists():
            PID_FILE.unlink()
    except Exception:
        pass

app = FastAPI(lifespan=lifespan)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AskRequest(BaseModel):
    prompt: str
    system_prompt: Optional[str] = None
    generation_params: Optional[Dict[str, Any]] = None
    conversation: Optional[List[Dict[str, str]]] = None
    # If provider is omitted or empty, server will choose a sensible default (local if loaded, else configured Gemini model)
    provider: Optional[str] = None # 'local', 'gemini-3-flash-preview', 'gemini-2.5-flash', 'gemini-2.5-flash-lite', 'gemma-3-27b'
    # Device preference: 'auto' (default), 'gpu' (GPU only), 'cpu' (CPU only)
    device_preference: Optional[str] = None

class AskResponse(BaseModel):
    response: str
    provider: str


def format_sse(data: str) -> str:
    if data is None:
        return 'data: \n\n'
    parts = data.split('\n')
    out_lines = [f"data: {p}" for p in parts]
    return '\n'.join(out_lines) + '\n\n'





# --- LLM pool helpers ---
async def acquire_llm_instance(force_cpu: bool = False):
    """Acquire a Llama instance from the pool or create one (async).

    If force_cpu=True, will create a CPU-only instance and not use pooled GPU instances.
    """
    if getattr(app.state, 'llm_model_path', None) is None:
        raise RuntimeError('No local model configured')
    pool = getattr(app.state, 'llm_pool', None)
    if pool is None:
        # No pool configured, create a temporary instance
        inst = await asyncio.to_thread(create_llama_instance, app.state.llm_model_path, verify_gpu=False, force_cpu=force_cpu)
        return inst
    try:
        inst = pool.get_nowait()
        return inst
    except asyncio.QueueEmpty:
        # create new instance if pool has capacity
        if getattr(app.state, 'llm_pool_count', 0) < getattr(app.state, 'llm_pool_max', 1):
            inst = await asyncio.to_thread(create_llama_instance, app.state.llm_model_path, verify_gpu=False, force_cpu=force_cpu)
            app.state.llm_pool_count = getattr(app.state, 'llm_pool_count', 0) + 1
            return inst
        # otherwise wait for one to be released
        inst = await pool.get()
        return inst


def release_llm_instance(inst):
    """Return instance to pool or drop it if pool not configured."""
    pool = getattr(app.state, 'llm_pool', None)
    if pool is None:
        # let GC handle it
        return
    try:
        pool.put_nowait(inst)
    except Exception:
        # If queue full or closed, drop
        pass





@app.get('/', response_class=HTMLResponse)
def index():
    html = """
    <!doctype html>
    <html>
        <head>
        <meta charset="utf-8"><title>LLM Server</title>
        <style>
            .dot {
                height: 12px;
                width: 12px;
                border-radius: 50%;
                display: inline-block;
                margin-right: 8px;
            }
            .dot-green {
                background-color: #28a745;
                animation: blink-green 1s infinite;
            }
            .dot-red {
                background-color: #dc3545;
            }
            @keyframes blink-green {
                0% { opacity: 1; }
                50% { opacity: 0.5; }
                100% { opacity: 1; }
            }
        </style>
    </head>
    <body style="font-family: Arial, Helvetica, sans-serif; margin:20px;">
        <h2>LLM Server</h2>
            <div id="status-container" style="margin-bottom:8px; display: flex; align-items: center;">
                <span id="status-dot" class="dot"></span>
                <span id="status-text"></span>
            </div>
        <form id="frm">
            <label>Provider</label><br>
            <select id="provider" style="width:100%; margin-bottom:10px; padding:4px;">
                <option value="gemini-3-flash-preview">Google Gemini 3 Flash Preview</option>
                <option value="gemini-2.5-flash">Google Gemini 2.5 Flash</option>
                <option value="gemini-2.5-flash-lite" selected>Google Gemini 2.5 Flash Lite (Unlimited!)</option>
                <option value="gemma-3-27b">Gemma 3-27B (Google)</option>
                <option value="local">Local Model (Llama/Gemma)</option> 
            </select><br>
            <label>Device</label><br>
            <select id="device" style="width:100%; margin-bottom:10px; padding:4px;">
                <option value="auto" selected>Auto</option>
                <option value="gpu">GPU only</option>
                <option value="cpu">CPU only</option>
            </select><br>
            <label>System prompt (optional)</label><br>
            <input id="system" style="width:100%" placeholder="System prompt"><br><br>
            <label>Prompt</label><br>
            <textarea id="prompt" rows="4" style="width:100%" placeholder="Enter your prompt"></textarea><br>
            <button type="submit">Ask</button>
        </form>
        <h3>Response (<span id="active-provider">None</span>)</h3>
        <pre id="resp" style="white-space:pre-wrap; background:#f6f6f6; padding:12px; border-radius:6px; max-width:800px;"></pre>
        <div id="stats" style="margin-top:8px; color:#555; font-size:0.9em;"></div>

        <hr>
        <h3>Conversation</h3>
        <div id="conversation" style="white-space:pre-wrap; background:#eef6ff; padding:12px; border-radius:6px; max-width:800px; min-height:80px;"></div>
        <button id="clear-conv" style="margin-top:8px;">Clear Conversation</button>

        <script>
            function escapeHtml(s) { return (s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }
            function appendChunk(el, chunk) {
              if (chunk) {
                el.textContent += chunk;
              }
            }

            let conversation = [];
            const convEl = document.getElementById('conversation');

            function renderConversation() {
                convEl.innerHTML = '';
                for (const msg of conversation) {
                    const wrapper = document.createElement('div');
                    wrapper.style.padding = '6px 8px';
                    wrapper.style.marginBottom = '6px';
                    wrapper.style.borderRadius = '6px';
                    if (msg.role === 'user') {
                        wrapper.style.background = '#fff';
                        wrapper.style.border = '1px solid #ddd';
                        wrapper.innerHTML = '<strong>You:</strong> ' + escapeHtml(msg.content);
                    } else {
                        wrapper.style.background = '#f6f9ff';
                        wrapper.style.border = '1px solid #cfe0ff';
                        wrapper.innerHTML = '<strong>Assistant:</strong> ' + escapeHtml(msg.content);
                    }
                    convEl.appendChild(wrapper);
                }
                convEl.scrollTop = convEl.scrollHeight;
            }

            document.getElementById('clear-conv').addEventListener('click', () => {
                conversation = [];
                renderConversation();
            });

            document.getElementById('provider').addEventListener('change', updateStatus);

            document.getElementById('frm').addEventListener('submit', async function(e) {
                e.preventDefault();
                const promptEl = document.getElementById('prompt');
                const prompt = promptEl.value;
                const system = document.getElementById('system').value || undefined;
                const provider = document.getElementById('provider').value;
                const device = document.getElementById('device').value;
                if (!prompt || !prompt.trim()) return;

                const payload = {
                    prompt: prompt,
                    system_prompt: system,
                    provider: provider,
                    device_preference: device,
                    conversation: conversation.slice() // Send history, server adds current prompt
                };

                const respEl = document.getElementById('resp');
                const providerEl = document.getElementById('active-provider');
                const statsEl = document.getElementById('stats');
                respEl.textContent = '';
                providerEl.textContent = '...';
                statsEl.textContent = '';

                // Optimistically update UI
                conversation.push({ role: 'user', content: prompt });
                renderConversation();

                try {
                    const r = await fetch('/ask?stream=1', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json', 'Accept': 'text/event-stream' },
                        body: JSON.stringify(payload)
                    });

                    if (!r.ok) {
                        const txt = await r.text();
                        respEl.textContent = 'Error: ' + r.status + '\\n' + txt;
                        conversation.pop(); // remove optimistic user message
                        renderConversation();
                        return;
                    }

                    const ct = (r.headers.get('content-type') || '').toLowerCase();
                    if (ct.includes('application/json')) {
                        const j = await r.json();
                        respEl.textContent = j.response || '';
                        providerEl.textContent = j.provider || 'unknown';
                        if (j.tokens) {
                            statsEl.textContent = `Tokens: ${j.tokens}, ${(j.tok_per_sec||0).toFixed(1)} tok/s, Time: ${(j.seconds||0).toFixed(2)}s`;
                        } else {
                            statsEl.textContent = '';
                        }
                        conversation.push({ role: 'assistant', content: j.response || '' });
                        renderConversation();
                        return;
                    }

                    const reader = r.body.getReader();
                    const dec = new TextDecoder();
                    let buf = '';
                    let finalText = '';

                    while (true) {
                        const { done, value } = await reader.read();
                        if (done) break;
                        buf += dec.decode(value, { stream: true });
                        const parts = buf.split('\\n\\n');
                        buf = parts.pop();
                        for (const part of parts) {
                            if (part.startsWith('event: provider')) {
                                providerEl.textContent = part.split('data: ')[1] || 'unknown';
                                continue;
                            }
                            if (part.startsWith('event: stats')) {
                                const data = part.split('data: ')[1] || '';
                                try {
                                    const s = JSON.parse(data);
                                    statsEl.textContent = `Tokens: ${s.tokens}, ${s.tok_per_sec.toFixed(1)} tok/s, Time: ${s.seconds.toFixed(2)}s`;
                                } catch(e) {}
                                continue;
                            }
                            const lines = part.split('\\n');
                            const dataLines = lines.filter(l => l.startsWith('data:'));
                            if (dataLines.length) {
                                const data = dataLines.map(l => l.slice(6)).join('\\n');
                                if (data !== '[DONE]') {
                                    appendChunk(respEl, data);
                                    finalText += data;
                                }
                            } else {
                                appendChunk(respEl, part);
                                finalText += part;
                            }
                            respEl.scrollTop = respEl.scrollHeight;
                        }
                    }

                    if (buf && buf.trim()) {
                        appendChunk(respEl, buf);
                        finalText += buf;
                    }

                    if (finalText && finalText.trim()) {
                        conversation.push({ role: 'assistant', content: finalText });
                        renderConversation();
                    }
                } catch (err) {
                    respEl.textContent = 'Request failed: ' + String(err);
                    statsEl.textContent = '';
                    conversation.pop(); // remove optimistic user message
                    renderConversation();
                }
            });
            function updateStatus() {
                const providerSelect = document.getElementById('provider');
                const selectedProvider = providerSelect.options[providerSelect.selectedIndex].text;
                const deviceSelect = document.getElementById('device');
                const selectedDevice = deviceSelect ? deviceSelect.options[deviceSelect.selectedIndex].text : '';
                const statusDot = document.getElementById('status-dot');
                const statusText = document.getElementById('status-text');

                if (providerSelect.value) {
                    statusDot.className = 'dot dot-green';
                    statusText.innerHTML = `Connected: <strong>${selectedProvider}</strong> ${selectedDevice ? ' — ' + selectedDevice : ''}`;
                } else {
                    statusDot.className = 'dot dot-red';
                    statusText.innerHTML = 'Not connected';
                }
            }

            window.addEventListener('load', updateStatus);
        </script>
    </body>
    </html>
    """
    # Compute LLM status for display
    llm = getattr(app.state, 'llm', None)
    gemini = getattr(app.state, 'gemini_model', None)
    if llm:
        provider = 'local'
    elif gemini:
        provider = getattr(app.state, 'gemini_model_name', GEMINI_MODEL_NAME)
    else:
        provider = ''
    llm_url = os.environ.get('LLM_SERVER_URL', 'http://ashy.tplinkdns.com:5005/ask')

    # Determine GPU display info
    gpu_enabled = bool(getattr(app.state, 'llm_use_gpu', False))
    gpu_param = getattr(app.state, 'llm_last_gpu_param', None)
    gpu_layers = getattr(app.state, 'llm_last_gpu_layers', None)

    if provider:
        if provider == 'local':
            # Show a concise label: either GPU (if GPU is actually in use) or CPU ONLY
            if getattr(app.state, 'llm_has_gpu', False):
                provider_label = 'GPU'
                # show details in a smaller subtle span, and indicate verification status
                details = ''
                try:
                    p = getattr(app.state, 'llm_last_gpu_param', None)
                    l = getattr(app.state, 'llm_last_gpu_layers', None)
                    v = getattr(app.state, 'llm_gpu_verified', False)
                    if p and l:
                        details = f' <small>({p}={l}{" verified" if v else ""})</small>'
                    else:
                        if v:
                            details = ' <small>(verified)</small>'
                except Exception:
                    details = ''
                provider_label_html = f'{provider_label}{details}'
            else:
                provider_label_html = 'CPU ONLY'
        else:
            provider_label_html = provider
        status_html = f'''
        <div style="margin-bottom:8px; display: flex; align-items: center;">
            <span class="dot dot-green"></span>
            <span>Connected: <strong>{provider_label_html}</strong></span>
        </div>'''
    else:
        status_html = f'''
        <div style="margin-bottom:8px; color:#b00; display: flex; align-items: center;">
            <span class="dot dot-red"></span>
            <span>Not connected</span>
        </div>'''
    # Diagnostics display
    diag = getattr(app.state, 'llm_diag', None)
    last_diag_text = 'No tests run yet'
    if diag:
        try:
            ts = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(diag.get('time', time.time())))
            last_diag_text = f"{'OK' if diag.get('ok') else 'FAIL'} at {ts}: {diag.get('msg','')}. Took {diag.get('seconds',0):.2f}s"
        except Exception:
            last_diag_text = str(diag)

    pool_q = getattr(app.state, 'llm_pool', None)
    pool_idle = pool_q.qsize() if pool_q is not None else (1 if getattr(app.state, 'llm', None) else 0)
    pool_count = getattr(app.state, 'llm_pool_count', 0)

    # Connection label reflects whether GPU is verified
    conn_label = 'CPU ONLY'
    try:
        if getattr(app.state, 'llm_has_gpu', False):
            conn_label = 'GPU (verified)' if getattr(app.state, 'llm_gpu_verified', False) else 'GPU (not verified)'
    except Exception:
        pass

    diag_html = f'''
        <div style="margin-top:8px; padding:8px; border-radius:6px; background:#f8f9fa; border:1px solid #ddd; max-width:800px;">
            <strong>Diagnostics</strong>
            <div style="margin-top:6px; font-size:0.95em; color:#333;">
                <div id="diag-connection">Connection: <strong>{conn_label}</strong></div>
                <div id="diag-gpu">GPU param: <strong>{getattr(app.state,'llm_last_gpu_param',None)}</strong> layers: <strong>{getattr(app.state,'llm_last_gpu_layers',None)}</strong></div>
                <div id="diag-pool">Pool: {pool_count} total, {pool_idle} idle</div>
                <div id="diag-last-test">Last test: <span id="diag-last-test-text">{last_diag_text}</span></div>
                <div style="margin-top:6px;">
                    <button id="run-diag">Run GPU test</button>
                    <button id="run-autotune" style="margin-left:8px;">Auto-tune GPU</button>
                    <button id="run-probe" style="margin-left:8px;">Run GPU probe</button>
                    <small style="color:#666; margin-left:8px;">(Auto-tune finds the largest safe GPU layer count; probe checks instance attributes and a short timing)</small>
                </div>
                <div style="margin-top:8px; font-family: monospace; font-size:0.92em; color:#222;"><pre id="autotune-log" style="white-space:pre-wrap; background:#fff; border:1px solid #ddd; padding:8px; max-height:200px; overflow:auto;">Autotune log...</pre></div>
                <div style="margin-top:8px;"><pre id="probe-log" style="white-space:pre-wrap; background:#fff; border:1px solid #ddd; padding:8px; max-height:200px; overflow:auto;">Probe results will appear here.</pre></div>
            </div>
        </div>
    '''

    html = html.replace('<h2>LLM Server</h2>', f'<h2>LLM Server</h2>\n        {status_html}\n        {diag_html}')

    # Inject a small script to handle the Run GPU test button
    html = html.replace('</body>', '''
        <script>
        document.getElementById('run-diag').addEventListener('click', async function () {
            const btn = this;
            const textEl = document.getElementById('diag-last-test-text');
            btn.disabled = true; textEl.textContent = 'Running test...';
            try {
                const r = await fetch('/llm_diag', {method: 'POST'});
                const j = await r.json();
                if (r.ok) {
                    textEl.textContent = `Result: ${j.ok ? 'OK' : 'FAIL'} ${j.param ? j.param+ '=' + j.layers : ''} Time: ${j.seconds ? j.seconds.toFixed(2)+'s' : ''} ${j.msg || ''}`;
                    // Update connection label if success
                    if (j.ok) {
                        const conn = document.getElementById('diag-connection');
                        conn.innerHTML = 'Connection: <strong>GPU</strong>';
                    }
                } else {
                    textEl.textContent = 'Error: ' + r.status + ' ' + JSON.stringify(j);
                }
            } catch (e) {
                textEl.textContent = 'Request failed: ' + String(e);
            } finally {
                btn.disabled = false;
            }
        });

        // Auto-tune button - uses EventSource to stream progress
        document.getElementById('run-autotune').addEventListener('click', function () {
            const btn = this;
            const log = document.getElementById('autotune-log');
            log.textContent = 'Starting autotune...\n';
            btn.disabled = true;
            // Open EventSource (GET streaming endpoint)
            const es = new EventSource('/llm_autotune?stream=1&mode=fast'); // fast mode by default (coarse doubling only)
            es.addEventListener('progress', function(evt) {
                try {
                    const d = JSON.parse(evt.data);
                    log.textContent += `[${new Date().toLocaleTimeString()}] TRY param=${d.param} layers=${d.layers} ok=${d.ok} t=${d.seconds ? d.seconds.toFixed(2)+'s' : ''} ${d.msg||''}\n`;
                    log.scrollTop = log.scrollHeight;
                } catch(e) {
                    log.textContent += 'Malformed progress: ' + evt.data + '\n';
                }
            });
            es.addEventListener('result', function(evt) {
                try {
                    const d = JSON.parse(evt.data);
                    log.textContent += `RESULT: recommended layers=${d.recommended} param=${d.param} took=${d.total_seconds ? d.total_seconds.toFixed(2)+'s' : ''}\n`;
                    // If success, update UI labels
                    if (d.recommended) {
                        const conn = document.getElementById('diag-connection');
                        conn.innerHTML = 'Connection: <strong>GPU</strong>';
                        const gparam = document.getElementById('diag-gpu');
                        gparam.innerHTML = `GPU param: <strong>${d.param}</strong> layers: <strong>${d.recommended}</strong>`;
                        const textEl = document.getElementById('diag-last-test-text');
                        textEl.textContent = `Auto-tune result: ${d.recommended} layers (param ${d.param})`;
                    }
                } catch (e) {
                    log.textContent += 'Malformed result: ' + evt.data + '\n';
                }
                es.close();
                btn.disabled = false;
            });
            es.addEventListener('error', function(evt) {
                log.textContent += 'Autotune failed or closed.\n';
                es.close();
                btn.disabled = false;
            });

        // Probe button
        document.getElementById('run-probe').addEventListener('click', async function () {
            const btn = this;
            const plog = document.getElementById('probe-log');
            plog.textContent = 'Running probe...';
            btn.disabled = true;
            try {
                const r = await fetch('/llm_probe');
                const j = await r.json();
                if (r.ok) {
                    plog.textContent = JSON.stringify(j, null, 2);
                } else {
                    plog.textContent = 'Error: ' + r.status + ' ' + JSON.stringify(j);
                }
            } catch (e) {
                plog.textContent = 'Request failed: ' + String(e);
            } finally {
                btn.disabled = false;
            }
        });
        });
        </script>
        </body>''')

    return HTMLResponse(content=html)


@app.post('/ask')
async def ask(request: Request, req: AskRequest):
    # Normalize provider selection:
    # - If client omits provider or sends '', choose local if loaded, else configured Gemini model
    # - Accept shorthand 'gemini' and map to configured Gemini model name
    # - Treat 'gemma-*' and 'local' as local model requests
    req_prov = (req.provider or '').strip()
    gemini_default = getattr(app.state, 'gemini_model', None)
    # Whether a local model *could* be used (path configured and llama_cpp present)
    llm_available = (getattr(app.state, 'llm_model_path', None) is not None) and (Llama is not None)
    # Backwards compat: provide a convenience reference to any preloaded instance
    llm = getattr(app.state, 'llm', None) if getattr(app.state, 'llm', None) else None

    if not req_prov:
        # Auto-select default: default to configured Gemini model name (gemini-2.5-flash-lite)
        provider = getattr(app.state, 'gemini_model_name', None) or GEMINI_MODEL_NAME
    else:
        provider = req_prov

    original_provider = provider

    # Map gemma-* and explicit 'local' to local logic
    if provider.startswith('gemma-') or provider == 'local':
        provider_logic = 'local'
    else:
        provider_logic = provider

    # Determine whether we're targeting Gemini (provider starts with 'gemini')
    is_gemini = isinstance(provider_logic, str) and provider_logic.startswith('gemini')

    # Validation
    if provider_logic == 'local' and not llm_available:
        if gemini_default:
            logging.info('Local LLM not available, falling back to Gemini')
            provider = 'gemini-2.5-flash-lite'
            is_gemini = True
        else:
            raise HTTPException(status_code=500, detail='Local LLM not loaded and no Gemini fallback available')
    
    if is_gemini and not gemini_default:
        if llm:
            logging.warning('Gemini not available, falling back to local')
            provider = 'local'
            is_gemini = False
        else:
            raise HTTPException(status_code=500, detail='Gemini API not initialized (no key) and no local LLM available')

    client_ip = (request.headers.get('X-Forwarded-For') or (request.client.host if request.client else 'unknown')).split(',')[0].strip()
    logging.info('Received /ask (%s) from %s', provider, client_ip)

    system_prompt = req.system_prompt or 'You are a helpful assistant. Keep answers concise.'
    
    # Prepare messages
    messages = []
    if system_prompt:
        messages.append({'role':'system', 'content': system_prompt})
    
    last_role = 'system'
    if req.conversation:
        for m in req.conversation:
            if m.get('role') in ('user','assistant') and 'content' in m:
                if m['role'] == last_role: continue
                messages.append(m)
                last_role = m['role']
    
    if last_role == 'user':
        if messages and messages[-1]['role'] == 'user':
            messages[-1] = {'role':'user', 'content': req.prompt}
        else:
            messages.append({'role':'user', 'content': req.prompt})
    else:
        messages.append({'role':'user', 'content': req.prompt})

    accept = request.headers.get('accept','') or ''
    stream_requested = 'text/event-stream' in accept or request.query_params.get('stream') in ('1','true')

    # --- GEMINI HANDLER ---
    if is_gemini:
        # Map provider string to actual model name
        # If provider begins with 'gemini', use it directly to support variants like 'gemini-2.5-flash-lite'
        # Map 'gemini' shorthand to configured Gemini model name; allow full model names like 'gemini-2.5-flash-lite'
        if provider.startswith('gemini'):
            if provider == 'gemini':
                model_name = getattr(app.state, 'gemini_model_name', None) or GEMINI_MODEL_NAME
            else:
                model_name = provider
        else:
            # Fall back to server configuration or the GEMINI_MODEL_NAME env var
            model_name = getattr(app.state, 'gemini_model_name', None) or GEMINI_MODEL_NAME

        # Convert history for Gemini
        gemini_history = []
        for m in messages:
            if m['role'] != 'system':
                role = 'user' if m['role'] == 'user' else 'model'
                gemini_history.append({'role': role, 'parts': [m['content']]})
        
        user_msg = gemini_history.pop() if gemini_history and gemini_history[-1]['role'] == 'user' else {'role':'user', 'parts':[req.prompt]}
        
        try:
            chat_model = genai.GenerativeModel(
                model_name=model_name,
                system_instruction=system_prompt
            )
            chat = chat_model.start_chat(history=gemini_history)
        except Exception as e:
            logging.error(f"Error initializing Gemini chat with model {model_name}: {e}")
            # Fallback to default if specific failed? Or just error?
            # Let's try to just raise for now or fallback if we were smart, but keep it simple.
            raise HTTPException(status_code=500, detail=f"Gemini init error: {e}")

        if stream_requested:
            async def gemini_stream():
                try:
                    yield f"event: provider\ndata: {provider}\n\n"
                    start = time.time()
                    final_text = ''
                    response = await asyncio.to_thread(chat.send_message, user_msg['parts'][0], stream=True)
                    for chunk in response:
                        try:
                            if chunk.text:
                                final_text += chunk.text
                                yield format_sse(chunk.text)
                        except (ValueError, IndexError, AttributeError):
                            pass
                        await asyncio.sleep(0)
                    elapsed = time.time() - start
                    tokens_est = max(1, int(len(final_text) / 4))
                    tok_per_sec = (tokens_est / elapsed) if elapsed > 0 else tokens_est
                    yield f"event: stats\ndata: {json.dumps({'tokens': tokens_est, 'seconds': elapsed, 'tok_per_sec': tok_per_sec})}\n\n"
                    yield 'event: done\ndata: [DONE]\n\n'
                except Exception as e:
                    logging.exception('Gemini streaming error: %s', e)
                    yield f"event: error\ndata: {str(e)}\n\n"
            return StreamingResponse(gemini_stream(), media_type='text/event-stream')
        
        try:
            start = time.time()
            response = await asyncio.to_thread(chat.send_message, user_msg['parts'][0])
            elapsed = time.time() - start
            try:
                text = response.text
            except (ValueError, IndexError, AttributeError) as ve:
                reason = "unknown"
                try:
                    if response.candidates:
                        reason = str(response.candidates[0].finish_reason)
                except: pass
                text = f"[Gemini Blocked/Empty] Finish reason: {reason}. Detail: {str(ve)}"
            tokens_est = max(1, int(len(text) / 4))
            tok_per_sec = (tokens_est / elapsed) if elapsed > 0 else tokens_est
            return {'response': text, 'provider': provider, 'tokens': tokens_est, 'seconds': elapsed, 'tok_per_sec': tok_per_sec}
        except Exception as e:
            logging.exception('Gemini error: %s', e)
            raise HTTPException(status_code=500, detail=str(e))

    # --- LOCAL HANDLER ---
    if provider_logic == 'local':
        gen = {
            'temperature': float(os.environ.get('LLM_TEMPERATURE', '0.2')),
            'top_k': int(os.environ.get('LLM_TOP_K', '40')),
            'top_p': float(os.environ.get('LLM_TOP_P', '0.95')),
            'repeat_penalty': float(os.environ.get('LLM_REPEAT_PENALTY', '1.1')),
            'max_tokens': int(os.environ.get('LLM_MAX_TOKENS', '512')),
            'stop': ["<|eot_id|>"]
        }
        if req.generation_params:
            gen.update(req.generation_params)

        # Respect device preference per-request: 'auto' (default), 'gpu', or 'cpu'
        device_pref = (req.device_preference or 'auto').strip().lower()
        tmp_llm_inst = None
        tmp_created = False
        if device_pref == 'gpu':
            if not llm_available:
                raise HTTPException(status_code=500, detail='Local LLM not loaded')
            # If server already has a verified GPU, proceed; otherwise attempt a one-off verified GPU instantiation
            if not (getattr(app.state, 'llm_has_gpu', False) and getattr(app.state, 'llm_gpu_verified', False)):
                try:
                    tmp_llm_inst = await asyncio.to_thread(create_llama_instance, app.state.llm_model_path, verify_gpu=True)
                    tmp_created = True
                except Exception as e:
                    logging.warning('Per-request GPU verification failed: %s', str(e))
                    raise HTTPException(status_code=503, detail='GPU not available or could not be verified')
        elif device_pref == 'cpu':
            try:
                tmp_llm_inst = await asyncio.to_thread(create_llama_instance, app.state.llm_model_path, force_cpu=True)
                tmp_created = True
            except Exception as e:
                logging.warning('Per-request CPU instance creation failed: %s', str(e))
                raise HTTPException(status_code=500, detail='Failed to create CPU instance')

        if stream_requested:
            async def event_stream():
                # Determine display provider for UI (GPU vs CPU ONLY) using request preference and tmp instance
                if tmp_llm_inst is not None:
                    display_provider = 'GPU' if device_pref == 'gpu' else 'CPU ONLY'
                else:
                    display_provider = 'GPU' if getattr(app.state, 'llm_has_gpu', False) and getattr(app.state, 'llm_gpu_verified', False) and device_pref != 'cpu' else 'CPU ONLY'
                yield f"event: provider\ndata: {display_provider}\n\n"
                # Acquire a model instance for the duration of the stream
                try:
                    if tmp_llm_inst is not None:
                        llm_inst = tmp_llm_inst
                    else:
                        llm_inst = await acquire_llm_instance(force_cpu=(device_pref == 'cpu'))
                except Exception as e:
                    logging.exception('Failed to acquire LLM instance for streaming: %s', e)
                    yield f"event: error\ndata: {str(e)}\n\n"
                    return
                try:
                    start = time.time()
                    final_text = ''
                    for chunk in llm_inst.create_chat_completion(messages=messages, stream=True, **gen):
                        try:
                            choice = (chunk.get('choices') or [{}])[0]
                            delta = choice.get('delta') or choice.get('message') or {}
                            text_part = ''
                            if isinstance(delta, dict):
                                text_part = delta.get('content') or ''
                            else:
                                text_part = str(delta)
                            if text_part:
                                final_text += text_part
                                yield format_sse(text_part)
                                await asyncio.sleep(0)
                        except Exception:
                            continue
                    elapsed = time.time() - start
                    tokens_est = max(1, int(len(final_text) / 4))
                    tok_per_sec = (tokens_est / elapsed) if elapsed > 0 else tokens_est
                    yield f"event: stats\ndata: {json.dumps({'tokens': tokens_est, 'seconds': elapsed, 'tok_per_sec': tok_per_sec})}\n\n"
                    yield 'event: done\ndata: [DONE]\n\n'
                except Exception as e:
                    logging.exception('LLM generation error (stream): %s', e)
                    yield f"event: error\ndata: {str(e)}\n\n"
                finally:
                    try:
                        if tmp_created:
                            # close and drop the temporary instance
                            if hasattr(llm_inst, 'close'):
                                try:
                                    llm_inst.close()
                                except Exception:
                                    pass
                        else:
                            release_llm_instance(llm_inst)
                    except Exception:
                        pass
            return StreamingResponse(event_stream(), media_type='text/event-stream')

        try:
            # Acquire instance and run blocking creation in a threadpool to avoid blocking the event loop
            try:
                if tmp_llm_inst is not None:
                    llm_inst = tmp_llm_inst
                else:
                    llm_inst = await acquire_llm_instance(force_cpu=(device_pref == 'cpu'))
            except Exception as e:
                logging.exception('Failed to acquire LLM instance: %s', e)
                raise HTTPException(status_code=500, detail=str(e))
            try:
                start = time.time()
                response = await asyncio.to_thread(llm_inst.create_chat_completion, messages=messages, **gen)
                elapsed = time.time() - start
            finally:
                try:
                    if tmp_created:
                        if hasattr(llm_inst, 'close'):
                            try:
                                llm_inst.close()
                            except Exception:
                                pass
                    else:
                        release_llm_instance(llm_inst)
                except Exception:
                    pass
            content = response['choices'][0]['message'].get('content') if response and 'choices' in response else ''
            text = content.strip() if content else ''
            tokens_est = max(1, int(len(text) / 4))
            tok_per_sec = (tokens_est / elapsed) if elapsed > 0 else tokens_est
            if tmp_llm_inst is not None:
                display_provider = 'GPU' if device_pref == 'gpu' else 'CPU ONLY'
            else:
                display_provider = 'GPU' if getattr(app.state, 'llm_has_gpu', False) and getattr(app.state, 'llm_gpu_verified', False) and device_pref != 'cpu' else 'CPU ONLY'
            return {'response': text, 'provider': display_provider, 'tokens': tokens_est, 'seconds': elapsed, 'tok_per_sec': tok_per_sec}
        except Exception as e:
            logging.exception('LLM generation error: %s', e)
            raise HTTPException(status_code=500, detail=str(e))


@app.post('/shutdown')
def shutdown(request: Request):
    if SHUTDOWN_TOKEN:
        token = request.query_params.get('token') or request.headers.get('X-Shutdown-Token')
        if token != SHUTDOWN_TOKEN:
            raise HTTPException(status_code=403, detail='Invalid shutdown token')

    def _exit():
        try:
            if PID_FILE.exists():
                PID_FILE.unlink()
        except Exception:
            pass
        os._exit(0)

    threading.Timer(0.2, _exit).start()
    return {'status': 'shutting_down'}


@app.post('/llm_diag')
async def llm_diag(request: Request):
    """Run a brief diagnostic GPU instantiation test and return results.

    This attempts to instantiate a temporary Llama model with common GPU kwargs and returns
    timing/error information. Only run on-demand via the UI to avoid startup overhead.
    """
    if Llama is None:
        raise HTTPException(status_code=500, detail='llama_cpp not installed')
    model_path = getattr(app.state, 'llm_model_path', None)
    if not model_path or not Path(model_path).exists():
        raise HTTPException(status_code=500, detail='Local model not configured or not found')

    # Decide layers to try
    try:
        layers = int(os.environ.get('LLM_GPU_LAYERS', os.environ.get('LLM_N_GPU_LAYERS', '8')))
    except Exception:
        layers = 8

    params = ['gpu_layers', 'n_gpu_layers']
    tried = []
    start_total = time.time()
    for p in params:
        start = time.time()
        try:
            # Try instantiation with this param on a thread to avoid blocking event loop
            inst = await asyncio.to_thread(Llama, model_path=model_path, **{p: layers}, n_ctx=int(os.environ.get('LLM_N_CTX', '8192')), n_threads=int(os.environ.get('LLM_N_THREADS', '4')))
            elapsed = time.time() - start
            # Mark server state as GPU-capable and verified
            try:
                app.state.llm_last_gpu_param = p
                app.state.llm_last_gpu_layers = layers
                app.state.llm_has_gpu = True
                app.state.llm_gpu_verified = True
            except Exception:
                pass
            # Cleanup
            try:
                # llama-cpp python sometimes exposes a .close or similar; attempt gentle cleanup
                if hasattr(inst, 'close'):
                    try:
                        inst.close()
                    except Exception:
                        pass
            except Exception:
                pass
            try:
                del inst
            except Exception:
                pass
            import gc; gc.collect()
            result = {'ok': True, 'param': p, 'layers': layers, 'seconds': elapsed, 'msg': 'OK', 'time': time.time()}
            app.state.llm_diag = result
            return result
        except TypeError as e:
            tried.append({'param': p, 'error': str(e)})
            continue
        except Exception as e:
            tried.append({'param': p, 'error': str(e)})
            continue

    elapsed_total = time.time() - start_total
    result = {'ok': False, 'param': None, 'layers': None, 'seconds': elapsed_total, 'msg': 'All attempts failed', 'tried': tried, 'time': time.time()}
    app.state.llm_diag = result
    return result


@app.get('/llm_autotune')
async def llm_autotune(request: Request):
    """Query param: mode=fast to skip binary search (fast doubling only)."""
    """Auto-tune GPU layer count. Streams progress as SSE if `stream=1` query parameter is present.

    Algorithm: find param that works at a small layer count, then exponentially increase until failure or max,
    then binary-search the maximum working layer count.
    """
    if Llama is None:
        raise HTTPException(status_code=500, detail='llama_cpp not installed')
    model_path = getattr(app.state, 'llm_model_path', None)
    if not model_path or not Path(model_path).exists():
        raise HTTPException(status_code=500, detail='Local model not configured or not found')

    # config
    max_layers = int(os.environ.get('LLM_AUTOTUNE_MAX', os.environ.get('LLM_GPU_LAYERS_MAX', '4096')))
    start = int(os.environ.get('LLM_AUTOTUNE_START', os.environ.get('LLM_GPU_LAYERS', '8')))
    max_time = float(os.environ.get('LLM_AUTOTUNE_MAX_TIME', '30'))  # per trial seconds timeout heuristic

    async def autotune_stream():
        tried = []
        start_time = time.time()
        # First, find which param name works at `start`
        param_name = None
        for p in ('gpu_layers', 'n_gpu_layers'):
            try:
                t0 = time.time()
                inst = await asyncio.to_thread(Llama, model_path=model_path, **{p: start}, n_ctx=int(os.environ.get('LLM_N_CTX', '8192')), n_threads=int(os.environ.get('LLM_N_THREADS', '4')))
                elapsed = time.time() - t0
                # close if possible
                try:
                    if hasattr(inst, 'close'):
                        inst.close()
                except Exception:
                    pass
                del inst
                tried.append({'param': p, 'layers': start, 'ok': True, 'seconds': elapsed})
                yield f"event: progress\ndata: {json.dumps(tried[-1])}\n\n"
                param_name = p
                break
            except TypeError as e:
                tried.append({'param': p, 'layers': start, 'ok': False, 'error': str(e)})
                yield f"event: progress\ndata: {json.dumps(tried[-1])}\n\n"
                continue
            except Exception as e:
                tried.append({'param': p, 'layers': start, 'ok': False, 'error': str(e)})
                yield f"event: progress\ndata: {json.dumps(tried[-1])}\n\n"
                continue

        if not param_name:
            yield f"event: result\ndata: {json.dumps({'ok': False, 'msg': 'No supported GPU param found'})}\n\n"
            return

        # exponential growth phase
        lo = start
        hi = None
        cur = start
        last_ok = start
        fast_mode = (request.query_params.get('mode') == 'fast') or (request.query_params.get('fast') in ('1','true'))
        while cur <= max_layers:
            try:
                t0 = time.time()
                inst = await asyncio.to_thread(Llama, model_path=model_path, **{param_name: cur}, n_ctx=int(os.environ.get('LLM_N_CTX', '8192')), n_threads=int(os.environ.get('LLM_N_THREADS', '4')))
                elapsed = time.time() - t0
                try:
                    if hasattr(inst, 'close'):
                        inst.close()
                except Exception:
                    pass
                del inst
                tried.append({'param': param_name, 'layers': cur, 'ok': True, 'seconds': elapsed})
                last_ok = cur
                yield f"event: progress\ndata: {json.dumps(tried[-1])}\n\n"
                # increase
                if cur == 0:
                    cur = 1
                else:
                    cur = cur * 2
                if cur > max_layers:
                    hi = max_layers + 1
                    break
            except Exception as e:
                tried.append({'param': param_name, 'layers': cur, 'ok': False, 'error': str(e)})
                yield f"event: progress\ndata: {json.dumps(tried[-1])}\n\n"
                hi = cur
                break

        # Fast mode: do not run binary search; accept last_ok as recommended (very fast)
        if fast_mode:
            recommended = last_ok
            total_seconds = time.time() - start_time
            result = {'ok': True, 'recommended': recommended, 'param': param_name, 'total_seconds': total_seconds, 'fast_mode': True}
            app.state.llm_diag = {'ok': True, 'param': param_name, 'layers': recommended, 'seconds': total_seconds, 'msg': 'autotune-fast', 'time': time.time()}
            yield f"event: result\ndata: {json.dumps(result)}\n\n"
            return

        if hi is None:
            # never failed up to max_layers; use last_ok
            recommended = last_ok
            total_seconds = time.time() - start_time
            result = {'ok': True, 'recommended': recommended, 'param': param_name, 'total_seconds': total_seconds}
            app.state.llm_diag = {'ok': True, 'param': param_name, 'layers': recommended, 'seconds': total_seconds, 'msg': 'autotune', 'time': time.time()}
            yield f"event: result\ndata: {json.dumps(result)}\n\n"
            return

        # binary search between last_ok and hi-1
        lo = last_ok
        hi = hi
        # ensure lo < hi
        if lo >= hi:
            recommended = lo
        else:
            while lo + 1 < hi:
                mid = (lo + hi) // 2
                try:
                    t0 = time.time()
                    inst = await asyncio.to_thread(Llama, model_path=model_path, **{param_name: mid}, n_ctx=int(os.environ.get('LLM_N_CTX', '8192')), n_threads=int(os.environ.get('LLM_N_THREADS', '4')))
                    elapsed = time.time() - t0
                    try:
                        if hasattr(inst, 'close'):
                            inst.close()
                    except Exception:
                        pass
                    del inst
                    tried.append({'param': param_name, 'layers': mid, 'ok': True, 'seconds': elapsed})
                    yield f"event: progress\ndata: {json.dumps(tried[-1])}\n\n"
                    lo = mid
                except Exception as e:
                    tried.append({'param': param_name, 'layers': mid, 'ok': False, 'error': str(e)})
                    yield f"event: progress\ndata: {json.dumps(tried[-1])}\n\n"
                    hi = mid
            recommended = lo

        total_seconds = time.time() - start_time
        result = {'ok': True, 'recommended': recommended, 'param': param_name, 'tried': tried, 'total_seconds': total_seconds}
        app.state.llm_diag = {'ok': True, 'param': param_name, 'layers': recommended, 'seconds': total_seconds, 'msg': 'autotune', 'time': time.time()}
        # Optionally preload: environment variable LLM_AUTOTUNE_PRELOAD=1 will preload instance with recommended layers
        try:
            if os.environ.get('LLM_AUTOTUNE_PRELOAD', '').strip().lower() in ('1', 'true', 'yes'):
                try:
                    # Temporarily set chosen GPU layers so create_llama_instance will use them
                    old_val = os.environ.get('LLM_GPU_LAYERS')
                    os.environ['LLM_GPU_LAYERS'] = str(recommended)
                    inst = create_llama_instance(model_path, verify_gpu=True)
                    # Restore old env
                    if old_val is None:
                        del os.environ['LLM_GPU_LAYERS']
                    else:
                        os.environ['LLM_GPU_LAYERS'] = old_val
                    # Drop existing preloaded instance and replace
                    try:
                        q = app.state.llm_pool
                        q.put_nowait(inst)
                        app.state.llm_pool_count = getattr(app.state, 'llm_pool_count', 0) + 1
                        app.state.llm = inst
                        logging.info('Preloaded Llama instance after autotune: %s=%s (verified=%s)', param_name, recommended, getattr(app.state, 'llm_gpu_verified', False))
                    except Exception:
                        # If pool full, just set llm for compatibility
                        app.state.llm = inst
                except Exception:
                    logging.warning('Autotune preload failed')
        except Exception:
            pass
        yield f"event: result\ndata: {json.dumps(result)}\n\n"

    if request.query_params.get('stream') in ('1','true'):
        return StreamingResponse(autotune_stream(), media_type='text/event-stream')
    # otherwise run a full autotune synchronously and return json
    # Run and collect
    out = []
    async for ev in autotune_stream():
        out.append(ev)
    # last event contains the result JSON inside 'event: result\ndata: ...'
    # extract final result from app.state.llm_diag and return it
    return app.state.llm_diag


@app.get('/llm_probe')
async def llm_probe():
    """Probe a Llama instance: report GPU-related attributes and run a short timed generation (small token count).

    Returns JSON with attributes, a short timing, and optional nvidia-smi before/after memory snapshots.
    """
    if Llama is None:
        raise HTTPException(status_code=500, detail='llama_cpp not installed')
    model_path = getattr(app.state, 'llm_model_path', None)
    if not model_path or not Path(model_path).exists():
        raise HTTPException(status_code=500, detail='Local model not configured or not found')

    # Optional small generation to measure speed: short prompt, few tokens
    probe_prompt = os.environ.get('LLM_PROBE_PROMPT', 'The quick brown fox')
    probe_max_tokens = int(os.environ.get('LLM_PROBE_TOKENS', '24'))

    smi_before = None
    smi_after = None
    # Capture nvidia-smi memory used if available
    try:
        p = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=3)
        if p.returncode == 0 and p.stdout.strip():
            smi_before = [int(x.strip()) for x in p.stdout.strip().splitlines()]
    except Exception:
        smi_before = None

    llm_inst = None
    inst_info = {'attrs': {}, 'has_gpu_attr': False}
    start = time.time()
    try:
        # Acquire an instance (may create temporary)
        llm_inst = await acquire_llm_instance()
        # Inspect instance attributes for GPU indicators
        for name in dir(llm_inst):
            if 'gpu' in name.lower() or 'cuda' in name.lower() or 'device' in name.lower():
                try:
                    val = getattr(llm_inst, name)
                    # Represent simple values only
                    if isinstance(val, (int, float, str, bool)):
                        inst_info['attrs'][name] = val
                    else:
                        # For callables/complex attrs, mark presence
                        inst_info['attrs'][name] = str(type(val))
                    inst_info['has_gpu_attr'] = True
                except Exception:
                    inst_info['attrs'][name] = 'ERROR'
        # Try a short generation to measure tokens/sec
        t0 = time.time()
        response = await asyncio.to_thread(llm_inst.create_chat_completion, messages=[{'role':'user','content':probe_prompt}], max_tokens=probe_max_tokens, temperature=0.1)
        t1 = time.time()
        content = response['choices'][0]['message'].get('content') if response and 'choices' in response else ''
        elapsed = t1 - t0
        tokens_est = max(1, int(len(content) / 4))
        tok_per_sec = (tokens_est / elapsed) if elapsed > 0 else tokens_est
        probe_res = {'seconds': elapsed, 'tokens': tokens_est, 'tok_per_sec': tok_per_sec, 'content_sample': (content or '')[:200]}
    except Exception as e:
        probe_res = {'error': str(e)}
    finally:
        try:
            if llm_inst:
                release_llm_instance(llm_inst)
        except Exception:
            pass

    try:
        p = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=3)
        if p.returncode == 0 and p.stdout.strip():
            smi_after = [int(x.strip()) for x in p.stdout.strip().splitlines()]
    except Exception:
        smi_after = None

    result = {
        'timestamp': time.time(),
        'llm_has_gpu': getattr(app.state, 'llm_has_gpu', False),
        'llm_use_gpu': getattr(app.state, 'llm_use_gpu', False),
        'llm_last_gpu_param': getattr(app.state, 'llm_last_gpu_param', None),
        'llm_last_gpu_layers': getattr(app.state, 'llm_last_gpu_layers', None),
        'inst_info': inst_info,
        'probe': probe_res,
        'smi_before': smi_before,
        'smi_after': smi_after,
    }
    app.state.llm_last_probe = result
    return result


@app.get('/llm_status')
def llm_status():
    gemini = getattr(app.state, 'gemini_model', None)
    llm_path = getattr(app.state, 'llm_model_path', None)
    pool_max = getattr(app.state, 'llm_pool_max', 0)
    pool_count = getattr(app.state, 'llm_pool_count', 0)
    pool_q = getattr(app.state, 'llm_pool', None)
    pool_idle = pool_q.qsize() if pool_q is not None else (1 if getattr(app.state, 'llm', None) else 0)

    if llm_path:
        provider = 'local'
        ok = True
    elif gemini:
        provider = getattr(app.state, 'gemini_model_name', GEMINI_MODEL_NAME)
        ok = True
    else:
        provider = ''
        ok = False

    url = os.environ.get('LLM_SERVER_URL', 'http://ashy.tplinkdns.com:5005/ask')
    return {
        'ok': ok,
        'status': 'Connected' if ok else 'Not connected',
        'provider': provider,
        'url': url,
        'pool_max': pool_max,
        'pool_count': pool_count,
        'pool_idle': pool_idle,
        'gpu': getattr(app.state, 'llm_use_gpu', False),
        'gpu_param': getattr(app.state, 'llm_last_gpu_param', None),
        'gpu_layers': getattr(app.state, 'llm_last_gpu_layers', None),
        'gpu_verified': getattr(app.state, 'llm_gpu_verified', False),
        'last_diag': getattr(app.state, 'llm_diag', None)
    }


if __name__ == '__main__':
    import argparse
    import subprocess
    import signal
    import sys

    def start_foreground():
        import uvicorn
        uvicorn.run('llm_server:app', host=HOST, port=PORT, log_level='info')

    def start_background(logfile='/tmp/llm_server.log'):
        if PID_FILE.exists():
            try:
                existing = int(PID_FILE.read_text().strip())
                os.kill(existing, 0)
                print(f"Server already running with PID {existing}")
                return
            except Exception:
                pass
        cmd = [sys.executable, str(__file__), 'start']
        with open(logfile, 'ab') as out:
            p = subprocess.Popen(cmd, stdout=out, stderr=out, cwd=str(APP_DIR))
        time.sleep(0.5)
        if p.poll() is None:
            print(f"Started background server (PID {p.pid}), logging to {logfile}")
            try:
                PID_FILE.write_text(str(p.pid))
            except Exception:
                pass
        else:
            print('Failed to start server. Check log:', logfile)

    def stop_server(timeout=3):
        if not PID_FILE.exists():
            print('No PID file found; attempting to locate running server processes...')
            # Try to find likely server processes (fallback when PID file missing)
            try:
                p = subprocess.run(['pgrep', '-f', 'llm_server'], capture_output=True, text=True)
                out = (p.stdout or '').strip()
                if not out:
                    print('No running server processes found.')
                    return
                pids = [int(x) for x in out.split() if x.strip().isdigit()]
            except Exception:
                print('Failed to query processes for llm_server. No PID file and cannot discover process.')
                return

            stopped_any = False
            for pid in pids:
                try:
                    print(f'Attempting to stop process PID {pid}...')
                    os.kill(pid, signal.SIGTERM)
                    for _ in range(timeout * 10):
                        time.sleep(0.1)
                        try:
                            os.kill(pid, 0)
                        except Exception:
                            break
                    else:
                        os.kill(pid, signal.SIGKILL)
                    print(f'Stopped server (PID {pid}).')
                    stopped_any = True
                except ProcessLookupError:
                    print(f'Process {pid} not found.')
                except PermissionError:
                    print(f'Permission denied when trying to stop process {pid}.')
                except Exception as e:
                    print(f'Failed to stop process {pid}: {e}')
            if not stopped_any:
                print('No processes were stopped.')
            return

        try:
            pid = int(PID_FILE.read_text().strip())
        except Exception:
            print('Invalid PID file. Removing it.')
            PID_FILE.unlink(missing_ok=True)
            return
        try:
            os.kill(pid, signal.SIGTERM)
            for _ in range(timeout * 10):
                time.sleep(0.1)
                try:
                    os.kill(pid, 0)
                except Exception:
                    break
            else:
                os.kill(pid, signal.SIGKILL)
            print(f'Stopped server (PID {pid}).')
        except ProcessLookupError:
            print('Process not found; removing stale PID file.')
        except PermissionError:
            print('Permission denied when trying to stop process.')
        finally:
            PID_FILE.unlink(missing_ok=True)

    def status_server():
        if not PID_FILE.exists():
            print('No PID file found; server not running.')
            return
        try:
            pid = int(PID_FILE.read_text().strip())
            os.kill(pid, 0)
            print(f'Server appears to be running with PID {pid}.')
        except Exception:
            print('PID file exists but process not running. PID file may be stale.')

    parser = argparse.ArgumentParser(description='LLM server control')
    sub = parser.add_subparsers(dest='cmd')
    sub.add_parser('start', help='Start server in foreground')
    sbg = sub.add_parser('start-bg', help='Start server in background (detached)')
    sbg.add_argument('--log', default='/tmp/llm_server.log', help='Log file for background server')
    sub.add_parser('stop', help='Stop server using PID file')
    sub.add_parser('status', help='Show server status')
    args = parser.parse_args()

    if args.cmd == 'start-bg':
        start_background(logfile=args.log)
    elif args.cmd == 'stop':
        stop_server()
    elif args.cmd == 'status':
        status_server()
    else:
        start_foreground()
