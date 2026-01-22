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

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

try:
    from llama_cpp import Llama
except Exception:
    Llama = None

try:
    import google.generativeai as genai
except Exception as e:
    logging.error(f"Failed to import google.generativeai: {e}")
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
app = FastAPI()

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

class AskResponse(BaseModel):
    response: str
    provider: str


def format_sse(data: str) -> str:
    if data is None:
        return 'data: \n\n'
    parts = data.split('\n')
    out_lines = [f"data: {p}" for p in parts]
    return '\n'.join(out_lines) + '\n\n'


@app.on_event('startup')
def startup_event():
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
            # Pool configuration
            app.state.llm_pool = asyncio.Queue()
            app.state.llm_pool_max = int(os.environ.get('LLM_POOL_MAX', '2'))
            app.state.llm_pool_count = 0
            preload = int(os.environ.get('LLM_POOL_PRELOAD', '1'))
            # Optionally preload a single instance (default: yes)
            if preload and app.state.llm_pool_max > 0:
                try:
                    inst = Llama(model_path=model_path,
                                 n_ctx=int(os.environ.get('LLM_N_CTX', '8192')),
                                 n_threads=int(os.environ.get('LLM_N_THREADS', '4')),
                                 verbose=False)
                    app.state.llm_pool.put_nowait(inst)
                    app.state.llm_pool_count = 1
                    # Keep this instance in app.state.llm for backwards compatibility
                    app.state.llm = inst
                    logging.info('Preloaded 1 Llama model from %s (pool max=%s)', model_path, app.state.llm_pool_max)
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


# --- LLM pool helpers ---
async def acquire_llm_instance():
    """Acquire a Llama instance from the pool or create one (async)."""
    if getattr(app.state, 'llm_model_path', None) is None:
        raise RuntimeError('No local model configured')
    pool = getattr(app.state, 'llm_pool', None)
    if pool is None:
        # No pool configured, create a temporary instance
        inst = await asyncio.to_thread(Llama, model_path=app.state.llm_model_path,
                                       n_ctx=int(os.environ.get('LLM_N_CTX', '8192')),
                                       n_threads=int(os.environ.get('LLM_N_THREADS', '4')),
                                       verbose=False)
        return inst
    try:
        inst = pool.get_nowait()
        return inst
    except asyncio.QueueEmpty:
        # create new instance if pool has capacity
        if getattr(app.state, 'llm_pool_count', 0) < getattr(app.state, 'llm_pool_max', 1):
            inst = await asyncio.to_thread(Llama, model_path=app.state.llm_model_path,
                                           n_ctx=int(os.environ.get('LLM_N_CTX', '8192')),
                                           n_threads=int(os.environ.get('LLM_N_THREADS', '4')),
                                           verbose=False)
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


@app.on_event('shutdown')
def shutdown_event():
    try:
        if PID_FILE.exists():
            PID_FILE.unlink()
    except Exception:
        pass


@app.get('/', response_class=HTMLResponse)
def index():
    html = """
    <!doctype html>
    <html>
    <head><meta charset="utf-8"><title>LLM Server</title></head>
    <body style="font-family: Arial, Helvetica, sans-serif; margin:20px;">
        <h2>LLM Server</h2>
        <form id="frm">
            <label>Provider</label><br>
            <select id="provider" style="width:100%; margin-bottom:10px; padding:4px;">
                <option value="gemini-3-flash-preview">Google Gemini 3 Flash Preview</option>
                <option value="gemini-2.5-flash">Google Gemini 2.5 Flash</option>
                <option value="gemini-2.5-flash-lite" selected>Google Gemini 2.5 Flash Lite (Unlimited!)</option>
                <option value="gemma-3-27b">Gemma 3-27B (Google)</option>
                <option value="local">Local Model (Llama/Gemma)</option> 
            </select><br>
            <label>System prompt (optional)</label><br>
            <input id="system" style="width:100%" placeholder="System prompt"><br><br>
            <label>Prompt</label><br>
            <textarea id="prompt" rows="4" style="width:100%" placeholder="Enter your prompt"></textarea><br>
            <button type="submit">Ask</button>
        </form>
        <h3>Response (<span id="active-provider">None</span>)</h3>
        <pre id="resp" style="white-space:pre-wrap; background:#f6f6f6; padding:12px; border-radius:6px; max-width:800px;"></pre>

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

            document.getElementById('frm').addEventListener('submit', async function(e) {
                e.preventDefault();
                const promptEl = document.getElementById('prompt');
                const prompt = promptEl.value;
                const system = document.getElementById('system').value || undefined;
                const provider = document.getElementById('provider').value;
                if (!prompt || !prompt.trim()) return;

                const payload = {
                    prompt: prompt,
                    system_prompt: system,
                    provider: provider,
                    conversation: conversation.slice() // Send history, server adds current prompt
                };

                const respEl = document.getElementById('resp');
                const providerEl = document.getElementById('active-provider');
                respEl.textContent = '';
                providerEl.textContent = '...';

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
                    conversation.pop(); // remove optimistic user message
                    renderConversation();
                }
            });
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
    if provider:
        status_html = f'<div style="margin-bottom:8px;"><strong>LLM:</strong> Connected [{provider}] (<a href="{llm_url}" target="_blank">{llm_url}</a>)</div>'
    else:
        status_html = '<div style="margin-bottom:8px; color:#b00"><strong>LLM:</strong> Not connected</div>'
    html = html.replace('<h2>LLM Server</h2>', f'<h2>LLM Server</h2>\n        {status_html}')
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
                    response = await asyncio.to_thread(chat.send_message, user_msg['parts'][0], stream=True)
                    for chunk in response:
                        try:
                            if chunk.text:
                                yield format_sse(chunk.text)
                        except (ValueError, IndexError, AttributeError):
                            pass
                        await asyncio.sleep(0)
                    yield 'event: done\ndata: [DONE]\n\n'
                except Exception as e:
                    logging.exception('Gemini streaming error: %s', e)
                    yield f"event: error\ndata: {str(e)}\n\n"
            return StreamingResponse(gemini_stream(), media_type='text/event-stream')
        
        try:
            response = await asyncio.to_thread(chat.send_message, user_msg['parts'][0])
            try:
                text = response.text
            except (ValueError, IndexError, AttributeError) as ve:
                reason = "unknown"
                try:
                    if response.candidates:
                        reason = str(response.candidates[0].finish_reason)
                except: pass
                text = f"[Gemini Blocked/Empty] Finish reason: {reason}. Detail: {str(ve)}"
            return {'response': text, 'provider': provider}
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

        if stream_requested:
            async def event_stream():
                yield f"event: provider\ndata: local\n\n"
                # Acquire a model instance for the duration of the stream
                try:
                    llm_inst = await acquire_llm_instance()
                except Exception as e:
                    logging.exception('Failed to acquire LLM instance for streaming: %s', e)
                    yield f"event: error\ndata: {str(e)}\n\n"
                    return
                try:
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
                                yield format_sse(text_part)
                                await asyncio.sleep(0)
                        except Exception:
                            continue
                    yield 'event: done\ndata: [DONE]\n\n'
                except Exception as e:
                    logging.exception('LLM generation error (stream): %s', e)
                    yield f"event: error\ndata: {str(e)}\n\n"
                finally:
                    try:
                        release_llm_instance(llm_inst)
                    except Exception:
                        pass
            return StreamingResponse(event_stream(), media_type='text/event-stream')

        try:
            # Acquire instance and run blocking creation in a threadpool to avoid blocking the event loop
            try:
                llm_inst = await acquire_llm_instance()
            except Exception as e:
                logging.exception('Failed to acquire LLM instance: %s', e)
                raise HTTPException(status_code=500, detail=str(e))
            try:
                response = await asyncio.to_thread(llm_inst.create_chat_completion, messages=messages, **gen)
            finally:
                try:
                    release_llm_instance(llm_inst)
                except Exception:
                    pass
            content = response['choices'][0]['message'].get('content') if response and 'choices' in response else ''
            text = content.strip() if content else ''
            return {'response': text, 'provider': provider}
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
    return {'ok': ok, 'status': 'Connected' if ok else 'Not connected', 'provider': provider, 'url': url, 'pool_max': pool_max, 'pool_count': pool_count, 'pool_idle': pool_idle}


if __name__ == '__main__':
    import argparse
    import subprocess
    import signal
    import sys
    import time

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
            print('No PID file found; server may not be running.')
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
