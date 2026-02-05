import os
import sys
from fastapi.testclient import TestClient

# Ensure repo root is on sys.path so tests can import `llm_server`
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import llm_server

client = TestClient(llm_server.app)


def assert_json_object(resp):
    """Assert response content-type is JSON and json() returns a dict-like object."""
    assert 'application/json' in resp.headers.get('content-type', ''), f"Not JSON content-type: {resp.headers.get('content-type')!r}"
    j = resp.json()
    assert isinstance(j, dict), f"Expected dict JSON response, got {type(j)}: {j!r}"
    return j


def test_llm_diag_returns_json_object():
    r = client.post('/llm_diag')
    # Should return JSON (error or result)
    j = assert_json_object(r)
    # If an error occurs (e.g., llama_cpp not installed), FastAPI provides a 'detail' key
    assert 'detail' in j or 'ok' in j


def test_llm_probe_returns_json_object():
    r = client.get('/llm_probe')
    j = assert_json_object(r)
    assert 'probe' in j or 'detail' in j


def test_ask_returns_json_object_on_basic_request():
    payload = {
        'prompt': 'Hello',
        'system_prompt': 'You are helpful.',
        'conversation': [],
        'provider': ''
    }
    r = client.post('/ask', json=payload)
    j = assert_json_object(r)
    # Either we get an error detail, or a response object (on success)
    assert 'detail' in j or 'response' in j or 'provider' in j


def test_gemini_test_endpoint():
    r = client.get('/gemini_test')
    # Should return JSON and either an error detail (if no key) or ok result
    j = assert_json_object(r)
    assert 'detail' in j or 'ok' in j
