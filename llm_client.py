"""
llm_client.py - one small interface so the model can be swapped without touching the pipeline.

    client = make_client("anthropic")                    # needs ANTHROPIC_API_KEY, optional LLM_MODEL
    client = make_client("openai-compat")                # LLM_API_KEY, LLM_BASE_URL, LLM_MODEL
    client = make_client("replay:judgements.jsonl")      # serve saved answers by key (no network)
    client = make_client("none")                         # rules-only runs

Every client implements complete_json(system, user, key=None) -> dict and never needs an SDK
(plain HTTPS via urllib). Wrap any client in CachedClient to make runs repeatable and cheap.

Gemini and most other providers expose an OpenAI-compatible endpoint, so 'openai-compat' with a
different LLM_BASE_URL / LLM_MODEL is how a stronger model supplied at interview time plugs in.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.request
import urllib.error
from typing import Any, Dict, Optional


class LLMError(RuntimeError):
    pass


def extract_json(text: str) -> Dict[str, Any]:
    """Parse the first {...} object in a model reply (tolerates code fences / chatter)."""
    a, b = text.find("{"), text.rfind("}")
    if a < 0 or b <= a:
        raise LLMError("no JSON object in model reply")
    try:
        return json.loads(text[a:b + 1])
    except json.JSONDecodeError as exc:
        raise LLMError(f"invalid JSON from model: {exc}") from exc


class LLMClient:
    name = "base"
    model = "none"

    def complete_json(self, system: str, user: str, key: Optional[str] = None) -> Dict[str, Any]:
        raise NotImplementedError


class NullClient(LLMClient):
    name = "none"

    def complete_json(self, system, user, key=None):
        raise LLMError("no LLM configured (rules-only run)")


def _post_json(url: str, headers: Dict[str, str], body: Dict[str, Any], retries: int = 3,
               timeout: int = 90) -> Dict[str, Any]:
    data = json.dumps(body).encode("utf-8")
    last: Optional[Exception] = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, data=data, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            last = exc
            if exc.code not in (429, 500, 502, 503, 529):   # not retryable
                raise LLMError(f"HTTP {exc.code}: {exc.read().decode('utf-8', 'replace')[:200]}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            last = exc
        time.sleep(2 ** attempt)
    raise LLMError(f"request failed after {retries} attempts: {last}")


class AnthropicClient(LLMClient):
    name = "anthropic"

    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None,
                 max_tokens: int = 700):
        self.model = model or os.environ.get("LLM_MODEL", "claude-sonnet-5-5")
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.max_tokens = max_tokens
        if not self.api_key:
            raise LLMError("ANTHROPIC_API_KEY is not set")

    def complete_json(self, system, user, key=None):
        out = _post_json("https://api.anthropic.com/v1/messages",
                         {"x-api-key": self.api_key, "anthropic-version": "2023-06-01",
                          "content-type": "application/json"},
                         {"model": self.model, "max_tokens": self.max_tokens, "temperature": 0,
                          "system": system, "messages": [{"role": "user", "content": user}]})
        try:
            return extract_json(out["content"][0]["text"])
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"unexpected Anthropic response shape: {exc}") from exc


class OpenAICompatClient(LLMClient):
    name = "openai-compat"

    def __init__(self, model: Optional[str] = None, base_url: Optional[str] = None,
                 api_key: Optional[str] = None):
        self.model = model or os.environ.get("LLM_MODEL")
        self.base_url = (base_url or os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self.api_key = api_key or os.environ.get("LLM_API_KEY")
        if not (self.model and self.api_key):
            raise LLMError("set LLM_MODEL and LLM_API_KEY (and LLM_BASE_URL for non-OpenAI providers)")

    def complete_json(self, system, user, key=None):
        out = _post_json(f"{self.base_url}/chat/completions",
                         {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                         {"model": self.model, "temperature": 0,
                          "response_format": {"type": "json_object"},
                          "messages": [{"role": "system", "content": system},
                                       {"role": "user", "content": user}]})
        try:
            return extract_json(out["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"unexpected response shape: {exc}") from exc


class ReplayClient(LLMClient):
    """Serves saved judgements keyed by `key` (candidate_id). Lets a run be reproduced offline."""
    name = "replay"

    def __init__(self, path: str):
        self.path = path
        self.model = "replay"
        self.table: Dict[str, Dict[str, Any]] = {}
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    rec = json.loads(line)
                    self.table[rec["key"]] = rec["result"]
                    self.model = rec.get("model", self.model)

    def has(self, key: str) -> bool:
        return key in self.table

    def complete_json(self, system, user, key=None):
        if key not in self.table:
            raise LLMError(f"no saved judgement for {key}")
        return self.table[key]


class CachedClient(LLMClient):
    """Disk cache around any client: same (model, system, user) is never paid for twice."""

    def __init__(self, inner: LLMClient, cache_path: str):
        self.inner, self.cache_path = inner, cache_path
        self.name, self.model = inner.name, inner.model
        self.hits = self.misses = 0
        self.table: Dict[str, Dict[str, Any]] = {}
        if os.path.exists(cache_path):
            with open(cache_path, encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        rec = json.loads(line)
                        self.table[rec["hash"]] = rec["result"]

    def complete_json(self, system, user, key=None):
        h = hashlib.sha256(f"{self.inner.model}\n{system}\n{user}".encode()).hexdigest()
        if h in self.table:
            self.hits += 1
            return self.table[h]
        result = self.inner.complete_json(system, user, key)
        self.misses += 1
        self.table[h] = result
        with open(self.cache_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"hash": h, "key": key, "model": self.inner.model, "result": result}) + "\n")
        return result


def make_client(spec: str, cache_path: Optional[str] = None) -> LLMClient:
    """spec: none | anthropic | openai-compat | replay:<path>"""
    if spec in ("none", "", None):
        return NullClient()
    if spec.startswith("replay:"):
        return ReplayClient(spec.split(":", 1)[1])
    clients = {"anthropic": AnthropicClient, "openai-compat": OpenAICompatClient}
    if spec not in clients:
        raise LLMError(f"unknown LLM spec {spec!r}; use none | anthropic | openai-compat | replay:<file>")
    client: LLMClient = clients[spec]()
    return CachedClient(client, cache_path) if cache_path else client
