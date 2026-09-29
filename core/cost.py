"""LLM 토큰 사용량·비용 집계. LangSmith 없이도 실행마다 비용을 기록한다.

캐시(SQLite)에서 꺼낸 응답은 실제 API 호출이 아니므로 비용에 넣지 않는다(llm_output 이 없으면 캐시 응답).
"""
from __future__ import annotations

import threading

from langchain_core.callbacks import BaseCallbackHandler

# USD / 100만 토큰 (OpenAI 공개 가격, config.yaml pricing 으로 덮어쓸 수 있음)
DEFAULT_PRICE = {"gpt-4.1-mini": (0.40, 1.60), "gpt-4.1-nano": (0.10, 0.40)}


class CostTracker(BaseCallbackHandler):
    def __init__(self):
        self.lock = threading.Lock()
        self.calls = self.cached = self.input_tokens = self.output_tokens = 0
        self.usd = 0.0

    def on_llm_end(self, response, **kwargs) -> None:
        out = response.llm_output or {}
        usage = out.get("token_usage") or {}
        with self.lock:
            if not out:
                self.cached += 1
                return
            model = str(out.get("model_name", ""))
            pi, po = next((v for k, v in DEFAULT_PRICE.items() if model.startswith(k)), (0.40, 1.60))
            i, o = usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
            self.calls += 1
            self.input_tokens += i
            self.output_tokens += o
            self.usd += i / 1e6 * pi + o / 1e6 * po

    def summary(self) -> dict:
        return {"api_calls": self.calls, "cache_hits": self.cached, "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens, "usd": round(self.usd, 4)}


TRACKER = CostTracker()
