"""LLM 생성 및 캐시 설정. 모델 이름은 config.yaml 에서만 가져온다."""
from __future__ import annotations

import os
from functools import lru_cache
from typing import TypeVar

from langchain_core.globals import set_llm_cache
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from core.config import get_config, path
from core.cost import TRACKER

T = TypeVar("T", bound=BaseModel)
_cache_ready = False


def _ensure_cache() -> None:
    global _cache_ready
    if _cache_ready:
        return
    cfg = get_config()
    if cfg.cache.llm:
        from langchain_community.cache import SQLiteCache

        set_llm_cache(SQLiteCache(database_path=str(path(f"{cfg.cache.dir}/llm_cache.sqlite"))))
    _cache_ready = True


def _block_network() -> None:
    """--offline: 캐시에 없는 LLM 호출만 실패시킨다. 모델 설정(=캐시 키)은 그대로 둬야 캐시가 맞는다."""
    def _refuse(self, *args, **kwargs):
        raise RuntimeError("--offline: 재현용 캐시에 없는 LLM 호출입니다 (프롬프트나 근거가 제출본과 다름)")

    ChatOpenAI._generate = _refuse


@lru_cache(maxsize=4)
def get_llm(role: str = "generator") -> ChatOpenAI:
    """role: generator(분석·작성) | judge(판정)."""
    _ensure_cache()
    cfg = get_config()
    if not os.getenv("OPENAI_API_KEY"):  # 캐시 재현 모드: 키 없이 만들되, 캐시에 없는 호출이면 그때 명확히 실패
        os.environ["OPENAI_API_KEY"] = "sk-replay-only-no-key"
    if os.getenv("REPLAY_OFFLINE"):
        _block_network()
    return ChatOpenAI(model=cfg.models[role], temperature=cfg.models.temperature, max_retries=3, timeout=120,
                      callbacks=[TRACKER])


def structured(schema: type[T], role: str = "generator"):
    """Pydantic 스키마로 출력을 강제한 LLM (JSON Schema strict)."""
    return get_llm(role).with_structured_output(schema, method="json_schema", strict=True)
