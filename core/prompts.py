"""prompts/*.md 템플릿을 불러와 채운다. 프롬프트는 코드와 분리해 관리한다."""
from __future__ import annotations

from functools import lru_cache

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from core.config import ROOT


@lru_cache(maxsize=1)
def _env() -> Environment:
    return Environment(loader=FileSystemLoader(ROOT / "prompts"), undefined=StrictUndefined,
                       trim_blocks=True, lstrip_blocks=True)


def render(_template: str, **kwargs) -> str:
    return _env().get_template(f"{_template}.md").render(**kwargs)
