"""LLM 이 내놓은 인용문이 실제 근거 본문에 있는지 코드로 확인한다 (환각 인용 차단).

LLM 은 제목과 본문을 섞거나 조사 한두 글자를 바꿔 인용하는 경우가 많아 완전 일치 대신
글자 3-gram 의 80% 이상이 같은 근거 안에 있으면 인정한다.
"""
from __future__ import annotations

import re


def norm(s: str) -> str:
    return re.sub(r"[\W_]+", "", s or "").lower()


def _grams(s: str, n: int = 3) -> set[str]:
    return {s[i:i + n] for i in range(max(0, len(s) - n + 1))}


def _piece_in(q: str, t: str, threshold: float) -> bool:
    if q in t:
        return True
    g = _grams(q)
    return bool(g) and len(g & _grams(t)) / len(g) >= threshold


def fuzzy_in(quote: str, text: str, threshold: float = 0.8) -> bool:
    """인용문이 "…" 로 이어 붙인 여러 조각이면 조각마다 확인한다."""
    t = norm(text)
    pieces = [norm(p) for p in re.split(r"…|\.\.\.|·{3}|\s/\s", quote or "")]
    pieces = [p for p in pieces if len(p) >= 6]
    return bool(pieces) and all(_piece_in(p, t, threshold) for p in pieces)


STAGE_TERMS = {
    "Seed": ["seed", "시드", "tips", "팁스", "엔젤"], "Pre-A": ["prea", "프리a", "프리시리즈a", "preseriesa"],
    "Series A": ["seriesa", "시리즈a"], "Pre-B": ["preb", "프리b", "프리시리즈b"], "Series B": ["seriesb", "시리즈b"],
    "Pre-C": ["prec", "프리c", "프리시리즈c"], "Series C": ["seriesc", "시리즈c"],
    "Pre-IPO": ["preipo", "프리ipo"], "Series D 이상": ["seriesd", "seriese", "seriesf", "시리즈d", "시리즈e", "시리즈f"],
}
