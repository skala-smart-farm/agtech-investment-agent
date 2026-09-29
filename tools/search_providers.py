"""검색 공급자 여러 개 (한 곳이 막혀도 계속 동작하도록).

- Serper (SERPER_API_KEY): 구글 검색·뉴스. 한국어 질의는 gl=kr·hl=ko 로 국내 결과. 가입 시 무료 2,500건
- Tavily (TAVILY_API_KEY): 처음부터 쓰던 공급자 (무료 한도를 다 써서 예비로 둔다)
키가 있는 공급자만 Serper → Tavily 순서로 시도한다.
네이버 검색 API 는 2026-07-31 부터 개발자센터 신규 발급이 끝나(NAVER API HUB 로 이관) 넣지 않았다.
모든 공급자의 결과는 같은 형식 {url, title, content, published_date} 으로 맞춘다.
"""
from __future__ import annotations

import os
import re
from datetime import date, datetime, timedelta

import requests

HANGUL = re.compile(r"[가-힣]")


def order_for(query: str) -> list[str]:
    return [p for p, key in (("serper", "SERPER_API_KEY"), ("tavily", "TAVILY_API_KEY")) if os.getenv(key)]


def _serper_date(raw: str | None, today: str) -> str | None:
    """Serper 의 날짜 표기("3 days ago", "3일 전", "Jan 5, 2025", "2025. 1. 5.")를 YYYY-MM-DD 로 바꾼다.
    "N일 전" 은 검색한 날(today)에서 뺀다. 변환한 날짜째로 캐시에 저장되므로 재현 때도 같다."""
    if not raw:
        return None
    base = date.fromisoformat(today)
    m = re.search(r"(\d+)\s*(minute|hour|day|week|month|year|분|시간|일|주|개월|달|년)", raw)
    if m and ("ago" in raw or "전" in raw):
        n, unit = int(m.group(1)), m.group(2)
        days = {"minute": 0, "분": 0, "hour": 0, "시간": 0, "day": 1, "일": 1, "week": 7, "주": 7,
                "month": 30, "개월": 30, "달": 30, "year": 365, "년": 365}[unit]
        return (base - timedelta(days=n * days)).isoformat()
    m = re.search(r"(20\d{2})\s*[.\-/]\s*(\d{1,2})\s*[.\-/]\s*(\d{1,2})", raw)
    if m:
        return f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
    for fmt in ("%b %d, %Y", "%d %b %Y", "%B %d, %Y"):
        try:
            return datetime.strptime(raw.strip(), fmt).date().isoformat()
        except ValueError:
            pass
    return None


def serper(query: str, topic: str, include_domains: list[str] | None,
           time_range: str | None, today: str) -> list[dict]:
    q = query + (" (" + " OR ".join(f"site:{d}" for d in include_domains) + ")" if include_domains else "")
    body = {"q": q, "num": 10}  # 제외 도메인을 결과에서 거르므로 넉넉히 받는다
    if HANGUL.search(query):
        body.update(gl="kr", hl="ko")
    if time_range == "year":
        body["tbs"] = "qdr:y"
    endpoint = "news" if topic == "news" else "search"
    r = requests.post(f"https://google.serper.dev/{endpoint}", json=body,
                      headers={"X-API-KEY": os.environ["SERPER_API_KEY"]}, timeout=20)
    r.raise_for_status()
    items = r.json().get("news" if endpoint == "news" else "organic", [])
    return [{"url": it["link"], "title": it.get("title", ""), "content": it.get("snippet", ""),
             "published_date": _serper_date(it.get("date"), today)} for it in items if it.get("link")]
