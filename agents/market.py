"""📊 시장성 평가 에이전트 (Agentic RAG 의 주 사용처).

시장 규모·성장률·정책·도입 장벽처럼 "공신력 있는 수치"가 필요한 질문은 공공기관·국제기구 보고서 코퍼스에서
Agentic RAG 로 찾는다. 같은 세부 분야는 한 번만 분석하고 캐시를 재사용한다.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from core.config import get_segment
from core.llm import structured
from core.prompts import render
from rag.agentic_rag import agentic_rag
from tools.sources import SourceRegistry
from tools.web_search import web_search

AGENT = "market"


class MarketAnalysis(BaseModel):
    market_size: str = Field(description="국내·글로벌 시장 규모와 기준 연도 (수치마다 근거 id)")
    growth: str = Field(description="성장률·전망 (근거 id)")
    demand_drivers: list[str] = Field(description="수요 요인과 고객의 실제 문제 (근거 id)")
    willingness_to_pay: str = Field(description="농가·기업의 지불 의향·도입 사례 (근거 id), 없으면 '확인 불가'")
    policy_regulation: str = Field(description="정책 지원과 규제 (근거 id)")
    adoption_barriers: list[str] = Field(description="도입 장벽·시장 리스크 (근거 id)")
    evidence_ids: list[str]


def market_node(state: dict) -> dict:
    c = state["current"]
    seg = get_segment(c["segment_id"])
    cache = state.get("market_cache", {})
    if seg["id"] in cache:
        msg = f"[시장성] {seg['name']}: 캐시 재사용"
        print(msg)
        return {"market": cache[seg["id"]], "log": [msg]}

    reg = SourceRegistry(state.get("registry"))
    questions = [
        ("국내 스마트농업·스마트팜 시장 규모 (기관별 추정치, 기준 연도)", "국내 시장 규모"),
        (f"{seg['name']} 및 스마트농업·애그테크 글로벌 시장 규모와 연평균 성장률", "글로벌 시장 규모"),
        (f"{seg['name']} 수요 요인: 농가 고령화·인력난·기후변화 등 해결하려는 문제", "수요 근거"),
        ("스마트농업 육성 정책, 지원 제도, 관련 규제", "정책·규제 리스크"),
        (f"{seg['name']} 도입 장벽: 비용, 투자 회수 기간, 농가 수용성", "시장 리스크"),
        ("애그테크 벤처 투자 동향 (투자액 추이, 분야별 비중)", "투자 환경"),
    ]
    ids, traces = [], []
    for q, purpose in questions:
        got, trace = agentic_rag(q, purpose, reg, AGENT)
        ids += got
        traces.append({"agent": AGENT, "question": q, "trace": trace})
    ids += web_search(f"{seg['ko']} 시장 전망 규모", reg, AGENT, topic="news")
    ids = list(dict.fromkeys(ids))
    res: MarketAnalysis = structured(MarketAnalysis).invoke(
        render("market", segment=seg["name"], evidence=reg.brief(ids, 700)))
    out = res.model_dump()
    out["evidence_ids"] = [i for i in dict.fromkeys(res.evidence_ids) if i in set(ids)]
    out["pool_ids"] = ids
    out["segment"] = seg["name"]
    msg = f"[시장성] {seg['name']}: 근거 {len(out['evidence_ids'])}건 (문서 {sum(i.startswith('D') for i in out['evidence_ids'])}건)"
    print(msg)
    return {"registry": reg.data, "market": out, "market_cache": {seg["id"]: out}, "log": [msg],
            "rag_traces": traces}
