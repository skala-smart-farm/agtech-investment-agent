"""📊 시장성 평가 에이전트 (Agentic RAG 의 주 사용처).

에이전트가 검색 방법을 스스로 정한다.
  1) 질의 분해: LLM 이 세부 분야의 시장성 질문을 검색 가능한 하위 질문 3~6개로 나눈다.
  2) 라우팅: LLM 이 하위 질문마다 docs(문서 코퍼스 RAG) / web(뉴스 검색) / both 중 하나를 고르고 이유를 한 줄 남긴다.
     라우터는 코퍼스 문서 목록(기관·연도·제목)을 보고, 코퍼스에 있을 수치인지 최근 뉴스가 필요한지 판단한다.
  3) 실행: docs 는 교정형 RAG 서브그래프(retrieve→grade→rewrite→web 보완), web 은 뉴스 검색(최근 1년 우선).
하위 질문 수(최대 6)와 서브그래프 재작성 횟수(max_rewrites)로 반복 횟수를 제한한다.
분해·라우팅 결정은 rag_traces 에 남겨 실행 기록(run_log)에서 확인할 수 있다. 같은 세부 분야는 한 번만 분석하고 캐시를 재사용한다.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from agents.tech import evidence_blocks
from core.config import get_segment, run_date
from core.llm import structured
from core.prompts import render
from rag.agentic_rag import agentic_rag
from rag.index import get_chunks
from tools.sources import SourceRegistry
from tools.web_search import web_search

AGENT = "market"
MIN_SUBQ, MAX_SUBQ = 3, 6
# LLM 이 하위 질문을 3개 미만으로 내면 채우는 기본 질문 (분해 실패 대비)
DEFAULT_SUBQ = [("국내 스마트농업·스마트팜 시장 규모 (기관별 추정치, 기준 연도)", "시장 규모"),
                ("{seg} 글로벌 시장 규모와 연평균 성장률 (2024년 이후 발표)", "성장률"),
                ("애그테크 벤처 투자 동향 ({prev}년 투자액 증감, 국가별 증감)", "투자 동향")]


class SubQuestion(BaseModel):
    question: str = Field(description="검색할 수 있는 구체 하위 질문 (지표·지역·연도 포함)")
    purpose: str = Field(description="분석 항목: 시장 규모 / 성장률 / 수요 / 지불 의향 / 정책·규제 / 도입 장벽 / 투자 동향")


class Decomposition(BaseModel):
    sub_questions: list[SubQuestion] = Field(description="하위 질문 3~6개")


class Route(BaseModel):
    idx: int = Field(description="하위 질문 번호")
    route: Literal["docs", "web", "both"]
    reason: str = Field(description="이 경로를 고른 이유 한 줄 (어느 문서가 답할 수 있는지 또는 왜 최신 뉴스가 필요한지)")
    web_query: str = Field(description="web·both 일 때 뉴스 검색어 (짧게), docs 면 빈 문자열")


class Routing(BaseModel):
    routes: list[Route]


class MarketAnalysis(BaseModel):
    market_size: str = Field(description="국내·글로벌 시장 규모와 기준 연도·추정 기관 (수치마다 근거 id)")
    growth: str = Field(description="성장률·전망과 발표 연도 (근거 id), 지난 기간 전망은 '(YYYY년 발표 전망)'")
    investment_trend: str = Field(description="최근 애그테크 투자 동향: 연도·증감률·국가별 차이 (근거 id), 없으면 '확인 불가'")
    demand_drivers: list[str] = Field(description="수요 요인과 고객의 실제 문제 (근거 id)")
    willingness_to_pay: str = Field(description="농가·기업의 지불 의향·도입 사례 (근거 id), 없으면 '확인 불가'")
    policy_regulation: str = Field(description="정책 지원과 규제 (근거 id)")
    adoption_barriers: list[str] = Field(description="도입 장벽·시장 리스크 (근거 id)")
    evidence_ids: list[str]


def corpus_catalog() -> str:
    """라우터가 코퍼스에 무엇이 있는지 알도록 문서 목록(기관·연도·제목)을 만든다."""
    docs: dict[str, str] = {}
    for ch in get_chunks():
        m = ch.metadata
        docs.setdefault(m.get("doc_id"), f"- {m.get('publisher')} ({m.get('year')}) {m.get('title')}")
    return "\n".join(docs.values())


def plan(seg: dict, date: str, catalog: str) -> tuple[str, list[SubQuestion], list[Route | None], bool]:
    """큰 질문 → 하위 질문(최대 6개) → 하위 질문별 경로. 분해가 3개 미만이면 기본 질문으로 채운다."""
    question = (f"{seg['name']} 분야 스타트업의 시장성: 국내·글로벌 시장 규모와 성장률, 수요 요인, 지불 의향, "
                f"정책·규제, 도입 장벽, 최근 투자 동향")
    dec: Decomposition = structured(Decomposition).invoke(
        render("market_decompose", segment=seg["name"], question=question, run_date=date, catalog=catalog))
    subs = [s for s in dec.sub_questions if s.question.strip()][:MAX_SUBQ]
    padded = len(subs) < MIN_SUBQ
    for q, purpose in DEFAULT_SUBQ:
        if len(subs) >= MIN_SUBQ:
            break
        subs.append(SubQuestion(question=q.format(seg=seg["name"], prev=int(date[:4]) - 1), purpose=purpose))
    listing = "\n".join(f"[{i}] {s.question} (용도: {s.purpose})" for i, s in enumerate(subs))
    rt: Routing = structured(Routing).invoke(
        render("market_route", segment=seg["name"], run_date=date, catalog=catalog, questions=listing))
    by_idx = {r.idx: r for r in rt.routes}
    return question, subs, [by_idx.get(i) for i in range(len(subs))], padded


def market_node(state: dict) -> dict:
    c = state["current"]
    seg = get_segment(c["segment_id"])
    cache = state.get("market_cache", {})
    if seg["id"] in cache:
        msg = f"[시장성] {seg['name']}: 캐시 재사용"
        print(msg)
        return {"market": cache[seg["id"]], "log": [msg]}

    reg = SourceRegistry(state.get("registry"))
    date = state.get("run_date") or run_date()
    question, subs, routes, padded = plan(seg, date, corpus_catalog())
    traces = [{"agent": AGENT, "question": question, "step": "decompose", "trace": [],
               "sub_questions": [s.question for s in subs], "padded_with_defaults": padded}]
    ids: list[str] = []
    for s, r in zip(subs, routes):
        route = r.route if r else "docs"  # 라우터가 빠뜨린 질문은 공공 문서(코퍼스)를 기본으로
        entry = {"agent": AGENT, "question": s.question, "purpose": s.purpose, "step": "route", "route": route,
                 "reason": r.reason if r else "라우팅 결과 없음 → 문서 코퍼스 기본", "trace": []}
        if route in ("docs", "both"):  # 교정형 RAG 서브그래프 (관련 조각 부족 → 재작성 → 웹 보완)
            got, entry["trace"] = agentic_rag(s.question, s.purpose, reg, AGENT)
            ids += got
        if route in ("web", "both"):   # 뉴스 검색: 최근 1년 우선, 부족하면 기간 제한 해제
            q = (r.web_query.strip() if r and r.web_query.strip() else s.question)[:80]
            got = web_search(q, reg, AGENT, topic="news")
            entry.update(web_query=q, web_results=len(got))
            ids += got
        traces.append(entry)
    ids = list(dict.fromkeys(ids))
    evidence = "\n\n".join(evidence_blocks(reg, ids, keys=[], max_chars=700).values())
    res: MarketAnalysis = structured(MarketAnalysis).invoke(
        render("market", segment=seg["name"], run_date=date, evidence=evidence))
    out = res.model_dump()
    out["evidence_ids"] = [i for i in dict.fromkeys(res.evidence_ids) if i in set(ids)]
    out["pool_ids"] = ids
    out["segment"] = seg["name"]
    n = {k: sum(t["route"] == k for t in traces[1:]) for k in ("docs", "web", "both")}
    msg = (f"[시장성] {seg['name']}: 하위 질문 {len(subs)}개 (문서 {n['docs']}·웹 {n['web']}·둘 다 {n['both']}), "
           f"근거 {len(out['evidence_ids'])}건 (문서 {sum(i.startswith('D') for i in out['evidence_ids'])}건)")
    print(msg)
    return {"registry": reg.data, "market": out, "market_cache": {seg["id"]: out}, "log": [msg],
            "rag_traces": traces}
