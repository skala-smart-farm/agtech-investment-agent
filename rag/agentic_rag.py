"""Agentic RAG 서브그래프.

retrieve(하이브리드 검색) → grade(Judge LLM 이 조각별 관련성 O/X) →
  관련 조각이 충분하면 finish
  부족하면 rewrite(질의 재작성) 후 다시 retrieve  (최대 max_rewrites 회)
  그래도 부족하면 web(웹 검색으로 보완, Corrective RAG) → finish
"""
from __future__ import annotations

from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from core.config import get_config
from core.llm import structured
from core.prompts import render
from rag.index import get_hybrid_retriever
from tools.sources import SourceRegistry
from tools.web_search import web_search


class RAGState(TypedDict):
    question: str
    purpose: str
    query: str
    retrieved: list[dict]
    relevant: list[dict]
    rewrites: int
    web_ids: list[str]
    evidence_ids: list[str]
    registry: dict
    trace: list[dict]
    agent: str


class Judgment(BaseModel):
    idx: int = Field(description="조각 번호")
    relevant: bool = Field(description="질문에 답하는 데 직접 쓸 수 있는 정보가 있으면 true")


class Grades(BaseModel):
    judgments: list[Judgment]


class Rewrite(BaseModel):
    query: str = Field(description="검색에 더 잘 걸리도록 고친 질의 (핵심 명사·수치 표현 포함)")


def _retrieve(state: RAGState) -> dict:
    cfg = get_config()
    docs = get_hybrid_retriever().invoke(state["query"])[: cfg.rag.candidate_k]
    got = [{"text": d.page_content, "meta": d.metadata} for d in docs]
    return {"retrieved": got}


def _grade(state: RAGState) -> dict:
    cfg = get_config()
    chunks = state["retrieved"]
    if not chunks:
        return {"trace": state["trace"] + [{"query": state["query"], "retrieved": 0, "relevant": 0}]}
    listing = "\n\n".join(
        f"[{i}] ({c['meta'].get('publisher')} {c['meta'].get('year')}, p.{c['meta'].get('page')})\n{c['text'][:900]}"
        for i, c in enumerate(chunks))
    res: Grades = structured(Grades, "judge").invoke(
        render("rag_grade", question=state["question"], purpose=state["purpose"], chunks=listing))
    ok = {j.idx for j in res.judgments if j.relevant}
    seen = {c["meta"]["chunk_id"] for c in state["relevant"]}
    new = [c for i, c in enumerate(chunks) if i in ok and c["meta"]["chunk_id"] not in seen]
    relevant = (state["relevant"] + new)[: cfg.rag.top_k]
    trace = state["trace"] + [{"query": state["query"], "retrieved": len(chunks), "relevant": len(new)}]
    return {"relevant": relevant, "trace": trace}


def _route(state: RAGState) -> Literal["finish", "rewrite", "web"]:
    cfg = get_config()
    if len(state["relevant"]) >= cfg.rag.min_relevant:
        return "finish"
    if state["rewrites"] < cfg.rag.max_rewrites:
        return "rewrite"
    return "web" if cfg.rag.web_fallback else "finish"


def _rewrite(state: RAGState) -> dict:
    tried = [t["query"] for t in state["trace"]]
    res: Rewrite = structured(Rewrite).invoke(
        render("rag_rewrite", question=state["question"], purpose=state["purpose"], tried="\n".join(tried)))
    return {"query": res.query, "rewrites": state["rewrites"] + 1}


def _web(state: RAGState) -> dict:
    reg = SourceRegistry(state["registry"])
    ids = web_search(state["question"], reg, agent=state["agent"], topic="general", recent=False, max_results=4)
    return {"web_ids": ids, "registry": reg.data,
            "trace": state["trace"] + [{"query": state["question"], "web_fallback": len(ids)}]}


def _finish(state: RAGState) -> dict:
    reg = SourceRegistry(state["registry"])
    ids = [reg.add_doc(c["meta"], c["text"], agent=state["agent"], query=state["question"])
           for c in state["relevant"]]
    return {"evidence_ids": ids + state["web_ids"], "registry": reg.data}


def build_rag_graph():
    g = StateGraph(RAGState)
    g.add_node("retrieve", _retrieve)
    g.add_node("grade", _grade)
    g.add_node("rewrite", _rewrite)
    g.add_node("web", _web)
    g.add_node("finish", _finish)
    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges("grade", _route, {"finish": "finish", "rewrite": "rewrite", "web": "web"})
    g.add_edge("rewrite", "retrieve")
    g.add_edge("web", "finish")
    g.add_edge("finish", END)
    return g.compile()


_RAG = None


def agentic_rag(question: str, purpose: str, registry: SourceRegistry, agent: str) -> tuple[list[str], list[dict]]:
    """질문 하나에 대한 근거 id 목록과 수행 기록(trace)을 돌려준다. registry 는 제자리에서 갱신된다."""
    global _RAG
    if _RAG is None:
        _RAG = build_rag_graph()
    out = _RAG.invoke({"question": question, "purpose": purpose, "query": question, "retrieved": [],
                       "relevant": [], "rewrites": 0, "web_ids": [], "evidence_ids": [],
                       "registry": registry.data, "trace": [], "agent": agent})
    registry.data.update(out["registry"])
    return out["evidence_ids"], out["trace"]
