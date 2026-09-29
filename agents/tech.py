"""🔬 기술·팀 분석 에이전트.

- 회사 고유 정보(제품, 핵심 기술, 특허·논문, 창업자 이력)는 웹 근거로 모은다.
  창업자는 인물 검색이 아니라 "회사가 알려진 기사·인터뷰 속 창업자 이력"으로 확인한다.
- 기술 수준 비교의 기준선(해당 분야 기술 동향·상용화 수준)은 문서 코퍼스에서 Agentic RAG 로 가져온다.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from core.config import get_segment
from core.llm import structured
from core.prompts import render
from rag.agentic_rag import agentic_rag
from rag.index import get_chunks
from tools.fetch import enrich
from tools.grounding import norm
from tools.sources import SourceRegistry
from tools.web_search import web_search

AGENT = "tech"


class Founder(BaseModel):
    name: str
    role: str
    background: str = Field(description="경력·학위·이전 창업 등, 근거 id 포함")


class TechAnalysis(BaseModel):
    product: str = Field(description="주요 제품·서비스 (근거 id 인용 [W..])")
    core_technology: str = Field(description="핵심 기술과 AI 가 하는 일")
    maturity: Literal["연구", "시제품", "실증", "상용", "확인 불가"]
    maturity_evidence: str
    differentiators: list[str] = Field(description="기술적 차별점, 각 항목 끝에 근거 id")
    weaknesses: list[str] = Field(description="기술 한계·위험, 각 항목 끝에 근거 id")
    ip_evidence: str = Field(description="특허·논문·인증 근거, 없으면 '확인 불가'")
    founders: list[Founder]
    team_assessment: str = Field(description="팀 역량 평가 (근거 id 인용)")
    trend_fit: str = Field(description="문서 코퍼스의 기술 동향 대비 위치 (근거 id 인용 [D..])")
    evidence_ids: list[str] = Field(description="실제로 인용한 근거 id 전체")


def tech_node(state: dict) -> dict:
    c = state["current"]
    reg = SourceRegistry(state.get("registry"))
    name = c["official_name"]
    seg = get_segment(c["segment_id"])
    ids = list(c.get("evidence_ids", []))
    # 평가표 문항별로 필요한 근거를 겨냥한 검색 (창업자 이력·기술 책임자·특허·실증·실적·파트너)
    if c["region"] == "KR":
        queries = [(f"{name} 대표 창업자 이력 인터뷰", True), (f"{name} CTO 연구소장 기술 개발", False),
                   (f"{name} 수상 선정 혁신상 우수기업 출시", False),
                   (f"{name} 특허 등록 기술", False), (f"{name} 실증 농가 효과 수확량 절감", True),
                   (f"{name} 매출 고객 농가 수 설치", True), (f"{name} 협약 계약 공급 농협 지자체 수출", False)]
    else:
        q = c.get("name_en") or name
        queries = [(f"{q} founder CEO background interview", True), (f"{q} CTO technology team", False),
                   (f"{q} patent", False), (f"{q} field trial results yield savings", True),
                   (f"{q} revenue customers farms deployed", True), (f"{q} partnership contract distribution", False)]
    for query, deep in queries:
        ids += web_search(query, reg, AGENT, topic="news", recent=False, deep=deep, raw=True)
    # 회사 기사 원문을 받아 창업자 이력·실적·계약처럼 스니펫에 없는 사실을 보강 (키 불필요)
    enrich(reg, ids, [norm(name), norm(c.get("name_en") or "")], limit=12)
    # 코퍼스(공공 문서)에서 회사 이름이 직접 나오는 조각 (예: 정부 우수기업 선정 목록) → 날짜 있는 제3자 근거
    keys = [k for k in (norm(name), norm(c.get("name_en") or "")) if len(k) >= 2]
    for ch in get_chunks():
        if any(k in norm(ch.page_content) for k in keys):
            ids.append(reg.add_doc(ch.metadata, ch.page_content, agent=AGENT, query=f"코퍼스 속 {name}"))
    rag_ids, trace = agentic_rag(f"{seg['name']} 분야의 기술 동향, 상용화 수준, 기술적 과제",
                                 "스타트업 기술 수준을 비교할 기준선", reg, AGENT)
    ids = list(dict.fromkeys(ids + rag_ids))
    res: TechAnalysis = structured(TechAnalysis).invoke(
        render("tech", name=name, one_line=c.get("one_line", ""), segment=seg["name"], evidence=reg.brief(ids, 650)))
    out = res.model_dump()
    out["evidence_ids"] = [i for i in dict.fromkeys(res.evidence_ids) if i in set(ids)]
    out["pool_ids"] = ids
    msg = f"[기술·팀] {name}: 성숙도 {res.maturity}, 창업자 {len(res.founders)}명, 근거 {len(out['evidence_ids'])}건"
    print(msg)
    return {"registry": reg.data, "tech": out, "log": [msg],
            "rag_traces": [{"agent": AGENT, "question": "기술 동향", "trace": trace}]}
