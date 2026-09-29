"""🥊 경쟁사 비교 에이전트.

기술·팀 분석 결과(차별점 주장)를 받아, 국내·해외 경쟁사와 실제로 비교해 차별성과 진입장벽을 검증한다.
"""
from __future__ import annotations

import re

from pydantic import BaseModel, Field

from core.config import get_segment
from core.llm import structured
from core.prompts import render
from tools.sources import SourceRegistry
from tools.web_search import web_search

AGENT = "competition"


class Competitor(BaseModel):
    name: str
    country: str
    offering: str = Field(description="제품·접근 방식")
    scale: str = Field(description="투자 단계·매출·고객 규모 등 알려진 규모, 모르면 '확인 불가'")
    vs_target: str = Field(description="대상 스타트업과 비교한 강점·약점 (근거 id)")


class CompetitionAnalysis(BaseModel):
    competitors: list[Competitor] = Field(description="3~5곳")
    differentiation: str = Field(description="대상 스타트업의 실제 차별성 판단 (근거 id)")
    entry_barriers: str = Field(description="특허·데이터·네트워크 효과·인증 등 진입장벽 (근거 id), 약하면 약하다고 쓴다")
    threats: list[str] = Field(description="경쟁 위협 (근거 id)")
    evidence_ids: list[str]


def competition_node(state: dict) -> dict:
    c, tech = state["current"], state.get("tech", {})
    reg = SourceRegistry(state.get("registry"))
    seg = get_segment(c["segment_id"])
    name = c["official_name"]
    ids = []
    # 제품 기능이 겹치는 경쟁사를 찾도록 기술·팀 분석의 제품 설명을 검색어에 쓴다 (분야 이름만 쓰면 엉뚱한 기업이 걸림)
    product = re.sub(r"\[[^\]]*\]", "", tech.get("product", ""))[:60]
    ids += web_search(f"{product} 국내 스타트업 경쟁", reg, AGENT, topic="general", recent=False)
    ids += web_search(f"{seg['ko']} 스타트업 경쟁 기업 비교", reg, AGENT, topic="news", recent=False)
    ids += web_search(f"{name} 경쟁사", reg, AGENT, topic="news", recent=False, raw=True)
    ids += web_search(f"{seg['en']} startups competitors", reg, AGENT, topic="general", recent=False)
    ids = list(dict.fromkeys(ids + tech.get("evidence_ids", [])))
    res: CompetitionAnalysis = structured(CompetitionAnalysis).invoke(
        render("competition", name=name, segment=seg["name"], product=tech.get("product", ""),
               differentiators="\n".join(tech.get("differentiators", [])), evidence=reg.brief(ids, 600)))
    out = res.model_dump()
    out["evidence_ids"] = [i for i in dict.fromkeys(res.evidence_ids) if i in set(ids)]
    out["pool_ids"] = ids
    msg = f"[경쟁사] {name}: 경쟁사 {len(res.competitors)}곳 비교"
    print(msg)
    return {"registry": reg.data, "competition": out, "log": [msg]}
