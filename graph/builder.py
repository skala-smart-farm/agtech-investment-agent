"""메인 그래프.

discover → verify ─(대기열 있음)→ select ─┬→ tech ──────┐
   ↑           ├─(없음·라운드 남음)→ discover  └→ market ─┐ │
   │           └─(없음·라운드 소진)→ report (적격 후보 없음 보고서)
   │                                          competition ← (tech·market 모두 끝나면)
   │                                               ↓
   │                                            decide ─(투자)→ report → END
   └──────────(보류 · 대기열 비었고 라운드 남음)────┤
                           (보류 · 대기열 남음) → select
                           (보류 · 평가 상한 도달/후보 소진) → report
"""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from agents.competition import competition_node
from agents.decision import decision_node
from agents.discovery import discovery_node
from agents.eligibility import eligibility_node
from agents.market import market_node
from agents.report import report_node
from agents.tech import tech_node
from core.config import get_config
from graph.state import InvestmentState


def select_node(state: dict) -> dict:
    """대기열 맨 앞 후보를 꺼내 평가 대상으로 정하고, 이전 후보의 분석 결과를 비운다."""
    queue = list(state.get("queue", []))
    current = queue.pop(0)
    n = state.get("iterations", 0) + 1
    msg = f"[평가 {n}] {current['official_name']} ({current['region']}, {current['stage']}, {current['segment_id']})"
    print(msg)
    return {"current": current, "queue": queue, "iterations": n, "tech": {}, "market": {}, "competition": {},
            "scorecard": {}, "log": [msg]}


def route_after_verify(state: dict) -> str:
    cfg = get_config()
    if state.get("queue"):
        return "select"
    if state.get("discovery_rounds", 0) < cfg.workflow.max_discovery_rounds:
        return "discover"
    return "report"  # 후보 소진: 평가 결과가 있으면 보류 보고서, 없으면 "적격 후보 없음" 보고서


def route_after_decide(state: dict) -> str:
    cfg = get_config()
    if state.get("decision") == "투자":
        return "report"
    if state.get("iterations", 0) >= cfg.workflow.max_evaluations:  # 반복 상한 (무한 루프 방지)
        return "report"
    if state.get("queue"):
        return "select"
    if state.get("discovery_rounds", 0) < cfg.workflow.max_discovery_rounds:
        return "discover"  # 가이드: 보류면 다른 스타트업 탐색으로 돌아간다
    return "report"        # 모두 보류: 루프 종료 후 보고서 생성


def build_graph():
    g = StateGraph(InvestmentState)
    g.add_node("discover", discovery_node)      # 🔭 스타트업 발굴 에이전트
    g.add_node("verify", eligibility_node)      # ✅ 적격성 검증 에이전트
    g.add_node("select", select_node)           # (제어 노드) 다음 평가 대상 선택
    g.add_node("tech", tech_node)               # 🔬 기술·팀 분석 에이전트
    g.add_node("market", market_node)           # 📊 시장성 평가 에이전트
    g.add_node("competition", competition_node)  # 🥊 경쟁사 비교 에이전트
    g.add_node("decide", decision_node)         # 🧮 투자 판단 에이전트
    g.add_node("report", report_node)           # 📝 보고서 생성 에이전트

    g.add_edge(START, "discover")
    g.add_edge("discover", "verify")
    g.add_conditional_edges("verify", route_after_verify,
                            {"select": "select", "discover": "discover", "report": "report"})
    g.add_edge("select", "tech")                # 기술·팀 분석과 시장성 평가는 서로 독립 → 병렬 실행
    g.add_edge("select", "market")
    g.add_edge(["tech", "market"], "competition")  # 둘 다 끝나면 경쟁사 비교 (기술 차별점 검증에 tech 결과 사용)
    g.add_edge("competition", "decide")
    g.add_conditional_edges("decide", route_after_decide,
                            {"report": "report", "select": "select", "discover": "discover"})
    g.add_edge("report", END)
    return g.compile()
