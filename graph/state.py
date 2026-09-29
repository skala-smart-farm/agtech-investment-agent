"""그래프 전체가 공유하는 State.

- 병렬로 실행되는 노드(기술·팀 분석, 시장성 평가)가 같은 키를 동시에 갱신해도 안전하도록
  누적되는 키에는 Reducer 를 붙였다.
- 후보별 분석 결과(tech/market/competition/scorecard)는 후보가 바뀔 때 select 노드가 초기화한다.
"""
from __future__ import annotations

import operator
from typing import Annotated, TypedDict


def merge_dict(a: dict | None, b: dict | None) -> dict:
    return {**(a or {}), **(b or {})}


def add_unique(a: list | None, b: list | None) -> list:
    out = list(a or [])
    for x in b or []:
        if x not in out:
            out.append(x)
    return out


class InvestmentState(TypedDict, total=False):
    # ── 실행 설정
    domain: str                                        # "AgTech"
    run_date: str                                      # 실행일 (YYYY-MM-DD), 근거 조회일로도 쓴다

    registry: Annotated[dict, merge_dict]              # 근거 저장소 {근거 id: 문서·웹 메타데이터}

    # ── 발굴·검증 (스타트업 발굴 에이전트 → 적격성 검증 에이전트)
    discovery_rounds: int                              # 발굴 라운드 수 (max_discovery_rounds 로 제한)
    raw_candidates: list[dict]                         # 이번 라운드 발굴 후보 (이름, 분야, 발굴 채널, 근거 id)
    seen: Annotated[list[str], add_unique]             # 이미 다룬 후보의 정규화 이름 (중복 발굴 방지)
    screened: Annotated[list[dict], operator.add]      # 적격성 판정 기록 전체 (통과·탈락 사유)
    queue: list[dict]                                  # 적격 판정 후 우선순위 순으로 정렬된 평가 대기열

    # ── 현재 평가 중인 후보와 분석 결과
    current: dict                                      # 후보 프로필 (이름, 투자 단계, 설립연도, 근거 id)
    tech: dict                                         # 기술·팀 분석 결과
    market: dict                                       # 시장성 평가 결과
    market_cache: Annotated[dict, merge_dict]          # 세부 분야별 시장 분석 캐시 (같은 분야 재분석 방지)
    competition: dict                                  # 경쟁사 비교 결과
    scorecard: dict                                    # 평가표 24문항 판정(YES·NO·UNKNOWN), 점수, 판단

    # ── 반복 제어
    iterations: int                                    # 심층 평가한 후보 수 (max_evaluations 로 제한)
    evaluations: Annotated[list[dict], operator.add]   # 후보별 최종 평가 요약 (보고서 비교표에 사용)
    decision: str                                      # 마지막 판단: 투자 또는 보류

    # ── 산출물
    report: dict                                       # 보고서 파일 경로, 페이지 수, 검증 결과
    rag_traces: Annotated[list[dict], operator.add]    # Agentic RAG 수행 기록 (재작성·웹 보완 여부)
    log: Annotated[list[str], operator.add]            # 진행 기록
