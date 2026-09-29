"""✅ 적격성 검증 에이전트.

과제의 스타트업 기준(비상장 · Seed~Series C · Exit 전)을 "관문(gate)"으로 검사한다.
- 상장 여부: 한국거래소 상장 종목 목록과 이름을 코드로 직접 대조 (LLM 판단에 의존하지 않음)
- 투자 단계·Exit 여부: 투자 뉴스·기업 DB 근거를 모아 Judge LLM 이 판정하고, 판정에는 근거 id 가 반드시 있어야 한다
- 단계가 확인되지 않으면(Unknown) 통과시키지 않는다 (보수적 판단)
통과한 후보는 정보가 풍부하고 최근 신호가 있는 순서로 평가 대기열에 넣는다.
"""
from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor
from typing import Literal

from pydantic import BaseModel, Field

from core.config import get_config
from core.llm import structured
from core.prompts import render
from tools.grounding import STAGE_TERMS, fuzzy_in, norm
from tools.fetch import enrich
from tools.listing_check import is_krx_listed
from tools.nps import as_evidence, summarize
from tools.nps import lookup as nps_lookup
from tools.sources import SourceRegistry, today
from tools.web_search import web_search

AGENT = "eligibility"
ALLOWED = {"Seed", "Pre-A", "Series A", "Pre-B", "Series B", "Pre-C", "Series C"}
Stage = Literal["Seed", "Pre-A", "Series A", "Pre-B", "Series B", "Pre-C", "Series C",
                "Pre-IPO", "Series D 이상", "Unknown"]


class Eligibility(BaseModel):
    official_name: str = Field(description="근거상 공식 회사명")
    founded_year: int = Field(description="설립연도, 모르면 0")
    latest_stage: Stage = Field(description="근거로 확인되는 가장 최근 투자 단계")
    latest_round_date: str = Field(description="최근 라운드 시점 YYYY-MM 또는 YYYY, 모르면 빈 문자열")
    latest_round_amount: str = Field(description="최근 라운드 금액(원문 표기), 모르면 빈 문자열")
    stage_quote: str = Field(description="최근 단계가 적힌 근거 원문 그대로의 짧은 인용 (회사명·단계 표현 포함, 80자 이내). 없으면 빈 문자열")
    stage_evidence_ids: list[str] = Field(description="투자 단계를 뒷받침하는 근거 id")
    listed: bool = Field(description="증권거래소 상장(또는 상장 완료) 근거가 있으면 true")
    exited: bool = Field(description="인수합병·상장 등으로 Exit 이 완료된 근거가 있으면 true")
    exit_evidence_ids: list[str] = Field(description="상장·Exit 판단 근거 id (없으면 빈 목록)")
    ai_core: Literal["YES", "NO", "UNKNOWN"] = Field(
        description="AI·컴퓨터비전·자율주행·로봇·데이터 기반 자동화가 제품 핵심이면 YES, 기술 기업이 아님이 분명하면 NO, 근거 부족이면 UNKNOWN")
    agtech_fit: Literal["YES", "NO", "UNKNOWN"] = Field(
        description="농업·축산 생산성 향상 분야면 YES, 유통·식품·다른 산업이 주력임이 분명하면 NO, 근거 부족이면 UNKNOWN")
    corporate_affiliate: bool = Field(description="상장사·대기업의 자회사·사내 부서·합작사면 true (독립 스타트업 아님)")
    distress: bool = Field(description="법정관리·대규모 구조조정·폐업·사업 중단 등 치명적 신호가 있으면 true")
    distress_note: str = Field(description="치명적 신호 요약, 없으면 빈 문자열")
    one_line: str = Field(description="무엇을 하는 회사인지 한 줄")
    info_richness: int = Field(description="평가에 쓸 공개 정보의 풍부함 0~3")


def _stage_grounded(quote: str, stage: str, ids: list[str], reg: SourceRegistry, names: list[str]) -> list[str]:
    """인용문이 (1) 근거 본문에 있고 (2) 같은 근거에 회사명과 (3) 해당 단계 표현이 함께 있는 근거 id 목록."""
    keys = [norm(n) for n in names if n and len(norm(n)) >= 2]
    terms = STAGE_TERMS.get(stage, [])
    ok = []
    for i in ids:
        s = reg.get(i)
        if not s:
            continue
        text = reg.text(i)
        t = norm(text)
        if fuzzy_in(quote, text) and any(k in t for k in keys) and any(term in t for term in terms):
            ok.append(i)
    return ok


def _check_one(cand: dict, registry: dict) -> tuple[dict, dict]:
    reg = SourceRegistry(registry)
    name, en = cand["name"], cand.get("name_en") or ""
    record = {"name": name, "name_en": en, "region": cand["region"], "segment_id": cand["segment_id"],
              "channels": cand.get("channels", [])}

    listed_krx = is_krx_listed(name, [en]) if cand["region"] == "KR" else None
    if listed_krx:
        record.update(eligible=False, reason="G1 한국거래소 상장 종목 목록에 있음 (비상장 조건 위반)")
        return record, reg.data
    if cand["region"] == "KR" and listed_krx is None:  # 상장 목록을 못 불러오면 통과시키지 않는다
        record.update(eligible=False, reason="G1 상장 목록을 불러오지 못해 비상장 여부 확인 불가 (fail-closed)")
        return record, reg.data

    ids = list(cand.get("evidence_ids", []))
    if cand["region"] == "KR":
        ids += web_search(f"{name} 투자 유치", reg, AGENT, topic="news", recent=False, deep=True, raw=True)
        ids += web_search(f"{name} 시리즈 프리A 시드 투자 단계", reg, AGENT, topic="general", recent=False, deep=True,
                          raw=True)
        ids += web_search(f"{name} 기업정보 설립 투자 단계", reg, AGENT, topic="general", recent=False,
                          include_domains=["thevc.kr", "innoforest.co.kr"], min_results=1)
        # 과거 사건(인수·폐업)은 news 색인보다 general 검색이 잘 찾는다 (실측)
        ids += web_search(f"{name} 인수 경영권 상장", reg, AGENT, topic="general", recent=False)
        ids += web_search(f"{name} 회생 폐업 구조조정 사업 중단", reg, AGENT, topic="general", recent=False)  # G4 반증
    else:
        q = en or name
        ids += web_search(f"{q} raises funding", reg, AGENT, topic="news", deep=True, raw=True)  # 최근 1년 라운드 우선
        ids += web_search(f"{q} Series funding round total raised", reg, AGENT, topic="general", recent=False, deep=True,
                          raw=True)
        ids += web_search(f"{q} acquired acquisition IPO", reg, AGENT, topic="general", recent=False)
        ids += web_search(f"{q} layoffs shut down closes operations", reg, AGENT, topic="general", recent=False)  # G4
    # 국민연금 가입 사업장: 실재·창업 시기(최초 가입일)·현재 인원·탈퇴 여부를 공공데이터로 확인 (키 불필요)
    nps = summarize(nps_lookup([name, en]))
    if ev := as_evidence(name, nps):
        nps["evidence_id"] = reg.add_web(ev, AGENT, "국민연금 가입 사업장 내역", today(), key=f"nps:{norm(name)}")
        ids.append(nps["evidence_id"])
    ids = list(dict.fromkeys(ids))
    # 투자 기사 원문을 받아 단계·금액·날짜 근거를 보강 (검색 스니펫만으로는 단계가 안 보이는 경우가 많음)
    enrich(reg, ids, [norm(name), norm(en)], limit=6)

    res: Eligibility = structured(Eligibility, "judge").invoke(
        render("eligibility", name=name, name_en=en, allowed=", ".join(sorted(ALLOWED)),
               evidence=reg.brief(ids, 600)))
    valid = set(ids)
    stage = res.latest_stage
    # 단계 판정은 인용문이 "회사명 + 단계 표현"이 함께 있는 근거 본문에서 확인될 때만 인정 (다른 회사 기사 혼동 방지)
    stage_ids = _stage_grounded(res.stage_quote, stage, ids, reg, [name, en, res.official_name]) if stage != "Unknown" else []
    if stage != "Unknown" and not stage_ids:
        stage = "Unknown"
    reasons = []
    if res.listed or res.exited:
        reasons.append("G1/G3 상장 또는 Exit 완료 근거 있음")
    if stage not in ALLOWED:
        reasons.append(f"G2 투자 단계 '{stage}' (Seed~Series C 아님 또는 확인 불가)")
    elif not stage_ids:
        reasons.append("투자 단계 근거 id 없음")
    if res.listed or res.exited:
        pass
    if res.corporate_affiliate:
        reasons.append("G5 상장사·대기업 계열 (독립 스타트업 아님)")
    if res.ai_core == "NO":
        reasons.append("G5 AI 가 핵심 기술이 아님")
    if res.agtech_fit == "NO":
        reasons.append("G5 AgTech 분야가 아님")
    if res.distress:
        reasons.append(f"G4 중대한 부정 신호: {res.distress_note}")
    if nps.get("withdrawn"):
        reasons.append("G4 국민연금 사업장 탈퇴 (폐업·휴업 가능성)")
    hosts = {reg.get(i)["url"].split("/")[2].lower() for i in ids if reg.get(i) and reg.get(i)["url"].count("/") >= 2}
    own = [k for k in (norm(en), norm(name)) if len(k) >= 3]
    third = {h for h in hosts if not any(k in norm(h) for k in own)}
    if len(hosts) < 3 or not third:
        reasons.append(f"G6 근거 출처 {len(hosts)}곳(제3자 {len(third)}곳) — 3곳 이상·제3자 1곳 이상 필요")
    record.update(
        official_name=res.official_name or name, founded_year=res.founded_year, stage=stage, stage_quote=res.stage_quote,
        round_date=res.latest_round_date, round_amount=res.latest_round_amount, stage_evidence_ids=stage_ids,
        exit_evidence_ids=[i for i in res.exit_evidence_ids if i in valid], distress=res.distress,
        distress_note=res.distress_note, one_line=res.one_line, info_richness=res.info_richness,
        raw_stage=res.latest_stage, krx_listed=listed_krx, third_party_hosts=sorted(third), nps=nps, evidence_ids=ids, eligible=not reasons, reason="; ".join(reasons) or "통과",
    )
    return record, reg.data


def eligibility_node(state: dict) -> dict:
    cfg = get_config()
    cands = state.get("raw_candidates", [])[: cfg.workflow.max_candidates_per_round]
    snapshot = copy.deepcopy(state.get("registry", {}))
    registry = copy.deepcopy(snapshot)
    records = []
    # 후보별 검색은 서로 독립 → 병렬. 각 작업은 같은 시작 스냅샷의 복사본을 받아 실행 순서와 무관하게 같은 결과를 낸다
    with ThreadPoolExecutor(max_workers=4) as ex:
        for rec, reg_data in ex.map(lambda c: _check_one(c, copy.deepcopy(snapshot)), cands):
            records.append(rec)
            for k, v in reg_data.items():  # 후보 순서대로 병합, 본문을 새로 채운 근거는 갱신
                if k not in registry or (v.get("body") and not registry[k].get("body")):
                    registry[k] = v

    region_rank = {r: i for i, r in enumerate(cfg.domain.region_priority)}
    passed = [r for r in records if r["eligible"]]
    # 공개 정보가 풍부한 후보부터 평가한다 (근거가 적으면 평가표 대부분이 UNKNOWN 이 되어 판단 자체가 어려움)
    for r in passed:
        r["richness_score"] = (r["info_richness"] * 2 + len(r["channels"]) + min(len(r.get("third_party_hosts", [])), 6)
                               + (1 if r.get("nps", {}).get("status") == "matched" else 0))
    passed.sort(key=lambda r: (region_rank.get(r["region"], 9), -r["richness_score"],
                               -(int(r["round_date"][:4]) if r.get("round_date", "")[:4].isdigit() else 0)))
    queue = list(state.get("queue", [])) + passed
    msg = f"[적격성 검증] {len(records)}곳 검사 → 통과 {len(passed)}곳, 탈락 {len(records) - len(passed)}곳"
    print(msg)
    for r in records:
        print(f"   - {r['name']}: {'통과' if r['eligible'] else '탈락'} ({r.get('stage', '-')}) {r['reason']}")
    return {"registry": registry, "screened": records, "queue": queue, "log": [msg]}
