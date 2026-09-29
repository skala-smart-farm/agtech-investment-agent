"""✅ 적격성 검증 에이전트.

과제의 스타트업 기준(비상장 · Seed~Series C · Exit 전)을 "관문(gate)"으로 검사한다.
- 상장 여부: 한국거래소 상장 종목 목록과 이름을 코드로 직접 대조 (LLM 판단에 의존하지 않음)
- 투자 단계·Exit 여부: 투자 뉴스·기업 DB 근거를 모아 Judge LLM 이 판정하고, 판정에는 근거 id 가 반드시 있어야 한다
- 단계가 확인되지 않으면(Unknown) 통과시키지 않는다 (보수적 판단)
- 검증 검색이 실패(한도 초과 등)하면 "반증 없음"이 아니라 "확인 불가"로 탈락시킨다 (G1 과 같은 fail-closed)
- AI·AgTech 적합성(G5)은 둘 다 YES 로 확인될 때만 통과 (UNKNOWN 도 탈락)
검증 슬롯과 평가 대기열은 지역별 몫(국내·해외)으로 나눠, 해외 후보가 국내 뒤로 밀려 빠지지 않게 한다.
통과한 후보는 지역 안에서 정보가 풍부하고 최근 신호가 있는 순서로 평가 대기열에 넣는다.
"""
from __future__ import annotations

import copy
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Literal

from pydantic import BaseModel, Field

from core.config import get_config
from core.llm import structured
from core.prompts import render
from tools.channels import tips_profile
from tools.grounding import STAGE_TERMS, fuzzy_in, norm
from tools.fetch import enrich
from tools.listing_check import is_krx_listed, normalize
from tools.nps import as_evidence, summarize
from tools.nps import lookup as nps_lookup
from tools.sources import SourceRegistry, today
from tools.web_search import FAILED_QUERIES, web_search

AGENT = "eligibility"
ALLOWED = {"Seed", "Pre-A", "Series A", "Pre-B", "Series B", "Pre-C", "Series C"}
# 구조화 스냅샷 근거(국민연금 조회·TIPS 목록): 검색이 모두 실패해도 생기므로 G6 출처 수에서 뺀다
SNAPSHOT_KEYS = ("nps:", "tips:")
STALE_MONTHS = 60  # 최근 라운드가 기준일보다 5년 넘게 전이면 기록만 남긴다 (관문 판정은 그대로)
Stage = Literal["Seed", "Pre-A", "Series A", "Pre-B", "Series B", "Pre-C", "Series C",
                "Pre-IPO", "Series D 이상", "Unknown"]


class Eligibility(BaseModel):
    official_name: str = Field(description="근거상 공식 회사명")
    ceo: str = Field(description="근거에 적힌 대표(CEO) 이름, 없으면 빈 문자열")
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
        description="농업·축산 생산 현장의 생산성 향상 기술이면 YES, 유통·중개·거래 플랫폼·수산 양식·식품 가공·배달·"
                    "다른 산업이 주력이면 NO, 근거 부족이면 UNKNOWN")
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


def _search(query: str, reg: SourceRegistry, gate: str, tried: list, **kw) -> list[str]:
    """검증 검색 + 실패 기록. tried 에 (관문, 쿼리, 실패 여부)를 붙인다.
    실패 여부는 호출 전후 FAILED_QUERIES 에 이 쿼리가 새로 붙었는지로 안다 (후보별 병렬 실행이라 쿼리로 구분)."""
    n0 = len(FAILED_QUERIES)
    ids = web_search(query, reg, AGENT, **kw)
    tried.append((gate, query, any(f["query"] == query and f["agent"] == AGENT for f in FAILED_QUERIES[n0:])))
    return ids


def _age_months(round_date: str, base: str) -> int | None:
    """최근 라운드(YYYY-MM 또는 YYYY)가 기준일(YYYY-MM-DD)보다 몇 달 전인지. 연도만 있으면 그해 12월로 본다."""
    m = re.match(r"((?:19|20)\d{2})(?:\s*[-./년]\s*(\d{1,2}))?", round_date or "")
    if not m:
        return None
    return (int(base[:4]) - int(m.group(1))) * 12 + int(base[5:7]) - int(m.group(2) or 12)


def region_quota_order(items: list[dict]) -> list[dict]:
    """지역별 몫(workflow.screen_kr : screen_global) 비율로 섞은 순서. 같은 지역 안의 순위는 그대로 둔다.
    국내를 통째로 앞에 두면 해외 후보가 검증·평가 슬롯에 오지 못한다(구조적 누락). 앞 screen_kr + screen_global 개는
    지역별 몫만큼 채워지고, 한 지역 후보가 모자라면 다른 지역의 다음 순위가 남은 자리를 채운다."""
    cfg = get_config()
    quota = {"KR": cfg.workflow.screen_kr, "GLOBAL": cfg.workflow.screen_global}
    rank = {r: i for i, r in enumerate(cfg.domain.region_priority)}
    seen: dict[str, int] = {}
    keyed = []
    for pos, it in enumerate(items):
        j, share = seen.get(it["region"], 0), quota.get(it["region"], 0)
        seen[it["region"]] = j + 1
        keyed.append((j / share if share else float("inf"), rank.get(it["region"], 9), pos))
    return [items[pos] for *_, pos in sorted(keyed)]


def _check_one(cand: dict, registry: dict) -> tuple[dict, dict]:
    reg = SourceRegistry(registry)
    name, en = cand["name"], cand.get("name_en") or ""
    # TIPS 공개 목록의 대표자·설립일 (창업 시기는 국민연금 최초 가입일이 아니라 설립일로 본다)
    tips = (tips_profile([name]) if cand["region"] == "KR" else None) or {}
    record = {"name": name, "name_en": en, "region": cand["region"], "segment_id": cand["segment_id"],
              "channels": cand.get("channels", []), "founded_date": tips.get("founded") or None,
              "ceo": tips.get("ceo") or None, "gate_search_failed": False}

    listed_krx = is_krx_listed(name, [en]) if cand["region"] == "KR" else None
    if listed_krx:
        record.update(eligible=False, reason="G1 한국거래소 상장 종목 목록에 있음 (비상장 조건 위반)")
        return record, reg.data
    if cand["region"] == "KR" and listed_krx is None:  # 상장 목록을 못 불러오면 통과시키지 않는다
        record.update(eligible=False, reason="G1 상장 목록을 불러오지 못해 비상장 여부 확인 불가 (fail-closed)")
        return record, reg.data

    ids = list(cand.get("evidence_ids", []))
    tried: list[tuple[str, str, bool]] = []  # (관문, 쿼리, 검색 실패 여부)
    if cand["region"] == "KR":
        ids += _search(f"{name} 투자 유치", reg, "G2", tried, topic="news", recent=False, deep=True, raw=True)
        ids += _search(f"{name} 시리즈 프리A 시드 투자 단계", reg, "G2", tried, topic="general", recent=False, deep=True,
                       raw=True)
        ids += _search(f"{name} 기업정보 설립 투자 단계", reg, "G2", tried, topic="general", recent=False,
                       include_domains=["thevc.kr", "innoforest.co.kr"], min_results=1)
        # 과거 사건(인수·폐업)은 news 색인보다 general 검색이 잘 찾는다 (실측)
        ids += _search(f"{name} 인수 경영권 상장", reg, "G3", tried, topic="general", recent=False)
        ids += _search(f"{name} 회생 폐업 구조조정 사업 중단", reg, "G4", tried, topic="general", recent=False)  # G4 반증
    else:
        q = en or name
        ids += _search(f"{q} raises funding", reg, "G2", tried, topic="news", deep=True, raw=True)  # 최근 1년 라운드 우선
        ids += _search(f"{q} Series funding round total raised", reg, "G2", tried, topic="general", recent=False,
                       deep=True, raw=True)
        ids += _search(f"{q} acquired acquisition IPO", reg, "G3", tried, topic="general", recent=False)
        ids += _search(f"{q} layoffs shut down closes operations", reg, "G4", tried, topic="general", recent=False)  # G4
    failed = [(gate, q) for gate, q, bad in tried if bad]
    record["failed_searches"] = [q for _, q in failed]
    if len(failed) == len(tried):  # 회사 검증 검색이 전부 실패: 판정할 근거가 없으므로 LLM 판정 없이 탈락 (fail-closed)
        record.update(eligible=False, gate_search_failed=True, evidence_ids=list(dict.fromkeys(ids)),
                      reason="G0 검증 검색 실패(한도 초과 등) — 확인 불가")
        return record, reg.data
    # 국민연금 가입 사업장(국내 법인만): 실재·첫 고용 시점(최초 가입일)·현재 인원·탈퇴 여부를 공공데이터로 확인 (키 불필요)
    nps = summarize(nps_lookup([name, en])) if cand["region"] == "KR" else {"status": "not_applicable", "ym": ""}
    if ev := as_evidence(name, nps):
        nps["evidence_id"] = reg.add_web(ev, AGENT, "국민연금 가입 사업장 내역", today(), key=f"nps:{norm(name)}")
        ids.append(nps["evidence_id"])
    ids = list(dict.fromkeys(ids))
    # 투자 기사 원문을 받아 단계·금액·날짜 근거를 보강 (검색 스니펫만으로는 단계가 안 보이는 경우가 많음)
    enrich(reg, ids, [norm(name), norm(en)], limit=6)

    res: Eligibility = structured(Eligibility, "judge").invoke(
        render("eligibility", name=name, name_en=en, allowed=", ".join(sorted(ALLOWED)),
               domain=get_config().domain.description, evidence=reg.brief(ids, 600)))
    valid = set(ids)
    stage = res.latest_stage
    # 단계 판정은 인용문이 "회사명 + 단계 표현"이 함께 있는 근거 본문에서 확인될 때만 인정 (다른 회사 기사 혼동 방지)
    stage_ids = _stage_grounded(res.stage_quote, stage, ids, reg, [name, en, res.official_name]) if stage != "Unknown" else []
    if stage != "Unknown" and not stage_ids:
        stage = "Unknown"
    reasons = []
    if any(gate in ("G3", "G4") for gate, _ in failed):  # 반증 검색이 실패했으면 "부정 근거 없음"이 아니다
        record["gate_search_failed"] = True
        reasons.append("G3/G4 반증 검색 실패 — 확인 불가")
    if res.listed or res.exited:
        reasons.append("G1/G3 상장 또는 Exit 완료 근거 있음")
    if stage not in ALLOWED:
        miss = sum(gate == "G2" for gate, _ in failed)
        reasons.append(f"G2 투자 단계 '{stage}' (Seed~Series C 아님 또는 확인 불가)"
                       + (f" — 투자 검색 {miss}건 실패" if miss else ""))
    elif not stage_ids:
        reasons.append("투자 단계 근거 id 없음")
    if res.corporate_affiliate:
        reasons.append("G5 상장사·대기업 계열 (독립 스타트업 아님)")
    if res.ai_core == "NO":
        reasons.append("G5 AI 가 핵심 기술이 아님")
    if res.agtech_fit == "NO":
        reasons.append("G5 AgTech 분야가 아님")
    if "NO" not in (res.ai_core, res.agtech_fit) and "UNKNOWN" in (res.ai_core, res.agtech_fit):
        reasons.append("G5 AI 농업 생산성 기술 확인 불가")  # 근거가 빈약할수록 통과하던 fail-open 을 막는다
    if res.distress:
        reasons.append(f"G4 중대한 부정 신호: {res.distress_note}")
    if nps.get("withdrawn"):
        reasons.append("G4 국민연금 사업장 탈퇴 (폐업·휴업 가능성)")
    # G6 최소 근거량: 회사명이 제목·스니펫·본문에 실제로 나오는 웹 근거만 센다. 구조화 스냅샷은 제외
    # (동명·무관 검색 결과나 국민연금 조회 결과가 출처 수를 채우지 않게)
    keys = [k for k in dict.fromkeys(norm(normalize(n)) for n in (name, en, res.official_name)) if len(k) >= 2]
    named = [s for s in map(reg.get, ids) if s and s["kind"] in ("web", "doc") and not s["key"].startswith(SNAPSHOT_KEYS)
             and any(k in norm(reg.text(s["id"])) for k in keys)]
    hosts = {s["url"].split("/")[2].lower() for s in named if s["url"].count("/") >= 2}
    own = [k for k in (norm(en), norm(name)) if len(k) >= 3]
    third = {h for h in hosts if not any(k in norm(h) for k in own)}
    if len(hosts) < 3 or not third:
        reasons.append(f"G6 회사명이 나오는 근거 출처 {len(hosts)}곳(제3자 {len(third)}곳) — 3곳 이상·제3자 1곳 이상 필요")
    # 최근 라운드가 5년 넘게 전이면 관문은 그대로 두고 기록만 남긴다 (현재 단계가 그 뒤로 바뀌었을 수 있음)
    age = _age_months(res.latest_round_date, today())
    stale = age is not None and age > STALE_MONTHS
    reason = "; ".join(reasons) or "통과"
    if stale:
        reason += f" (참고: 최근 라운드 {res.latest_round_date} — 기준일보다 5년 넘게 전)"
    # 대표자: TIPS 목록 우선, 없으면 LLM 이 근거에서 찾은 이름이 실제 근거 본문에 있을 때만
    ceo_key = norm(res.ceo)
    if not record["ceo"] and len(ceo_key) >= 2 and any(ceo_key in norm(reg.text(i)) for i in ids):
        record["ceo"] = res.ceo.strip()
    founded_year = int(record["founded_date"][:4]) if record["founded_date"] else res.founded_year
    record.update(
        official_name=res.official_name or name, founded_year=founded_year, stage=stage, stage_quote=res.stage_quote,
        round_date=res.latest_round_date, round_amount=res.latest_round_amount, stage_evidence_ids=stage_ids,
        exit_evidence_ids=[i for i in res.exit_evidence_ids if i in valid], distress=res.distress,
        distress_note=res.distress_note, one_line=res.one_line, info_richness=res.info_richness,
        raw_stage=res.latest_stage, krx_listed=listed_krx, third_party_hosts=sorted(third), nps=nps, evidence_ids=ids,
        round_stale=stale, eligible=not reasons, reason=reason,
    )
    return record, reg.data


def eligibility_node(state: dict) -> dict:
    cfg = get_config()
    # 검증 슬롯을 지역별 몫으로 나눈다 (발굴 대기열이 이미 같은 순서면 그대로)
    cands = region_quota_order(state.get("raw_candidates", []))[: cfg.workflow.max_candidates_per_round]
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

    passed = [r for r in records if r["eligible"]]
    # 공개 정보가 풍부한 후보부터 평가한다 (근거가 적으면 평가표 대부분이 UNKNOWN 이 되어 판단 자체가 어려움)
    for r in passed:
        r["richness_score"] = (r["info_richness"] * 2 + len(r["channels"]) + min(len(r.get("third_party_hosts", [])), 6)
                               + (1 if r.get("nps", {}).get("status") == "matched" else 0))
    passed.sort(key=lambda r: (-r["richness_score"],
                               -(int(r["round_date"][:4]) if r.get("round_date", "")[:4].isdigit() else 0)))
    # 평가 대기열도 지역별 몫 비율로 섞는다 (평가 상한 안에 해외 후보가 들어오게)
    queue = list(state.get("queue", [])) + region_quota_order(passed)
    n_failed = sum(r.get("gate_search_failed", False) for r in records)
    msg = (f"[적격성 검증] {len(records)}곳(국내 {sum(r['region'] == 'KR' for r in records)}·해외 "
           f"{sum(r['region'] != 'KR' for r in records)}) 검사 → 통과 {len(passed)}곳, 탈락 {len(records) - len(passed)}곳"
           + (f" (검증 검색 실패로 확인 불가 {n_failed}곳)" if n_failed else ""))
    print(msg)
    for r in records:
        print(f"   - {r['name']}: {'통과' if r['eligible'] else '탈락'} ({r.get('stage', '-')}) {r['reason']}")
    return {"registry": registry, "screened": records, "queue": queue, "log": [msg]}
