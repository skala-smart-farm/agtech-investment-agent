"""📝 보고서 생성 에이전트.

과제 조건과 평가 결과와의 일치를 코드로 검증한다.
- 맨 앞 SUMMARY: 상황 → 결론 → 근거 → 조건(→ 요청). 결론·요청 줄은 코드가 평가 결과로 쓴다. 개요 문장 금지,
  A4 절반 이내(렌더링 후 높이 측정)
- 투자 후보가 있으면 그 회사의 투자 검토 보고서. 모두 보류면 "후보마다 왜 안 되는지"로 구성한다
  (교수님: 모두 안 되면 종료하고, 그 보고서는 각 후보가 왜 안 되는지 그 이유들로 구성)
  → 후보 풀과 선정 과정 / 후보별 보류 사유 / 최고점 후보 상세 / 투자 판단 요약 / 한계점
- 본문과 표가 평가표와 어긋나면(예: 특허 문항이 UNKNOWN 인데 "특허 보유", C2 가 YES 가 아닌데 "앞선다") 다시 쓰게 한다
- 맨 끝 REFERENCE: 본문에 실제로 인용된 근거만, 기관 보고서 / 학술 논문 / 웹페이지로 나눠 지정 형식으로
- 전체 5쪽 이내: 넘으면 조판 밀도를 높이고, 그래도 넘으면 줄여 쓴다
"""
from __future__ import annotations

import json
import re
import shutil
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from core.config import get_config, get_segment, path
from core.llm import structured
from core.prompts import render
from report.render import html_to_pdf, render_html
from tools import web_search as search_tool
from tools.grounding import norm
from tools.sources import (GROUPS, SourceRegistry, citable, format_reference, merge_duplicates, reference_group,
                           reference_key, title_key)

AGENT = "report"
CITE = re.compile(r"\[\s*([WD][0-9a-f]{5}(?:\s*[,，]\s*[WD][0-9a-f]{5})*)\s*\]")
BANNED = ["본 보고서", "이 보고서", "보고서는", "보고서에서", "보고서의 목적", "판단하는 보고서", "평가하는 보고서",
          "목적으로 작성", "살펴보", "다음과 같", "개요", "소개하", "UNKNOWN"]  # UNKNOWN 은 내부 용어 → "미확인"
EVALUATIVE = ["우수", "탁월", "뛰어난", "선도적", "독보적", "혁신적인", "압도적"]
# 평가표에서 확인되지 않은 사실을 본문이 단정하는지 찾는 규칙: (문항, 패턴, 설명)
CLAIMS = [
    ("C1", r"특허[^.。\n]{0,15}(보유|확인|등록|출원|확보)", "특허"),
    ("R1", r"매출[^.。\n]{0,12}(\d[\d,.]*\s*(억|만|천|원|달러|%)|기록|달성)", "매출"),  # "제3자" 같은 숫자는 제외
    ("R2", r"(유료 고객|고객 수|설치 농가|도입 농가)[^.。\n]{0,12}\d[\d,.]*\s*(곳|명|개|농가|호|대|ha|㎡|%|만|천|억)",
     "고객·설치 규모"),
    ("P4", r"(인허가|검정|인증)[^.。\n]{0,10}(획득|취득|받았|완료)", "인허가·검정"),
    ("C2", r"차별화(된다|되어 있|를 갖|했다)|경쟁 우위(를|가)? ?(확보|갖)", "경쟁사 대비 차별성"),
    ("P1", r"상용 (운영|판매)[^.。\n]{0,6}(중이|하고 있)", "상용 운영"),
]
NEGATION = re.compile(r"않|불가|미확인|없|못|필요|여부|되면|이면|하면|경우|예정|계획")  # 부정·조건(재검토 조건)은 단정이 아니다
SUPERIORITY = re.compile(r"앞선다|앞서 있|우위|차별화된다|차별화되어")  # C2 가 YES 가 아니면 우열 단정 금지
# 판정 단계의 내부 기록(강등 사유)을 임원이 읽을 짧은 이유로 바꾼다
DEMOTED = {"인용문이 근거 본문에서 확인되지 않음": "근거 원문에서 해당 내용이 확인되지 않음",
           "인용 주변": "근거가 이 회사가 아닌 업계 일반 내용임",
           "공공·연구기관 문서 근거 없음": "공공·연구기관 문서 근거 없음",
           "회사 자체 발표만 있음": "회사 자체 발표만 있고 제3자 근거 없음",
           "최근 24개월 이내 근거 아님": "최근 24개월 이내 근거 없음",
           "언급된 사건 날짜가 모두": "언급된 사건이 모두 24개월보다 오래됨"}


class Risk(BaseModel):
    type: Literal["시장", "기술", "규제", "경쟁"]
    content: str = Field(description="리스크 내용 (근거 id 인용)")
    evidence_ids: list[str] = Field(description="리스크 근거 id (1개 이상)")
    due_diligence: str = Field(description="실사에서 확인할 항목 또는 투자 조건")


class CompetitorNote(BaseModel):
    name: str = Field(description="경쟁사 표의 회사명 그대로")
    vs_target: str = Field(description="대상과 비교한 사실 한 문장 (근거 id). C2 가 YES 가 아니면 우열 표현 없이 차이만")


class CandidateNote(BaseModel):
    name: str = Field(description="심층 평가한 후보 회사명 그대로")
    business: str = Field(description="사업 한 줄 요약 (한국어, 50자 이내, 근거가 영어여도 한국어로)")
    why_not: str = Field(description="보류 사유 2~3문장: 결정적 반대 근거와 확인되지 않은 핵심 항목 (근거 id)")
    recheck: str = Field(description="재검토 조건 한 문장")


class _Body(BaseModel):
    situation: str = Field(description="SUMMARY 상황: 현재 상태 한 문장 (근거 id, 110자 이내)")
    key_evidence: str = Field(description="SUMMARY 근거: 결론을 가른 확인된 사실과 확인되지 않은 핵심 정보 대비 (근거 id, 160자 이내)")
    condition: str = Field(description="SUMMARY 조건: 투자 조건 또는 재검토 조건 (120자 이내)")
    problem: str = Field(description="사업 아이디어: 해결하는 문제 (근거 id)")
    product: str = Field(description="사업 아이디어: 제품·핵심 컨셉 (근거 id)")
    revenue_model: str = Field(description="사업 아이디어: 수익 방식, 확인 안 되면 '확인 불가'")
    market: str = Field(description="시장 규모·성장성 (국내·글로벌 수치는 [D..] 문서 인용)")
    tech_team: str = Field(description="기술력과 팀: 대표·핵심 인력 이름과 경력, 설립일, 핵심 기술 (근거 id)")
    industry_baseline: str = Field(description="업계 기술 수준 대비 위치 한 문장 ([D..] 문서 인용 필수)")
    competition: str = Field(description="경쟁 구도 요약 (표는 코드가 붙임)")
    competitor_notes: list[CompetitorNote] = Field(description="경쟁사 표의 회사마다 대상과 비교한 한 문장")
    risks: list[Risk] = Field(description="시장·기술·규제·경쟁 리스크 각 1개 이상")
    data_limits: list[str] = Field(description="이 회사·시장 데이터의 한계 1개")


class Draft(_Body):
    """투자 권고 보고서 문안."""
    decision_rationale: str = Field(description="투자 판단 근거와 조건")


class HoldDraft(_Body):
    """모두 보류 보고서 문안: 후보별 보류 사유가 더해진다."""
    candidates: list[CandidateNote] = Field(description="심층 평가한 후보마다 보류 사유와 재검토 조건")


# ── 후보 풀·선정 과정 (코드가 State 로 계산)

def _same(a: str, b: str) -> bool:
    x, y = norm((a or "").split("(")[0]), norm((b or "").split("(")[0])
    return bool(x) and bool(y) and (x == y or (min(len(x), len(y)) >= 2 and (x in y or y in x)))


def _names(e: dict) -> list[str]:
    p = e.get("profile") or {}
    return [x for x in (e.get("name"), p.get("official_name"), p.get("name"), p.get("name_en")) if x]


def _pool(state: dict, evals: list[dict], screened: list[dict], cfg, invested: bool) -> dict:
    """발굴 → 검증 → 적격 → 심층 평가 단계별 후보 수와, 적격인데 평가하지 않은 후보."""
    def split(rows, key="region"):
        return sum(r.get(key) == "KR" for r in rows), sum(r.get(key) != "KR" for r in rows)

    rounds = state.get("discovery_rounds") or 1
    counts = [int(m.group(1)) for x in state.get("log", []) if (m := re.match(r"\[발굴 \d+라운드\].*?후보 (\d+)곳", x))]
    raw = state.get("raw_candidates") or []
    discovered = sum(counts) if counts else (len(raw) if rounds == 1 else None)
    disc_split = split(raw) if rounds == 1 and raw and discovered == len(raw) else (None, None)
    eligible = [r for r in screened if r.get("eligible")]
    done = [n for e in evals for n in _names(e)]
    left = [r for r in eligible if not any(_same(r.get("official_name") or r["name"], n) or _same(r["name"], n)
                                           for n in done)]
    if invested:
        why = "투자 권고 후보가 나와 평가 종료"
    elif len(evals) >= cfg.workflow.max_evaluations:
        why = f"평가 상한({cfg.workflow.max_evaluations}곳, 비용 관리) 도달"
    else:
        why = "평가 전 종료"
    return {
        "rounds": rounds, "discovered": discovered, "disc_split": disc_split,
        "screened": len(screened), "scr_split": split(screened),
        "rejected": [(r.get("official_name") or r["name"], re.sub(r"^G\d(?:/G\d)?\s*", "", r.get("reason") or ""))
                     for r in screened if not r.get("eligible")],
        "eligible": len(eligible), "el_split": split(eligible),
        "evaluated": len(evals), "ev_split": split([e.get("profile") or e for e in evals]),
        "unevaluated": [{"name": r.get("official_name") or r["name"], "stage": r.get("stage") or "-",
                         "search_failed": bool(r.get("gate_search_failed"))} for r in left],
        "why": why, "per_round": cfg.workflow.max_candidates_per_round,
    }


def _pool_rows(p: dict) -> list[dict]:
    def n(x):
        return "-" if x is None else x

    rej = "; ".join(f"{name} — {reason}" for name, reason in p["rejected"])
    return [
        {"stage": "발굴", "total": n(p["discovered"]), "kr": n(p["disc_split"][0]), "gl": n(p["disc_split"][1]),
         "note": f"발굴 {p['rounds']}라운드" + (" (라운드 합계, 중복 포함)" if p["rounds"] > 1 else "")
                 + f", 라운드당 최대 {p['per_round']}곳을 적격성 검증으로 넘김"},
        {"stage": "적격성 검증", "total": p["screened"], "kr": p["scr_split"][0], "gl": p["scr_split"][1],
         "note": f"탈락 {len(p['rejected'])}곳" + (f": {rej}" if rej else "")},
        {"stage": "적격 (비상장·Seed~C·Exit 전)", "total": p["eligible"], "kr": p["el_split"][0], "gl": p["el_split"][1],
         "note": f"미평가 {len(p['unevaluated'])}곳" if p["unevaluated"] else "모두 심층 평가"},
        {"stage": "심층 평가", "total": p["evaluated"], "kr": p["ev_split"][0], "gl": p["ev_split"][1],
         "note": "기술·시장·경쟁 분석 후 평가표 24문항 판정"},
    ]


def _conclusion(target: dict, pool: dict, any_invest: bool) -> str:
    sc = target["scorecard"]
    if any_invest:
        return (f"투자 권고 — {target['name']} 총점 {sc['total']}/100 (기준 {sc['threshold']}점), "
                f"핵심 항목·Deal-killer·정보량 기준 충족")
    k, m, rest = pool["evaluated"], pool["eligible"], len(pool["unevaluated"])
    scope = f"적격 {m}곳 중 {k}곳 평가; 미평가 {rest}곳은 {pool['why']}" if rest else f"적격 {m}곳 모두 평가"
    return f"투자 권고 없음 — 심층 평가 {k}곳 모두 보류 ({scope})"


def _request(pool: dict) -> str:
    rest = len(pool["unevaluated"])
    return "재검토 조건 충족 시 재평가 승인 여부" + (f", 미평가 적격 {rest}곳 추가 심층 평가 승인 여부" if rest else "")


def _situation(pool: dict) -> str:
    """모두 보류일 때 SUMMARY 상황 줄: 평가 범위를 코드가 사실대로 쓴다."""
    ev_kr, ev_gl = pool["ev_split"]
    disc = f"{pool['discovered']}곳을 발굴해 " if pool.get("discovered") else ""
    return (f"국내외 AgTech AI 스타트업 {disc}{pool['screened']}곳을 검증했고, 적격 {pool['eligible']}곳 중 "
            f"{pool['evaluated']}곳(국내 {ev_kr}·해외 {ev_gl})을 심층 평가했다.")


def summary_lines(draft: _Body, conclusion: str, request: str = "", situation: str | None = None) -> list[str]:
    """SUMMARY 4~5줄. 결론·요청 줄(모두 보류면 상황 줄도)은 코드가 평가 결과로 쓴다 (순서·수치 오류 방지)."""
    def bare(x: str) -> str:  # LLM 이 칸 이름을 앞에 또 붙인 경우 ("재검토 조건: …")
        return re.sub(r"^\s*(상황|근거|조건|재검토 조건|투자 조건)\s*[:：]\s*", "", x)

    return [f"상황: {situation or bare(draft.situation)}", f"결론: {conclusion}", f"근거: {bare(draft.key_evidence)}",
            f"조건: {bare(draft.condition)}"] + ([f"요청: {request}"] if request else [])


# ── 검사: SUMMARY 규칙, 평가표와의 일치

def _summary_problems(lines: list[str], limit: int) -> list[str]:
    text = " ".join(lines)
    probs = [f"SUMMARY 에 금지 표현 '{b}'" for b in BANNED if b in text]
    plain = CITE.sub("", text)
    if len(plain) > limit:
        probs.append(f"SUMMARY {len(plain)}자 > {limit}자 (각 칸 길이 상한을 지켜라)")
    return probs


SUP_NEGATION = re.compile(r"않|없|미확인|불가|못|확인되지")  # 우열 단정을 부정하는 말만 (조건·필요는 면제하지 않음)


def _superiority(text: str) -> re.Match | None:
    return next((m for m in SUPERIORITY.finditer(text) if not SUP_NEGATION.search(text[m.end(): m.end() + 12])), None)


def _claim_problems(text: str, verdict: dict, who: str = "") -> list[str]:
    """평가표에서 YES 가 아닌 사실을 단정하는 문장."""
    probs = []
    for qid, pat, label in CLAIMS:
        if verdict.get(qid) == "YES":
            continue
        for m in re.finditer(pat, text):
            if not NEGATION.search(text[m.start(): m.end() + 12]):
                probs.append(f"{who}'{m.group(0)}' — {qid}({label})는 평가표에서 {verdict.get(qid)} 이므로 확인됐다고 쓰지 마라")
                break
    if verdict.get("C2") != "YES" and (m := _superiority(text)):
        probs.append(f"{who}'{m.group(0)}' — 경쟁사 대비 차별성(C2)이 평가표에서 {verdict.get('C2')} 이므로 "
                     f"'앞선다·우위·차별화된다' 같은 우열 표현 없이 차이만 사실로 써라 (경쟁사 표·리스크 표 포함)")
    return probs


def _maturity_problems(situation: str, p1: str | None) -> list[str]:
    """SUMMARY 상황의 성숙도 표현이 평가표 P1(상용 운영)과 맞는지."""
    if p1 == "YES":
        if m := re.search(r"PoC|실증|시제품", situation):
            return [f"SUMMARY 상황의 '{m.group(0)}' — 평가표 P1(상용 운영)이 YES 이므로 성숙도를 평가표와 같게 써라"]
        return []
    for m in re.finditer(r"상용", situation):
        if not re.search(r"예정|계획|목표|앞두|준비|않|미확인|불가|없|이전", situation[m.end(): m.end() + 12]):
            return [f"SUMMARY 상황의 '상용' — 평가표 P1(상용 운영)이 {p1} 이므로 상용 운영 중이라고 쓰지 마라"]
    return []


def _stale_forecasts(text: str, run_date: str) -> list[str]:
    """이미 지난 기간의 전망('2025년 … 전망')을 미래처럼 쓴 문장. '(2022년 발표 전망)'처럼 발표 시점을 밝히면 허용."""
    year, out = int(run_date[:4]), []
    for sent in re.split(r"(?<=[.。])\s+|(?<=다)\s+|\n", CITE.sub("", text)):
        if "발표" in sent or re.search(r"확인 불가|미확인", sent):
            continue
        for m in re.finditer(r"전망", sent):
            years = [int(y) for y in re.findall(r"(20\d{2})년?(?!\s*(?:기준|에서|대비))", sent[max(0, m.start() - 60): m.start()])]
            if years and max(years) < year:
                out.append(f"'{sent.strip()[:40]}…' — {max(years)}년은 이미 지났다. 지난 기간의 전망이면 "
                           f"'(20XX년 발표 전망)'처럼 발표 시점을 밝히거나 최신 수치로 바꿔라")
                break
    return out[:3]


def _consistency_problems(draft: _Body, target: dict, evals: list[dict], run_date: str) -> list[str]:
    """본문·표가 평가표와 어긋나는 단정, 평가성 형용사, 지난 전망, 근거 없는 리스크·시장 수치를 찾는다."""
    sc = target["scorecard"]
    verdict = {r["qid"]: r["answer"] for r in sc["rows"]}
    fields = [draft.situation, draft.key_evidence, draft.problem, draft.product, draft.revenue_model, draft.market,
              draft.tech_team, draft.industry_baseline, draft.competition]
    if isinstance(draft, Draft):
        fields.append(draft.decision_rationale)
    tables = [c.vs_target for c in draft.competitor_notes] + [r.content for r in draft.risks]  # 실사 항목 칸은 확인할 일이라 제외
    text = "\n".join(fields + tables)
    notes = ""
    if isinstance(draft, HoldDraft):
        body = "\n".join(fields[2:] + tables)  # 최고점 후보 상세(3장)는 그 후보의 평가표로
        probs = _claim_problems(body, verdict)
        verdicts = {e["name"]: {r["qid"]: r["answer"] for r in e["scorecard"]["rows"]} for e in evals}
        # 후보 이름이 없는 요약 문장은 한 후보라도 YES 면 허용
        any_yes = {q: "YES" for v in verdicts.values() for q, a in v.items() if a == "YES"}
        for sent in re.split(r"(?<=[.。])\s+|(?<=다)\s+", f"{draft.situation} {draft.key_evidence}"):
            named = [e for e in evals if any(n and n in sent for n in _names(e))]
            for e in named:
                probs += _claim_problems(sent, verdicts[e["name"]], f"[{e['name']}] ")
            if not named:
                probs += _claim_problems(sent, any_yes)
    else:
        probs = _claim_problems(text, verdict)
    if isinstance(draft, HoldDraft):  # 후보별 보류 사유는 그 후보의 평가표와 맞춘다
        for note in draft.candidates:
            e = next((e for e in evals if any(_same(note.name, n) for n in _names(e))), None)
            if e is None:
                continue
            rows = e["scorecard"]["rows"]
            probs += _claim_problems(f"{note.why_not} {note.recheck}", {r["qid"]: r["answer"] for r in rows}, f"[{e['name']}] ")
            if any(r["answer"] == "NO" and r.get("evidence_ids") for r in rows) and not CITE.search(note.why_not):
                probs.append(f"[{e['name']}] 보류 사유에 반대 근거의 근거 id 가 없다")
        missing = [e["name"] for e in evals if not any(_same(n.name, x) for n in draft.candidates for x in _names(e))]
        if missing:
            probs.append(f"candidates 에 {', '.join(missing)} 가 없다 — 심층 평가한 후보마다 하나씩 써라")
        notes = "\n".join(f"{n.why_not} {n.recheck}" for n in draft.candidates)
    top_sents = " ".join(s for s in re.split(r"(?<=[.。])\s+|(?<=다)\s+", draft.situation)
                         if not isinstance(draft, HoldDraft) or any(n and n in s for n in _names(target)))
    probs += _maturity_problems(top_sents, verdict.get("P1"))
    probs += _stale_forecasts(f"{text}\n{notes}", run_date)
    for rj in sc.get("rejected_yes", []):  # 판정 단계에서 원문 확인에 실패해 기각된 주장이 다시 나오면 안 된다
        q = norm(rj.get("quote", ""))
        if len(q) >= 15 and q[:15] in norm(f"{text}\n{notes}"):
            probs.append(f"판정 단계에서 기각된 주장('{rj['quote'][:30]}…')을 쓰지 마라")
    plain = re.sub(r"우수기업|우수 기업|우수벤처|우수 벤처", "", f"{text}\n{notes}")  # 공식 프로그램 이름은 평가성 표현이 아니다
    probs += [f"평가성 표현 '{w}' 대신 사실과 판정 근거로 써라" for w in EVALUATIVE if w in plain]
    if not re.search(r"\[[^\]]*D[0-9a-f]{5}", draft.market):
        probs.append("시장 수치에 공공·연구기관 문서([D..]) 인용이 없다")
    if not re.search(r"\[[^\]]*D[0-9a-f]{5}", draft.industry_baseline):
        probs.append("업계 기술 수준 문장에 문서([D..]) 인용이 없다")
    if any(not r.evidence_ids for r in draft.risks):
        probs.append("근거 id 가 없는 리스크가 있다")
    if {r.type for r in draft.risks} < {"시장", "기술", "규제", "경쟁"}:
        probs.append("리스크 유형(시장·기술·규제·경쟁)을 하나씩은 다 써라")
    return probs


# ── 표 (코드가 평가 결과로 만든다)

def _human_reason(text: str) -> str:
    """판정 이유에서 내부 기록('→ UNKNOWN 강등 (…)', '(코드 판정)')을 빼고 짧은 이유만 남긴다."""
    if "→ UNKNOWN 강등" in text:
        head = text.split(" → ")[0]
        return next((v for k, v in DEMOTED.items() if head.startswith(k)), re.sub(r"\s*\(.*\)\s*$", "", head))
    if "→ NO 대신 UNKNOWN" in text:
        return "반대 근거가 원문에서 확인되지 않음"
    text = text.removeprefix("N/A 불가 문항 — ")
    return re.sub(r"\s*\((?:코드 판정|코드 날짜 검사)\)", "", text).strip()


def _human_knockouts(reasons: list[str]) -> list[str]:
    return [r.replace("UNKNOWN", "근거 미확인") for r in reasons]


def _judgment_rows(sc: dict) -> list[dict]:
    """24문항 판정표 (투자 권고 보고서). 판정 근거 id 를 이유 끝에 붙여 REFERENCE 로 이어지게 한다."""
    out = []
    for r in sc["rows"]:
        ids = r.get("evidence_ids") or []
        cite = f" [{', '.join(ids)}]" if ids and r["answer"] in ("YES", "NO") else ""
        out.append({**r, "rationale": _human_reason(r["rationale"]) + cite})
    return out


def _decisive_rows(sc: dict, short: dict, limit: int = 6) -> list[dict]:
    """최고점 후보의 결정적 미충족 문항: 반대 근거(NO) 먼저, 다음은 가중치가 큰 항목부터 항목마다 하나씩."""
    weight = {d["id"]: d["weight"] for d in sc["dims"]}
    rows = [r for r in sc["rows"] if r["answer"] in ("NO", "UNKNOWN")]
    no = [r for r in rows if r["answer"] == "NO"]
    unk = sorted((r for r in rows if r["answer"] == "UNKNOWN"), key=lambda r: -weight.get(r["dim"], 0))
    first = [r for i, r in enumerate(unk) if r["dim"] not in {x["dim"] for x in unk[:i]}]
    by_qid = {r["qid"]: r for r in rows}
    out = []
    for r in (by_qid[q] for q in list(dict.fromkeys(r["qid"] for r in no + first + unk))[:limit]):
        ids = r.get("evidence_ids") or []
        cite = f" [{', '.join(ids)}]" if ids and r["answer"] == "NO" else ""
        out.append({"qid": r["qid"], "short": short.get(r["qid"], r["qid"]), "question": r["question"],
                    "answer": "반대 근거" if r["answer"] == "NO" else "미확인",
                    "rationale": _human_reason(r["rationale"]) + cite})
    return out


def _score_rows(evals: list[dict]) -> list[dict]:
    return [{"name": e["name"], "dims": e["scorecard"]["dims"], "total": e["total"], "decision": e["decision"],
             "unknown": round(e["scorecard"]["unknown_ratio"] * 100)} for e in evals]


def _competitor_rows(target: dict, reg: SourceRegistry, notes: list[CompetitorNote], c2_yes: bool) -> list[dict]:
    """근거 본문에 이름이 실제로 나오는 경쟁사만 표에 남기고(지어낸 경쟁사 차단), 그 근거를 인용으로 붙인다.
    비교 문장은 보고서 문안(평가표 일치 검사를 거친 것)을 쓰고, C2 가 YES 가 아닌데 남은 우열 표현은 회사 측 주장으로 표시한다."""
    comp = target["competition"]
    ids = [i for i in dict.fromkeys(comp.get("evidence_ids", []) + comp.get("pool_ids", []))
           if reg.get(i) and citable(reg.get(i))]
    texts = {i: norm(reg.text(i)) for i in ids}
    rows, found = [], []
    for c in comp.get("competitors", []):
        key = norm(c["name"].split("(")[0])
        hit = [i for i in ids if len(key) >= 2 and key in texts[i]][:2]
        vs = next((n.vs_target for n in notes if _same(n.name, c["name"])), c["vs_target"])
        if not c2_yes and _superiority(vs):
            vs = f"(회사 측 주장, 제3자 비교 근거 없음) {vs}"
        row = {**c, "vs_target": vs, "cite": f"[{', '.join(hit)}]" if hit else ""}
        rows.append(row)
        if hit:
            found.append(row)
    return found or rows


def _risk_rows(risks: list[Risk], reg: SourceRegistry) -> list[dict]:
    """리스크 근거 id 를 내용 끝에 붙여 REFERENCE 로 이어지게 한다."""
    out = []
    for r in risks:
        ids = [i for i in r.evidence_ids if reg.get(i) and citable(reg.get(i))]
        cite = f" [{', '.join(ids)}]" if ids and not CITE.search(r.content) else ""
        out.append({"type": r.type, "content": r.content + cite, "due_diligence": r.due_diligence})
    return out


def _first_source(reg: SourceRegistry, ids: list[str], needle: str) -> str | None:
    return next((i for i in ids if needle and reg.get(i) and citable(reg.get(i)) and needle in reg.text(i)), None)


def _team_line(e: dict, reg: SourceRegistry) -> str:
    """'대표 OOO (설립 YYYY-MM-DD) [근거]'. 대표·설립일은 적격성 검증(TIPS 목록 등)과 기술·팀 분석에서 온다."""
    p, t = e.get("profile") or {}, e.get("tech") or {}
    founders = t.get("founders") or []
    ceo = p.get("ceo") or next((f["name"] for f in founders
                                if re.search(r"대표|CEO|창업자", f.get("role") or "", re.I)), None)
    founded = p.get("founded_date") or (f"{p['founded_year']}년" if p.get("founded_year") else None)
    tips = next((x for n in (p.get("name"), p.get("official_name")) if n and (x := reg._find(f"tips:{n}"))), None)
    ids = ([tips] if tips else []) + list(dict.fromkeys(
        p.get("evidence_ids", []) + [i for f in founders for i in f.get("evidence_ids") or []] + t.get("evidence_ids", [])))
    cites = [x for x in (_first_source(reg, ids, ceo), _first_source(reg, ids, p.get("founded_date"))) if x]
    line = f"대표 {ceo}" if ceo else "대표 확인 불가"
    line += f" (설립 {founded})" if founded else " (설립일 확인 불가)"
    return line + (f" [{', '.join(dict.fromkeys(cites))}]" if cites else "")


def _nps_line(p: dict) -> str:
    n = p.get("nps") or {}
    if n.get("status") != "matched" or not n.get("evidence_id"):
        return ""
    return f"국민연금 가입자 {n['members']}명({n['ym']} 기준) [{n['evidence_id']}]"


def _stage_line(p: dict, reg: SourceRegistry) -> str:
    ids = [i for i in p.get("stage_evidence_ids") or [] if reg.get(i) and citable(reg.get(i))][:2]
    return (f"{p.get('stage') or '단계 미상'} ({p.get('round_date') or '시점 미상'}, {p.get('round_amount') or '금액 미공개'})"
            + (f" [{', '.join(ids)}]" if ids else ""))


def _founder_lines(t: dict) -> list[str]:
    out = []
    for f in t.get("founders") or []:
        ids = f.get("evidence_ids") or []
        bg = f.get("background") or ""
        out.append(f"{f['name']}({f.get('role') or '역할 미상'}): {bg}" + (f" [{', '.join(ids)}]" if ids and not CITE.search(bg) else ""))
    return out


def _failed_by_company(failed: list[dict], companies: list[list[str]]) -> dict[str, int]:
    """실패한 검색 질의(중복 제외)를 회사별로 센다 (companies: [대표 이름, 다른 표기...]). 회사명이 없는 질의는 '분야·경쟁사 검색'."""
    alias = sorted(((a, c[0]) for c in companies for a in c if len(norm(a)) >= 2), key=lambda x: -len(norm(x[0])))
    out: dict[str, int] = {}
    for q in dict.fromkeys(f.get("query", "") for f in failed):
        who = next((c for a, c in alias if norm(a) in norm(q)), "분야·경쟁사 검색")
        out[who] = out.get(who, 0) + 1
    return dict(sorted(out.items()))  # 병렬 실행 순서와 관계없이 같은 보고서가 나오게


def _candidate_blocks(evals: list[dict], notes: list[CandidateNote], reg: SourceRegistry, short: dict,
                      failed_by: dict[str, int]) -> list[dict]:
    """후보별 보류 사유 블록: 사실 줄은 코드가, 보류 사유 문장은 LLM 이 (없으면 코드가 평가표로) 쓴다."""
    out = []
    for e in evals:
        p, rows = e["profile"], e["scorecard"]["rows"]
        note = next((n for n in notes if any(_same(n.name, x) for x in _names(e))), None)
        one_line = p.get("one_line") or "-"
        if note:
            why, recheck = note.why_not, note.recheck
            if note.business and re.search(r"[가-힣]", note.business):  # 영어 원문 요약 대신 한국어 한 줄
                one_line = note.business
        else:
            no = [short[r["qid"]] for r in rows if r["answer"] == "NO"]
            unk = [short[r["qid"]] for r in rows if r["answer"] == "UNKNOWN"]
            why = " / ".join(([f"반대 근거: {', '.join(no)}"] if no else []) + ([f"미확인: {', '.join(unk[:6])}"] if unk else []))
            recheck = "-"
        fail = next((v for k, v in failed_by.items() if any(_same(k, x) for x in _names(e))), 0)
        facts = [_team_line(e, reg), _nps_line(p), _stage_line(p, reg)] + ([f"대상 웹 검색 {fail}건 실패"] if fail else [])
        out.append({"name": e["name"], "kind": get_segment(e["segment_id"])["name"], "total": e["total"],
                    "decision": e["decision"], "one_line": one_line, "facts": " · ".join(x for x in facts if x),
                    "reasons": ", ".join(_human_knockouts(e["knockouts"])) or "-", "why_not": why, "recheck": recheck})
    return out


def _candidate_rows(evals: list[dict], screened: list[dict], pool: dict) -> list[dict]:
    """투자 권고 보고서의 후보 비교표: 평가한 후보 + 탈락 후보 + 미평가 적격 후보."""
    from agents.decision import load_rubric

    short = {q["id"]: q["short"] for d in load_rubric()["dimensions"] for q in d["questions"]}
    out = []
    for e in sorted(evals, key=lambda e: -e["total"]):
        rows = e["scorecard"]["rows"]
        no = [short[r["qid"]] for r in rows if r["answer"] == "NO"]
        unk = [short[r["qid"]] for r in rows if r["answer"] == "UNKNOWN"]
        gaps = ([f"반대 근거: {', '.join(no)}"] if no else []) + (
            [f"미확인 {len(unk)}개: {', '.join(unk[:5])}" + (" 등" if len(unk) > 5 else "")] if unk else [])
        out.append({"name": e["name"], "kind": f"{get_segment(e['segment_id'])['ko']} · {e.get('stage') or '-'}",
                    "total": e["total"], "decision": e["decision"], "reasons": ", ".join(_human_knockouts(e["knockouts"])) or "-",
                    "gaps": " / ".join(gaps) or "-"})
    for c in screened:
        if not c.get("eligible"):
            out.append({"name": c.get("official_name") or c["name"], "kind": f"적격성 검증 · {c.get('stage') or '-'}",
                        "total": "-", "decision": "탈락", "reasons": c.get("reason") or "-", "gaps": "심층 평가 대상 아님"})
    for u in pool["unevaluated"]:
        out.append({"name": u["name"], "kind": f"적격 · {u['stage']}", "total": "-", "decision": "미평가",
                    "reasons": pool["why"], "gaps": "심층 평가하지 않음"})
    return out


def _limitations(cfg, pool: dict, failed_by: dict[str, int], data_limits: list[str], hold: bool) -> list[str]:
    """한계점 2~4개: 검색 실패, 미평가 적격 후보, 방법 한계, 데이터 한계 순."""
    out = []
    if failed_by:
        n = sum(failed_by.values())
        parts = ", ".join(f"{k} {v}건" for k, v in sorted(failed_by.items(), key=lambda kv: (kv[0] == "분야·경쟁사 검색", -kv[1])))
        out.append(f"웹 검색 질의 {n}건이 실패(검색 한도 초과 등)해 빈 결과로 진행했다 — {parts}. "
                   f"해당 후보의 미확인·반대 판정 일부는 근거 부재가 아니라 검색 실패 때문일 수 있다")
    if pool["unevaluated"]:
        out.append(f"적격 후보 {pool['eligible']}곳 중 {len(pool['unevaluated'])}곳은 {pool['why']}로 심층 평가하지 않았다"
                   + (f" — '투자 권고 없음'은 평가한 {pool['evaluated']}곳에 대한 결론이다" if hold else ""))
    out.append("공개 근거를 찾지 못한 문항(미확인)을 0점으로 처리하는 보수적 채점이라, 공개 정보가 적은 초기 기업일수록 점수가 낮게 나온다")
    out += data_limits[:1]
    out.append(f"판정은 {cfg.models.judge} 가 하며, 코드는 근거 인용이 원문에 있는지만 검증한다 (근거 해석 오류는 남을 수 있음)")
    return out[:4]


# ── REFERENCE: 인용 번호 매기기

def _same_title(a: str, b: str) -> bool:
    return a == b or (min(len(a), len(b)) >= 15 and (a.startswith(b) or b.startswith(a)))


def _renumber(draft: dict, tables: dict, reg: SourceRegistry) -> tuple[dict, dict, list[dict]]:
    """[W..]/[D..] 인용을 REFERENCE 번호로 바꾸고, 실제 인용된 근거만 REFERENCE 로 만든다.
    - 같은 자료(같은 URL, 포털 전재본처럼 제목이 같은 기사)는 한 번호로 묶는다
    - 번호는 기관 보고서 → 학술 논문 → 웹페이지 순으로 이어지고, 같은 유형 안에서는 처음 인용된 순서
    - 뉴스레터·모음 메일은 인용하지 않는다 (인용 표시에서도 뺀다)"""
    cited: list[str] = []

    def collect(x):
        if isinstance(x, str):
            cited.extend(i for m in CITE.finditer(x) for i in re.split(r"\s*[,，]\s*", m.group(1)))
        elif isinstance(x, list):
            for v in x:
                collect(v)
        elif isinstance(x, dict):
            for v in x.values():
                collect(v)

    collect(draft)
    collect(tables)
    groups: list[dict] = []
    for sid in dict.fromkeys(cited):
        s = reg.get(sid)
        if not s or not citable(s):
            continue
        k, tk = reference_key(s), title_key(s)
        g = next((g for g in groups if k in g["keys"] or (tk and any(_same_title(tk, x) for x in g["titles"]))), None)
        if g is None:
            g = {"ids": [], "srcs": [], "keys": set(), "titles": []}
            groups.append(g)
        g["ids"].append(sid)
        g["srcs"].append(s)
        g["keys"].add(k)
        if tk:
            g["titles"].append(tk)
    groups.sort(key=lambda g: GROUPS.index(reference_group(g["srcs"][0])))  # 안정 정렬: 유형 안에서는 인용 순서 유지
    num, refs = {}, []
    for n, g in enumerate(groups, 1):
        num.update({sid: n for sid in g["ids"]})
        refs.append({"n": n, "group": reference_group(g["srcs"][0]), "text": format_reference(merge_duplicates(g["srcs"]))})

    def repl(m: re.Match) -> str:
        nums = sorted({num[i] for i in re.split(r"\s*[,，]\s*", m.group(1)) if i in num})
        return "[" + ", ".join(map(str, nums)) + "]" if nums else ""

    def walk(x):
        if isinstance(x, str):
            x = CITE.sub(repl, x)
            x = re.sub(r"\[(?![\d, ]+\])[^\[\]]{1,20}\]", "", x)  # 근거 id 가 아닌 임의 대괄호 표기 제거
            x = re.sub(r"\[\d+(?:, \d+)*\](?:\s*\[\d+(?:, \d+)*\])+",  # "[1][2, 3]" → "[1, 2, 3]"
                       lambda m: "[" + ", ".join(map(str, sorted({int(n) for n in re.findall(r"\d+", m.group(0))}))) + "]",
                       x)
            return re.sub(r"\s+\.", ".", x).strip()
        if isinstance(x, list):
            return [walk(i) for i in x]
        if isinstance(x, dict):
            return {k: walk(v) for k, v in x.items()}
        return x

    return walk(draft), walk(tables), refs


def _ref_groups(refs: list[dict]) -> list[dict]:
    """과제 형식의 세 소제목을 항상 둔다 (인용한 자료가 없는 유형은 '없음'으로 표시)."""
    return [{"name": g, "items": [r for r in refs if r["group"] == g]} for g in GROUPS]


# ── 보고서 노드

def _candidates_context(evals: list[dict], short: dict, reg: SourceRegistry, failed_by: dict[str, int]) -> str:
    """모두 보류 보고서용: 후보마다 평가표 요약 (반대 근거·확인된 사실·미확인 항목)."""
    blocks = []
    for e in evals:
        p, rows = e["profile"], e["scorecard"]["rows"]

        def fmt(r):
            ids = ", ".join(i for i in r.get("evidence_ids") or [] if reg.get(i) and citable(reg.get(i)))
            return f"{r['qid']} {short[r['qid']]}: \"{(r.get('quote') or '')[:80]}\"" + (f" [{ids}]" if ids else "")

        fail = next((v for k, v in failed_by.items() if any(_same(k, x) for x in _names(e))), 0)
        blocks.append("\n".join([
            f"### {e['name']} — 보류 사유: {', '.join(_human_knockouts(e['knockouts']))}",
            f"- 사업: {p.get('one_line')} / 단계: {p.get('stage')} ({p.get('round_date') or '시점 미상'}, "
            f"{p.get('round_amount') or '금액 미공개'}) / 대표: {p.get('ceo') or '확인 불가'} / 설립: "
            f"{p.get('founded_date') or p.get('founded_year') or '확인 불가'}",
            "- 반대 근거(NO): " + ("; ".join(f"{fmt(r)} — {_human_reason(r['rationale'])}" for r in rows if r["answer"] == "NO") or "없음"),
            "- 확인된 사실(YES): " + ("; ".join(fmt(r) for r in rows if r["answer"] == "YES") or "없음"),
            "- 공개 근거 없음(미확인): " + (", ".join(f"{r['qid']} {short[r['qid']]}" for r in rows if r["answer"] == "UNKNOWN") or "없음"),
        ] + ([f"- 이 회사 대상 웹 검색 {fail}건이 실패했다 (근거 부족의 일부는 검색 실패 탓일 수 있음)"] if fail else [])))
    return "\n\n".join(blocks)


def report_node(state: dict) -> dict:
    from agents.decision import load_rubric

    cfg = get_config()
    reg = SourceRegistry(state.get("registry"))
    evals = state.get("evaluations", [])
    screened = state.get("screened", [])
    if not evals:
        return _no_candidate_report(state, cfg, screened)
    run_date = state.get("run_date") or datetime.now().strftime("%Y-%m-%d")
    invested = [e for e in evals if e["decision"] == "투자"]
    hold = not invested
    ranked = sorted(evals, key=lambda e: -e["total"])
    target = invested[0] if invested else ranked[0]
    prof, sc = target["profile"], target["scorecard"]
    short = {q["id"]: q["short"] for d in load_rubric()["dimensions"] for q in d["questions"]}
    pool = _pool(state, evals, screened, cfg, bool(invested))
    companies = [_names(e) for e in evals] + [
        [n for n in (r.get("official_name") or r.get("name"), r.get("name"), r.get("name_en")) if n] for r in screened]
    failed_by = _failed_by_company(list(getattr(search_tool, "FAILED_QUERIES", []) or []), companies)
    conclusion = _conclusion(target, pool, bool(invested))
    request = _request(pool) if hold else ""
    situation = _situation(pool) if hold else None
    verdict = {r["qid"]: r["answer"] for r in sc["rows"]}
    tech = {k: v for k, v in target["tech"].items() if k != "pool_ids"}
    if verdict.get("C1") != "YES":
        tech["ip_evidence"] = "확인 불가 (평가표 C1 미확인)"
    if verdict.get("C2") != "YES":
        tech["differentiators"] = [f"(회사 측 주장, 제3자 비교 근거 없음) {x}" for x in tech.get("differentiators", [])]

    def usable(ids):
        return [i for i in dict.fromkeys(ids) if reg.get(i) and citable(reg.get(i))]

    def scrub(x):  # 분석 결과 속 뉴스레터 근거 id 를 빼고 LLM 에 넘긴다 (인용 후보에서 제외)
        if isinstance(x, str):
            return CITE.sub(lambda m: f"[{', '.join(k)}]" if (k := usable(re.split(r"\s*[,，]\s*", m.group(1)))) else "", x)
        if isinstance(x, list):
            return usable(x) if x and all(isinstance(i, str) and re.fullmatch(r"[WD][0-9a-f]{5}", i) for i in x) else [scrub(i) for i in x]
        if isinstance(x, dict):
            return {k: scrub(v) for k, v in x.items() if k != "pool_ids"}
        return x

    pool_ids = usable(prof.get("evidence_ids", []) + target["tech"].get("evidence_ids", [])
                      + target["market"].get("evidence_ids", []) + target["competition"].get("evidence_ids", [])
                      + [i for r in sc["rows"] for i in (r.get("evidence_ids") or [])])
    other_ids = usable([i for e in ranked if e is not target for i in
                        (e["profile"].get("stage_evidence_ids") or []) + [(e["profile"].get("nps") or {}).get("evidence_id")]
                        + [x for r in e["scorecard"]["rows"] for x in (r.get("evidence_ids") or [])] if i])
    other_ids = [i for i in other_ids if i not in pool_ids]
    comp_rows = _competitor_rows(target, reg, [], verdict.get("C2") == "YES")
    ctx = {
        "domain": cfg.domain.name, "run_date": run_date, "decision": target["decision"], "any_invest": bool(invested),
        "hold": hold,
        "conclusion": conclusion, "target": prof["official_name"],
        "pool": (f"발굴 {pool['discovered'] if pool['discovered'] is not None else '-'}곳 → 검증 {pool['screened']}곳 → "
                 f"적격 {pool['eligible']}곳 → 심층 평가 {pool['evaluated']}곳 (미평가 적격 {len(pool['unevaluated'])}곳: {pool['why']})"),
        "profile": json.dumps({k: prof.get(k) for k in ("official_name", "region", "one_line", "founded_year", "founded_date",
                                                        "ceo", "stage", "round_date", "round_amount", "nps")}, ensure_ascii=False),
        "tech": json.dumps(scrub(tech), ensure_ascii=False),
        "market": json.dumps(scrub(target["market"]), ensure_ascii=False),
        "competition": json.dumps(scrub(target["competition"]), ensure_ascii=False),
        "competitor_names": ", ".join(c["name"] for c in comp_rows),
        "scorecard": json.dumps({"total": sc["total"], "threshold": sc["threshold"], "reasons": sc["reasons"],
                                 "dims": sc["dims"]}, ensure_ascii=False),
        "verified": "\n".join(f"- {r['qid']}: \"{r['quote']}\" [{', '.join(usable(r['evidence_ids']))}]"
                              for r in sc["rows"] if r["answer"] == "YES" and r.get("quote")) or "(없음)",
        "unverified": ", ".join(f"{r['qid']}({r['question'][:26]})" for r in sc["rows"] if r["answer"] != "YES"),
        "rejected": "\n".join(f"- {r['qid']}: \"{r['quote']}\"" for r in sc.get("rejected_yes", []) if r.get("quote")) or "(없음)",
        "others": json.dumps([{k: e[k] for k in ("name", "total", "decision")} for e in evals], ensure_ascii=False),
        "candidates": _candidates_context(ranked, short, reg, failed_by) if hold else "",
        "evidence": reg.brief(pool_ids, 420) + ("\n\n" + reg.brief(other_ids, 300) if hold and other_ids else ""),
    }
    schema = HoldDraft if hold else Draft

    # 1) 초안 → SUMMARY 규칙 + 평가표 일치 검사, 어긋나면 이유를 알려 주고 다시 쓴다 (최대 3회)
    draft, feedback, probs = None, "", []
    for _ in range(3):
        draft = structured(schema).invoke(render("report", **ctx, feedback=feedback, shorten=False))
        probs = (_summary_problems(summary_lines(draft, conclusion, request, situation), cfg.report.summary_max_chars)
                 + _consistency_problems(draft, target, evals, run_date))
        if not probs:
            break
        feedback = "직전 초안의 문제를 모두 고쳐라:\n- " + "\n- ".join(probs)

    if hold:
        title = f"AgTech AI 스타트업 투자 검토 — 투자 권고 없음 (심층 평가 {len(evals)}곳 모두 보류)"
    else:
        title = f"{prof['official_name']} 투자 검토 — 투자 권고"
    team = cfg.submission
    out_dir = path(f"{cfg.report.output_dir}/.keep").parent
    pdf_path = out_dir / "investment_report.pdf"

    # 2) 렌더링 → 분량 검증. 5쪽을 넘으면 조판 밀도를 올리고, 그래도 넘으면 줄여 쓴다
    result, density, shorten_round = {}, 0, 0
    while True:
        dd = draft.model_dump()
        dd["summary"] = summary_lines(draft, conclusion, request, situation)
        dd["limitations"] = _limitations(cfg, pool, failed_by, dd["data_limits"], hold)
        tables = {
            "competitors": _competitor_rows(target, reg, draft.competitor_notes, verdict.get("C2") == "YES"),
            "risks": _risk_rows(draft.risks, reg),
            "team": _team_line(target, reg), "nps": _nps_line(prof), "stage": _stage_line(prof, reg),
            "founders": _founder_lines(target["tech"]), "growth": list(target["tech"].get("growth_signals") or [])[:3],
        }
        if hold:
            tables |= {"pool": _pool_rows(pool), "unevaluated": pool["unevaluated"],
                       "blocks": _candidate_blocks(ranked, draft.candidates, reg, short, failed_by),
                       "scores": _score_rows(ranked), "decisive": _decisive_rows(sc, short)}
        else:
            tables |= {"rows": _judgment_rows(sc), "candidates": _candidate_rows(evals, screened, pool)}
        shown = {k: v for k, v in dd.items() if k not in ("competitor_notes", "candidates", "risks", "data_limits")}
        d, t, refs = _renumber(shown, tables, reg)
        view = {
            "title": title, "run_date": run_date, "domain": cfg.domain.name, "decision": target["decision"], "hold": hold,
            "badge": "투자 권고 없음" if hold else "투자 권고", "target": target, "profile": prof, "draft": d,
            "tables": t, "scorecard": sc, "pool": pool, "ref_groups": _ref_groups(refs),
            "dim_names": [f"{x['name'].split(' (')[0]} {x['weight']}" for x in sc["dims"]],
            "c2_verified": verdict.get("C2") == "YES",
        }
        html = render_html(view, density=density)
        result = html_to_pdf(html, pdf_path)
        if result["pages"] <= cfg.report.max_pages and result["summary_ratio"] <= 0.5:
            break
        if density < 3:
            density += 1
            continue
        if shorten_round >= 2:
            break
        shorten_round += 1
        density = 1
        draft = structured(schema).invoke(render("report", **ctx, feedback="분량 초과", shorten=True))

    submit_name = f"RAG-Output_{team.campus}-{team['class']}_{'+'.join(sorted(team.members))}.pdf"
    shutil.copyfile(pdf_path, out_dir / submit_name)
    md_path = out_dir / "investment_report.md"
    md_path.write_text(_to_markdown(view), encoding="utf-8")
    checks = {
        "structure": "모두 보류(후보별 보류 사유)" if hold else "투자 권고",
        "pages": result["pages"], "max_pages": cfg.report.max_pages, "pages_ok": result["pages"] <= cfg.report.max_pages,
        "summary_ratio_of_a4": result["summary_ratio"], "summary_ok": result["summary_ratio"] <= 0.5,
        "summary_rule_violations": _summary_problems(summary_lines(draft, conclusion, request), cfg.report.summary_max_chars),
        "scorecard_consistency_violations": _consistency_problems(draft, target, evals, run_date),
        "references": len(refs), "reference_groups": {g["name"]: len(g["items"]) for g in view["ref_groups"]},
        "limitations": len(d["limitations"]), "density_level": density,
        "first_section": "SUMMARY", "last_section": "REFERENCE",
    }
    msg = (f"[보고서] {pdf_path.name} {result['pages']}쪽 (≤{cfg.report.max_pages}), SUMMARY A4 대비 "
           f"{result['summary_ratio']:.0%}, REFERENCE {len(refs)}건, 평가표 불일치 "
           f"{len(checks['scorecard_consistency_violations'])}건 → {submit_name}")
    print(msg)
    return {"report": {"pdf": str(pdf_path), "submit_pdf": str(out_dir / submit_name), "html": result["html"],
                       "markdown": str(md_path), "title": title, "target": prof["official_name"],
                       "decision": "투자 권고" if invested else "투자 권고 대상 없음", "checks": checks,
                       "generated_at": datetime.now().isoformat(timespec="seconds")},
            "log": [msg]}


def _to_markdown(v: dict) -> str:
    d, t, sc, prof = v["draft"], v["tables"], v["scorecard"], v["profile"]
    hold = v["hold"]
    h, n = ("###", ["3.1", "3.2", "3.3", "3.4", "3.5"]) if hold else ("##", ["1.", "2.", "3.", "4.", "5."])
    vs = "대상 대비" if v["c2_verified"] else "대상 대비 (회사 측 주장 포함)"
    L = [f"# {v['title']} ({v['run_date']})", "", "## SUMMARY", *[f"- {s}" for s in d["summary"]], ""]
    if hold:
        p = v["pool"]
        L += ["## 1. 후보 풀과 선정 과정", "", "| 단계 | 후보 수 | 국내 | 해외 | 비고 |", "|---|---|---|---|---|",
              *[f"| {r['stage']} | {r['total']} | {r['kr']} | {r['gl']} | {r['note']} |" for r in t["pool"]], ""]
        if t["unevaluated"]:
            L += [f"미평가 적격 후보 {len(t['unevaluated'])}곳 ({p['why']}): "
                  + ", ".join(f"{u['name']}({u['stage']}{', 검증 검색 실패' if u['search_failed'] else ''})" for u in t["unevaluated"]), ""]
        L += ["## 2. 후보별 보류 사유", ""]
        for i, b in enumerate(t["blocks"], 1):
            L += [f"### 2.{i} {b['name']} — {b['total']}점 {b['decision']} ({b['kind']})", f"- 사업: {b['one_line']}",
                  f"- 팀·단계: {b['facts']}", f"- 보류 사유: {b['why_not']} ({b['reasons']})", f"- 재검토 조건: {b['recheck']}", ""]
        L += [f"## 3. 최고점 후보 상세 — {prof['official_name']}", ""]
    L += [f"{h} {n[0]} 사업 아이디어", f"- 해결하는 문제: {d['problem']}", f"- 제품·핵심 컨셉: {d['product']}",
          f"- 수익 방식: {d['revenue_model']}", f"- 투자 단계: {t['stage']}",
          *([f"- 성장 신호: {' / '.join(t['growth'])}"] if t["growth"] else []), "",
          f"{h} {n[1]} 시장 규모와 성장성", d["market"], "",
          f"{h} {n[2]} 기술력과 팀", f"- 팀: {t['team']}" + (f" · {t['nps']}" if t["nps"] else ""),
          *[f"- {x}" for x in t["founders"]], "", d["tech_team"], "", f"업계 기술 수준 대비: {d['industry_baseline']}", "",
          f"{h} {n[3]} 경쟁 구도", d["competition"], "",
          f"| 경쟁사 | 국가 | 제품·접근 | 규모 | {vs} | 근거 |", "|---|---|---|---|---|---|",
          *[f"| {c['name']} | {c['country']} | {c['offering']} | {c['scale']} | {c['vs_target']} | {c['cite']} |"
            for c in t["competitors"]], "",
          f"{h} {n[4]} 사업 리스크", "| 유형 | 내용 | 실사 항목·투자 조건 |", "|---|---|---|",
          *[f"| {r['type']} | {r['content']} | {r['due_diligence']} |" for r in t["risks"]], ""]
    if hold:
        L += ["## 4. 투자 판단 요약", "", "| 후보 | " + " | ".join(v["dim_names"]) + " | 총점 | 미확인 | 판단 |",
              "|---" * (len(v["dim_names"]) + 4) + "|",
              *[f"| {s['name']} | " + " | ".join(f"{x['score']}" for x in s["dims"]) + f" | {s['total']} | {s['unknown']}% | {s['decision']} |"
                for s in t["scores"]], "",
              f"{prof['official_name']}의 결정적 미충족 문항", "", "| ID | 문항 | 판정 | 이유 |", "|---|---|---|---|",
              *[f"| {r['qid']} | {r['short']} | {r['answer']} | {r['rationale']} |" for r in t["decisive"]], "",
              "## 5. 한계점"]
    else:
        L += [f"## 6. 투자 판단: {v['decision']} ({sc['total']}점 / 기준 {sc['threshold']}점)", d["decision_rationale"], "",
              "기준점 민감도: " + ", ".join(f"{k}점 {x}" for k, x in sc["sensitivity"].items()), "",
              "| 항목 | 가중치 | 확인/판정 문항 | 미확인 | 점수 |", "|---|---|---|---|---|",
              *[f"| {x['name']} | {x['weight']} | {x['yes']}/{x['n']} | {x['unknown']} | {x['score']} |" for x in sc["dims"]],
              "", "| ID | 판정 | 근거·이유 |", "|---|---|---|", *[f"| {r['qid']} | {r['answer']} | {r['rationale']} |" for r in t["rows"]],
              "", "### 후보별 판단", "", "| 후보 | 분야·단계 | 총점 | 판단 | 사유 | 확인되지 않은 근거 |", "|---|---|---|---|---|---|",
              *[f"| {c['name']} | {c['kind']} | {c['total']} | {c['decision']} | {c['reasons']} | {c['gaps']} |"
                for c in t["candidates"]], "", "## 7. 한계점"]
    L += [*[f"- {x}" for x in d["limitations"]], "", "## REFERENCE"]
    for g in v["ref_groups"]:
        L += ["", f"### {g['name']}", *([f"{r['n']}. {r['text']}" for r in g["items"]] or ["- 본문에 인용한 자료 없음"])]
    return "\n".join(L) + "\n"


def _no_candidate_report(state: dict, cfg, screened: list[dict]) -> dict:
    """적격 후보가 하나도 없을 때: 발굴·검증 결과와 탈락 사유만 담은 보고서 (LLM 없이 코드로 작성)."""
    reasons: dict[str, int] = {}
    for r in screened:
        key = r["reason"].split("(")[0].split(":")[0][:30]
        reasons[key] = reasons.get(key, 0) + 1
    rows = "".join(f"<tr><td>{r['name']}</td><td>{r.get('stage', '-')}</td><td>{r['reason']}</td></tr>" for r in screened)
    html = f"""<html><head><meta charset="utf-8"><style>body{{font-family:sans-serif;font-size:10pt}}
    table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #ccc;padding:3px}}</style></head><body>
    <h1>AgTech AI 스타트업 투자 검토 — 적격 후보 없음</h1><section id="summary"><h2>SUMMARY</h2><ul>
    <li>상황: 발굴 후보 {len(screened)}곳을 검증했으나 과제 기준(비상장·Seed~Series C·Exit 전)을 모두 충족한 곳이 없다.</li>
    <li>결론: 투자 권고 대상 없음 — 심층 평가 대상 0곳</li>
    <li>근거: 탈락 사유 {', '.join(f'{k} {v}곳' for k, v in reasons.items())}</li>
    <li>조건: 발굴 채널·세부 분야를 넓혀 다시 실행 (config.yaml workflow)</li></ul></section>
    <h2>1. 발굴·검증 결과</h2><table><tr><th>후보</th><th>단계</th><th>판정</th></tr>{rows}</table>
    <h2>2. 한계점</h2><p>공개 정보만으로 투자 단계를 확인하지 못한 후보는 보수적으로 제외했다.</p>
    <h2>REFERENCE</h2><p>본문에 인용한 외부 자료 없음</p></body></html>"""
    out_dir = path(f"{cfg.report.output_dir}/.keep").parent
    pdf_path = out_dir / "investment_report.pdf"
    result = html_to_pdf(html, pdf_path)
    team = cfg.submission
    submit_name = f"RAG-Output_{team.campus}-{team['class']}_{'+'.join(sorted(team.members))}.pdf"
    shutil.copyfile(pdf_path, out_dir / submit_name)
    msg = f"[보고서] 적격 후보 없음 보고서 {result['pages']}쪽 → {submit_name}"
    print(msg)
    return {"report": {"pdf": str(pdf_path), "submit_pdf": str(out_dir / submit_name), "html": result["html"],
                       "decision": "적격 후보 없음", "checks": {"pages": result["pages"]}}, "log": [msg]}
