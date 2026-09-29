"""🧮 투자 판단 에이전트.

LLM 은 평가표 질문에 판정(YES/NO/UNKNOWN/N/A)·근거 id·근거 원문 인용만 답한다. 점수와 결론은 코드가 정한다.
낙관 편향을 막는 장치
- 항목(차원)별로 따로 호출한다 (한 항목의 인상이 다른 항목으로 번지는 후광 효과 방지)
- 점수·기준점·"유망" 같은 표현을 LLM 에 보여 주지 않는다
- YES 는 인용문이 실제 근거 본문에 있어야 인정한다 (코드가 문자열로 검사). 실패하면 UNKNOWN 으로 강등
- NO 도 반대 사실을 적은 문장의 인용이 본문에 있어야 인정한다. "근거가 없다"는 NO 가 아니라 UNKNOWN
  (판정 이유가 "추정·확인되지 않음·근거가 없음" 같은 근거 부족이면 코드가 UNKNOWN 으로 바꾼다)
- 제3자 근거·최근 24개월·문서 근거 요구 조건을 코드가 검사한다
  (제3자 문항은 기사에 실렸어도 인용 주변에 대표·회사 측 발언이나 목표·계획·예정 표현이 있으면 인정하지 않는다)
- 상용 운영 문항은 인용 주변에 예정·목표·실증·PoC·시범 표현이 있거나 인용에 수량·고객이 없으면 인정하지 않는다
- UNKNOWN 은 0점이고 분모를 줄이지 않는다 (정보를 감춘 회사가 유리해지지 않게)
"""
from __future__ import annotations

import re
from datetime import date, datetime
from functools import lru_cache
from typing import Literal

import yaml
from pydantic import BaseModel, Field

from core.config import ROOT, get_config
from core.llm import structured
from core.prompts import render
from rank_bm25 import BM25Okapi

from rag.index import kiwi_tokenize
from rag.loader import load_manifest
from tools.grounding import fuzzy_in, norm
from tools.sources import SourceRegistry

AGENT = "decision"
DB_SITES = {"THE VC", "혁신의숲"}
# 아래 표지들은 공백·기호를 뺀 소문자 글자(norm)에서 찾는다
# 제3자 근거 문항: 기사에 실렸어도 대표·회사 측 발언이나 목표·계획이면 제3자가 확인한 사실이 아니다
COMPANY_VOICE = ("대표는", "대표가", "대표이사는", "대표의설명", "라고말했다", "라고밝혔다", "설명했다", "설명이다",
                 "회사측", "회사는", "목표", "계획", "예정")
# 영어 발화 표지는 공백을 지우면 'has aided' 같은 오탐이 생겨 원문에서 단어 경계로 찾는다
COMPANY_VOICE_EN = re.compile(r"\b(said|says|plans to|aims to|according to the company)\b", re.I)
# 상용 운영 문항: 예정·실증 단계를 뜻하는 표현
PLANNED = re.compile(r"상용화를기점|상용화예정|출시예정|목표|실증|poc|proofofconcept|시범|(?<!auto)pilot")
# 상용 운영 문항 YES 요건(설치 수·면적·두수·고객명): 숫자, 한글 수량 표현, 농협·법인 같은 고객 이름
SCALE = re.compile(r"\d|농협|영농조합|농업회사법인")
SCALE_WORD = re.compile(r"(?<![가-힣])(한|두|세|네|다섯|여섯|일곱|여덟|아홉|수십|수백|수천)\s*(곳|개소|농가|농장|마리)")
# NO 판정 이유에 이런 말이 있으면 반대 사실이 아니라 근거 부족이다
NO_HEDGE = re.compile(r"추정|불분명|명확하지않|확인되지않|없어|없으므로|근거가없")
UNDISCLOSED = ("비공개", "미공개", "undisclosed", "비밀", "n/a")


class Answer(BaseModel):
    qid: str
    quote: str = Field(description="판정을 뒷받침하는 근거 원문 그대로의 짧은 인용 (80자 이내). 없으면 빈 문자열")
    evidence_ids: list[str] = Field(description="인용이 나온 근거 id")
    verdict: Literal["YES", "NO", "UNKNOWN", "N/A"]
    rationale: str = Field(description="한 문장 이유")


class Answers(BaseModel):
    answers: list[Answer]


def load_rubric() -> dict:
    with open(ROOT / "rubric.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


_norm = norm


def _quote_sources(quote: str, cited: list[str], pool: list[str], reg: SourceRegistry) -> list[str]:
    """인용문이 실제로 들어 있는 근거 id. LLM 이 id 를 잘못 적어도 후보의 근거 전체에서 찾아 바로잡는다."""
    hit = [i for i in cited if fuzzy_in(quote, reg.text(i))]
    return hit or [i for i in pool if fuzzy_in(quote, reg.text(i))]


def _around(quote: str, text: str, window: int) -> str | None:
    """인용문과 그 앞뒤 window 글자(공백·기호를 뺀 글자 기준). 근거 본문에서 위치를 못 찾으면 None."""
    t, q = _norm(text), _norm(quote)
    pos = t.find(q[:12]) if len(q) >= 12 else t.find(q)
    if pos < 0:  # 인용이 조금 달라 위치를 못 찾으면 인용 앞부분 여러 조각으로 다시 찾는다
        pos = next((t.find(q[i:i + 10]) for i in range(0, max(1, len(q) - 10), 10) if t.find(q[i:i + 10]) >= 0), -1)
    if pos < 0:
        return None
    return t[max(0, pos - window): pos + len(q) + window]


def _near_company(quote: str, text: str, keys: list[str], window: int = 300) -> bool:
    """인용문이 나온 위치 앞뒤 window 글자 안에 회사명이 있는지 (문서 어딘가에 회사명만 있으면 되는 허점 차단)."""
    near = _around(quote, text, window)
    return near is not None and any(k in near for k in keys)


def _company_voice(quote: str, text: str, keys: list[str], window: int = 120) -> bool:
    """인용 앞뒤 window 글자 안에 대표·회사 측 발언이나 목표·계획 표지가 있으면 True (제3자 근거 아님).
    신문 기사라도 대표 인터뷰를 옮긴 문장이면 회사 자체 주장이다. 인용 위치를 못 찾으면 발화자를 확인할 수 없어 True."""
    near = _around(quote, text, window)
    if near is None:
        return True
    raw_pos = text.find(quote[:20])
    raw = text[max(0, raw_pos - window): raw_pos + len(quote) + window] if raw_pos >= 0 else ""
    return (any(v in near for v in COMPANY_VOICE) or any(k + "에따르면" in near for k in [*keys, "회사", "업체"])
            or bool(COMPANY_VOICE_EN.search(raw)))


def _planned(quote: str, texts: list[str], window: int = 120) -> bool:
    """상용 운영 문항: 인용이나 그 앞뒤 window 글자에 예정·목표·실증·PoC·시범 표현이 있으면 True."""
    zones = [_norm(quote)] + [z for t in texts if (z := _around(quote, t, window))]
    return any(PLANNED.search(z) for z in zones)


def _has_scale(quote: str, keys: list[str]) -> bool:
    """상용 운영 문항 YES 요건: 인용에 설치 수·면적·두수 같은 수량이나 고객 이름이 있는지 (회사 자신의 이름은 빼고 본다)."""
    q = _norm(quote)
    for k in keys:
        q = q.replace(k, "")
    return bool(SCALE.search(q) or SCALE_WORD.search(quote))


def _passages(pool: list[str], reg: SourceRegistry, size: int = 520, step: int = 420) -> list[tuple[str, str]]:
    out = []
    for sid in pool:
        t = reg.text(sid)
        for k in range(0, max(1, len(t) - 80), step):
            out.append((sid, t[k:k + size]))
    return out


def _evidence_for(dim: dict, company: str, passages: list[tuple[str, str]], bm25, reg: SourceRegistry,
                  per_q: int = 6, cap: int = 24) -> str:
    """문항마다 관련 깊은 근거 문단을 따로 찾아 합친다 (후보가 모은 근거에 대한 작은 RAG).
    항목 전체를 한 번에 검색하면 특정 문항(예: 날짜가 있는 마일스톤)의 근거가 밀려나기 때문."""
    picked: list[int] = []
    for q in dim["questions"]:
        scores = bm25.get_scores(kiwi_tokenize(f"{company} {q['text']} {q['need']}"))
        for i in sorted(range(len(passages)), key=lambda i: -scores[i])[:per_q]:
            if i not in picked:
                picked.append(i)
    lines = []
    for i in picked[:cap]:
        sid, text = passages[i]
        s = reg.get(sid)
        head = (f"{s['site']}, {s['date'] or '게시일 미상'}" if s["kind"] == "web"
                else f"{s['publisher']} {s['year']}, p.{s['page']}")
        lines.append(f"[{sid}] ({head}) {text}")
    return "\n\n".join(lines)


def _is_third_party(s: dict, company_keys: list[str]) -> bool:
    if s["kind"] == "doc":
        return True
    host = s["url"].split("/")[2].lower() if s["url"].count("/") >= 2 else ""
    return not any(k and k in _norm(host) for k in company_keys)


@lru_cache(maxsize=1)
def _doc_years() -> dict:
    """문서 코퍼스(data/manifest.yaml)의 doc_id → 발행 연도."""
    return {m["doc_id"]: m.get("year") for m in load_manifest()}


def _is_recent(s: dict, run_date: str) -> bool:
    if s["kind"] == "doc" or s.get("id", "").startswith("D"):  # 문서는 manifest 의 발행 연도 ≥ 기준 연도 - 2 (쪽마다 날짜가 없음)
        year = _doc_years().get(s.get("doc_id")) or s.get("year")
        try:
            return int(year) >= int(run_date[:4]) - 2
        except (TypeError, ValueError):
            return False
    if s["kind"] == "web" and s.get("site") in DB_SITES:
        return True  # 기업 DB 프로필은 현재 정보
    d = s.get("date") if s["kind"] == "web" else None
    if not d:
        return False
    try:
        days = (datetime.strptime(run_date, "%Y-%m-%d").date() - date.fromisoformat(d)).days
    except ValueError:
        return False
    return 0 <= days <= 730


def _events_too_old(text: str, run_date: str) -> bool:
    """문장 속 사건 날짜(2024년 6월, 2024-06, 2024.06)가 있고, 그 모두가 24개월보다 오래됐으면 True.
    LLM 은 날짜 계산을 자주 틀리므로 코드로 확인한다. 날짜가 없으면 판단하지 않는다(False)."""
    run = datetime.strptime(run_date, "%Y-%m-%d")
    months = [(int(y), int(m)) for y, m in re.findall(r"(20\d{2})\s*[년.\-/]\s*(\d{1,2})", text) if 1 <= int(m) <= 12]
    if not months:
        return False
    return all((run.year - y) * 12 + (run.month - m) > 24 for y, m in months)


def _round_rule(c: dict, run_date: str) -> tuple[str, list[str], str]:
    """D1: 최근 24개월 라운드의 단계·금액·시점. 적격성 검증 단계에서 원문 인용으로 확인된 값만 쓴다."""
    ids, rd, amount = c.get("stage_evidence_ids") or [], str(c.get("round_date") or ""), str(c.get("round_amount") or "")
    m = re.match(r"(20\d{2})(?:-(\d{1,2}))?", rd)
    if not ids or not m:
        return "UNKNOWN", [], "라운드 시점이 확인되지 않음 (코드 판정)"
    y, mo = int(m.group(1)), int(m.group(2) or 6)
    run = datetime.strptime(run_date, "%Y-%m-%d")
    months = (run.year - y) * 12 + (run.month - mo)
    if months > 24:
        return "NO", ids, f"최근 라운드가 {rd} 로 24개월보다 오래됨 (코드 판정)"
    # 금액은 숫자가 있고 '비공개·미공개' 같은 표현이 없을 때만 확인된 것으로 본다 ('비공개' 문자열을 금액으로 세지 않게)
    if not re.search(r"\d", amount) or any(u in amount.lower() for u in UNDISCLOSED):
        return "UNKNOWN", [], f"{rd} {c.get('stage')} 라운드 금액 비공개{f' ({amount})' if amount else ''} (코드 판정)"
    return "YES", ids, f"{rd} {c.get('stage')} {amount} — 적격성 검증에서 원문 인용으로 확인 (코드 판정)"


def _nps_line(n: dict) -> str:
    if n.get("status") != "matched":
        return "국민연금 가입 사업장 목록에서 확인 안 됨 (3인 미만 법인이거나 사명 다름)"
    return (f"국민연금 가입자 {n['members']}명({n['ym']}), 최초 가입 {n['first_date']}, "
            f"{'탈퇴' if n['withdrawn'] else '가입 중'}")


def _analysis_text(state: dict) -> str:
    c, t, m, k = state["current"], state.get("tech", {}), state.get("market", {}), state.get("competition", {})
    founders = "; ".join(f"{f['name']}({f['role']}): {f['background']}" for f in t.get("founders", [])) or "확인 불가"
    return "\n".join([
        f"[후보] {c['official_name']} | 단계 {c.get('stage')} ({c.get('round_date') or '시점 미상'}, "
        f"{c.get('round_amount') or '금액 미상'}) | 설립 {c.get('founded_year') or '확인 불가'} | {c.get('one_line')}",
        f"[기술] 제품: {t.get('product')} / 핵심 기술: {t.get('core_technology')} / 성숙도: {t.get('maturity')} "
        f"({t.get('maturity_evidence')}) / 특허·인증: {t.get('ip_evidence')}",
        f"[팀] {t.get('team_assessment')} / 창업자: {founders}",
        f"[고용·창업 시기] {_nps_line(c.get('nps') or {})}",
        f"[시장] 규모: {m.get('market_size')} / 성장: {m.get('growth')} / 지불 의향: {m.get('willingness_to_pay')}",
        f"[경쟁] 차별성: {k.get('differentiation')} / 진입장벽: {k.get('entry_barriers')}",
    ])


def decision_node(state: dict) -> dict:
    cfg = get_config()
    rubric = load_rubric()
    reg = SourceRegistry(state.get("registry"))
    c = state["current"]
    run_date = state.get("run_date") or datetime.now().strftime("%Y-%m-%d")
    pool = list(dict.fromkeys(
        c.get("evidence_ids", []) + state.get("tech", {}).get("pool_ids", [])
        + state.get("market", {}).get("pool_ids", []) + state.get("competition", {}).get("pool_ids", [])))
    pool = [i for i in pool if reg.get(i)]
    company_keys = [_norm(x) for x in (c.get("name_en"), c.get("official_name"), c.get("name")) if x and len(_norm(x)) >= 2]
    analysis = _analysis_text(state)
    passages = _passages(pool, reg)
    bm25 = BM25Okapi([kiwi_tokenize(p[1]) for p in passages])
    judge = structured(Answers, "judge")

    rows, dims, rejected, retried = [], [], [], 0
    for d in rubric["dimensions"]:
        qlist = "\n".join(f"- {q['id']}: {q['text']} (YES 요건: {q['need']})" for q in d["questions"])
        evidence = _evidence_for(d, c["official_name"], passages, bm25, reg)
        res: Answers = judge.invoke(render("decision", dimension=d["name"], questions=qlist, run_date=run_date,
                                           analysis=analysis, evidence=evidence))
        got = {a.qid: a for a in res.answers}
        missing = [q for q in d["questions"] if q["id"] not in got]
        if missing:  # 판정을 빠뜨린 문항만 다시 묻는다
            again: Answers = judge.invoke(render(
                "decision", dimension=d["name"], analysis=analysis, evidence=evidence, run_date=run_date,
                questions="\n".join(f"- {q['id']}: {q['text']} (YES 요건: {q['need']})" for q in missing)))
            got.update({a.qid: a for a in again.answers})
        # 인용이 [근거] 원문에 없는 YES·NO 는 한 번만 다시 묻는다 (분석 요약에서 베껴 온 인용 등). 그래도 없으면 아래에서 UNKNOWN
        bad = [q for q in d["questions"] if q["id"] != "D1" and (a := got.get(q["id"])) and a.verdict in ("YES", "NO")
               and a.quote.strip()
               and not _quote_sources(a.quote, [i for i in a.evidence_ids if i in pool], pool, reg)]
        if bad:
            again = judge.invoke(render(
                "decision", dimension=d["name"], analysis=analysis, evidence=evidence, run_date=run_date,
                questions="\n".join(f"- {q['id']}: {q['text']} (YES 요건: {q['need']}) — 직전 인용 \"{got[q['id']].quote[:60]}\" 은 "
                                    f"[근거] 원문에 없다. [근거]에서 글자 그대로 다시 복사하거나, 없으면 UNKNOWN" for q in bad)))
            got.update({a.qid: a for a in again.answers if a.qid in {q["id"] for q in bad}})
            retried += len(bad)
        yes = unknown = na = 0
        for q in d["questions"]:
            a = got.get(q["id"])
            verdict = a.verdict if a else "UNKNOWN"
            ev = [i for i in (a.evidence_ids if a else []) if i in pool]
            note = a.rationale if a else "판정 누락"
            quote = a.quote if a else ""
            if verdict == "N/A" and not q.get("na_allowed"):
                verdict, note = "UNKNOWN", "N/A 불가 문항 — " + note
            if q["id"] == "D1":  # 투자 라운드는 적격성 검증에서 인용 검증을 마친 값으로 코드가 판정 (LLM 판정은 쓰지 않음)
                verdict, ev, note = _round_rule(c, run_date)
                quote = c.get("stage_quote") or ""  # 적격성 검증에서 원문 확인을 마친 인용 (LLM 인용은 검증 전이라 쓰지 않음)
            elif verdict == "YES":
                fail = None
                ev = _quote_sources(a.quote, ev, pool, reg)
                if not ev:
                    fail = "인용문이 근거 본문에서 확인되지 않음"
                elif not q.get("market_level") and not any(_near_company(a.quote, reg.text(i), company_keys) for i in ev):
                    fail = "인용 주변(±300자)에 회사명이 없음 (업계 일반론이 아니라 이 회사 이야기여야 함)"
                elif q.get("doc_required") and not any(i.startswith("D") for i in ev):
                    fail = "공공·연구기관 문서 근거 없음"
                elif q.get("third_party") and not any(_is_third_party(reg.get(i), company_keys) for i in ev):
                    fail = "회사 자체 발표만 있음 (제3자 근거 필요)"
                elif q.get("third_party") and not any(_is_third_party(reg.get(i), company_keys)
                                                      and not _company_voice(a.quote, reg.text(i), company_keys)
                                                      for i in ev):
                    fail = "회사 측 발언·계획(제3자 근거 아님)"
                elif q.get("commercial") and _planned(a.quote, [reg.text(i) for i in ev]):
                    fail = "실증·예정 단계 (인용 주변에 예정·목표·실증·PoC·시범 표현)"
                elif q.get("commercial") and not _has_scale(a.quote, company_keys):
                    fail = "상용 규모 근거 없음 (인용에 설치 수·면적·두수·고객명이 없음)"
                elif q.get("recent") and not any(_is_recent(reg.get(i), run_date) for i in ev):
                    fail = "최근 24개월 이내 근거 아님"
                elif q.get("recent") and not q.get("market_level") and _events_too_old(f"{a.quote} {a.rationale}", run_date):
                    fail = "언급된 사건 날짜가 모두 평가 기준일로부터 24개월보다 오래됨 (코드 날짜 검사)"
                if fail:
                    verdict, note = "UNKNOWN", f"{fail} → UNKNOWN 강등 ({note})"
                    rejected.append({"qid": q["id"], "reason": fail, "quote": a.quote})
            elif verdict == "NO":  # "근거가 없다"는 NO 가 아니라 UNKNOWN. 반대 사실을 적은 문장이 원문에 있어야 NO
                ev = _quote_sources(a.quote, ev, pool, reg) if a.quote.strip() else []
                if not ev:
                    verdict, note = "UNKNOWN", f"반대 근거 인용이 원문에서 확인되지 않음 → NO 대신 UNKNOWN ({note})"
                elif NO_HEDGE.search(re.sub(r"\s+", "", a.rationale)):  # 이유가 "추정·확인되지 않음·근거가 없어" 류
                    verdict, note = "UNKNOWN", f"반대 사실이 아니라 근거 부족 → NO 대신 UNKNOWN ({note})"
            yes += verdict == "YES"
            unknown += verdict == "UNKNOWN"
            na += verdict == "N/A"
            rows.append({"dim": d["id"], "qid": q["id"], "question": q["text"], "bessemer": q.get("bessemer", ""),
                         "answer": verdict, "evidence_ids": ev if verdict in ("YES", "NO") else [],
                         "quote": quote if verdict in ("YES", "NO") else "", "rationale": note})
        n = len(d["questions"]) - na
        score = d["weight"] * yes / n if n else 0.0
        dims.append({"id": d["id"], "name": d["name"], "weight": d["weight"], "yes": yes, "unknown": unknown,
                     "n": n, "score": round(score, 1)})

    total = round(sum(x["score"] for x in dims), 1)
    verdict_of = {r["qid"]: r["answer"] for r in rows}
    reasons = []
    for k in rubric["deal_killers"]:
        if all(verdict_of.get(q) == v for q, v in k["when"].items()):
            reasons.append(f"Deal-killer {k['id']}")
    ratio = {x["id"]: (x["yes"] / x["n"] if x["n"] else 0) for x in dims}
    for dim_id, min_r in cfg.decision.min_dimension_ratio.items():
        if ratio.get(dim_id, 0) < min_r:
            label = {"founder": "창업자", "market": "시장성"}.get(dim_id, dim_id)
            reasons.append(f"핵심 항목 미달({label} {ratio.get(dim_id, 0):.0%})")
    judged = sum(x["n"] for x in dims)
    unknown_ratio = round(sum(x["unknown"] for x in dims) / judged, 3) if judged else 1.0
    if unknown_ratio > cfg.decision.max_unknown_ratio:
        reasons.append(f"정보 부족(UNKNOWN {unknown_ratio:.0%})")
    threshold = cfg.decision.invest_threshold
    if total < threshold:
        reasons.insert(0, f"점수 미달({total}점)")
    decision = "보류" if reasons else "투자"
    sensitivity = {str(t): ("투자" if total >= t and not [r for r in reasons if not r.startswith("점수 미달")] else "보류")
                   for t in (60, 70, 80)}
    scorecard = {"dims": dims, "rows": rows, "total": total, "threshold": threshold, "reasons": reasons,
                 "rejected_yes": rejected, "quote_retried": retried,
                 "knockouts": reasons, "unknown_ratio": unknown_ratio, "decision": decision,
                 "sensitivity": sensitivity}
    msg = (f"[투자 판단] {c['official_name']}: {total}점, UNKNOWN {unknown_ratio:.0%} → {decision}"
           + (f" ({'; '.join(reasons)})" if reasons else ""))
    print(msg)
    summary = {"name": c["official_name"], "region": c["region"], "segment_id": c["segment_id"],
               "stage": c.get("stage"), "round_date": c.get("round_date"), "total": total, "decision": decision,
               "knockouts": reasons, "dims": dims, "unknown_ratio": unknown_ratio, "sensitivity": sensitivity}
    return {"scorecard": scorecard, "decision": decision, "log": [msg],
            "evaluations": [{**summary, "tech": state.get("tech"), "market": state.get("market"),
                             "competition": state.get("competition"), "scorecard": scorecard, "profile": c}]}
