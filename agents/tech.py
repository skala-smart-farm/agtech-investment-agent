"""🔬 기술·팀 분석 에이전트.

- 회사 고유 정보(제품, 핵심 기술, 특허·논문, 창업자 이력)는 웹 근거로 모은다.
  창업자는 인물 검색이 아니라 "회사가 알려진 기사·인터뷰 속 창업자 이력"으로 확인한다.
- 검색 스니펫에는 창업자 이름·이력이 잘 안 나온다(예: CTO 이름이 기사 본문 1,300자 뒤에만 있음).
  그래서 기사 본문에서 회사명·창업자 표현(대표, CTO, 창업 …) 주변 문단을 골라 스니펫과 함께 LLM 에 넘긴다.
- 기술 수준 비교의 기준선(해당 분야 기술 동향·상용화 수준)은 문서 코퍼스에서 Agentic RAG 로 가져온다.
"""
from __future__ import annotations

import re
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

# 창업자·핵심 인력이 나오는 문단을 찾는 표현 ("대표적"은 제외)
FOUNDER_TERMS = re.compile(r"대표(?!적)|CEO|CTO|공동\s?창업|창업자|창업|[Cc]o-?[Ff]ounder|[Ff]ounder|기술이사|연구소장")
_TITLE = r"(?:대표이사|대표(?!적)|CEO|CTO|COO|공동창업자|창업자|기술이사|연구소장)"
# 사람 이름(한글 3자) + 직함: "이규화 대표", "이규화(28) 메타파머스 대표", "(대표 이규화)", "대표자는 이원준입니다",
# "윤원재 CTO", "Jane Doe, CEO". 회사명이 사이에 끼는 "이원준 조벡스 대표"는 _mentions 가 회사명으로 따로 찾는다
PERSON = re.compile(
    rf"(?<![가-힣])(?P<ko>[가-힣]{{3}})(?:\(\d{{2}}\) (?:[가-힣A-Za-z]{{2,12}} )?| ){_TITLE}"
    rf"|(?:대표이사|대표|CEO|CTO) (?P<ko2>[가-힣]{{3}})(?=[),])"
    r"|대표자는 (?P<ko3>[가-힣]{3})(?=입니다|[\s.,)])"
    r"|(?P<en>[A-Z][a-z]+ [A-Z][a-z]+),? (?:the )?(?:CEO|CTO|[Cc]o-?[Ff]ounder|[Ff]ounder)"
    r"|(?:CEO|CTO|[Cc]o-?[Ff]ounder|[Ff]ounder)(?: and CEO)?,? (?P<en2>[A-Z][a-z]+ [A-Z][a-z]+)")
# 직함 앞에 오지만 사람 이름이 아닌 말
NOT_NAME = {"비롯한", "투자사", "관계자", "운영사", "스타트", "창업주", "공동의", "신임의"}
# 창업자 이력 문단에 자주 나오는 말 (학력·경력) — 이름만 나열된 문단보다 이력 문단을 먼저 고르게 한다
BACKGROUND = re.compile(r"대학|학과|학부|박사|석사|전공|출신|경력|경험|근무|졸업|University|PhD|former", re.I)
CITE_ID = re.compile(r"\b[WD][0-9a-f]{5}\b")


def _covered(passage: str, skip: str) -> bool:
    """문단의 절반 이상이 이미 스니펫에 있으면 True (같은 내용을 두 번 넘기지 않게)."""
    parts = [passage[i:i + 30] for i in range(0, max(1, len(passage) - 30), 30)]
    return bool(skip) and sum(x in skip for x in parts) > len(parts) / 2


def body_passages(body: str, keys: list[str], terms: re.Pattern, n: int = 2, window: int = 400,
                  skip: str = "", boost: re.Pattern | None = BACKGROUND) -> list[str]:
    """본문에서 회사명(keys)·용어(terms) 주변 ±window 자 문단을 최대 n 개 고른다 (서로 겹치지 않게).
    용어·사람 이름+직함·boost 표현이 많이 모이고 회사명이 함께 있는 문단을 먼저 고르고,
    스니펫(skip)에 이미 있는 문단은 건너뛴다."""
    low = body.lower()
    hits = {m.start() for m in terms.finditer(body)}
    for k in keys:
        i = low.find(k.lower())
        while i >= 0:
            hits.add(i)
            i = low.find(k.lower(), i + 1)

    def win(p: int) -> str:
        return body[max(0, p - window): p + window]

    def score(p: int) -> int:
        w = win(p)
        return (len(terms.findall(w)) + 2 * len(PERSON.findall(w)) + (len(boost.findall(w)) if boost else 0)
                + 2 * any(k.lower() in w.lower() for k in keys))

    skip = re.sub(r"\s+", " ", skip or "")
    chosen: list[int] = []
    for p in sorted(hits, key=lambda p: (-score(p), p)):
        if len(chosen) >= n:
            break
        if any(abs(p - q) < 2 * window for q in chosen) or _covered(win(p), skip):
            continue
        chosen.append(p)
    return [("…" if p > window else "") + win(p) + ("…" if p + window < len(body) else "") for p in sorted(chosen)]


def evidence_blocks(reg: SourceRegistry, ids: list[str], keys: list[str], terms: re.Pattern = FOUNDER_TERMS,
                    max_chars: int = 650, n: int = 2, window: int = 400,
                    boost: re.Pattern | None = BACKGROUND) -> dict[str, str]:
    """근거 id → LLM 에 넘길 텍스트 (기술·경쟁·시장 에이전트 공용).
    - 웹 근거: 스니펫(max_chars) + 회사명(keys)이 나오는 근거는 본문 핵심 문단 최대 n 개(±window 자)
    - 문서 조각: 자르지 않고 그대로 (조각 800자 기준, 끝부분 수치가 잘리지 않게)"""
    keys = [k for k in keys if k and len(norm(k)) >= 2]
    out: dict[str, str] = {}
    for sid in dict.fromkeys(ids):
        s = reg.get(sid)
        if not s:
            continue
        if s["kind"] == "doc":
            out[sid] = reg.brief([sid], 1200)
            continue
        block = reg.brief([sid], max_chars)
        body = s.get("body") or ""
        if body and keys and any(norm(k) in norm(reg.text(sid)) for k in keys):
            ps = body_passages(body, keys, terms, n, window, skip=s["snippet"][:max_chars], boost=boost)
            if ps:
                block += "\n" + "\n".join(f"(본문) {p}" for p in ps)
        out[sid] = block
    return out


def _mentions(text: str, companies: list[str] | None = None) -> list[re.Match]:
    """이름+직함 표현. companies 가 있으면 "이원준 조벡스 대표", "조벡스 대표 이원준" 형태도 찾고, 회사명 자체(3자)는 뺀다."""
    companies = [c for c in companies or [] if c]
    pats = [PERSON] + [re.compile(rf"(?<![가-힣])(?P<ko>[가-힣]{{3}}) {re.escape(c)} {_TITLE}"
                                  rf"|{re.escape(c)} {_TITLE} (?P<ko2>[가-힣]{{3}})(?=(?:는|은|이|가|의)?(?![가-힣]))")
                       for c in companies]
    own = {norm(c) for c in companies}
    out = []
    for pat in pats:
        for m in pat.finditer(text or ""):
            who = next(v for v in m.groupdict().values() if v)
            if who not in NOT_NAME and norm(who) not in own:
                out.append(m)
    return sorted(out, key=lambda m: m.start())


def person_mentions(text: str, companies: list[str] | None = None) -> list[str]:
    """사람 이름과 직함이 붙은 표현 (창업자 누락 재시도 판단용)."""
    return list(dict.fromkeys(m.group(0) for m in _mentions(text, companies)))


def _leader_hints(blocks: dict[str, str], keys: list[str], ceo: str | None, limit: int = 8) -> list[str]:
    """회사명이 나오는 근거 안의 '이름 + 직함' 표현. 프로필 대표자 → 회사명 바로 옆(±80자) → 나머지 순으로 고른다
    (같은 기사에 나온 투자사 대표 같은 다른 회사 사람이 앞에 오지 않게)."""
    nk = [norm(k) for k in keys if k and len(norm(k)) >= 2]
    ck = norm(ceo or "")
    ranked = []
    for sid, text in blocks.items():
        if not any(k in norm(text) for k in nk):
            continue
        found = _mentions(text, keys)
        for m in found:
            near = any(k in norm(text[max(0, m.start() - 80): m.end() + 80]) for k in nk)
            rank = 0 if len(ck) >= 2 and ck in norm(m.group(0)) else (1 if near else 2)
            ranked.append((rank, f"'{m.group(0)}' [{sid}]"))
        if len(ck) >= 2 and ck in norm(text) and not any(ck in norm(m.group(0)) for m in found):
            ranked.append((0, f"'{ceo}'(프로필 대표자) [{sid}]"))
    return list(dict.fromkeys(h for _, h in sorted(ranked, key=lambda x: x[0])))[:limit]


def _ground_founders(founders: list, blocks: dict[str, str]) -> list[dict]:
    """근거 텍스트에 이름이 실제로 있는 인물만 남기고, evidence_ids 는 그 이름이 나오는 근거로 맞춘다 (지어낸 인물 차단)."""
    out = []
    for f in founders:
        key = norm(re.split(r"[(/]", f.name)[0])
        found = [sid for sid, t in blocks.items() if len(key) >= 2 and key in norm(t)]
        if not found:
            continue
        ev = [i for i in f.evidence_ids if i in found] or found[:3]
        out.append({**f.model_dump(), "evidence_ids": ev})
    return out


class Founder(BaseModel):
    name: str = Field(description="근거에 나온 이름 그대로")
    role: str = Field(description="대표, 공동창업자, CTO 등")
    background: str = Field(description="학력·경력·이전 창업 등 (근거 id 포함), 모르면 '확인 불가'")
    evidence_ids: list[str] = Field(description="이 인물이 나오는 근거 id")


class TechAnalysis(BaseModel):
    product: str = Field(description="주요 제품·서비스 (근거 id 인용 [W..])")
    core_technology: str = Field(description="핵심 기술과 AI 가 하는 일")
    maturity: Literal["연구", "시제품", "실증", "상용", "확인 불가"]
    maturity_evidence: str = Field(description="성숙도 근거와 그 날짜 (근거 id), 앞으로의 계획은 '예정'으로 구분")
    differentiators: list[str] = Field(description="기술적 차별점, 각 항목 끝에 근거 id")
    weaknesses: list[str] = Field(description="기술 한계·위험, 각 항목 끝에 근거 id")
    ip_evidence: str = Field(description="특허·논문·인증 근거, 없으면 '확인 불가'")
    founders: list[Founder] = Field(description="근거에 이름이 나온 창업자·대표·기술 책임자")
    team_assessment: str = Field(description="팀 역량 평가 (근거 id 인용)")
    growth_signals: list[str] = Field(
        description="회사 성장 신호: 수상, 정부·기관 선정, 투자 유치, 상용화 일정, 인력 규모. 각 항목에 날짜와 근거 id")
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

    # 스니펫 + 본문 속 회사명·창업자 문단 (창업자 이름·이력은 대개 본문에만 있다)
    names = [name, c.get("name_en") or ""]
    blocks = evidence_blocks(reg, ids, names)
    # 적격성 검증 프로필의 대표자·설립일 (공공 데이터 기준, 병합 전에도 동작하도록 .get)
    ceo = c.get("ceo")
    founded = c.get("founded_date") or (str(c["founded_year"]) if c.get("founded_year") else None)
    ctx = dict(name=name, one_line=c.get("one_line", ""), segment=seg["name"], ceo=ceo or "미확인",
               founded=founded or "확인 불가", evidence="\n\n".join(blocks.values()))
    llm = structured(TechAnalysis)
    res: TechAnalysis = llm.invoke(render("tech", **ctx, feedback=""))
    hints = _leader_hints(blocks, names, ceo)
    retried = False
    if not res.founders and hints:  # 근거에 '이름 + 대표' 같은 표현이 있는데 창업자를 비웠으면 한 번만 다시 묻는다
        retried = True
        feedback = ("직전 답의 founders 가 비어 있다. 근거에 다음 인물 표현이 있다: " + "; ".join(hints)
                    + "\n근거 본문에서 이 회사의 대표·공동창업자·기술 책임자인지 확인해 founders 를 채워라. "
                      "사람 이름이 아니거나 다른 회사 사람이면 넣지 마라.")
        res = llm.invoke(render("tech", **ctx, feedback=feedback))

    valid = set(ids)
    out = res.model_dump()
    out["founders"] = _ground_founders(res.founders, blocks)
    # 근거 id 가 달린 성장 신호만 남긴다
    out["growth_signals"] = [g for g in res.growth_signals if set(CITE_ID.findall(g)) & valid]
    cited = (list(res.evidence_ids) + [i for f in out["founders"] for i in f["evidence_ids"]]
             + [i for g in out["growth_signals"] for i in CITE_ID.findall(g)])
    out["evidence_ids"] = [i for i in dict.fromkeys(cited) if i in valid]
    out["pool_ids"] = ids
    who = ", ".join(f"{f['name']}({f['role']})" for f in out["founders"]) or "없음"
    msg = (f"[기술·팀] {name}: 성숙도 {res.maturity}, 창업자 {len(out['founders'])}명({who})"
           f"{' — 재시도' if retried else ''}, 성장 신호 {len(out['growth_signals'])}건, 근거 {len(out['evidence_ids'])}건")
    print(msg)
    return {"registry": reg.data, "tech": out, "log": [msg],
            "rag_traces": [{"agent": AGENT, "question": "기술 동향", "trace": trace}]}
