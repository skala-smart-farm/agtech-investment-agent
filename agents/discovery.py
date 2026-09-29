"""🔭 스타트업 발굴 에이전트.

창업자 검색(링크드인 등)은 사람을 찾는 방법이지, 회사의 창업 시기·투자 단계를 알려 주지 않는다.
그래서 "투자 단계가 드러나는 공개 신호"가 생기는 채널에서 후보를 모은다.

  T. TIPS 공개 목록   : 운영사가 먼저 투자해야 선정되므로 "Seed 급 투자 이력"이 보장된 회사 목록 (대표자·설립일·선정연도 필드)
  W. 투자 기사 피드   : 와우테일 애그테크 카테고리의 투자 유치 기사 (정확한 게시일)
  A. 투자 유치 뉴스   : 라운드(시드·시리즈) 자체가 기사화된다 → 단계·시점이 함께 잡힌다
  B. 공공 프로그램    : 농식품 A-벤처스·스마트농업 우수기업 등 선정 기사 → 초기 기업이 모인다
  C. 스타트업 DB      : THE VC·혁신의숲 기업 페이지 → 설립일·투자 단계가 구조화되어 있다
  D. 수상·전시        : CES 혁신상, 농업기술 창업경진대회 → 기술력 있는 초기 기업
  E. 문서(RAG)       : 코퍼스 보고서의 스타트업 사례·투자 동향
  F. 해외 투자 뉴스   : 해외 라운드 기사 (과제: 국내외 스타트업 조사)

여러 채널에서 동시에 잡힌 후보일수록(교차 신호) 실재하고 활발한 기업일 가능성이 높아 우선순위를 올린다.
검증 슬롯은 지역별 몫(workflow.screen_kr·screen_global)으로 나눠, 해외 후보가 국내 뒤로 밀려 검증되지 못하는 일을 막는다.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from agents.eligibility import region_quota_order
from core.config import get_config, get_segment
from core.llm import structured
from core.prompts import render
from rag.agentic_rag import agentic_rag
from tools.channels import tips_agtech, wowtale_agtech_funding
from tools.listing_check import normalize
from tools.sources import SourceRegistry, today
from tools.web_search import web_search

AGENT = "discovery"


class Found(BaseModel):
    name: str = Field(description="회사명(한글 공식명 우선, 해외는 영문명)")
    name_en: str = Field(description="영문명, 모르면 빈 문자열")
    region: str = Field(description="KR 또는 GLOBAL")
    segment_id: str = Field(description="세부 분야 id 중 하나")
    what: str = Field(description="무엇을 하는 회사인지 한 줄")
    is_startup_candidate: bool = Field(description="대기업·상장사·공공기관·투자사·언론사가 아니고 AgTech 스타트업으로 보이면 true")
    evidence_ids: list[str] = Field(description="이 회사가 언급된 근거 id")


class FoundList(BaseModel):
    companies: list[Found]


def _channel_queries(cfg, round_no: int) -> list[tuple[str, str, dict]]:
    """(채널, 쿼리, 검색 옵션). 2라운드는 표현을 바꾸고 해외 비중을 늘린다."""
    segs = cfg.domain.segments
    q: list[tuple[str, str, dict]] = []
    if round_no == 1:
        for s in segs:
            q.append(("A.투자뉴스", f"{s['ko']} 스타트업 투자 유치 시리즈", {"topic": "news"}))
        q += [
            ("B.공공프로그램", "TIPS 선정 애그테크 스마트팜 AI 스타트업", {"topic": "general", "recent": False}),
            ("B.공공프로그램", "농식품 벤처육성 선정 AI 애그테크 스타트업", {"topic": "general", "recent": False}),
            ("B.공공프로그램", "이달의 A-벤처스 선정 농식품 AI 기업", {"topic": "general", "recent": False}),
            ("B.공공프로그램", "스마트농업 우수기업 선정 AI 스타트업", {"topic": "news"}),
            ("C.스타트업DB", "스마트팜 AI 스타트업 투자 단계", {"topic": "general", "recent": False,
                                                    "include_domains": ["thevc.kr", "innoforest.co.kr"]}),
            ("C.스타트업DB", "농업 로봇 축산 AI 스타트업", {"topic": "general", "recent": False,
                                                 "include_domains": ["thevc.kr", "innoforest.co.kr"]}),
            ("D.수상전시", "CES 혁신상 애그테크 한국 스타트업 AI", {"topic": "news"}),
            ("F.해외투자뉴스", "AI agtech startup raises Series A", {"topic": "news"}),
            ("F.해외투자뉴스", "agricultural robotics startup raises seed funding", {"topic": "news"}),
        ]
    else:
        for s in segs:
            q.append(("A.투자뉴스", f"{s['ko']} 기업 프리A 시드 투자", {"topic": "news", "recent": False}))
            q.append(("F.해외투자뉴스", f"{s['en']} startup funding round", {"topic": "news"}))
        q.append(("B.공공프로그램", "스마트팜 혁신밸리 입주 스타트업 AI", {"topic": "general", "recent": False}))
    return q


def _tips_evidence(t: dict) -> dict:
    """TIPS 목록 한 행 → 웹 근거. 설립일은 연도만 적는다: 본문의 'YYYY-MM-DD' 는 근거 저장소가 게시일로 읽기 때문
    (tools/sources.py _date_from). 정확한 설립일은 적격성 에이전트가 TIPS 스냅샷에서 따로 기록한다(founded_date)."""
    founded = str(t.get("founded") or "")[:4]
    return {"url": "https://jointips.or.kr/network/startups", "title": f"TIPS 창업기업: {t['name']}",
            "content": f"{t['name']} — 대표 {t.get('ceo') or '미상'}, 설립 {founded + '년' if founded else '미상'}, "
                       f"TIPS 선정 {t['sel_year']}년, 운영사(선투자) {t['operator']}, 홈페이지 {t['homepage']}. {t['intro']}"}


def discovery_node(state: dict) -> dict:
    cfg = get_config()
    reg = SourceRegistry(state.get("registry"))
    round_no = state.get("discovery_rounds", 0) + 1
    by_channel: dict[str, list[str]] = {}

    for channel, query, opt in _channel_queries(cfg, round_no):
        ids = web_search(query, reg, AGENT, **opt)
        by_channel.setdefault(channel, []).extend(ids)

    if round_no == 1:  # 구조화 채널: 무료·결정적이라 먼저 쓴다
        for t in sorted(tips_agtech(), key=lambda x: -x["sel_year"])[:40]:
            sid = reg.add_web(_tips_evidence(t), AGENT, "TIPS 공개 목록", today(), key=f"tips:{t['name']}")
            reg.data[sid].update(date=None, date_is_access=True)  # 목록 행이라 게시일이 없다 (소개문 속 날짜를 게시일로 쓰지 않게)
            by_channel.setdefault("T.TIPS선정", []).append(sid)
        for w in wowtale_agtech_funding():
            sid = reg.add_web({"url": w["url"], "title": w["title"], "content": w["title"],
                               "published_date": w["date"]}, AGENT, "와우테일 애그테크 피드", today())
            by_channel.setdefault("W.투자기사피드", []).append(sid)

    rag_traces = []
    if round_no == 1:
        ids, trace = agentic_rag("국내외 애그테크 스타트업 사례와 투자 동향 (기업명 포함)",
                                 "투자 후보 스타트업 발굴", reg, AGENT)
        by_channel["E.문서RAG"] = ids
        rag_traces.append({"agent": AGENT, "question": "스타트업 사례", "trace": trace})

    # 근거 id → 채널 역색인
    id2ch: dict[str, set[str]] = {}
    for ch, ids in by_channel.items():
        for i in ids:
            id2ch.setdefault(i, set()).add(ch)

    all_ids = list(id2ch)
    seg_list = "\n".join(f"- {s['id']}: {s['name']}" for s in cfg.domain.segments)
    extractor = structured(FoundList)
    found: dict[str, dict] = {}
    en_key: dict[str, str] = {}  # 영문명 → 후보 키: 표기만 다른 같은 회사를 하나로 (예: 오차드 로보틱스 = Orchard Robotics)
    for i in range(0, len(all_ids), 12):  # 근거 12개씩 나눠 추출 (긴 입력에서 누락 방지)
        batch = all_ids[i:i + 12]
        res: FoundList = extractor.invoke(render("discovery_extract", domain=cfg.domain.description,
                                                 segments=seg_list, evidence=reg.brief(batch, 700)))
        for c in res.companies:
            if not c.is_startup_candidate:
                continue
            key, alias = normalize(c.name), normalize(c.name_en)
            valid = [e for e in c.evidence_ids if e in id2ch]
            if not key or not valid:
                continue
            key = en_key.setdefault(alias, key) if alias else key
            # 한글 이름만으로는 판단하지 않는다(해외 기업을 한글로 적은 기사도 있음). 지역 값의 한국 표기만 KR 로 맞춘다
            region = "KR" if c.region.strip().upper() in ("KR", "KOREA", "SOUTH KOREA", "한국", "국내") else "GLOBAL"
            item = found.setdefault(key, {"name": c.name, "name_en": c.name_en, "region": region,
                                          "segment_id": get_segment(c.segment_id)["id"], "what": c.what,
                                          "evidence_ids": [], "channels": set()})
            item["evidence_ids"] = list(dict.fromkeys(item["evidence_ids"] + valid))
            for e in valid:
                item["channels"] |= id2ch[e]

    seen = set(state.get("seen", []))
    region_rank = {r: i for i, r in enumerate(cfg.domain.region_priority)}
    cands = [dict(v, channels=sorted(v["channels"]), key=k) for k, v in found.items()
             if k not in seen and normalize(v["name_en"]) not in seen]
    # 교차 신호(채널 수)와 근거량으로 검증 순서를 정한다
    cands.sort(key=lambda c: (region_rank.get(c["region"], 9), -(2 * len(c["channels"]) + min(len(c["evidence_ids"]), 6))))
    # 같은 지역 안의 순위는 두고, 검증 슬롯을 지역별 몫으로 나눈다 (국내를 통째로 앞에 두면 해외 후보가 검증에 오지 못함)
    cands = region_quota_order(cands)
    n = cfg.workflow.max_candidates_per_round
    n_kr = sum(c["region"] == "KR" for c in cands)
    msg = (f"[발굴 {round_no}라운드] 채널 {len(by_channel)}개·근거 {len(all_ids)}건에서 후보 {len(cands)}곳 "
           f"(국내 {n_kr}·해외 {len(cands) - n_kr}, 교차 신호 2채널 이상 {sum(len(c['channels']) >= 2 for c in cands)}곳) "
           f"→ 검증 {min(n, len(cands))}곳(해외 {sum(c['region'] != 'KR' for c in cands[:n])})")
    print(msg)
    return {"registry": reg.data, "raw_candidates": cands, "discovery_rounds": round_no,
            "seen": [k for c in cands[:n] for k in (c["key"], normalize(c["name_en"])) if k],
            "rag_traces": rag_traces, "log": [msg]}
