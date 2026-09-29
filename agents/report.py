"""📝 보고서 생성 에이전트.

과제 조건과 평가 결과와의 일치를 코드로 검증한다.
- 맨 앞 SUMMARY: 상황 → 결론 → 근거 → 조건. 결론 줄은 코드가 평가 결과로 쓴다. 개요 문장 금지, A4 절반 이내(렌더링 후 높이 측정)
- 본문이 평가표와 어긋나면(예: 특허 문항이 UNKNOWN 인데 "특허 보유"라고 씀) 다시 쓰게 한다
- 맨 끝 REFERENCE: 본문에 실제로 인용된 근거만, 지정 형식으로
- 전체 5쪽 이내: 넘으면 조판 밀도를 높이고, 그래도 넘으면 줄여 쓴다
"""
from __future__ import annotations

import json
import re
import shutil
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from core.config import get_config, path
from core.llm import structured
from core.prompts import render
from report.render import html_to_pdf, render_html
from tools.grounding import norm
from tools.sources import SourceRegistry, format_reference, reference_key

AGENT = "report"
CITE = re.compile(r"\[\s*([WD][0-9a-f]{5}(?:\s*[,，]\s*[WD][0-9a-f]{5})*)\s*\]")
BANNED = ["본 보고서", "이 보고서", "보고서는", "보고서에서", "보고서의 목적", "판단하는 보고서", "평가하는 보고서",
          "목적으로 작성", "살펴보", "다음과 같", "개요", "소개하"]
EVALUATIVE = ["우수", "탁월", "뛰어난", "선도적", "독보적", "혁신적인", "압도적"]
# 평가표에서 확인되지 않은 사실을 본문이 단정하는지 찾는 규칙: (문항, 패턴, 설명)
CLAIMS = [
    ("C1", r"특허[^.。\n]{0,15}(보유|확인|등록|출원|확보)", "특허"),
    ("R1", r"매출[^.。\n]{0,12}(\d|기록|달성)", "매출"),
    ("R2", r"(유료 고객|고객 수|설치 농가|도입 농가)[^.。\n]{0,12}\d", "고객·설치 규모"),
    ("P4", r"(인허가|검정|인증)[^.。\n]{0,10}(획득|취득|받았|완료)", "인허가·검정"),
    ("C2", r"차별화(된다|되어 있|를 갖|했다)|경쟁 우위(를|가)? ?(확보|갖)", "경쟁사 대비 차별성"),
    ("P1", r"상용 (운영|판매)[^.。\n]{0,6}(중이|하고 있)", "상용 운영"),
]
NEGATION = re.compile(r"않|불가|미확인|없|못|필요|여부")


class Risk(BaseModel):
    type: Literal["시장", "기술", "규제", "경쟁"]
    content: str = Field(description="리스크 내용 (근거 id 인용)")
    evidence_ids: list[str] = Field(description="리스크 근거 id (1개 이상)")
    due_diligence: str = Field(description="실사에서 확인할 항목 또는 투자 조건")


class Draft(BaseModel):
    situation: str = Field(description="SUMMARY 상황: 대상 회사와 시장의 현재 상태 한 문장 (근거 id, 110자 이내)")
    key_evidence: str = Field(description="SUMMARY 근거: 확인된 강점과 확인되지 않은 핵심 정보 대비 (근거 id, 160자 이내)")
    condition: str = Field(description="SUMMARY 조건: 투자 조건 또는 실사에서 확인할 항목 (120자 이내)")
    problem: str = Field(description="1장 해결하는 문제 (근거 id)")
    product: str = Field(description="1장 제품·핵심 컨셉 (근거 id)")
    revenue_model: str = Field(description="1장 수익 방식, 확인 안 되면 '확인 불가'")
    market: str = Field(description="2장 시장 규모·성장성 (국내·글로벌 수치는 [D..] 문서 인용)")
    tech_team: str = Field(description="3장 기술력과 팀 (근거 id)")
    industry_baseline: str = Field(description="3장 업계 기술 수준 대비 위치 한 문장 ([D..] 문서 인용 필수)")
    competition: str = Field(description="4장 경쟁 구도 요약 (표는 코드가 붙임)")
    risks: list[Risk] = Field(description="시장·기술·규제·경쟁 리스크 각 1개 이상")
    decision_rationale: str = Field(description="6장 투자 판단 근거와 조건")
    data_limits: list[str] = Field(description="7장 데이터 한계 1~2개")


def _conclusion(target: dict, evals: list[dict], any_invest: bool) -> str:
    sc = target["scorecard"]
    if any_invest:
        return (f"투자 권고 — {target['name']} 총점 {sc['total']}/100 (기준 {sc['threshold']}점), "
                f"핵심 항목·Deal-killer·정보량 기준 충족")
    best = f"최고점 {target['name']} {sc['total']}/100, 기준 {sc['threshold']}점"
    return f"투자 권고 대상 없음 — 심층 평가 {len(evals)}곳 모두 보류 ({best}; {', '.join(sc['reasons'])})"


def summary_lines(draft: Draft, conclusion: str) -> list[str]:
    """SUMMARY 4줄. 결론 줄은 코드가 평가 결과로 쓴다 (순서·수치 오류 방지)."""
    return [f"상황: {draft.situation}", f"결론: {conclusion}", f"근거: {draft.key_evidence}",
            f"조건: {draft.condition}"]


def _summary_problems(lines: list[str], limit: int) -> list[str]:
    text = " ".join(lines)
    probs = [f"SUMMARY 에 금지 표현 '{b}'" for b in BANNED if b in text]
    plain = CITE.sub("", text)
    if len(plain) > limit:
        probs.append(f"SUMMARY {len(plain)}자 > {limit}자 (각 칸 길이 상한을 지켜라)")
    return probs


def _consistency_problems(draft: Draft, rows: list[dict]) -> list[str]:
    """본문이 평가표와 어긋나는 단정, 평가성 형용사, 근거 없는 리스크·시장 수치를 찾는다."""
    verdict = {r["qid"]: r["answer"] for r in rows}
    fields = [draft.situation, draft.key_evidence, draft.problem, draft.product, draft.revenue_model, draft.market,
              draft.tech_team, draft.industry_baseline, draft.competition, draft.decision_rationale]
    text = "\n".join(fields)
    probs = []
    for qid, pat, label in CLAIMS:
        if verdict.get(qid) == "YES":
            continue
        for m in re.finditer(pat, text):
            if not NEGATION.search(text[m.start(): m.end() + 12]):
                probs.append(f"'{m.group(0)}' — {qid}({label})는 평가표에서 {verdict.get(qid)} 이므로 확인됐다고 쓰지 마라")
                break
    plain = re.sub(r"우수기업|우수 기업|우수벤처|우수 벤처", "", text)  # 공식 프로그램 이름은 평가성 표현이 아니다
    probs += [f"평가성 표현 '{w}' 대신 사실과 판정 근거로 써라" for w in EVALUATIVE if w in plain]
    if not re.search(r"\[[^\]]*D[0-9a-f]{5}", draft.market):
        probs.append("2장 시장 수치에 공공·연구기관 문서([D..]) 인용이 없다")
    if not re.search(r"\[[^\]]*D[0-9a-f]{5}", draft.industry_baseline):
        probs.append("3장 업계 기술 수준 문장에 문서([D..]) 인용이 없다")
    if any(not r.evidence_ids for r in draft.risks):
        probs.append("근거 id 가 없는 리스크가 있다")
    if {r.type for r in draft.risks} < {"시장", "기술", "규제", "경쟁"}:
        probs.append("리스크 유형(시장·기술·규제·경쟁)을 하나씩은 다 써라")
    return probs


def _method_limits(cfg, evals: list[dict], screened: list[dict]) -> list[str]:
    eligible = sum(r["eligible"] for r in screened)
    out = ["근거를 찾지 못한 문항(UNKNOWN)을 0점으로 처리하는 보수적 채점이라, 공개 정보가 적은 초기 기업일수록 점수가 낮게 나온다",
           f"판정은 {cfg.models.judge} 가 하며, 코드는 근거 인용이 원문에 있는지만 검증한다 (근거 해석 오류는 남을 수 있음)"]
    if eligible > len(evals):
        out.append(f"평가 상한({cfg.workflow.max_evaluations}곳) 때문에 적격 후보 {eligible - len(evals)}곳은 심층 평가하지 않았다")
    return out


def _renumber(draft: dict, tables: dict, reg: SourceRegistry) -> tuple[dict, dict, list[dict]]:
    """[W..]/[D..] 인용을 등장 순서대로 [1], [2] 로 바꾸고, 실제 인용된 근거만 REFERENCE 로 만든다."""
    order: dict[str, int] = {}
    refs: list[dict] = []

    def repl(m: re.Match) -> str:
        nums = []
        for sid in re.split(r"\s*[,，]\s*", m.group(1)):
            s = reg.get(sid)
            if not s:
                continue
            k = reference_key(s)
            if k not in order:
                order[k] = len(order) + 1
                refs.append({"n": order[k], "text": format_reference(s), "kind": s["kind"]})
            if order[k] not in nums:
                nums.append(order[k])
        return "[" + ", ".join(map(str, sorted(nums))) + "]" if nums else ""

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


def _competitor_rows(target: dict, reg: SourceRegistry) -> list[dict]:
    """근거 본문에 이름이 실제로 나오는 경쟁사만 표에 남긴다 (지어낸 경쟁사 차단)."""
    comp = target["competition"]
    pool_text = norm(" ".join(reg.text(i) for i in comp.get("pool_ids", [])))
    rows = []
    for c in comp.get("competitors", []):
        key = norm(c["name"].split("(")[0])
        if len(key) >= 2 and key in pool_text:
            rows.append(c)
    return rows or comp.get("competitors", [])


def _judgment_rows(sc: dict) -> list[dict]:
    """판정 근거 id 를 이유 끝에 붙여 REFERENCE 로 이어지게 한다."""
    out = []
    for r in sc["rows"]:
        ids = r.get("evidence_ids") or []
        cite = f" [{', '.join(ids)}]" if ids and r["answer"] in ("YES", "NO") else ""
        out.append({**r, "rationale": r["rationale"] + cite})
    return out


def report_node(state: dict) -> dict:
    cfg = get_config()
    reg = SourceRegistry(state.get("registry"))
    evals = state.get("evaluations", [])
    screened = state.get("screened", [])
    if not evals:
        return _no_candidate_report(state, cfg, screened)
    invested = [e for e in evals if e["decision"] == "투자"]
    target = invested[0] if invested else max(evals, key=lambda e: e["total"])
    prof, sc = target["profile"], target["scorecard"]
    conclusion = _conclusion(target, evals, bool(invested))
    verdict = {r["qid"]: r["answer"] for r in sc["rows"]}
    tech = {k: v for k, v in target["tech"].items() if k != "pool_ids"}
    if verdict.get("C1") != "YES":
        tech["ip_evidence"] = "확인 불가 (평가표 C1 미확인)"
    if verdict.get("C2") != "YES":
        tech["differentiators"] = [f"(회사 측 주장, 제3자 비교 근거 없음) {x}" for x in tech.get("differentiators", [])]
    pool = list(dict.fromkeys(prof.get("evidence_ids", []) + target["tech"].get("evidence_ids", [])
                              + target["market"].get("evidence_ids", []) + target["competition"].get("evidence_ids", [])
                              + [i for r in sc["rows"] for i in (r.get("evidence_ids") or [])]))
    ctx = {
        "domain": cfg.domain.name, "decision": target["decision"], "any_invest": bool(invested), "conclusion": conclusion,
        "profile": json.dumps({k: prof.get(k) for k in ("official_name", "region", "one_line", "founded_year", "stage",
                                                        "round_date", "round_amount")}, ensure_ascii=False),
        "tech": json.dumps(tech, ensure_ascii=False),
        "market": json.dumps({k: v for k, v in target["market"].items() if k != "pool_ids"}, ensure_ascii=False),
        "competition": json.dumps({k: v for k, v in target["competition"].items() if k != "pool_ids"},
                                  ensure_ascii=False),
        "scorecard": json.dumps({"total": sc["total"], "threshold": sc["threshold"], "reasons": sc["reasons"],
                                 "dims": sc["dims"]}, ensure_ascii=False),
        "verified": "\n".join(f"- {r['qid']}: \"{r['quote']}\" [{', '.join(r['evidence_ids'])}]"
                              for r in sc["rows"] if r["answer"] == "YES" and r.get("quote")) or "(없음)",
        "unverified": ", ".join(f"{r['qid']}({r['question'][:26]})" for r in sc["rows"] if r["answer"] != "YES"),
        "others": json.dumps([{k: e[k] for k in ("name", "total", "decision")} for e in evals], ensure_ascii=False),
        "evidence": reg.brief(pool, 420),
    }

    # 1) 초안 → SUMMARY 규칙 + 평가표 일치 검사, 어긋나면 이유를 알려 주고 다시 쓴다 (최대 3회)
    draft, feedback, probs = None, "", []
    for _ in range(3):
        draft = structured(Draft).invoke(render("report", **ctx, feedback=feedback, shorten=False))
        probs = (_summary_problems(summary_lines(draft, conclusion), cfg.report.summary_max_chars)
                 + _consistency_problems(draft, sc["rows"]))
        if not probs:
            break
        feedback = "직전 초안의 문제를 모두 고쳐라:\n- " + "\n- ".join(probs)

    tables = {"competitors": _competitor_rows(target, reg), "rows": _judgment_rows(sc)}
    title = (f"{prof['official_name']} 투자 검토 — 투자 권고" if invested
             else f"AgTech AI 스타트업 투자 검토 — 투자 권고 대상 없음 (최고점: {prof['official_name']})")
    team = cfg.submission
    out_dir = path(f"{cfg.report.output_dir}/.keep").parent
    pdf_path = out_dir / "investment_report.pdf"
    method_limits = _method_limits(cfg, evals, screened)

    # 2) 렌더링 → 분량 검증. 5쪽을 넘으면 조판 밀도를 올리고, 그래도 넘으면 줄여 쓴다
    result, density, shorten_round = {}, 0, 0
    while True:
        dd = draft.model_dump()
        dd["summary"] = summary_lines(draft, conclusion)
        dd["limitations"] = dd["data_limits"][:2] + method_limits
        d, t, refs = _renumber(dd, tables, reg)
        html = render_html({
            "title": title, "run_date": state.get("run_date"), "domain": cfg.domain.name, "decision": target["decision"],
            "badge": "투자 권고" if invested else "투자 권고 대상 없음", "target": target, "profile": prof, "draft": d,
            "tables": t, "scorecard": sc, "evals": sorted(evals, key=lambda e: -e["total"]), "refs": refs,
            "c2_verified": verdict.get("C2") == "YES",
        }, density=density)
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
        draft = structured(Draft).invoke(render("report", **ctx, feedback="분량 초과", shorten=True))

    submit_name = f"RAG-Output_{team.campus}-{team['class']}_{'+'.join(sorted(team.members))}.pdf"
    shutil.copyfile(pdf_path, out_dir / submit_name)
    md_path = out_dir / "investment_report.md"
    md_path.write_text(_to_markdown(title, d, t, refs, sc, target, evals, state.get("run_date"), verdict),
                       encoding="utf-8")
    checks = {
        "pages": result["pages"], "max_pages": cfg.report.max_pages, "pages_ok": result["pages"] <= cfg.report.max_pages,
        "summary_ratio_of_a4": result["summary_ratio"], "summary_ok": result["summary_ratio"] <= 0.5,
        "summary_rule_violations": _summary_problems(summary_lines(draft, conclusion), cfg.report.summary_max_chars),
        "scorecard_consistency_violations": _consistency_problems(draft, sc["rows"]),
        "references": len(refs), "density_level": density, "first_section": "SUMMARY", "last_section": "REFERENCE",
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


def _to_markdown(title, d, t, refs, sc, target, evals, run_date, verdict) -> str:
    vs = "대상 대비" if verdict.get("C2") == "YES" else "대상 대비 (회사 측 주장 포함)"
    L = [f"# {title} ({run_date})", "", "## SUMMARY", *[f"- {s}" for s in d["summary"]], "",
         "## 1. 사업 아이디어", f"- 해결하는 문제: {d['problem']}", f"- 제품·핵심 컨셉: {d['product']}",
         f"- 수익 방식: {d['revenue_model']}",
         f"- 투자 단계: {target['profile'].get('stage')} ({target['profile'].get('round_date') or '시점 미상'}, "
         f"{target['profile'].get('round_amount') or '금액 미공개'})", "",
         "## 2. 시장 규모와 성장성", d["market"], "",
         "## 3. 기술력과 팀", d["tech_team"], "", f"업계 기술 수준 대비: {d['industry_baseline']}", "",
         "## 4. 경쟁 구도", d["competition"], "",
         f"| 경쟁사 | 국가 | 제품·접근 | 규모 | {vs} |", "|---|---|---|---|---|",
         *[f"| {c['name']} | {c['country']} | {c['offering']} | {c['scale']} | {c['vs_target']} |" for c in t["competitors"]],
         "", "## 5. 사업 리스크", "| 유형 | 내용 | 실사 항목·투자 조건 |", "|---|---|---|",
         *[f"| {r['type']} | {r['content']} | {r['due_diligence']} |" for r in d["risks"]], "",
         f"## 6. 투자 판단: {target['decision']} ({sc['total']}점 / 기준 {sc['threshold']}점)", d["decision_rationale"], "",
         "| 항목 | 가중치 | YES/판정 문항 | UNKNOWN | 점수 |", "|---|---|---|---|---|",
         *[f"| {x['name']} | {x['weight']} | {x['yes']}/{x['n']} | {x['unknown']} | {x['score']} |" for x in sc["dims"]],
         "", f"보류 사유: {', '.join(sc['reasons']) or '없음'} / 기준점 민감도: "
             + ", ".join(f"{k}점 {v}" for k, v in sc["sensitivity"].items()), "",
         "| ID | 판정 | 근거·이유 |", "|---|---|---|", *[f"| {r['qid']} | {r['answer']} | {r['rationale']} |" for r in t["rows"]],
         "", "| 심층 평가 후보 | 총점 | 판단 |", "|---|---|---|",
         *[f"| {e['name']} | {e['total']} | {e['decision']} |" for e in sorted(evals, key=lambda e: -e["total"])], "",
         "## 7. 한계점", *[f"- {x}" for x in d["limitations"]], "", "## REFERENCE",
         *[f"{r['n']}. {r['text']}" for r in refs]]
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
