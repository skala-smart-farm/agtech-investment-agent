"""설계 산출물 생성: docs/design.md (GitHub 에서 읽는 용) + RAG-Design_...pdf (제출용).

표는 rubric.yaml · data/manifest.yaml · graph/state.py · outputs/eval/ 에서 자동으로 채워
설계 문서와 코드가 어긋나지 않게 한다.

    uv run python -m docs.build_design
"""
from __future__ import annotations

import json
import re
from datetime import datetime

import markdown
import pandas as pd
import yaml
from jinja2 import Environment, FileSystemLoader

from core.config import ROOT, get_config, path
from rag.loader import load_manifest, total_pages

def _main_mermaid(cfg) -> str:
    w = cfg.workflow
    return f"""graph TD
  S([START]) --> D["🔭 스타트업 발굴<br/>TIPS·투자기사 피드·뉴스·공공 선정·기업DB·문서 RAG"]
  D --> V["✅ 적격성 검증<br/>G1 상장목록 · G2 단계 인용 · G3 Exit · G4 반증 · G5 분야 · G6 근거량<br/>검증 검색 실패 시 통과 불가"]
  V -->|대기열 있음| SEL["다음 후보 선택"]
  V -->|후보 없음 · 발굴 라운드 남음| D
  V -->|후보 없음 · 라운드 소진| R
  SEL --> T["🔬 기술·팀 분석<br/>기사 본문에서 창업자·기술 추출"]
  SEL --> M["📊 시장성 평가<br/>Agentic RAG: 질문 분해 → 문서/웹 선택"]
  T --> C["🥊 경쟁사 비교<br/>제품 유형 기준"]
  M --> C
  C --> J["🧮 투자 판단<br/>24문항 YES/NO/UNKNOWN · 코드 채점"]
  J -->|투자| R["📝 보고서 생성<br/>모두 보류면 후보별 불가 사유 · 5쪽 검증"]
  J -->|보류 · 대기열 남음| SEL
  J -->|보류 · 대기열 비고 라운드 남음| D
  J -->|"보류 · 평가 상한({w.max_evaluations}곳) 또는 후보 소진"| R
  R --> E([END])"""


def _rag_mermaid(cfg) -> str:
    r = cfg.rag
    return f"""graph LR
  Q["시장 질문"] --> DC["decompose<br/>LLM 하위 질문 3~6개"]
  DC --> RO{{"route<br/>LLM: 문서 / 웹 / 둘 다"}}
  RO -->|문서| RT["retrieve<br/>Kiwi BM25 + Dense · 후보 {r.candidate_k}"]
  RO -->|웹| WS["web 검색<br/>뉴스 최근 1년"]
  RT --> G["grade<br/>Judge LLM 조각별 O/X"]
  G -->|"관련 ≥ {r.min_relevant}"| F["finish<br/>상위 {r.top_k}개 근거 등록"]
  G -->|"부족 · 재작성 {r.max_rewrites}회 미만"| RW["rewrite<br/>질의 재작성"]
  RW --> RT
  G -->|부족 · 재작성 소진| W["web<br/>웹 검색으로 보완"]
  W --> F
  WS --> F"""


# 쓰는/읽는 노드 칸은 코드를 읽고 사람이 적은 값이다 (타입·설명 칸만 graph/state.py 에서 자동 추출)
WRITERS = {
    "domain": "app", "run_date": "app", "registry": "모든 에이전트", "discovery_rounds": "discover",
    "raw_candidates": "discover", "seen": "discover", "screened": "verify", "queue": "verify · select",
    "current": "select", "tech": "tech", "market": "market", "market_cache": "market", "competition": "competition",
    "scorecard": "decide", "iterations": "select", "evaluations": "decide", "decision": "decide", "report": "report",
    "rag_traces": "discover · tech · market", "log": "모든 노드",
}
READERS = {
    "domain": "없음 (실행 기록용, 에이전트는 config 를 읽음)", "run_date": "verify · decide(24개월 판정) · report(작성일)",
    "registry": "모든 에이전트 (근거 인용·REFERENCE)", "discovery_rounds": "라우터(발굴 반복 상한)",
    "raw_candidates": "verify", "seen": "discover(중복 발굴 방지)", "screened": "report(선정 과정·미평가 후보)",
    "queue": "select · 라우터", "current": "tech · market · competition · decide",
    "tech": "competition · decide", "market": "decide", "market_cache": "market(같은 분야 재사용)",
    "competition": "decide", "scorecard": "없음 (report 는 evaluations 안의 후보별 사본을 읽음)",
    "iterations": "라우터(평가 상한)", "evaluations": "report(후보별 사유·최고점 상세)",
    "decision": "라우터(투자면 보고서)", "report": "app(run_log)", "rag_traces": "app(run_log)", "log": "app(run_log)",
}
RESET_BY_SELECT = {"tech", "market", "competition", "scorecard"}


def _state_fields() -> list[dict]:
    src = (ROOT / "graph" / "state.py").read_text(encoding="utf-8")
    body = src[src.index("class InvestmentState"):]
    out = []
    for m in re.finditer(r"^ {4}(\w+): ([^#\n]+?)[ \t]*#[ \t]*([^\n]+)$", body, re.M):
        name, typ, desc = m.group(1), m.group(2).strip(), m.group(3).strip()
        am = re.match(r"Annotated\[(.+), ([\w.]+)\]$", typ)
        typ = f"{am.group(1)} (reducer: {am.group(2)})" if am else typ
        if name in RESET_BY_SELECT:
            desc += " — 후보가 바뀔 때 select 가 초기화"
        out.append({"name": name, "type": typ.replace("|", "\\|"), "writer": WRITERS.get(name, "-"),
                    "reader": READERS.get(name, "-"), "desc": desc.replace("|", "/")})
    return out


def _retrieval_table() -> str:
    f = path("outputs/eval/retrieval_eval.csv")
    if not f.exists():
        return "(eval/eval_retrieval.py 실행 결과 없음)"
    df = pd.read_csv(f).dropna(subset=["MRR@5"])
    piv = (df.groupby(["retriever", "embedding"])[["Hit@1", "Hit@3", "Hit@5", "MRR@5"]].mean().round(3)
           .sort_values("MRR@5", ascending=False).reset_index())
    piv["embedding"] = piv["embedding"].str.replace("dragonkue/", "").str.replace("intfloat/", "")
    return piv.to_markdown(index=False)


def _elig_stale() -> bool:
    """적격성 평가 결과가 현재 적격성 코드보다 오래됐으면 True (설계서에 '재측정 안 됨'을 자동으로 밝히기 위해)."""
    res = path("outputs/eval/eligibility_eval_gold.json")
    code = [path(p) for p in ("agents/eligibility.py", "prompts/eligibility.md", "tools/web_search.py")]
    return not res.exists() or any(c.stat().st_mtime > res.stat().st_mtime for c in code)


def _runtime_table() -> str:
    """실제 파이프라인 설정(후보 8개, 앞 4개 사용)으로 잰 검색기 비교 (eval/eval_final_retriever.py 결과)."""
    d = _json("outputs/eval/runtime_retriever.json")
    rows = {r["name"]: r for r in d.get("rows") or []}
    if not rows:
        return "(eval/eval_final_retriever.py 실행 결과 없음)"
    pick = d.get("recommendation", {}).get("name")
    show = ["final (hybrid)", "dense only", "hybrid 0.5:0.5", "KURE-v1 dense", "KURE-v1 hybrid", pick, "KURE-v1 hybrid 0.5:0.5",
            "Kiwi BM25 only"]
    label = {"final (hybrid)": "snowflake + 하이브리드 0.3:0.7 (처음 선택)", "dense only": "snowflake Dense",
             "hybrid 0.5:0.5": "snowflake + 하이브리드 0.5:0.5", "KURE-v1 dense": "KURE-v1 Dense",
             "KURE-v1 hybrid": "KURE-v1 + 하이브리드 0.3:0.7", "Kiwi BM25 only": "Kiwi BM25 단독"}
    out = ["| 설정 (후보 8개) | Hit@1 | Hit@3 | **Hit@4** | **MRR@4** | Hit@8 | 한국어 문서 Hit@4 | 영어 문서 Hit@4 |",
           "|---|---|---|---|---|---|---|---|"]
    for name in dict.fromkeys(n for n in show if n in rows):
        a, ko, en = rows[name]["all"], rows[name]["ko"], rows[name]["en"]
        lab = label.get(name, name.replace("KURE-v1 hybrid", "KURE-v1 + 하이브리드"))
        lab = f"**{lab} (선택)**" if name == pick else lab
        out.append(f"| {lab} | {a['Hit@1']:.3f} | {a['Hit@3']:.3f} | {a['Hit@4']:.3f} | {a['MRR@4']:.3f} | {a['Hit@8']:.3f} | "
                   f"{ko['Hit@4']:.3f} | {en['Hit@4']:.3f} |")
    return "\n".join(out) + "\n\n표의 수치는 질문 70개 기준이며 1문항 = 0.014. 한국어/영어는 정답 문서의 언어다(질문은 모두 한국어)."


def _json(rel: str) -> dict:
    f = path(rel)
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}


def _read(rel: str, strip_title: bool = True) -> str:
    f = path(rel)
    if not f.exists():
        return ""
    t = f.read_text(encoding="utf-8")
    return re.sub(r"^# .*\n", "", t) if strip_title else t


def build() -> tuple[str, str]:
    cfg = get_config()
    team = cfg.submission
    members = sorted(team.members)
    env = Environment(loader=FileSystemLoader(ROOT / "docs"))
    with open(ROOT / "rubric.yaml", encoding="utf-8") as f:
        rubric = yaml.safe_load(f)
    md = env.get_template("design.md.j2").render(
        team=team, members_line=" · ".join(members), today=datetime.now().strftime("%Y-%m-%d"), cfg=cfg,
        segments=cfg.domain.segments, corpus=load_manifest(), total_pages=total_pages(), rubric=rubric,
        state_fields=_state_fields(), retrieval_table=_retrieval_table(),
        retrieval_decision=_read("outputs/eval/retrieval_decision.md").replace("## ", "#### "),
        eligibility_history=_read("outputs/eval/eligibility_eval_history.md"),
        judge_line=(lambda j: f"Relevance {j['relevance']:.2f} · Faithfulness {j['faithfulness']:.2f} · "
                              f"Correctness {j['correctness']:.2f} · 질의 재작성 {j['rewrite_rate']:.0%} · "
                              f"웹 보완 {j['web_fallback_rate']:.0%}" if j else "(미실행)")(
            _json("outputs/eval/judge_eval.json").get("summary")),
        elig_gold=(lambda j: f"정확도 {j['accuracy']:.2f}" if j else "(미실행)")(
            _json("outputs/eval/eligibility_eval_gold.json").get("summary")),
        elig_holdout=(lambda j: f"정확도 {j['accuracy']:.2f}" if j else "(미실행)")(
            _json("outputs/eval/eligibility_eval_holdout.json").get("summary")),
        elig_stale=_elig_stale(), runtime_table=_runtime_table(), runtime=_json("outputs/eval/runtime_retriever.json"),
        main_mermaid=_main_mermaid(cfg), rag_mermaid=_rag_mermaid(cfg))
    md_path = path("docs/design.md")
    md_path.write_text(md, encoding="utf-8")

    html_body = markdown.markdown(md, extensions=["tables", "fenced_code", "toc", "sane_lists"])
    html_body = re.sub(r'<pre><code class="language-mermaid">(.*?)</code></pre>',
                       lambda m: f'<pre class="mermaid">{m.group(1)}</pre>', html_body, flags=re.S)
    html = HTML.replace("{{BODY}}", html_body)
    pdf = path(f"docs/RAG-Design_{team.campus}-{team['class']}_{'+'.join(members)}.pdf")
    _to_pdf(html, pdf)
    return str(md_path), str(pdf)


def _to_pdf(html: str, pdf) -> None:
    from playwright.sync_api import sync_playwright

    tmp = pdf.with_suffix(".html")
    tmp.write_text(html, encoding="utf-8")
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page()
        pg.goto(tmp.resolve().as_uri(), wait_until="networkidle")
        pg.wait_for_function("() => document.querySelectorAll('pre.mermaid svg').length === "
                             "document.querySelectorAll('pre.mermaid').length", timeout=30000)
        pg.pdf(path=str(pdf), format="A4", print_background=True, display_header_footer=True,
               header_template="<span></span>",
               footer_template='<div style="font-size:8px;width:100%;text-align:center;color:#888">'
                               '<span class="pageNumber"></span> / <span class="totalPages"></span></div>',
               margin={"top": "14mm", "bottom": "16mm", "left": "14mm", "right": "14mm"})
        b.close()
    tmp.unlink(missing_ok=True)


HTML = """<!DOCTYPE html><html lang="ko"><head><meta charset="utf-8">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/pretendard@1.3.9/dist/web/static/pretendard.css">
<script src="https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.min.js"></script>
<style>
 body{font-family:"Pretendard","Apple SD Gothic Neo","Malgun Gothic",sans-serif;font-size:9.6pt;line-height:1.55;color:#1b2330}
 h1{font-size:18pt;color:#12213f;border-bottom:3px solid #12213f;padding-bottom:6px;margin-top:0}
 h2{font-size:13.5pt;color:#12213f;border-bottom:1.5px solid #d8dee9;padding-bottom:3px;margin-top:22px;break-after:avoid}
 h3{font-size:11pt;color:#2f5bd3;margin:14px 0 4px;break-after:avoid}
 h4{font-size:10pt;margin:10px 0 4px}
 table{width:100%;border-collapse:collapse;font-size:8.4pt;margin:6px 0 10px}
 th,td{border:0.6pt solid #cfd6e2;padding:3px 6px;vertical-align:top;text-align:left}
 th{background:#eef2f8} tr{break-inside:avoid}
 code{font-family:Menlo,monospace;font-size:8.3pt;background:#f3f5f9;padding:0 3px;border-radius:3px}
 blockquote{border-left:4px solid #2f5bd3;background:#f5f7fb;margin:8px 0;padding:6px 12px;color:#344054}
 pre.mermaid{background:#fff;text-align:center;break-inside:avoid;page-break-inside:avoid}
 pre.mermaid svg{max-height:820px;max-width:100%}
 a{color:#2f5bd3;text-decoration:none}
</style></head><body>{{BODY}}
<script>mermaid.initialize({startOnLoad:true,theme:"default",flowchart:{htmlLabels:true,curve:"basis"}});</script>
</body></html>"""


if __name__ == "__main__":
    print(build())
