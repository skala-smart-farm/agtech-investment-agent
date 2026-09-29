"""README.md 생성: 발표 수치가 제출 실행 결과(outputs/run_log.json)·평가 결과(outputs/eval/)와 항상 같도록 템플릿으로 만든다.

    uv run python -m docs.build_readme
"""
from __future__ import annotations

import json
import re

from jinja2 import Environment, FileSystemLoader

from core.config import ROOT, get_config, path
from rag.loader import load_manifest, total_pages


def _search_providers() -> str:
    """재현용 캐시에 실제로 결과를 준 검색 공급자만 적는다 (키를 받지 못한 공급자를 쓴 것처럼 쓰지 않게)."""
    used = set()
    for f in path(f"{get_config().cache.dir}/search").glob("*.json"):
        used |= {r.get("provider", "tavily") for r in json.loads(f.read_text(encoding="utf-8"))}
    names = [n for p, n in (("serper", "Serper(구글)"), ("tavily", "Tavily")) if p in used]
    return "·".join(names) or "Tavily"


def _j(rel: str) -> dict:
    f = path(rel)
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}


def _runtime(cfg) -> tuple[str, str, str]:
    """지금 설정(임베딩·가중치)과 같은 행을 실제 파이프라인 설정 측정표(outputs/eval/runtime_retriever.json)에서 찾는다."""
    w = [float(x) for x in cfg.rag.ensemble_weights]
    label = "Dense 단독" if w[0] == 0 else f"하이브리드 BM25 {w[0]:g} : Dense {w[1]:g}"
    rows = _j("outputs/eval/runtime_retriever.json").get("rows", [])
    row = next((r for r in rows if r.get("embedding") == cfg.embedding.model
                and [float(x) for x in r.get("weights[bm25,dense]", [])] == w), None)
    if not row:
        return label, "-", "-"
    return label, f"{row['all']['Hit@4']:.3f}", f"{row['all']['MRR@4']:.3f}"


def _elig_stale() -> bool:
    res = path("outputs/eval/eligibility_eval_gold.json")
    code = [path(p) for p in ("agents/eligibility.py", "prompts/eligibility.md", "tools/web_search.py")]
    return not res.exists() or any(c.stat().st_mtime > res.stat().st_mtime for c in code)


def _pc_line() -> str:
    d = _j("outputs/eval/positive_control.json")
    rows = d.get("rows") or d.get("results") or []
    if not rows:
        return "(미실행)"
    inv = [r for r in rows if r.get("decision") == "투자"]
    best = max(rows, key=lambda r: r.get("total", 0))
    return (f"{len(rows)}곳 중 투자 {len(inv)}곳 · 최고 {best.get('name')} {best.get('total')}점"
            + ("" if inv else " — 공개 정보만으로는 기준(70점)을 넘지 못함"))


def _common_cause(evals: list[dict]) -> str:
    """보류 사유 중 가장 많이 겹친 것 (코드가 run_log 에서 센다)."""
    from collections import Counter

    c = Counter(re.sub(r"\(.*?\)", "", k).strip() for e in evals for k in e.get("knockouts", []))
    c.pop("점수 미달", None)
    return ", ".join(f"{k} {v}곳" for k, v in c.most_common(2)) or "점수 미달"


def _contributors() -> str:
    f = path("docs/contributors.md")
    return f.read_text(encoding="utf-8").strip() if f.exists() else "(조원별 수행 역할 확인 중)"


def build() -> str:
    cfg = get_config()
    run = _j("outputs/run_log.json")
    judge = _j("outputs/eval/judge_eval.json").get("summary", {})
    gold = _j("outputs/eval/eligibility_eval_gold.json").get("summary", {})
    hold = _j("outputs/eval/eligibility_eval_holdout.json").get("summary", {})
    m = re.search(r"후보 (\d+)곳", " ".join(run.get("log", [])))
    team = cfg.submission
    members = "+".join(sorted(team.members))
    evals = sorted(run.get("evaluations", []), key=lambda e: -e["total"])
    screened = run.get("screened", [])
    n_eligible = sum(1 for s in screened if s["eligible"])
    label, hit4, mrr4 = _runtime(cfg)
    md = Environment(loader=FileSystemLoader(ROOT / "docs")).get_template("README.md.j2").render(
        cfg=cfg, run=run, r=run.get("report", {}), evals=evals, search_providers=_search_providers(),
        disc={"candidates": m.group(1) if m else "-"}, n_eligible=n_eligible,
        n_unevaluated=max(0, n_eligible - len(evals)), common_cause=_common_cause(evals),
        retrieval_label=label, rt_hit4=hit4, rt_mrr4=mrr4, pc_line=_pc_line(), elig_stale=_elig_stale(),
        judge_line=(f"Relevance {judge['relevance']:.2f} · Faithfulness {judge['faithfulness']:.2f} · "
                    f"Correctness {judge['correctness']:.2f}") if judge else "-",
        elig_gold=f"{gold.get('accuracy', 0):.2f}", elig_holdout=f"{hold.get('accuracy', 0):.2f}",
        n_docs=len(load_manifest()), total_pages=total_pages(), run_cost=cfg.report.get("run_cost_usd", "0.4"),
        contributors=_contributors(),
        report_pdf=f"RAG-Output_{team.campus}-{team['class']}_{members}.pdf")
    out = ROOT / "README.md"
    out.write_text(md, encoding="utf-8")
    return str(out)


if __name__ == "__main__":
    print(build())
