"""README.md 생성: 발표 수치가 제출 실행 결과(outputs/run_log.json)·평가 결과(outputs/eval/)와 항상 같도록 템플릿으로 만든다.

    uv run python -m docs.build_readme
"""
from __future__ import annotations

import json
import re

from jinja2 import Environment, FileSystemLoader

from core.config import ROOT, get_config, path
from rag.loader import load_manifest, total_pages


def _labels() -> dict:
    from agents.decision import load_rubric

    return {q["id"]: q["short"] for d in load_rubric()["dimensions"] for q in d["questions"]}


LABELS = _labels()
GAP_KEYS = ["R1", "R2", "R3", "C1", "P4", "P1", "C2"]


def _nps_example(run: dict) -> str:
    for e in sorted(run.get("evaluations", []), key=lambda e: -e["total"]):
        n = e.get("nps") or {}
        if n.get("status") == "matched":
            return f"{e['name']} 가입자 {n['members']}명, 최초 가입 {n['first_date']}"
    return "평가 후보별 인원·최초 가입일"


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


def build() -> str:
    cfg = get_config()
    run = _j("outputs/run_log.json")
    fr = _j("outputs/eval/final_retriever.json")
    judge = _j("outputs/eval/judge_eval.json").get("summary", {})
    gold = _j("outputs/eval/eligibility_eval_gold.json").get("summary", {})
    hold = _j("outputs/eval/eligibility_eval_holdout.json").get("summary", {})
    m = re.search(r"후보 (\d+)곳 \(교차 신호 2채널 이상 (\d+)곳\)", " ".join(run.get("log", [])))
    team = cfg.submission
    members = "+".join(sorted(team.members))
    evals = sorted(run.get("evaluations", []), key=lambda e: -e["total"])
    top_rows = {r["qid"]: r["answer"] for r in (evals[0]["rows"] if evals else [])}
    strengths = ", ".join(LABELS[q] for q in LABELS if top_rows.get(q) == "YES") or "-"
    gaps = ", ".join(LABELS[q] for q in GAP_KEYS if top_rows.get(q) not in (None, "YES")) or "-"
    md = Environment(loader=FileSystemLoader(ROOT / "docs")).get_template("README.md.j2").render(
        cfg=cfg, run=run, strengths=strengths, gaps=gaps, nps_example=_nps_example(run), search_providers=_search_providers(), r=run.get("report", {}), evals=evals,
        disc={"candidates": m.group(1) if m else "-", "cross": m.group(2) if m else "-"},
        n_screened=len(run.get("screened", [])), n_eligible=sum(1 for s in run.get("screened", []) if s["eligible"]),
        rejected=sum(len(e.get("rejected_yes", [])) for e in run.get("evaluations", [])),
        fr=_A(fr.get("final (hybrid)", {})), fd=_A(fr.get("dense only", {})), fb=_A(fr.get("Kiwi BM25 only", {})),
        judge_line=(f"Relevance {judge['relevance']:.2f} · Faithfulness {judge['faithfulness']:.2f} · "
                    f"Correctness {judge['correctness']:.2f} (재작성 {judge['rewrite_rate']:.0%}, 웹 보완 "
                    f"{judge['web_fallback_rate']:.0%})") if judge else "-",
        elig_gold=f"{gold.get('accuracy', 0):.2f}", elig_holdout=f"{hold.get('accuracy', 0):.2f}",
        n_docs=len(load_manifest()), total_pages=total_pages(),
        design_pdf=f"RAG-Design_{team.campus}-{team['class']}_{members}.pdf",
        report_pdf=f"RAG-Output_{team.campus}-{team['class']}_{members}.pdf")
    out = ROOT / "README.md"
    out.write_text(md, encoding="utf-8")
    return str(out)


class _A(dict):
    """템플릿에서 fr.all / fr.ko / fr.en 으로 접근."""

    def __getattr__(self, k):
        return self[k]


if __name__ == "__main__":
    print(build())
