"""README.md 생성: 발표 수치가 제출 실행 결과(outputs/run_log.json)·평가 결과(outputs/eval/)와 항상 같도록 템플릿으로 만든다.

    uv run python -m docs.build_readme
"""
from __future__ import annotations

import json
import re

from jinja2 import Environment, FileSystemLoader

from core.config import ROOT, get_config, path
from rag.loader import load_manifest, total_pages


LABELS = {"F1": "창업팀 이력", "F2": "기술 책임자 이력", "F3": "최근 마일스톤", "M1": "시장 규모", "M2": "시장 성장",
          "M3": "경제적 효과", "M4": "수익 모델", "P1": "상용 운영", "P2": "제3자 실증", "P3": "자체 기술", "P4": "인허가",
          "C1": "특허", "C2": "경쟁사 대비 차별점", "C3": "데이터 축적 구조", "C4": "전략 파트너 계약", "R1": "매출",
          "R2": "유료 고객·설치 규모", "R3": "재계약", "R4": "민간 매출", "D1": "최근 라운드", "D2": "기관투자자",
          "D3": "기업가치", "D4": "자본집약도"}
GAP_KEYS = ["R1", "R2", "R3", "C1", "P4", "P1", "C2"]


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
    strengths = "·".join(LABELS[q] for q in LABELS if top_rows.get(q) == "YES") or "-"
    gaps = "·".join(LABELS[q] for q in GAP_KEYS if top_rows.get(q) not in (None, "YES")) or "-"
    md = Environment(loader=FileSystemLoader(ROOT / "docs")).get_template("README.md.j2").render(
        cfg=cfg, run=run, strengths=strengths, gaps=gaps, r=run.get("report", {}), evals=evals,
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
