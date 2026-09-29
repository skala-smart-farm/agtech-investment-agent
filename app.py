"""AgTech 스타트업 투자 평가 에이전트 실행 스크립트.

    uv run python app.py                 # 발굴 → 검증 → 평가 → 보고서(PDF) 생성
    uv run python app.py --graph-only    # 그래프 구조 이미지(docs/architecture.png)만 생성
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")  # 라이브러리 경고로 진행 로그가 묻히지 않게
sys.stdout.reconfigure(line_buffering=True)  # 파일로 내보낼 때도 진행 로그가 바로 보이게

from core.config import ROOT, get_config, path
from core.config import run_date as get_run_date
from core.cost import TRACKER
from graph.builder import build_graph
from tools import web_search as web_search_mod


def save_graph_image(app) -> str:
    """README Architecture 이미지 2종.
    - docs/architecture.png          : 설계서와 같은 한글 설명 그림
    - docs/architecture_langgraph.png: 컴파일된 LangGraph 가 직접 그린 그림 (코드와 설계가 같은지 확인용)"""
    from docs.build_design import MAIN_MERMAID
    from report.mermaid import mermaid_to_png

    mmd = app.get_graph().draw_mermaid()
    path("docs/architecture_langgraph.mmd").write_text(mmd, encoding="utf-8")
    mermaid_to_png(mmd, path("docs/architecture_langgraph.png"))
    out = path("docs/architecture.png")
    mermaid_to_png(MAIN_MERMAID, out)
    return str(out)


def _check_browser() -> None:
    """PDF 는 마지막 단계에서 만들기 때문에, Chromium 이 없으면 처음부터 알려 준다."""
    from pathlib import Path

    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        if not Path(p.chromium.executable_path).exists():
            raise SystemExit("PDF 생성용 브라우저가 없습니다. 먼저 `uv run playwright install chromium` 을 실행하세요.")


def main() -> None:
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph-only", action="store_true")
    parser.add_argument("--fresh", action="store_true",
                        help="저장소에 포함된 재현용 캐시(replay/) 대신 웹·LLM 을 새로 호출해 최신 정보로 평가")
    parser.add_argument("--offline", action="store_true",
                        help="재현 테스트: 캐시에 없는 검색·LLM 호출이 생기면 바로 실패 (API 키 없이 실행 가능)")
    parser.add_argument("--retry-failed", action="store_true",
                        help="재현용 캐시에 실패로 표시된 검색만 다시 시도 (검색 한도 초과로 비었던 근거를 새 키로 보강)")
    args = parser.parse_args()
    if args.retry_failed:
        os.environ["SEARCH_RETRY_FAILED"] = "1"
    if args.offline:
        os.environ["REPLAY_OFFLINE"] = "1"
    _check_browser()

    cfg = get_config()
    if args.fresh:
        cfg["cache"]["dir"] = "cache_fresh"
    app = build_graph()
    if args.graph_only:
        print(save_graph_image(app))
        return

    t0 = time.time()
    run_date = get_run_date()
    meta = path(f"{cfg.cache.dir}/run_meta.json")
    if not meta.exists():  # 처음 실행한 날을 기준일로 기록 → 같은 캐시로 다시 돌리면 같은 날짜 기준으로 판정
        meta.write_text(json.dumps({"run_date": run_date}), encoding="utf-8")
    print(f"== AgTech 스타트업 투자 평가 시작 ({run_date}, 생성 {cfg.models.generator}, 판정 {cfg.models.judge}, "
          f"임베딩 {cfg.embedding.model}, 캐시 {cfg.cache.dir}/) ==")
    state = app.invoke({"domain": cfg.domain.name, "run_date": run_date, "registry": {}, "discovery_rounds": 0,
                        "iterations": 0, "queue": [], "seen": []},
                       {"recursion_limit": 100})
    elapsed = round(time.time() - t0, 1)

    report = state.get("report") or {}
    run_log = {
        "run_date": run_date, "elapsed_sec": elapsed,
        "models": dict(cfg.models), "embedding": cfg.embedding.model,
        "discovery_rounds": state.get("discovery_rounds"), "evaluated": state.get("iterations"),
        "screened": [{k: r.get(k) for k in ("name", "region", "stage", "round_date", "founded_date", "ceo", "eligible",
                                            "reason", "gate_search_failed", "channels")}
                     for r in state.get("screened", [])],
        "evaluations": [{**{k: e[k] for k in ("name", "total", "decision", "knockouts", "dims", "unknown_ratio")},
                         "nps": (e.get("profile") or {}).get("nps"),
                         "rows": [{k: r.get(k) for k in ("qid", "answer", "rationale", "quote", "evidence_ids")}
                                  for r in e["scorecard"]["rows"]],
                         "rejected_yes": e["scorecard"].get("rejected_yes", []),
                         "quote_retried": e["scorecard"].get("quote_retried", 0)}
                        for e in state.get("evaluations", [])],
        "report": report, "rag_traces": state.get("rag_traces", []), "log": state.get("log", []),
        "sources_collected": len(state.get("registry", {})),
        "failed_searches": sorted({(f["agent"], f["query"]) for f in getattr(web_search_mod, "FAILED_QUERIES", [])}),
        "llm_cost": TRACKER.summary(),
    }
    # 공개 저장소에 로컬 절대경로가 남지 않게 저장소 기준 상대경로로 적는다
    text = json.dumps(run_log, ensure_ascii=False, indent=2).replace(str(ROOT) + "/", "")
    path(f"{cfg.report.output_dir}/run_log.json").write_text(text, encoding="utf-8")
    cost = TRACKER.summary()
    if cost.get("api_calls"):  # 실제로 API 를 부른 실행만 비용 기록을 남긴다 (캐시 재생 실행은 0원이라 남기지 않음)
        with open(path(f"{cfg.report.output_dir}/cost_history.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps({"run_date": run_date, "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                                "elapsed_sec": elapsed, "fresh": args.fresh, **cost}, ensure_ascii=False) + "\n")
    print(f"== 완료 ({elapsed}초, LLM {TRACKER.summary()}) ==")
    if report:
        print(f"보고서: {report['submit_pdf']}")
        print(f"검증: {json.dumps(report['checks'], ensure_ascii=False)}")
    else:
        print("평가를 통과한 후보가 없어 보고서를 만들지 못했습니다. outputs/run_log.json 을 확인하세요.")


if __name__ == "__main__":
    main()
