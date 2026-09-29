"""런타임 설정 그대로의 검색기 성능과 대안 비교: 질문 70개 전체·정답 문서 언어별·질문 세트별 Hit@1/3/4/5/8, MRR@4.

런타임 동작(rag/agentic_rag.py): get_hybrid_retriever().invoke(질의)[:candidate_k] → Judge 가 조각별 관련 O/X →
관련 조각을 검색 순서 그대로 두고 앞 top_k(4)개만 컨텍스트로 쓴다(재정렬 없음).
- Hit@4 · MRR@4 : 파이프라인 지표. 정답 조각이 검색 순서 4위 안이면 Judge 가 앞 조각을 모두 관련으로 판정해도 컨텍스트에 들어간다
- Hit@8 : Judge 가 정답만 골라낸다고 가정한 상한 (Judge 가 보는 후보 8개 안 적중)
모든 설정을 런타임과 같은 k=candidate_k 로 잰다 (BM25·Dense 가 각각 k개, 하이브리드는 RRF 로 합친 순서의 앞 k개).
질문은 모두 한국어이고, ko/en 은 정답 문서의 언어다.

추천 규칙 (측정 전에 정해 두고 코드가 기계적으로 적용):
  1) 전체 70문항 Hit@4 가 가장 높은 설정을 고른다
  2) Hit@4 가 같으면 MRR@4 가 높은 설정 (완전 동률이면 표의 앞쪽 = 현재 런타임 설정이 맨 앞)
  3) 1위가 현재 설정 임베딩이 아니고, 현재 임베딩으로 되는 설정(재색인 불필요) 중 최고가
     Hit@4 · MRR@4 모두 1위보다 0.01 이하로만 낮으면 그 설정을 추천한다
  (판정은 결과 파일에 적힌 소수 셋째 자리 값으로 한다)

    uv run python -m eval.eval_final_retriever
"""
from __future__ import annotations

import json

from langchain_classic.retrievers import EnsembleRetriever

from core.config import get_config, path
from eval.eval_retrieval import SETS, _keys, _load, _metrics
from rag.index import get_bm25, get_hybrid_retriever, get_vectorstore

ALT_EMB = ("nlpai-lab/KURE-v1", "", "")  # bake-off 상위 동률 후보 (eval_retrieval 과 같은 접두어)
GRID = (0.1, 0.2, 0.3, 0.4, 0.5)          # 하이브리드 BM25 가중치 (Dense = 1 - BM25)
TOL = 0.01                                 # 규칙 3의 허용 차이
RULE = ("1) 전체 Hit@4 최대 2) 동률이면 MRR@4 최대(완전 동률이면 현재 런타임 설정 우선) "
        f"3) 1위가 다른 임베딩이면, 현재 임베딩 설정 중 최고가 Hit@4·MRR@4 모두 {TOL} 이내일 때 그쪽(재색인 불필요)")


def _score(retriever, groups: dict[str, list[dict]], k: int) -> dict:
    out = {}
    for g, qs in groups.items():
        ranked = [_keys(retriever.invoke(q["question"])[:k]) for q in qs]  # 런타임과 같이 앞 k개만
        out[g] = {"n": len(qs), **_metrics(ranked, [(q["doc_id"], q["page"]) for q in qs])}
    return out


def _key(r: dict) -> tuple[float, float]:
    return r["all"]["Hit@4"], r["all"]["MRR@4"]


def _config_change(r: dict, cfg) -> str:
    if r["runtime"]:
        return "변경 없음 (현재 런타임 설정)"
    ws = r["weights[bm25,dense]"]
    w = f"rag.ensemble_weights: [{ws[0]}, {ws[1]}]" + (" (가중치 0 인 쪽은 앞 k개 순서에 영향 없음)" if 0.0 in ws else "")
    redo = "검색 순서가 바뀌므로 본 실행과 재현용 LLM 캐시는 다시 만들어야 함"
    if r["embedding"] in (cfg.embedding.model, "-"):
        return f"{w} — 재색인 불필요(같은 FAISS 색인), {redo}"
    return (f"embedding.model: {r['embedding']} (query_prefix '{ALT_EMB[1]}') + {w} — 임베딩 교체라 "
            f"본 실행 질의의 임베딩 캐시가 없어 모델이 필요하고, {redo}")


def _recommend(rows: list[dict], cfg) -> dict:
    best = max(rows, key=_key)  # max 는 동률이면 앞쪽을 고른다
    cur = max((r for r in rows if r["embedding"] in (cfg.embedding.model, "-")), key=_key)
    kept = best is not cur and all(b - c <= TOL + 1e-9 for b, c in zip(_key(best), _key(cur)))
    pick = cur if kept else best
    base = next(r for r in rows if r["runtime"])
    return {"rule": RULE, "name": pick["name"], "embedding": pick["embedding"],
            "weights[bm25,dense]": pick["weights[bm25,dense]"],
            "Hit@4": pick["all"]["Hit@4"], "MRR@4": pick["all"]["MRR@4"],
            "delta_vs_runtime": {m: round(pick["all"][m] - base["all"][m], 3) for m in ("Hit@4", "MRR@4", "Hit@8")},
            "rule3_applied": kept, "overall_best": best["name"],
            # 참고(규칙과 별개): 재색인 없이 현재 임베딩으로 되는 최고 설정
            "best_without_reindex": {"name": cur["name"], "Hit@4": cur["all"]["Hit@4"], "MRR@4": cur["all"]["MRR@4"],
                                     "config_change": _config_change(cur, cfg)},
            "config_change": _config_change(pick, cfg),
            "ranking": [f"{r['name']}: Hit@4 {r['all']['Hit@4']:.3f} / MRR@4 {r['all']['MRR@4']:.3f}"
                        for r in sorted(rows, key=_key, reverse=True)]}


def main() -> None:
    cfg = get_config()
    k = cfg.rag.candidate_k
    qs = {s: _load(f) for s, f in SETS.items() if path(f).exists()}
    allq = [q for v in qs.values() for q in v]
    groups = {"all": allq, **{lang: [q for q in allq if q["lang"] == lang] for lang in ("ko", "en")}, **qs}
    w = [float(x) for x in cfg.rag.ensemble_weights]
    bm25 = get_bm25(k)
    dense = get_vectorstore().as_retriever(search_kwargs={"k": k})
    alt = get_vectorstore(*ALT_EMB).as_retriever(search_kwargs={"k": k})
    alt_short = ALT_EMB[0].split("/")[-1]

    # (이름, 검색기, 임베딩, [BM25, Dense] 가중치, 런타임 여부). 맨 앞이 런타임 설정 그대로의 검색기
    specs = [("final (hybrid)", get_hybrid_retriever(), cfg.embedding.model, w, True),
             ("dense only", dense, cfg.embedding.model, [0.0, 1.0], False),
             ("Kiwi BM25 only", bm25, "-", [1.0, 0.0], False),
             (f"{alt_short} dense", alt, ALT_EMB[0], [0.0, 1.0], False),
             (f"{alt_short} hybrid", EnsembleRetriever(retrievers=[bm25, alt], weights=w), ALT_EMB[0], w, False)]
    for emb, name, r in ((cfg.embedding.model, "hybrid", dense), (ALT_EMB[0], f"{alt_short} hybrid", alt)):
        for b in GRID:
            if abs(b - w[0]) < 1e-9:
                continue  # 설정 가중치는 위 행과 같은 검색기
            gw = [b, round(1 - b, 1)]
            specs.append((f"{name} {gw[0]}:{gw[1]}", EnsembleRetriever(retrievers=[bm25, r], weights=gw), emb, gw, False))

    rows = [{"name": n, "embedding": e, "weights[bm25,dense]": ws, "k": k, "runtime": rt, **_score(r, groups, k)}
            for n, r, e, ws, rt in specs]
    rec = _recommend(rows, cfg)
    meta = {"embedding": cfg.embedding.model, "weights[bm25,dense]": w, "candidate_k": k, "top_k": cfg.rag.top_k,
            "measured_k": k, "questions": {g: len(v) for g, v in groups.items()},
            "hit_resolution": f"전체 1문항 = {1 / len(allq):.3f}",
            "groups": "all=전체, ko/en=정답 문서 언어(질문은 모두 한국어), natural/keyword=질문 세트",
            "pipeline_metric": "Hit@4·MRR@4 (관련 조각을 검색 순서대로 앞 top_k=4개만 사용), Hit@8 = Judge 상한"}
    runtime = {"config": meta, "rows": rows, "recommendation": rec}
    # 설계서·README 생성 스크립트가 읽는 이름별 형식 (all/ko/en)
    compat = {"config": meta, **{r["name"]: {g: r[g] for g in ("all", "ko", "en")} for r in rows[:5]}}
    path("outputs/eval/runtime_retriever.json").write_text(json.dumps(runtime, ensure_ascii=False, indent=2),
                                                           encoding="utf-8")
    path("outputs/eval/final_retriever.json").write_text(json.dumps(compat, ensure_ascii=False, indent=2),
                                                         encoding="utf-8")
    print(f"k={k} (런타임 candidate_k), top_k={cfg.rag.top_k}, 질문 {len(allq)}개")
    print(f"{'설정':<28} {'Hit@1':>6} {'Hit@3':>6} {'Hit@4':>6} {'Hit@5':>6} {'Hit@8':>6} {'MRR@4':>6}  en Hit@4")
    for r in rows:
        a = r["all"]
        print(f"{r['name']:<28} " + " ".join(f"{a[m]:>6.3f}" for m in ("Hit@1", "Hit@3", "Hit@4", "Hit@5", "Hit@8", "MRR@4"))
              + f"  {r['en']['Hit@4']:.3f}")
    print(json.dumps(rec, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
