"""최종 설정(config.yaml)의 검색기 성능: 질문 70개 전체·문서 언어별 Hit@1/3/5/8, MRR@5.

Agentic RAG 는 후보 8개(candidate_k)를 Judge LLM 이 다시 거르므로 Hit@8 이 실제 파이프라인 품질에 가장 가깝다.

    uv run python -m eval.eval_final_retriever
"""
from __future__ import annotations

import json

from langchain_classic.retrievers import EnsembleRetriever

from core.config import get_config, path
from eval.eval_retrieval import SETS, _keys, _load
from rag.index import get_bm25, get_hybrid_retriever, get_vectorstore


def _score(retriever, qs: list[dict]) -> dict:
    ranked = [_keys(retriever.invoke(q["question"])) for q in qs]
    gold = [(q["doc_id"], q["page"]) for q in qs]
    out = {f"Hit@{k}": round(sum(g in r[:k] for r, g in zip(ranked, gold)) / len(qs), 3) for k in (1, 3, 5, 8)}
    out["MRR@5"] = round(sum(next((1 / (i + 1) for i, x in enumerate(r[:5]) if x == g), 0.0)
                             for r, g in zip(ranked, gold)) / len(qs), 3)
    return {"n": len(qs), **out}


def main() -> None:
    cfg = get_config()
    allq = [q for f in SETS.values() if path(f).exists() for q in _load(f)]
    w = list(cfg.rag.ensemble_weights)
    kure = get_vectorstore("nlpai-lab/KURE-v1", "", "").as_retriever(search_kwargs={"k": 10})
    retrievers = {
        "final (hybrid)": get_hybrid_retriever(k=10),
        "dense only": get_vectorstore().as_retriever(search_kwargs={"k": 10}),
        "Kiwi BM25 only": get_bm25(10),
        # 동률 후보(KURE-v1)를 같은 방식으로 재서 최종 선택 근거를 남긴다
        "KURE-v1 hybrid": EnsembleRetriever(retrievers=[get_bm25(10), kure], weights=w),
        "KURE-v1 dense": kure,
    }
    result = {"config": {"embedding": cfg.embedding.model, "weights[bm25,dense]": list(cfg.rag.ensemble_weights),
                         "candidate_k": cfg.rag.candidate_k, "top_k": cfg.rag.top_k}}
    for name, r in retrievers.items():
        result[name] = {lang: _score(r, [q for q in allq if lang == "all" or q["lang"] == lang])
                        for lang in ("all", "ko", "en")}
    path("outputs/eval/final_retriever.json").write_text(json.dumps(result, ensure_ascii=False, indent=2),
                                                         encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
