"""LLM-as-a-Judge 로 Agentic RAG 답변 평가 (O/X 이진 판정).

- Relevance: 답변이 질문에 맞게 답했는가
- Faithfulness: 답변의 내용이 검색된 근거로 뒷받침되는가
- Correctness: 답변이 정답과 같은 사실을 말하는가
1~5점 척도 대신 이진 판정을 써서 판정 기준을 분명히 한다.

    uv run python -m eval.eval_judge --n 20
"""
from __future__ import annotations

import argparse
import json

from pydantic import BaseModel, Field

from core.config import get_config, path
from core.llm import get_llm, structured
from rag.agentic_rag import agentic_rag
from tools.sources import SourceRegistry


class Verdict(BaseModel):
    relevance: bool = Field(description="답변이 질문이 묻는 것에 직접 답하면 true")
    faithfulness: bool = Field(description="답변의 모든 사실이 근거에 있으면 true (근거 밖 내용이 있으면 false)")
    correctness: bool = Field(description="답변이 정답과 같은 사실을 말하면 true")
    reason: str


ANSWER = """아래 근거만 사용해 질문에 한두 문장으로 답하라. 근거에 없으면 "근거에서 확인되지 않음"이라고 답하라.
질문: {q}
근거:
{ctx}"""

JUDGE = """너는 RAG 시스템 평가자다. 각 기준을 O/X(true/false)로 판정하라.
질문: {q}
정답: {gold}
시스템 답변: {ans}
시스템이 사용한 근거:
{ctx}"""


def main(n: int) -> None:
    get_config()
    qa = [json.loads(l) for l in path("data/eval/qa_set.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()][:n]
    judge = structured(Verdict, "judge")
    rows = []
    for q in qa:
        reg = SourceRegistry()
        ids, trace = agentic_rag(q["question"], "평가 질문", reg, "eval")
        ctx = reg.brief(ids, 900)
        ans = get_llm("generator").invoke(ANSWER.format(q=q["question"], ctx=ctx)).content
        v: Verdict = judge.invoke(JUDGE.format(q=q["question"], gold=q["answer"], ans=ans, ctx=ctx))
        rows.append({"id": q["id"], "question": q["question"], "answer": ans, "gold": q["answer"],
                     "rewrites": sum(1 for t in trace if "retrieved" in t) - 1,
                     "web_fallback": any("web_fallback" in t for t in trace), **v.model_dump()})
        print(f"{q['id']}: R={v.relevance} F={v.faithfulness} C={v.correctness} (재작성 {rows[-1]['rewrites']}회)")
    k = len(rows)
    summary = {m: round(sum(r[m] for r in rows) / k, 3) for m in ("relevance", "faithfulness", "correctness")}
    summary.update(n=k, rewrite_rate=round(sum(r["rewrites"] > 0 for r in rows) / k, 3),
                   web_fallback_rate=round(sum(r["web_fallback"] for r in rows) / k, 3))
    path("outputs/eval/judge_eval.json").write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False,
                                                               indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    main(ap.parse_args().n)
