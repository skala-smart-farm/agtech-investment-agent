"""양성 대조 실험: 공개 정보가 풍부한 후기(Series B~C) AgTech 기업에도 평가표가 "투자"를 줄 수 있는가.

본 실행의 후보가 모두 보류로 끝났을 때, 그것이 "무조건 보류하는 평가표" 때문인지
"초기 기업의 공개 정보 부족" 때문인지 가르기 위한 실험이다.
적격성 검증 → 기술·팀 → 시장성 → 경쟁사 → 투자 판단을 그래프와 같은 순서로 실행한다.

    uv run python -m eval.eval_positive_control
"""
from __future__ import annotations

import json

from agents.competition import competition_node
from agents.decision import decision_node
from agents.eligibility import _check_one
from agents.market import market_node
from agents.tech import tech_node
from core.config import get_config, path, run_date

# 정보가 풍부하고 조사 단계에서 적격(비상장·Series B~C)으로 확인된 기업
CONTROLS = [
    {"name": "Source.ag", "name_en": "Source.ag", "region": "GLOBAL", "segment_id": "greenhouse"},
    {"name": "Agtonomy", "name_en": "Agtonomy", "region": "GLOBAL", "segment_id": "robotics"},
    {"name": "아이오크롭스", "name_en": "IOCROPS", "region": "KR", "segment_id": "greenhouse"},
]


def main() -> None:
    get_config()
    rows = []
    for c in CONTROLS:
        rec, reg = _check_one({**c, "evidence_ids": [], "channels": []}, {})
        if not rec["eligible"]:
            rows.append({"name": c["name"], "eligible": False, "reason": rec["reason"]})
            print(f"{c['name']}: 적격성 탈락 — {rec['reason']}")
            continue
        state = {"registry": reg, "current": rec, "run_date": run_date(), "market_cache": {}}
        for node in (tech_node, market_node):
            out = node(state)
            state["registry"] = {**state["registry"], **out.get("registry", {})}
            state.update({k: v for k, v in out.items() if k in ("tech", "market")})
        out = competition_node(state)
        state["registry"] = {**state["registry"], **out.get("registry", {})}
        state["competition"] = out["competition"]
        sc = decision_node(state)["scorecard"]
        rows.append({"name": c["name"], "eligible": True, "stage": rec["stage"], "total": sc["total"],
                     "decision": sc["decision"], "unknown_ratio": sc["unknown_ratio"], "reasons": sc["reasons"],
                     "dims": {d["id"]: f"{d['yes']}/{d['n']}" for d in sc["dims"]}})
    path("outputs/eval/positive_control.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2),
                                                          encoding="utf-8")
    for r in rows:
        print(json.dumps(r, ensure_ascii=False))


if __name__ == "__main__":
    main()
