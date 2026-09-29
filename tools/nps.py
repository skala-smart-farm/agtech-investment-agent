"""국민연금 가입 사업장 내역(공공데이터포털, 로그인·키 불필요)으로 회사의 실재·창업 시기·인원을 확인한다.

교수님 지적: 창업자 검색은 "창업 시기"와 다른 문제다. 이 데이터는 회사(사업장) 단위로
- 적용일자: 사업장이 국민연금에 처음 가입한 날 → 첫 고용 시점, 창업 시기에 가까운 값
- 가입자수: 지금 실제로 고용 중인 인원 (보도자료의 "직원 N명"을 제3자 자료로 교차 확인)
- 가입상태: 1 등록 / 2 탈퇴 → 탈퇴면 폐업·휴업 가능성 (적격성 관문 G4 신호)
- 신규취득자수·상실가입자수: 최근 달의 채용·퇴사 흐름
을 준다. 3인 미만 법인은 목록에 없으므로 "못 찾음"은 탈락 사유가 아니다(보류 신호만).

원본 CSV(약 115MB)는 저장소에 넣지 않고, 회사별 조회 결과만 스냅샷으로 남겨 재현한다.
"""
from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from pathlib import Path

from core.config import get_config, path
from tools.listing_check import normalize

PAGE = "https://www.data.go.kr/data/15083277/fileData.do"
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko)"}
COLS = {0: "ym", 1: "name", 2: "brn6", 3: "status", 12: "corp", 14: "industry", 15: "first_date",
        17: "withdraw_date", 18: "members", 20: "new", 21: "lost"}


def _snapshot_file() -> Path:
    return path(f"{get_config().cache.dir}/snapshots/nps_lookup.json")


def _load_snapshot() -> dict:
    f = _snapshot_file()
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}


def _csv_file() -> Path | None:
    d = path(f"{get_config().cache.dir}/nps/.keep").parent
    files = sorted(d.glob("nps_*.csv"))
    if files:
        return files[-1]
    if os.getenv("REPLAY_OFFLINE"):
        return None
    import requests  # 최신 파일을 받는다 (약 40초)

    page = requests.get(PAGE, headers=UA, timeout=30).text
    m = re.search(r"fileDownload\.do\?atchFileId=(FILE_\d+)&fileDetailSn=(\d+)", page)
    if not m:
        return None
    f = d / f"nps_{m.group(1)}.csv"
    url = f"https://www.data.go.kr/cmm/cmm/fileDownload.do?atchFileId={m.group(1)}&fileDetailSn={m.group(2)}"
    f.write_bytes(requests.get(url, headers=UA, timeout=300).content)
    return f


@lru_cache(maxsize=1)
def _table():
    import pandas as pd

    f = _csv_file()
    if f is None:
        return None, ""
    df = pd.read_csv(f, encoding="cp949", dtype=str, usecols=list(COLS))
    df.columns = [COLS[i] for i in sorted(COLS)]
    df["key"] = df["name"].map(_norm)
    return df, df["ym"].iloc[0]


def _norm(name: str) -> str:
    return normalize(re.sub(r"\(.*?\)", "", str(name or "")))


def lookup(names: list[str]) -> dict:
    """회사명(여러 표기)으로 사업장을 찾는다. 결과는 스냅샷에 남겨 재현 때 그대로 쓴다.
    반환: {"status": matched|ambiguous|not_found|unavailable, "matches": [...], "ym": "YYYY-MM"}"""
    keys = [k for k in dict.fromkeys(_norm(n) for n in names if n) if len(k) >= 2]
    if not keys:
        return {"status": "not_found", "matches": [], "ym": ""}
    snap = _load_snapshot()
    sid = "|".join(keys)
    if sid in snap:
        return snap[sid]
    df, ym = _table()
    if df is None:
        return {"status": "unavailable", "matches": [], "ym": ""}
    rows = df[df["key"].isin(keys)]
    matches = rows.drop(columns=["key"]).to_dict("records")
    brns = {m["brn6"] for m in matches}
    status = "not_found" if not matches else ("matched" if len(brns) == 1 else "ambiguous")
    out = {"status": status, "matches": matches[:5], "ym": ym}
    snap[sid] = out
    _snapshot_file().write_text(json.dumps(snap, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


def summarize(res: dict) -> dict:
    """적격성·보고서에서 쓰는 요약: 인원, 최초 가입일, 탈퇴 여부."""
    if res["status"] != "matched":
        return {"status": res["status"], "ym": res.get("ym", "")}
    ms = res["matches"]
    active = [m for m in ms if m["status"] == "1"]
    first = min((m["first_date"] for m in ms if m.get("first_date")), default="")
    return {"status": "matched", "ym": res["ym"], "members": sum(int(m["members"] or 0) for m in active),
            "first_date": first, "withdrawn": not active, "new": sum(int(m["new"] or 0) for m in active),
            "lost": sum(int(m["lost"] or 0) for m in active), "sites": len(ms),
            "industry": ms[0].get("industry", "")}


def as_evidence(name: str, s: dict) -> dict | None:
    """근거 저장소에 넣을 웹 근거 형식 (REFERENCE: 국민연금공단(YYYY-MM-DD). 제목. 공공데이터포털, URL)."""
    if s["status"] != "matched":
        return None
    state = "탈퇴(사업장 소멸)" if s["withdrawn"] else "가입 중"
    text = (f"{name} 국민연금 가입 사업장: 상태 {state}, 가입자 {s['members']}명({s['ym']} 기준), "
            f"최초 적용일 {s['first_date']}, 최근 한 달 신규 {s['new']}명·상실 {s['lost']}명, 업종 {s['industry']}")
    return {"url": PAGE + f"#{_norm(name)}", "title": f"국민연금 가입 사업장 내역 ({s['ym']} 자료)",
            "content": text, "published_date": f"{s['ym']}-01", "author": "국민연금공단"}
