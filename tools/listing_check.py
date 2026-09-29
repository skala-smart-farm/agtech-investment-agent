"""비상장 여부의 결정적(deterministic) 확인.

LLM 판단에만 맡기지 않고, 한국거래소(KOSPI·KOSDAQ·KONEX) 상장 종목 목록과 이름을 직접 대조한다.
해외 기업은 거래소 목록 대조가 어렵기 때문에 뉴스 근거(IPO·인수 보도)를 적격성 에이전트가 판정한다.
"""
from __future__ import annotations

import re
from datetime import datetime
from functools import lru_cache

import pandas as pd

from core.config import get_config, path

_SUFFIX = re.compile(r"\(주\)|㈜|주식회사|농업회사법인|유한회사|\(유\)|inc\.?|corp\.?|co\.,?\s*ltd\.?|ltd\.?|\s+", re.I)


def normalize(name: str) -> str:
    return _SUFFIX.sub("", name or "").lower()


def _kind_listing() -> pd.DataFrame:
    """한국거래소 KIND 상장법인 목록 (KOSPI·KOSDAQ·KONEX, 당일 반영)."""
    import io

    import requests

    r = requests.get("https://kind.krx.co.kr/corpgeneral/corpList.do?method=download&searchType=13",
                     headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    r.encoding = "euc-kr"
    df = pd.read_html(io.StringIO(r.text))[0]
    return df.rename(columns={"회사명": "Name", "시장구분": "Market"})[["Name", "Market"]]


def _fdr_listing() -> pd.DataFrame:
    import FinanceDataReader as fdr

    return pd.concat([fdr.StockListing(m)[["Name"]].assign(Market=m) for m in ("KRX", "KONEX")],
                     ignore_index=True)


@lru_cache(maxsize=1)
def _krx_names() -> set[str]:
    """KIND 목록을 기본으로, 실패하면 FinanceDataReader 로 대체. 스냅샷은 날짜별로 저장해 판정 근거로 남긴다."""
    d = path(f"{get_config().cache.dir}/snapshots/.keep").parent
    old = sorted(d.glob("krx_listing_*.csv"))
    f = old[-1] if old else d / f"krx_listing_{datetime.now():%Y%m%d}.csv"
    if f.exists():
        df = pd.read_csv(f)
    else:
        df = None
        for loader in (_kind_listing, _fdr_listing):
            try:
                df = loader()
                if len(df) > 1000:
                    break
            except Exception:
                df = None
        if df is None or len(df) <= 1000:
            return set()
        df.to_csv(f, index=False)
    return {normalize(n) for n in df["Name"].dropna().astype(str)}


def is_krx_listed(name: str, aliases: list[str] | None = None) -> bool | None:
    """True=상장, False=목록에 없음, None=목록을 불러오지 못함 (호출 쪽에서 통과시키지 않는다: fail-closed)."""
    try:
        names = _krx_names()
    except Exception:
        return None
    if not names:
        return None
    return any(normalize(n) in names for n in [name, *(aliases or [])] if n)
