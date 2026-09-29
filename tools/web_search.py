"""외부 정보 검색 도구 (Serper(구글) → Tavily, 키가 있는 공급자만 순서대로).

- 뉴스는 기본적으로 최근 1년(time_range)만 검색해 오래된 정보가 섞이지 않게 한다.
- 결과가 부족할 때만 기간 제한을 풀어 다시 검색한다.
- 같은 쿼리는 디스크 캐시를 재사용한다(비용·재현성). 캐시 키에 공급자를 넣지 않아, 어느 공급자로 받았든
  재현 때는 저장된 결과를 그대로 쓴다 (결과마다 provider 필드로 출처 공급자를 남긴다).
- 한 공급자가 한도 초과·오류면 다음 공급자로 넘어간다. 모두 실패하면 실패 표시(.failed)를 남기고,
  --retry-failed 로 실행하면 그 검색만 다시 시도한다.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from datetime import date

from langchain_tavily import TavilySearch

from core.config import get_config, path, require_keys
from tools import search_providers as providers
from tools.sources import SourceRegistry, today

# 보도자료 배포 사이트로 퍼진 유료 시장조사 홍보 기사 (예: "... Market Growing at 24% CAGR, Says Mordor Intelligence")
PR_MARKET = re.compile(r"market (size|share|growing|to reach|worth|report|intelligence|analysis|forecast)|cagr|"
                       r"says .*(research|intelligence|insights)|(mordor|marketsandmarkets|grand view|precedence|imarc|"
                       r"fortune business|researchandmarkets|research and markets)", re.I)
# 검색 결과 목록 페이지는 근거가 아니다 (예: search.zdnet.co.kr?kwd=...)
SEARCH_PAGE = re.compile(r"//search\.|/search[/?]|[?&](kwd|q|query|keyword)=", re.I)


def _cache_file(key: dict):
    h = hashlib.sha256(json.dumps(key, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:24]
    return path(f"{get_config().cache.dir}/search/{h}.json")


def _legacy_files(key: dict) -> list:
    """예전 캐시 키(제외 도메인 목록을 키에 넣던 시절)의 파일 후보. 찾으면 새 키로 옮겨 재현용 캐시를 살린다."""
    ex = list(get_config().search.exclude_domains)
    return [_cache_file({**key, "ex": ex[:n]}) for n in (15, 33, len(ex))]


def _raw_search(query: str, topic: str, time_range: str | None, max_results: int,
                include_domains: list[str] | None, depth: str = "basic", raw: bool = False) -> list[dict]:
    cfg = get_config()
    # 제외 도메인은 결과를 받은 뒤 다시 거르므로 캐시 키에 넣지 않는다 (목록을 고쳐도 재현용 캐시가 유지되게)
    key = {"q": query, "topic": topic, "tr": time_range, "n": max_results, "dom": include_domains or [],
           "depth": depth, "raw": raw}
    f = _cache_file(key)
    if cfg.cache.search and not f.exists():
        old = next((o for o in _legacy_files(key) if o.exists()), None)
        if old:
            f.write_text(old.read_text(encoding="utf-8"), encoding="utf-8")
    if cfg.cache.search and f.exists():
        return json.loads(f.read_text(encoding="utf-8"))
    failed = f.with_suffix(".failed")
    # 제출 실행 때 실패한 검색은 재현 때도 "결과 없음"으로 (누구 키로 돌려도 같은 결과). --retry-failed 면 다시 시도
    if cfg.cache.search and failed.exists() and not os.getenv("SEARCH_RETRY_FAILED"):
        return []
    if os.getenv("REPLAY_OFFLINE"):
        raise RuntimeError(f"--offline: 재현용 캐시에 없는 검색입니다 → {query!r}")
    require_keys()
    results, ok, errors = [], False, []
    for provider in providers.order_for(query):
        try:
            got = _call(provider, query, topic, time_range, max_results, include_domains, depth, raw)
        except Exception as e:  # 한도 초과·인증 오류 → 다음 공급자
            errors.append(f"{provider}: {str(e)[:60]}")
            continue
        ok = True
        results = [{**r, "provider": provider} for r in got]
        if results:
            break
    if not ok:
        print(f"   (검색 실패, 빈 결과로 진행: {query[:40]} — {'; '.join(errors)[:120]})")
    if cfg.cache.search and ok:
        f.write_text(json.dumps(results, ensure_ascii=False), encoding="utf-8")
        failed.unlink(missing_ok=True)
    elif cfg.cache.search:  # 오류(한도 초과 등)는 결과 대신 실패 표시만 남긴다. --fresh(새 캐시)에서는 다시 시도한다
        failed.write_text(json.dumps({"query": query, "error": "search failed"}, ensure_ascii=False), encoding="utf-8")
    return results


def _call(provider: str, query: str, topic: str, time_range: str | None, max_results: int,
          include_domains: list[str] | None, depth: str, raw: bool) -> list[dict]:
    cfg = get_config()
    if provider == "tavily":
        kwargs = dict(max_results=max_results, topic=topic, search_depth=depth)
        if raw:
            kwargs["include_raw_content"] = "text"  # 본문 전체: 창업자 이력·실적처럼 스니펫에 잘 안 나오는 사실 확보
        if time_range:
            kwargs["time_range"] = time_range
        if include_domains:
            kwargs["include_domains"] = include_domains
        else:
            kwargs["exclude_domains"] = list(cfg.search.exclude_domains)
        for attempt in range(3):
            try:
                res = TavilySearch(**kwargs).invoke({"query": query})
            except Exception as e:  # 일시적 네트워크 오류는 재시도, "결과 없음"은 정상 빈 결과
                if "No search results" in str(e):
                    return []
                if attempt == 2:
                    raise
                time.sleep(2 * (attempt + 1))
                continue
            if isinstance(res, dict) and "error" in res:  # 인증·한도 오류는 예외 대신 dict 로 온다
                raise RuntimeError(str(res["error"]))
            return res.get("results", []) if isinstance(res, dict) else []
    blocked = tuple(cfg.search.exclude_domains)
    got = [r for r in providers.serper(query, topic, include_domains, time_range, date.today().isoformat())
           if not _host(r["url"]).endswith(blocked) and not PR_MARKET.search(r["title"])
           and not SEARCH_PAGE.search(r["url"])][:max_results]
    if raw:  # 본문은 원문 페이지에서 직접 받는다 (Tavily raw_content 대신)
        from tools.fetch import fetch

        for r in got:
            if d := fetch(r["url"]):
                r["raw_content"] = d["text"]
                r["published_date"] = r["published_date"] or d["date"][:10] or None
    return got


def _host(url: str) -> str:
    return url.split("/")[2].lower() if url.count("/") >= 2 else ""


def web_search(query: str, registry: SourceRegistry, agent: str, *, topic: str = "news",
               recent: bool = True, max_results: int | None = None,
               include_domains: list[str] | None = None, min_results: int = 2, deep: bool = False,
               raw: bool = False) -> list[str]:
    """검색 결과를 근거로 등록하고 근거 id 목록을 돌려준다.
    deep=True: 특정 회사에 대한 검색처럼 본문 근거가 더 필요할 때 Tavily advanced 검색(관련 문단을 더 길게 가져옴).
    raw=True : 기사 본문 전체도 받아 근거 본문(body)으로 저장한다 (투자 판단 에이전트가 문항별로 다시 검색)."""
    cfg = get_config()
    n = max_results or cfg.search.max_results
    tr = cfg.search.news_time_range if (recent and topic == "news") else None
    depth = "advanced" if deep else "basic"
    results = _raw_search(query, topic, tr, n, include_domains, depth, raw)
    if len(results) < min_results and tr:
        results += _raw_search(query, topic, cfg.search.fallback_time_range, n, include_domains, depth, raw)
    ids: list[str] = []
    blocked = tuple(cfg.search.exclude_domains)
    for r in results:
        host = _host(r.get("url", ""))
        if (not r.get("url") or host.endswith(blocked) or PR_MARKET.search(r.get("title") or "")
                or SEARCH_PAGE.search(r["url"])):
            continue
        sid = registry.add_web(r, agent=agent, query=query, access_date=today())
        if sid not in ids:
            ids.append(sid)
    return ids
