"""근거(Evidence) 레지스트리와 REFERENCE 표기.

모든 에이전트가 모은 근거는 id(D=문서, W=웹)로 등록되고, 분석 결과는 이 id 로만 근거를 인용한다.
보고서 REFERENCE 에는 최종 본문에 실제로 인용된 id 만 들어간다(과제 조건: 실제 활용 자료만).
"""
from __future__ import annotations

import hashlib
import re
from email.utils import parsedate_to_datetime
from urllib.parse import unquote, urlparse

# 언론사·DB 도메인 → 사이트명 (없으면 제목 꼬리나 도메인으로 추정)
SITE_NAMES = {
    "wowtale.net": "와우테일", "startuprecipe.co.kr": "스타트업레시피", "platum.kr": "플래텀",
    "venturesquare.net": "벤처스퀘어", "thevc.kr": "THE VC", "innoforest.co.kr": "혁신의숲",
    "edaily.co.kr": "이데일리", "mk.co.kr": "매일경제", "hankyung.com": "한국경제", "sedaily.com": "서울경제",
    "chosun.com": "조선일보", "joongang.co.kr": "중앙일보", "donga.com": "동아일보", "yna.co.kr": "연합뉴스",
    "zdnet.co.kr": "지디넷코리아", "etnews.com": "전자신문", "nongmin.com": "농민신문", "aflnews.co.kr": "농수축산신문",
    "agrinet.co.kr": "한국농어민신문", "bloter.net": "블로터", "unicornfactory.co.kr": "머니투데이 유니콘팩토리",
    "news.mt.co.kr": "머니투데이", "mt.co.kr": "머니투데이", "hankookilbo.com": "한국일보", "khan.co.kr": "경향신문",
    "techcrunch.com": "TechCrunch", "agfundernews.com": "AgFunderNews", "reuters.com": "Reuters",
    "bloomberg.com": "Bloomberg", "businesswire.com": "Business Wire", "prnewswire.com": "PR Newswire",
    "forbes.com": "Forbes", "crunchbase.com": "Crunchbase", "futurefarming.com": "Future Farming",
    "jointips.or.kr": "TIPS 창업기업 목록", "mafra.go.kr": "농림축산식품부", "agnavigator.com": "AgNavigator",
    "daum.net": "다음뉴스", "news.naver.com": "네이버 뉴스", "aving.net": "에이빙(AVING)", "newsis.com": "뉴시스",
    "news1.kr": "뉴스1", "asiae.co.kr": "아시아경제", "fnnews.com": "파이낸셜뉴스", "heraldcorp.com": "헤럴드경제",
    "dt.co.kr": "디지털타임스", "etoday.co.kr": "이투데이", "sisajournal-e.com": "시사저널e", "the-pr.co.kr": "더피알",
    "cbinsights.com": "CB Insights", "financialcontent.com": "FinancialContent", "zdnet.co.kr": "지디넷코리아",
    "supplychangecapital.substack.com": "Supply Change Capital", "agfunder.com": "AgFunder", "weforum.org": "World Economic Forum",
}


def _site(url: str, title: str = "") -> str:
    host = urlparse(url).netloc.lower().removeprefix("www.").removeprefix("m.")
    for dom, name in SITE_NAMES.items():
        if host == dom or host.endswith("." + dom):
            return name
    for sep in (" - ", " | ", " : ", " – "):
        if sep in title:
            tail = title.rsplit(sep, 1)[-1].strip()
            if 1 < len(tail) <= 20:
                return tail
    return host


SITE_TAIL = re.compile(r"(news|korea|\.com|\.kr|일보|신문|뉴스|경제|innoforest|혁신의숲|the vc|times|herald|tribune|"
                       r"기업정보|매출·투자·고용)", re.I)


def _clean_title(title: str, site: str) -> str:
    title = title.strip()
    for sep in (" - ", " | ", " : ", " – "):  # 먼저 "제목 - 사이트명" 꼬리를 떼고
        if sep in title:
            tail = title.rsplit(sep, 1)[-1].strip()
            if tail == site or (len(tail) <= 28 and SITE_TAIL.search(tail)):
                title = title.rsplit(sep, 1)[0].strip()
    # 다음으로 "< 산업 < 기사본문" 같은 게시판 경로 꼬리를 뗀다
    return re.sub(r"(\s*<\s*[^<>]{1,20})+\s*<\s*기사본문\s*$", "", title).strip()


def _date(raw: str | None) -> str | None:
    if not raw:
        return None
    try:
        return parsedate_to_datetime(raw).strftime("%Y-%m-%d")
    except (TypeError, ValueError):
        pass
    m = re.search(r"(20\d{2})[-./](\d{1,2})[-./](\d{1,2})", raw)
    return f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}" if m else None


def _date_from(url: str, text: str) -> str | None:
    """게시일이 없을 때 URL(/2022/12/10/, /20221210) 이나 본문(2022.12.10)에서 날짜를 찾는다."""
    for pat in (r"/(20\d{2})/(\d{2})/(\d{2})(?:/|\b)", r"/(20\d{2})(\d{2})(\d{2})\d*", r"[?&]date=(20\d{2})-?(\d{2})-?(\d{2})"):
        m = re.search(pat, url or "")
        if m and 1 <= int(m.group(2)) <= 12 and 1 <= int(m.group(3)) <= 31:
            return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    head = (text or "")[:1500]
    for pat in (r"(?:입력|등록|게재|승인|발행|기사입력|Published|Posted)\s*[:：]?\s*(20\d{2})[.\-/년 ]+\s?(\d{1,2})[.\-/월 ]+\s?(\d{1,2})",
                r"(20\d{2})\s?년\s?(\d{1,2})\s?월\s?(\d{1,2})\s?일",
                r"(20\d{2})[.-]\s?(\d{1,2})[.-]\s?(\d{1,2})"):
        m = re.search(pat, head)
        if m and 1 <= int(m.group(2)) <= 12 and 1 <= int(m.group(3)) <= 31:
            return f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
    return None


def _author(text: str) -> str | None:
    m = re.search(r"([가-힣]{2,4})\s?기자", text or "")
    return f"{m.group(1)} 기자" if m else None


class SourceRegistry:
    """dict 기반이라 LangGraph State 에 그대로 담을 수 있다."""

    def __init__(self, data: dict | None = None):
        self.data: dict[str, dict] = dict(data or {})

    @staticmethod
    def _id(prefix: str, key: str) -> str:
        # 병렬 노드가 동시에 등록해도 충돌하지 않도록 순번 대신 키 해시로 id 를 만든다
        return prefix + hashlib.sha1(key.encode()).hexdigest()[:5]

    def _find(self, key: str) -> str | None:
        return next((sid for sid, s in self.data.items() if s.get("key") == key), None)

    def add_web(self, result: dict, agent: str, query: str, access_date: str, key: str | None = None) -> str:
        url = result.get("url", "")
        key = key or "web:" + url
        if sid := self._find(key):
            if result.get("raw_content") and not self.data[sid].get("body"):  # 같은 URL 을 본문과 함께 다시 받은 경우
                self.data[sid]["body"] = re.sub(r"\s+", " ", result["raw_content"])[:6000]
                if not self.data[sid]["date"] and (d := _date_from("", result["raw_content"])):
                    self.data[sid].update(date=d, date_is_access=False)
            return sid
        title = result.get("title", "") or url
        site = _site(url, title)
        pub = (_date(result.get("published_date")) or _date_from(url, result.get("content", ""))
               or _date_from("", result.get("raw_content") or ""))
        sid = self._id("W", key)
        self.data[sid] = {
            "id": sid, "key": key, "kind": "web", "url": url, "site": site,
            "title": _clean_title(title, site), "date": pub, "date_is_access": pub is None,
            "access_date": access_date, "author": _author(result.get("content", "")),
            "snippet": (result.get("content") or "")[:1200], "agent": agent, "query": query,
            "body": re.sub(r"\s+", " ", result.get("raw_content") or "")[:6000],
        }
        return sid

    def add_doc(self, meta: dict, text: str, agent: str, query: str) -> str:
        key = f"doc:{meta.get('doc_id')}:{meta.get('page')}:{hashlib.md5(text.encode()).hexdigest()[:8]}"
        if sid := self._find(key):
            return sid
        sid = self._id("D", key)
        self.data[sid] = {
            "id": sid, "key": key, "kind": "doc", "doc_id": meta.get("doc_id"), "type": meta.get("type", "report"),
            "title": meta.get("title"), "publisher": meta.get("publisher"), "year": meta.get("year"),
            "url": meta.get("url"), "page": meta.get("page"), "authors": meta.get("authors"),
            "journal": meta.get("journal"), "volume": meta.get("volume"), "issue": meta.get("issue"),
            "pages": meta.get("pages"), "snippet": text[:1200], "agent": agent, "query": query,
        }
        return sid

    def get(self, sid: str) -> dict | None:
        return self.data.get(sid)

    def text(self, sid: str) -> str:
        """근거 전체 텍스트 (제목 + 스니펫 + 본문). 인용문 검증과 문항별 재검색에 쓴다."""
        s = self.data.get(sid) or {}
        return " ".join(x for x in (s.get("title"), s.get("snippet"), s.get("body")) if x)

    def brief(self, ids: list[str], max_chars: int = 600) -> str:
        """LLM 에 넘길 근거 목록 텍스트."""
        lines = []
        for sid in ids:
            s = self.data.get(sid)
            if not s:
                continue
            if s["kind"] == "web":
                when = s["date"] or "게시일 미상"  # 조회일을 보여 주면 LLM 이 사건 날짜로 오인한다
                head = f"[{sid}] (웹, {s['site']}, {when}) {s['title']}"
            else:
                head = f"[{sid}] (문서, {s['publisher']} {s['year']}, p.{s['page']}) {s['title']}"
            lines.append(f"{head}\n{s['snippet'][:max_chars]}")
        return "\n\n".join(lines)


def format_reference(s: dict) -> str:
    """과제에서 지정한 REFERENCE 표기 형식. 게시일을 찾지 못한 웹페이지는 조회일을 쓰고 끝에 그 사실을 밝힌다."""
    if s["kind"] == "doc":
        if s.get("type") == "paper":
            vol = f"{s.get('volume')}({s.get('issue')})" if s.get("issue") else f"{s.get('volume')}"
            return f"{s['authors']}({s['year']}). {s['title']}. {s['journal']}, {vol}, {s['pages']}."
        return f"{s['publisher']}({s['year']}). {s['title']}. {s['url']}"
    who = s.get("author") or s["site"]
    when = s["date"] or s["access_date"]
    note = " (게시일 미상, 조회일 표기)" if not s["date"] else ""
    return f"{who}({when}). {s['title']}. {s['site']}, {unquote(s['url'])}{note}"


def reference_key(s: dict) -> str:
    """같은 문서의 여러 페이지는 한 줄로, 같은 웹페이지(인코딩·쿼리 차이 포함)도 한 줄로 합친다."""
    if s["kind"] == "doc":
        return f"doc:{s['doc_id']}"
    u = unquote(s["url"]).split("#")[0].split("?")[0].rstrip("/").lower()
    return "web:" + u.replace("://m.", "://").replace("://www.", "://")


def today() -> str:
    """조회일 = 평가 기준일 (재현 모드에서는 고정)."""
    from core.config import run_date

    return run_date()
