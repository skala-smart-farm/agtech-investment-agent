# 재현성 테스트 기록

| 항목 | 결과 |
|---|---|
| 방법 | 저장소를 새 폴더에 git clone → `uv sync` → **.env 없이(API 키 없음)** `uv run python app.py --offline` |
| --offline 의미 | 재현용 캐시(replay/)에 없는 검색·LLM 호출이 하나라도 생기면 즉시 실패 |
| 결과 | 성공 · LLM API 호출 0회 (캐시 적중 206회) · 12.1초 · 보고서 본문(`outputs/investment_report.md`)이 제출본과 **완전히 같음**, 실행 기록(`run_log.json`)도 실행 시간·비용 칸을 빼면 같음 |
| 커밋 | a5eeb34 (2026-09-30, 심층 평가 10곳 최종 실행) |
| 참고 | 문서 색인(KURE-v1)·질의 임베딩이 replay/ 에 있어 임베딩 모델(약 2GB)을 내려받지 않고 재현된다(HF_HUB_OFFLINE=1 로 확인). 국민연금 원본 CSV(약 115MB)는 저장소에 없고, 회사별 조회 결과 스냅샷(replay/snapshots/nps_lookup.json)과 기사 원문 캐시(replay/articles/)로 재현된다 |
| 새로 평가할 때 | `.env` 에 OPENAI_API_KEY 와 검색 키(SERPER_API_KEY 또는 TAVILY_API_KEY)를 넣고 `python app.py --fresh` — 공개 웹이 바뀌므로 결과는 달라질 수 있다 |
