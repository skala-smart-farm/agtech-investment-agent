# 재현성 테스트 기록

| 항목 | 결과 |
|---|---|
| 방법 | 저장소를 새 폴더에 git clone → `uv sync` → `playwright install chromium` → **.env 없이(API 키 없음)** `python app.py --offline` |
| --offline 의미 | 재현용 캐시(replay/)에 없는 검색·LLM 호출이 하나라도 생기면 즉시 실패 |
| 결과 | 성공 · LLM API 호출 0회 (캐시 적중 90회) · 8.2초 · 보고서 본문이 제출본과 **완전히 같음** (diff 없음) |
| 커밋 | 06380a1 |
| 참고 | 문서 색인·질의 임베딩도 replay/ 에 있어 임베딩 모델(약 2GB)을 내려받지 않고 재현된다 |
