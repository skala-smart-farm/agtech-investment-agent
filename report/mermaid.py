"""Mermaid 정의를 PNG 로 렌더링 (Playwright + mermaid.js)."""
from __future__ import annotations

import html
from pathlib import Path

PAGE = """<!DOCTYPE html><html><head><meta charset="utf-8">
<script src="https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.min.js"></script>
<style>body{margin:0;background:#fff;font-family:"Pretendard","Apple SD Gothic Neo","Malgun Gothic",sans-serif}
#c{display:inline-block;padding:16px}</style></head>
<body><div id="c"><pre class="mermaid">%s</pre></div>
<script>mermaid.initialize({startOnLoad:true, theme:"default", flowchart:{curve:"basis"}});</script></body></html>"""


def mermaid_to_png(mmd: str, out: Path, width: int = 1400) -> Path:
    from playwright.sync_api import sync_playwright

    tmp = out.with_suffix(".render.html")
    tmp.write_text(PAGE % html.escape(mmd), encoding="utf-8")
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page(viewport={"width": width, "height": 900}, device_scale_factor=2)
        pg.goto(tmp.resolve().as_uri(), wait_until="networkidle")
        pg.wait_for_selector("#c svg", timeout=20000)
        pg.locator("#c").screenshot(path=str(out))
        b.close()
    tmp.unlink(missing_ok=True)
    return out
