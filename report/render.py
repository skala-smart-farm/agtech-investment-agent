"""HTML → PDF 렌더링과 분량 검증 (Playwright + Chromium)."""
from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape
from pypdf import PdfReader

from core.config import ROOT

A4_HEIGHT_PX = 1122.5  # 297mm @ 96dpi
CONTENT_WIDTH_PX = 680  # 210mm - 좌우 여백 15mm×2 = 180mm @ 96dpi


def render_html(context: dict, density: int = 0) -> str:
    env = Environment(loader=FileSystemLoader(ROOT / "report" / "templates"), autoescape=select_autoescape(["html"]))
    return env.get_template("report.html.j2").render(**context, density=density)


def html_to_pdf(html: str, pdf_path: Path) -> dict:
    """PDF 로 저장하고 페이지 수와 SUMMARY 높이(A4 대비 비율)를 돌려준다."""
    from playwright.sync_api import sync_playwright

    html_path = pdf_path.with_suffix(".html")
    html_path.write_text(html, encoding="utf-8")
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": CONTENT_WIDTH_PX, "height": 1000})
        page.goto(html_path.resolve().as_uri(), wait_until="networkidle")
        page.emulate_media(media="print")
        summary_px = page.evaluate(
            "() => { const e = document.querySelector('#summary'); return e ? e.getBoundingClientRect().height : 0 }")
        page.pdf(path=str(pdf_path), format="A4", print_background=True,
                 margin={"top": "14mm", "bottom": "14mm", "left": "15mm", "right": "15mm"})
        browser.close()
    pages = len(PdfReader(str(pdf_path)).pages)
    return {"pages": pages, "summary_ratio": round(summary_px / A4_HEIGHT_PX, 3), "html": str(html_path)}
