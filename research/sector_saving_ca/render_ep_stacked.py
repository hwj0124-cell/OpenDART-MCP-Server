"""S−I 분해 누적 막대: 저축(+)과 −투자 등을 쌓고 S−I(점추정치·68% 구간)를 선으로 겹침.

입력: output/ep_report_fgls_nl.csv (S−I 정규화, 2단계 가중최소제곱). 사용: python render_ep_stacked.py [HMAX]
→ output/ep_report_stacked.png
"""
import sys

import numpy as np
import pandas as pd
from playwright.sync_api import sync_playwright

from make_report import HERE, OUT

HMAX = int(sys.argv[1]) if len(sys.argv) > 1 else 5
C_SV, C_INV, C_NL = "#2459a8", "#d1701f", "#222"


def panel(r, s, title, w=430, h=300):
    q = r[(r.sector == s) & (r.h <= HMAX)]
    get = lambda v: q[q["var"] == f"{v}_{s}"].sort_values("h")
    sv, inv, nl = get("sv").ratio.values, -get("invtot").ratio.values, get("nl")
    hs = np.arange(len(sv))
    pos, neg = np.clip(sv, 0, None) + np.clip(inv, 0, None), np.clip(sv, None, 0) + np.clip(inv, None, 0)
    lo = min(neg.min(), nl.lo68.min(), 0)
    hi = max(pos.max(), nl.hi68.max(), 0)
    step = 0.25 if hi - lo <= 2 else 0.5
    lo, hi = np.floor(lo / step) * step, np.ceil(hi / step) * step
    ml, mr, mt, mb = 48, 12, 14, 40
    X = lambda i: ml + (i + 0.5) / len(hs) * (w - ml - mr)
    Y = lambda v: mt + (hi - v) / (hi - lo) * (h - mt - mb)
    bw = 0.55 * (w - ml - mr) / len(hs)
    o = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{title}">']
    for t in np.arange(lo, hi + step / 2, step):
        cls = "zero" if abs(t) < 1e-9 else "grid"
        o.append(f'<line class="{cls}" x1="{ml}" x2="{w-mr}" y1="{Y(t):.1f}" y2="{Y(t):.1f}"/>')
        o.append(f'<text class="tick" x="{ml-6}" y="{Y(t)+4:.1f}" text-anchor="end">{0 if abs(t) < 1e-9 else round(t, 2):g}</text>')
    for i in hs:
        up = dn = 0.0
        for val, col in ((sv[i], C_SV), (inv[i], C_INV)):  # 양수는 0 위로, 음수는 0 아래로 쌓기
            base = up if val >= 0 else dn
            top, bot = (base + val, base) if val >= 0 else (base, base + val)
            o.append(f'<rect x="{X(i)-bw/2:.1f}" y="{Y(top):.1f}" width="{bw:.1f}" height="{max(Y(bot)-Y(top),0):.1f}" fill="{col}" opacity=".78"/>')
            if val >= 0:
                up += val
            else:
                dn += val
        o.append(f'<text class="tick" x="{X(i):.1f}" y="{h-mb+16}" text-anchor="middle">{i}</text>')
    o.append(f'<text class="axis" x="{(ml+w-mr)/2}" y="{h-4}" text-anchor="middle">h년 후 (t−1 = 0, t = 시작)</text>')
    for i, (a, b) in enumerate(zip(nl.lo68.values, nl.hi68.values)):  # S−I 68% 구간
        o.append(f'<line x1="{X(i):.1f}" x2="{X(i):.1f}" y1="{Y(a):.1f}" y2="{Y(b):.1f}" stroke="{C_NL}" stroke-width="1.4"/>')
    pts = " ".join(f"{X(i):.1f},{Y(v):.1f}" for i, v in enumerate(nl.ratio.values))
    o.append(f'<polyline points="{pts}" fill="none" stroke="{C_NL}" stroke-width="2"/>')
    o += [f'<circle cx="{X(i):.1f}" cy="{Y(v):.1f}" r="3.2" fill="#fff" stroke="{C_NL}" stroke-width="1.8"/>' for i, v in enumerate(nl.ratio.values)]
    o.append("</svg>")
    leg = (f'<span><i style="display:inline-block;width:12px;height:12px;background:{C_SV};opacity:.78"></i>저축</span>'
           f'<span><i style="display:inline-block;width:12px;height:12px;background:{C_INV};opacity:.78"></i>−투자 등</span>'
           f'<span><i style="display:inline-block;width:14px;border-top:2px solid {C_NL}"></i>저축−투자 차(S−I)·68% 구간</span>')
    return f'<figure><div class="panel-title">{title}</div>{"".join(o)}<div class="legend">{leg}</div></figure>'


def main():
    r = pd.read_csv(OUT / "ep_report_fgls_nl.csv")
    r = r[r.kind == "resp"]
    n_hh, n_corp = int(r.n_hh.iloc[0]), int(r.n_corp.iloc[0])
    cells = [panel(r, "hh", f"(a) 가계(소비행태 변화, {n_hh}건)"), panel(r, "corp", f"(b) 기업(교역조건, {n_corp}건)")]
    style = (HERE / "report_template.html").read_text()
    style = style[:style.index("<main>")]
    note = ("주: 시작 연도 자기 부문 저축−투자 차(S−I, 국민계정상 순대출) 반응을 1로 정규화한 누적 반응(GDP 대비). 막대는 저축과 −투자 등"
            "(고정투자·재고·토지 등 − 순자본이전)으로, 합이 S−I. 선은 S−I 추정치와 68% 구간(국가 단위 부트스트랩 499회). "
            "2단계 가중최소제곱 국소투영(국가·연도 고정효과). 사례 정의는 그림 1과 같음.")
    html = (f'<!doctype html><html><head><meta charset="utf-8">{style}</head><body><main style="max-width:900px">'
            f'<section id="F"><h2 style="font-size:18px">그림. 저축−투자 차의 분해: 저축과 투자</h2>'
            f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:10px">{"".join(cells)}</div>'
            f'<p class="ci" style="font-size:12px">{note}</p></section></main></body></html>')
    (OUT / "ep_report_stacked.html").write_text(html)
    with sync_playwright() as p:
        b = p.chromium.launch(executable_path="/opt/pw-browsers/chromium-1194/chrome-linux/chrome")
        pg = b.new_page(viewport={"width": 900, "height": 800}, device_scale_factor=2)
        pg.set_content(html)
        pg.wait_for_timeout(1200)
        pg.locator("#F").screenshot(path=str(OUT / "ep_report_stacked.png"))
        b.close()


if __name__ == "__main__":
    main()
