"""S−I 세부 분해 누적 막대(6년): 소득·배당(소비)·기타·투자를 쌓고 S−I(68% 구간)와 저축을 선으로 겹침.

기업: S−I = 영업잉여 − 순배당 + 기타(이자·세금·재산소득 수취 등) − 투자 등
가계: S−I = 가처분소득 − 소비 + 기타(연금 조정 등) − 투자 등
입력: output/ep_report_fgls_nl.csv (S−I 정규화, 2단계 가중최소제곱, 점추정치를 선형 결합). 사용: python render_ep_stacked_detail.py [HMAX]
→ output/ep_report_stacked_detail.png
"""
import sys

import numpy as np
import pandas as pd
from playwright.sync_api import sync_playwright

from make_report import HERE, OUT

HMAX = int(sys.argv[1]) if len(sys.argv) > 1 else 6
COL = {"inc": "#2459a8", "out": "#8e6cc0", "oth": "#a3a3a3", "inv": "#d1701f"}
C_NL, C_SV = "#222", "#2459a8"
SPEC = {"corp": [("inc", "영업잉여", lambda g: g("prof")), ("out", "−순배당", lambda g: -g("div_net")),
                 ("oth", "기타(이자·세금 등)", lambda g: g("sv") - g("prof") + g("div_net")), ("inv", "−투자 등", lambda g: -g("invtot"))],
        "hh": [("inc", "가처분소득", lambda g: g("yd")), ("out", "−소비", lambda g: -g("cons")),
               ("oth", "기타(연금 조정 등)", lambda g: g("sv") - g("yd") + g("cons")), ("inv", "−투자 등", lambda g: -g("invtot"))]}


def panel(r, s, title, w=430, h=320):
    q = r[(r.sector == s) & (r.h <= HMAX)]
    g = lambda v: q[q["var"] == f"{v}_{s}" if v not in ("prof", "div_net") else q["var"] == v].sort_values("h").ratio.values
    comps = [(k, lab, f(g)) for k, lab, f in SPEC[s]]
    nl = q[q["var"] == f"nl_{s}"].sort_values("h")
    sv = g("sv")
    n = len(sv)
    pos = sum(np.clip(c, 0, None) for _, _, c in comps)
    neg = sum(np.clip(c, None, 0) for _, _, c in comps)
    lo, hi = min(neg.min(), nl.lo68.min(), 0), max(pos.max(), nl.hi68.max(), sv.max(), 0)
    step = 0.25 if hi - lo <= 2.1 else 0.5
    lo, hi = np.floor(lo / step) * step, np.ceil(hi / step) * step
    ml, mr, mt, mb = 48, 12, 14, 40
    X = lambda i: ml + (i + 0.5) / n * (w - ml - mr)
    Y = lambda v: mt + (hi - v) / (hi - lo) * (h - mt - mb)
    bw = 0.58 * (w - ml - mr) / n
    o = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{title}">']
    for t in np.arange(lo, hi + step / 2, step):
        cls = "zero" if abs(t) < 1e-9 else "grid"
        o.append(f'<line class="{cls}" x1="{ml}" x2="{w-mr}" y1="{Y(t):.1f}" y2="{Y(t):.1f}"/>')
        o.append(f'<text class="tick" x="{ml-6}" y="{Y(t)+4:.1f}" text-anchor="end">{0 if abs(t) < 1e-9 else round(t, 2):g}</text>')
    for i in range(n):
        up = dn = 0.0
        for k, _, c in comps:  # 양수는 0 위로, 음수는 0 아래로 쌓기
            v = c[i]
            if v >= 0:
                top, bot, up = up + v, up, up + v
            else:
                top, bot, dn = dn, dn + v, dn + v
            o.append(f'<rect x="{X(i)-bw/2:.1f}" y="{Y(top):.1f}" width="{bw:.1f}" height="{max(Y(bot)-Y(top),0):.1f}" fill="{COL[k]}" opacity=".8"/>')
        o.append(f'<text class="tick" x="{X(i):.1f}" y="{h-mb+16}" text-anchor="middle">{i}</text>')
    o.append(f'<text class="axis" x="{(ml+w-mr)/2}" y="{h-4}" text-anchor="middle">h년 후 (t−1 = 0, t = 시작)</text>')
    for i, (a, b) in enumerate(zip(nl.lo68.values, nl.hi68.values)):
        o.append(f'<line x1="{X(i):.1f}" x2="{X(i):.1f}" y1="{Y(a):.1f}" y2="{Y(b):.1f}" stroke="{C_NL}" stroke-width="1.4"/>')
    for vals, col, dash in ((sv, C_SV, "5 4"), (nl.ratio.values, C_NL, "")):
        pts = " ".join(f"{X(i):.1f},{Y(v):.1f}" for i, v in enumerate(vals))
        o.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2" stroke-dasharray="{dash}"/>')
        o += [f'<circle cx="{X(i):.1f}" cy="{Y(v):.1f}" r="3" fill="#fff" stroke="{col}" stroke-width="1.8"/>' for i, v in enumerate(vals)]
    o.append("</svg>")
    sw = lambda c: f'<i style="display:inline-block;width:12px;height:12px;background:{c};opacity:.8"></i>'
    leg = "".join(f'<span>{sw(COL[k])}{lab}</span>' for k, lab, _ in comps)
    leg += (f'<span><i style="display:inline-block;width:14px;border-top:2px solid {C_NL}"></i>S−I·68% 구간</span>'
            f'<span><i style="display:inline-block;width:14px;border-top:2px dashed {C_SV}"></i>저축</span>')
    return f'<figure><div class="panel-title">{title}</div>{"".join(o)}<div class="legend">{leg}</div></figure>'


def main():
    r = pd.read_csv(OUT / "ep_report_fgls_nl.csv")
    r = r[r.kind == "resp"]
    n_hh, n_corp = int(r.n_hh.iloc[0]), int(r.n_corp.iloc[0])
    cells = [panel(r, "hh", f"(a) 가계(소비행태 변화, {n_hh}건)"), panel(r, "corp", f"(b) 기업(교역조건, {n_corp}건)")]
    style = (HERE / "report_template.html").read_text()
    style = style[:style.index("<main>")]
    note = ("주: 시작 연도 자기 부문 저축−투자 차(S−I, 국민계정상 순대출) 반응을 1로 정규화한 누적 반응(GDP 대비). "
            "막대 합 = S−I. 기업: 영업잉여 − 순배당(지급 − 수취) + 기타(이자·세금·기타 재산소득 등) − 투자 등. "
            "가계: 가처분소득 − 소비 + 기타(연금 조정 등) − 투자 등. 투자 등 = 고정투자·재고·토지 등 − 순자본이전. "
            "검은 선은 S−I 추정치와 68% 구간(국가 단위 부트스트랩 499회), 파란 점선은 저축. 2단계 가중최소제곱 국소투영(국가·연도 고정효과).")
    html = (f'<!doctype html><html><head><meta charset="utf-8">{style}</head><body><main style="max-width:900px">'
            f'<section id="F"><h2 style="font-size:18px">그림. 저축−투자 차의 세부 분해</h2>'
            f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:10px">{"".join(cells)}</div>'
            f'<p class="ci" style="font-size:12px">{note}</p></section></main></body></html>')
    (OUT / "ep_report_stacked_detail.html").write_text(html)
    with sync_playwright() as p:
        b = p.chromium.launch(executable_path="/opt/pw-browsers/chromium-1194/chrome-linux/chrome")
        pg = b.new_page(viewport={"width": 900, "height": 800}, device_scale_factor=2)
        pg.set_content(html)
        pg.wait_for_timeout(1200)
        pg.locator("#F").screenshot(path=str(OUT / "ep_report_stacked_detail.png"))
        b.close()


if __name__ == "__main__":
    main()
