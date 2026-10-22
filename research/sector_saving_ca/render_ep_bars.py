"""보고서 그림: 왼쪽 가계·기업 저축−투자 차(S−I) 선(68% 구간), 오른쪽 기업 S−I 분해 누적 막대(점추정치).

기업 S−I = 영업잉여 − 순배당(지급 − 수취) − 투자(총자본형성: 고정투자·재고·귀중품) + 기타
  기타 = 이자·세금·기타 재산소득·경상이전 − 토지 등 비생산자산 취득 + 순자본이전 수취
사양: ep_report와 같음(S−I 정규화, 2단계 가중최소제곱, 국가·연도FE). 사용: python render_ep_bars.py [HMAX]
→ output/ep_report_bars.csv, output/ep_report_bars_a.png(그림 1), output/ep_report_bars_b.png(그림 2)
"""
import sys

from pathlib import Path

import numpy as np
import pandas as pd
try:
    from playwright.sync_api import sync_playwright
except ImportError:  # 로컬 PC 등 playwright가 없으면 한글 HTML 미리보기 PNG만 건너뜀(분석 결과·논문 그림과 무관)
    sync_playwright = None

import ep_report as rp
import lp_tools as lt
from make_report import HERE, OUT, svg_lines

HMAX = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 6
K = rp.K
COL = {"prof": "#2459a8", "div": "#8e6cc0", "inv": "#d1701f", "oth": "#a3a3a3"}
XL = "h년 후 (t−1 = 0, t = 시작)"


CHROME = "/opt/pw-browsers/chromium-1194/chrome-linux/chrome"


def estimate():
    rp.LAGS = "--lags" in sys.argv
    rp.FGLS = rp.WSV = True
    rp.NORM = "nl"
    rp.HHV, rp.COV = ["nl_hh", "sv_hh"], ["nl_corp", "sv_corp", "prof", "div_net", "gcf_corp"]
    P, d = rp.prepare()
    rows = []
    for s, vs in (("hh", rp.HHV), ("corp", rp.COV)):
        w = f"w_{s}"
        g0 = lt.ols(d, f"nl_{s}_h0", [f"z_{s}"], rp.ctrl(s), se=False, w=w)[0][0]
        for v in vs:
            for h in range(K + 1):
                rows.append(dict(sector=s, var=v, h=h, n=len(P[s]),
                                 ratio=lt.ols(d, f"{v}_h{h}", [f"z_{s}"], rp.ctrl(s), se=False, w=w)[0][0] / g0))
    r = pd.DataFrame(rows)
    r.to_csv(OUT / f"ep_report_bars{'_lags' if rp.LAGS else ''}.csv", index=False)
    return r


def bars(r, title, w=640, h=360):
    g = lambda v: r[(r["var"] == v) & (r.h <= HMAX)].sort_values("h").ratio.values
    nl = g("nl_corp")
    comps = [("prof", "영업잉여", g("prof")), ("div", "−순배당", -g("div_net")), ("inv", "−투자", -g("gcf_corp"))]
    comps.append(("oth", "기타", nl - sum(c for _, _, c in comps)))  # 막대 합 = S−I가 되도록 나머지
    n = len(nl)
    pos = sum(np.clip(c, 0, None) for *_, c in comps)
    neg = sum(np.clip(c, None, 0) for *_, c in comps)
    step = 0.25
    lo, hi = np.floor(min(neg.min(), nl.min(), 0) / step) * step, np.ceil(max(pos.max(), nl.max(), 0) / step) * step
    ml, mr, mt, mb = 48, 12, 14, 40
    X = lambda i: ml + (i + 0.5) / n * (w - ml - mr)
    Y = lambda v: mt + (hi - v) / (hi - lo) * (h - mt - mb)
    bw = 0.58 * (w - ml - mr) / n
    o = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{title}">']
    for t in np.arange(lo, hi + step / 2, step):
        o.append(f'<line class="{"zero" if abs(t) < 1e-9 else "grid"}" x1="{ml}" x2="{w-mr}" y1="{Y(t):.1f}" y2="{Y(t):.1f}"/>')
        o.append(f'<text class="tick" x="{ml-6}" y="{Y(t)+4:.1f}" text-anchor="end">{0 if abs(t) < 1e-9 else round(t, 2):g}</text>')
    for i in range(n):
        up = dn = 0.0
        for k, _, c in comps:  # 양수는 0 위로, 음수는 0 아래로 쌓기
            v = c[i]
            top, bot = (up + v, up) if v >= 0 else (dn, dn + v)
            up, dn = (up + v, dn) if v >= 0 else (up, dn + v)
            o.append(f'<rect x="{X(i)-bw/2:.1f}" y="{Y(top):.1f}" width="{bw:.1f}" height="{max(Y(bot)-Y(top),0):.1f}" fill="{COL[k]}" opacity=".8"/>')
        o.append(f'<text class="tick" x="{X(i):.1f}" y="{h-mb+16}" text-anchor="middle">{i}</text>')
    o.append(f'<text class="axis" x="{(ml+w-mr)/2}" y="{h-4}" text-anchor="middle">{XL}</text>')
    pts = " ".join(f"{X(i):.1f},{Y(v):.1f}" for i, v in enumerate(nl))
    o.append(f'<polyline points="{pts}" fill="none" stroke="#222" stroke-width="2"/>')
    o += [f'<circle cx="{X(i):.1f}" cy="{Y(v):.1f}" r="3" fill="#fff" stroke="#222" stroke-width="1.8"/>' for i, v in enumerate(nl)]
    o.append("</svg>")
    sw = lambda c: f'<i style="display:inline-block;width:12px;height:12px;background:{c};opacity:.8"></i>'
    leg = "".join(f'<span>{sw(COL[k])}{lab}</span>' for k, lab, _ in comps)
    leg += '<span><i style="display:inline-block;width:14px;border-top:2px solid #222"></i>저축−투자 차(S−I)</span>'
    return f'<figure><div class="panel-title">{title}</div>{"".join(o)}<div class="legend">{leg}</div></figure>'


def render(r):
    n_hh, n_corp = int(r[r.sector == "hh"].n.iloc[0]), int(r[r.sector == "corp"].n.iloc[0])
    # 왼쪽 패널 68% 구간: 같은 사양의 부트스트랩 결과(ep_report.py --fgls --nl)
    sfx = "_lags" if "--lags" in sys.argv else ""
    ci = pd.read_csv(OUT / f"ep_report_fgls_nl{sfx}.csv")
    ci = ci[ci.kind == "resp"]
    q = lambda s, v: ci[(ci.sector == s) & (ci["var"] == v) & (ci.h <= HMAX)].sort_values("h")
    for s, v in (("hh", "nl_hh"), ("corp", "nl_corp")):  # 점추정치가 그림 오른쪽과 같은 사양인지 확인
        a = r[(r.sector == s) & (r["var"] == v) & (r.h <= HMAX)].sort_values("h").ratio.values
        assert np.allclose(a, q(s, v).ratio.values, atol=1e-6), f"{v}: 구간 파일과 점추정치가 다름"
    ser = [("s1", "", q("hh", "nl_hh").h.values, q("hh", "nl_hh").ratio.values, q("hh", "nl_hh").lo68.values, q("hh", "nl_hh").hi68.values),
           ("s2", "", q("corp", "nl_corp").h.values, q("corp", "nl_corp").ratio.values, q("corp", "nl_corp").lo68.values, q("corp", "nl_corp").hi68.values)]
    vals = np.r_[ser[0][3], ser[0][4], ser[0][5], ser[1][3], ser[1][4], ser[1][5]]
    ylim = (np.floor(min(vals.min(), 0) / 0.25) * 0.25, np.ceil(max(vals.max(), 0) / 0.25) * 0.25)
    t1 = "(a) 저축−투자 차(S−I): 가계 vs 기업"
    left = (f'<figure><div class="panel-title">{t1}</div>{svg_lines(ser, XL, t1, w=640, h=360, ylim=ylim)}'
            f'<div class="legend"><span><i class="sw s1"></i>가계(소비행태 변화, {n_hh}건)</span>'
            f'<span><i class="sw s2"></i>기업(교역조건, {n_corp}건)</span></div></figure>')
    right = bars(r[r.sector == "corp"], "(b) 기업 S−I 분해")
    style = (HERE / "report_template.html").read_text()
    style = style[:style.index("<main>")] + """<style>
svg .line.s1{stroke:#2459a8;fill:none;stroke-width:2.2} svg .dot.s1{fill:#2459a8} svg .band.s1{fill:#2459a8;opacity:.14}
svg .line.s2{stroke:#d1701f;fill:none;stroke-width:2.2;stroke-dasharray:6 4} svg .dot.s2{fill:#d1701f} svg .band.s2{fill:#d1701f;opacity:.14}
.sw.s1{border-color:#2459a8} .sw.s2{border-color:#d1701f;border-top-style:dashed}</style>"""
    note = ("주: 시작 연도 자기 부문 저축−투자 차(S−I, 국민계정상 순대출) 반응을 1로 정규화한 누적 반응(GDP 대비). "
            "(a) 음영은 68% 구간(국가 단위 부트스트랩 499회), (b)는 점추정치. "
            "(b) 막대 합 = 기업 S−I: 영업잉여 − 순배당(지급 − 수취) − 투자(총자본형성: 고정투자·재고) + 기타"
            "(이자·세금·기타 재산소득, 토지 등 비생산자산, 자본이전 등). "
            "가계 = 가계저축률 변화가 국가별 상·하위 10%이고 소비 비중 변화가 가처분소득 비중 변화보다 큰 해, "
            "기업 = 교역조건 변화가 국가별 상·하위 10%이고 영업잉여가 같은 방향인 해. 2단계 가중최소제곱 국소투영(국가·연도 고정효과).")
    common = ("주: 시작 연도 자기 부문 저축−투자 차(S−I, 국민계정상 순대출) 반응을 1로 정규화한 누적 반응(GDP 대비). "
              "가계 = 가계저축률 변화가 국가별 상·하위 10%이고 소비 비중 변화가 가처분소득 비중 변화보다 큰 해(소비행태 변화), "
              "기업 = 교역조건 변화가 국가별 상·하위 10%이고 영업잉여가 같은 방향인 해. OECD 42개국 1970~2025년, "
              "2단계 가중최소제곱 국소투영(국가·연도 고정효과).")
    note_a = common + " 음영은 68% 구간(국가 단위 부트스트랩 499회)."
    note_b = (common + " 막대 합 = 기업 S−I: 영업잉여 − 순배당(지급 − 수취) − 투자(총자본형성) + 기타"
              "(이자·세금·기타 재산소득, 토지 등, 자본이전). 점추정치.")
    left = left.replace(t1, "")
    right = right.replace("(b) 기업 S−I 분해", "")
    sec = lambda i, title, body, note: (f'<section id="{i}"><h2 style="font-size:19px">{title}</h2>{body}'
                                        f'<p class="ci" style="font-size:12px">{note}</p></section>')
    html = (f'<!doctype html><html><head><meta charset="utf-8">{style}</head><body><main style="max-width:700px">'
            + sec("A", "그림 1. 원천별 저축−투자 차(S−I)의 지속성", left, note_a)
            + sec("B", "그림 2. 기업 저축−투자 차의 해소 경로", right, note_b)
            + '</main></body></html>')
    (OUT / f"ep_report_bars{sfx}.html").write_text(html)
    if sync_playwright is None or not Path(CHROME).exists():
        return
    with sync_playwright() as p:
        b = p.chromium.launch(executable_path=CHROME)
        pg = b.new_page(viewport={"width": 700, "height": 800}, device_scale_factor=2)
        pg.set_content(html)
        pg.wait_for_timeout(1200)
        pg.locator("#A").screenshot(path=str(OUT / f"ep_report_bars{sfx}_a.png"))
        pg.locator("#B").screenshot(path=str(OUT / f"ep_report_bars{sfx}_b.png"))
        b.close()


if __name__ == "__main__":
    r = pd.read_csv(OUT / f"ep_report_bars{'_lags' if '--lags' in sys.argv else ''}.csv") if "--render" in sys.argv else estimate()
    render(r)
    c = r[r.sector == "corp"].pivot_table(index="var", columns="h", values="ratio").round(2)
    c.loc["기타"] = (c.loc["nl_corp"] - c.loc["prof"] + c.loc["div_net"] + c.loc["gcf_corp"]).round(2)
    print(c.to_string())
