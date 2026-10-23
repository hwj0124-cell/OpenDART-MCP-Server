"""ep_lpiv_fe 결과 → 한글 PNG. 모든 패널 같은 세로축, 68% 구간(국가 부트스트랩)."""
import numpy as np
import pandas as pd
from playwright.sync_api import sync_playwright

from make_report import HERE, OUT, svg_lines

r = pd.read_csv(OUT / "ep_lpiv_fe.csv")
f = r[r.model == "고정효과"]
XL = "h (t−1 = 0, t = 시작)"
LAB = {"nl": "순자금운용", "sv": "저축", "invtot": "투자 등", "inv": "고정투자"}


def ser(df, v, cls):
    x = df[df["var"] == v].sort_values("h")
    return (cls, "", x.h.values, x.beta.values, x.lo68.values, x.hi68.values)


def fig(title, series, items, ylim):
    leg = "".join(f'<span><i class="sw {c}"></i>{t}</span>' for c, t in items)
    return (f'<figure><div class="panel-title">{title}</div>{svg_lines(series, XL, title, w=420, h=240, ylim=ylim)}'
            f'<div class="legend">{leg}</div></figure>')


band = r[r.model.isin(["고정효과", "차이(가계−교역조건)"])][["beta", "lo68", "hi68"]].to_numpy()
YL = (np.floor(np.nanmin(band) * 2) / 2, np.ceil(np.nanmax(band) * 2) / 2)
n = {"hh": 86, "corp": 42}
cells = []
for a, b in (("nl", "sv"), ("invtot", "inv")):
    for s, who in (("hh", "가계 성향"), ("corp", "교역조건(비침체)")):
        cells.append(fig(f"{who} (n {n[s]}): {LAB[a]}·{LAB[b]}", [ser(f, f"{a}_{s}", "s1"), ser(f, f"{b}_{s}", "s2")],
                         [("s1", LAB[a]), ("s2", LAB[b])], YL))
cells.append(fig("순자금운용: 가계 성향 vs 교역조건", [ser(f, "nl_hh", "s1"), ser(f, "nl_corp", "s2")],
                 [("s1", "가계 성향"), ("s2", "교역조건")], YL))
cells.append(fig("차이: 가계 − 교역조건 (순자금운용)", [ser(r[r.model == "차이(가계−교역조건)"], "nl_diff", "s1")],
                 [("s1", "차이")], YL))
style = (HERE / "report_template.html").read_text()
style = style[:style.index("<main>")] + """<style>
svg .line.s1{stroke:#2459a8;fill:none} svg .band.s1{fill:#2459a8;opacity:.13} svg .dot.s1{fill:#2459a8}
svg .line.s2{stroke:#d1701f;fill:none;stroke-dasharray:6 4} svg .band.s2{fill:#d1701f;opacity:.13} svg .dot.s2{fill:#d1701f}
.sw.s1{border-color:#2459a8} .sw.s2{border-color:#d1701f;border-top-style:dashed}</style>"""
note = ("y<sub>t+h</sub> − y<sub>t−1</sub> = β<sub>h</sub>(nl<sub>t</sub> − nl<sub>t−1</sub>) + 국가FE + 연도FE. "
        "도구 = 에피소드 해의 충격 부호(±1), 그 외 0. β<sub>h</sub> = 시작 시점 자기 부문 순자금운용 1단위당 반응. "
        "에피소드: 1년 변화 국가별 상·하위 10%, 순자금운용이 충격과 같은 방향, 4년 내 중복 제외, 시작 2020–22 제외, 그해 성장률 ≥ 0 (가계·교역조건 모두). "
        "교역조건 = 수출/수입 디플레이터(상품·서비스), 영업잉여 같은 방향. 모두 GNI 대비 %. 음영 68% 구간(국가 단위 부트스트랩 499회).")
html = f"""<!doctype html><html><head><meta charset="utf-8">{style}</head><body><main style="max-width:900px">
<section id="A"><h2 style="font-size:18px">패널 LP-IV (국가·연도 고정효과): 가계 성향 vs 교역조건(비침체)</h2>
<div style="display:grid;grid-template-columns:1fr 1fr;gap:10px">{"".join(cells)}</div>
<p class="ci" style="font-size:12px">{note}</p></section></main></body></html>"""
(OUT / "ep_lpiv_figs.html").write_text(html)
with sync_playwright() as p:
    b = p.chromium.launch(executable_path="/opt/pw-browsers/chromium-1194/chrome-linux/chrome")
    pg = b.new_page(viewport={"width": 900, "height": 800}, device_scale_factor=2)
    pg.set_content(html)
    pg.wait_for_timeout(1500)
    pg.locator("#A").screenshot(path=str(OUT / "ep_lpiv_fig.png"))
    b.close()
print("ok", YL)
