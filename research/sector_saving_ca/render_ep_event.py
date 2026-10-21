"""ep_event 결과 → 한글 PNG. 정규화 패널은 모두 같은 세로축, 음영 68%(국가 부트스트랩)."""
import numpy as np
import pandas as pd
from playwright.sync_api import sync_playwright

from make_report import HERE, OUT, svg_lines

r = pd.read_csv(OUT / "ep_event.csv")
BASE = "상품 교역조건, 국가·연도FE (기본)"
XL = "h (t−1 = 0, t = 시작)"
LAB = {"nl": "순자금운용", "sv": "저축", "invtot": "투자 등", "inv": "고정투자"}


def ser(df, v, cls, raw=False):
    x = df[df["var"] == v].sort_values("h")
    if raw:  # 원단위 γ_h ± 1 표준오차(≈68%)
        return (cls, "", x.h.values, x.gamma.values, (x.gamma - x.gamma_se).values, (x.gamma + x.gamma_se).values)
    return (cls, "", x.h.values, x.ratio.values, x.lo68.values, x.hi68.values)


def fig(title, series, items, ylim):
    leg = "".join(f'<span><i class="sw {c}"></i>{t}</span>' for c, t in items)
    return (f'<figure><div class="panel-title">{title}</div>{svg_lines(series, XL, title, w=420, h=240, ylim=ylim)}'
            f'<div class="legend">{leg}</div></figure>')


def lim(a):
    return np.floor(np.nanmin(a) * 2) / 2, np.ceil(np.nanmax(a) * 2) / 2


YL = lim(r[["ratio", "lo68", "hi68"]].to_numpy())
n = r[r.h == 0].groupby(["spec", "sector"]).n_ep.first()
b = r[r.spec == BASE]
cells = []
for a, c in (("nl", "sv"), ("invtot", "inv")):
    for s, who in (("hh", "가계 성향"), ("corp", "상품 교역조건")):
        cells.append(fig(f"{who} (n {int(n[(BASE, s)])}): {LAB[a]}·{LAB[c]}", [ser(b, f"{a}_{s}", "s1"), ser(b, f"{c}_{s}", "s2")],
                         [("s1", LAB[a]), ("s2", LAB[c])], YL))
cells.append(fig("순자금운용: 가계 성향 vs 상품 교역조건", [ser(b, "nl_hh", "s1"), ser(b, "nl_corp", "s2")],
                 [("s1", "가계 성향"), ("s2", "교역조건")], YL))
cells.append(fig("차이: 가계 − 교역조건 (순자금운용)", [ser(b, "nl_diff", "s1")], [("s1", "차이")], YL))
raw = b[b["var"].isin(["nl_hh", "nl_corp"])]
RL = lim(
         np.r_[raw.gamma - raw.gamma_se, raw.gamma + raw.gamma_se])
cells.append(fig("원단위: 에피소드 1건당 순자금운용 반응 (GNI 대비 %p)", [ser(b, "nl_hh", "s1", True), ser(b, "nl_corp", "s2", True)],
                 [("s1", "가계 성향"), ("s2", "교역조건")], RL))
cells.append('<figure><p class="ci" style="font-size:12px;padding:12px">원단위 패널만 세로축이 다릅니다(%p). 음영 ±1 표준오차(국가 군집).<br>'
             '나머지 패널은 시작 시점 순자금운용 반응(γ<sub>0</sub>)으로 나눈 값(시작 = 1)이며 모두 같은 세로축입니다.</p></figure>')
spec_cells = []
for sp in r.spec.unique():
    q = r[r.spec == sp]
    spec_cells.append(fig(f"{sp}: 가계 (n {int(n[(sp, 'hh')])}) vs 교역조건 (n {int(n[(sp, 'corp')])})",
                          [ser(q, "nl_hh", "s1"), ser(q, "nl_corp", "s2")], [("s1", "가계 성향"), ("s2", "교역조건")], YL))
    spec_cells.append(fig(f"{sp}: 차이", [ser(q, "nl_diff", "s1")], [("s1", "가계 − 교역조건")], YL))
grid = lambda c: f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:10px">{"".join(c)}</div>'
style = (HERE / "report_template.html").read_text()
style = style[:style.index("<main>")] + """<style>
svg .line.s1{stroke:#2459a8;fill:none} svg .band.s1{fill:#2459a8;opacity:.13} svg .dot.s1{fill:#2459a8}
svg .line.s2{stroke:#d1701f;fill:none;stroke-dasharray:6 4} svg .band.s2{fill:#d1701f;opacity:.13} svg .dot.s2{fill:#d1701f}
.sw.s1{border-color:#2459a8} .sw.s2{border-color:#d1701f;border-top-style:dashed}</style>"""
note = ("y<sub>t+h</sub> − y<sub>t−1</sub> = γ<sub>h</sub>·z<sub>it</sub> + 국가FE + 연도FE, z = 에피소드 해의 충격 부호(±1), 그 외 0. "
        "정규화 = γ<sub>h</sub> ÷ γ<sub>0</sub>(자기 부문 순자금운용). 에피소드: 1년 변화 국가별 상·하위 10%, 순자금운용이 충격과 같은 방향, "
        "4년 내 중복 제외, 시작 2020–22 제외, 그해 성장률 ≥ 0(가계·교역조건 모두). 교역조건 = OECD 상품 수출 디플레이터 ÷ 상품 수입 디플레이터, "
        "영업잉여 같은 방향. 모두 GNI 대비 %. 음영 68% 구간(국가 단위 부트스트랩 499회).")
html = f"""<!doctype html><html><head><meta charset="utf-8">{style}</head><body><main style="max-width:900px">
<section id="A"><h2 style="font-size:18px">이벤트 더미 국소투영(정규화): 가계 성향 vs 상품 교역조건</h2>{grid(cells)}
<p class="ci" style="font-size:12px">{note}</p></section>
<section id="B"><h2 style="font-size:18px">사양 비교: 자기 부문 순자금운용 (같은 세로축)</h2>{grid(spec_cells)}
<p class="ci" style="font-size:12px">국가FE만: 연도 고정효과 제외. 상품·서비스: WDI 국민계정 수출입 디플레이터. 개선·상승만: 가계저축률 상승·교역조건 개선 사례만. {note}</p></section>
</main></body></html>"""
(OUT / "ep_event_figs.html").write_text(html)
with sync_playwright() as p:
    br = p.chromium.launch(executable_path="/opt/pw-browsers/chromium-1194/chrome-linux/chrome")
    pg = br.new_page(viewport={"width": 900, "height": 800}, device_scale_factor=2)
    pg.set_content(html)
    pg.wait_for_timeout(1500)
    for sec in ("A", "B"):
        pg.locator(f"#{sec}").screenshot(path=str(OUT / f"ep_event_fig{sec}.png"))
    br.close()
print("ok", YL, RL)
