"""ep_detail 결과 → 한글 PNG. 정규화 변수는 모든 패널이 같은 세로축."""
import numpy as np
import pandas as pd
from playwright.sync_api import sync_playwright

from make_report import HERE, OUT, svg_lines

r = pd.read_csv(OUT / "ep_detail.csv")
s = pd.read_csv(OUT / "ep_detail_sens.csv")
XL = "t (t−1 = 0, t = 시작)"
LAB = {"nl": "순자금운용", "sv": "저축", "invtot": "투자 등", "inv": "고정투자", "yd_hh": "가처분소득", "cons_hh": "소비",
       "prof": "영업잉여", "div_paid": "배당 지급", "int_paid": "이자 지급", "tax_paid": "법인세 등 지급"}
n = r[(r.k == 0) & r["var"].str.startswith("nl")].set_index("sector").n


def ser(df, sec, v, cls):
    x = df[(df.sector == sec) & (df["var"] == v)].sort_values("k")
    return (cls, "", x.k.values, x.ratio.values, x.lo68.values, x.hi68.values)


def lab(v):
    return LAB.get(v, LAB.get(v.rsplit("_", 1)[0], v))


def fig(title, series, items, ylim):
    leg = "".join(f'<span><i class="sw {c}"></i>{t}</span>' for c, t in items)
    return (f'<figure><div class="panel-title">{title}</div>{svg_lines(series, XL, title, w=420, h=240, ylim=ylim)}'
            f'<div class="legend">{leg}</div></figure>')


def lim(df):
    v = df[["ratio", "lo68", "hi68"]].to_numpy()
    return np.floor(np.nanmin(v) * 2) / 2, np.ceil(np.nanmax(v) * 2) / 2


ROWS = [  # (가계 변수 쌍, 교역조건 변수 쌍)
    (("nl_hh", "sv_hh"), ("nl_corp", "sv_corp")),
    (("invtot_hh", "inv_hh"), ("invtot_corp", "inv_corp")),
    (("yd_hh", "cons_hh"), ("prof", "div_paid")),
    (None, ("int_paid", "tax_paid")),
]
YL = lim(r[r["var"] != "src"])
cells = []
for hh, co in ROWS:
    for sec, pair, who in (("hh", hh, "가계 성향"), ("corp", co, "교역조건")):
        if pair is None:
            cells.append('<figure><p class="ci" style="font-size:12px;padding:12px">가계 저축 ≈ 가처분소득 − 소비 (연금 조정 제외).<br>'
                         '교역조건 쪽은 영업잉여가 배당·이자·법인세로 얼마나 나가는지 확인.</p></figure>')
            continue
        cells.append(fig(f"{who} (n {n[sec]}): {lab(pair[0])}·{lab(pair[1])}",
                         [ser(r, sec, pair[0], "s1"), ser(r, sec, pair[1], "s2")],
                         [("s1", lab(pair[0])), ("s2", lab(pair[1]))], YL))
src = [fig("가계 성향: 가계저축률 변화 (%p, 원단위)", [ser(r, "hh", "src", "s1")], [("s1", "가계저축률, %p")],
           lim(r[(r["var"] == "src") & (r.sector == "hh")])),
       fig("교역조건: 교역조건 누적 변화 (%, 원단위)", [ser(r, "corp", "src", "s1")], [("s1", "교역조건, %")],
           lim(r[(r["var"] == "src") & (r.sector == "corp")]))]
# 조건별 민감도: 순자금운용
# '모든 사례 충격 부호 정렬'은 분모(평균 시작 반응)가 작아 구간이 −4까지 벌어져 그림에서 제외(표에는 남김)
sn = s[s["var"].str.startswith("nl") & (s.spec != "모든 사례 충격 부호 정렬")]
YS = lim(sn)
nn = sn[sn.k == 0].set_index(["spec", "sector"]).n
sens = [fig(f"{who}: {sp} (n {nn[(sp, sec)]})", [ser(sn[sn.spec == sp], sec, f"nl_{sec}", "s1")], [("s1", "순자금운용")], YS)
        for sp in sn.spec.unique() for sec, who in (("hh", "가계"), ("corp", "교역조건"))]

grid = lambda c: f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:10px">{"".join(c)}</div>'
style = (HERE / "report_template.html").read_text()
style = style[:style.index("<main>")] + """<style>
svg .line.s1{stroke:#2459a8;fill:none} svg .band.s1{fill:#2459a8;opacity:.13} svg .dot.s1{fill:#2459a8}
svg .line.s2{stroke:#d1701f;fill:none;stroke-dasharray:6 4} svg .band.s2{fill:#d1701f;opacity:.13} svg .dot.s2{fill:#d1701f}
.sw.s1{border-color:#2459a8} .sw.s2{border-color:#d1701f;border-top-style:dashed}</style>"""
note = ("값 = 사례 평균[부호 맞춘 v<sub>t+k</sub> − v<sub>t−1</sub>] / 사례 평균[부호 맞춘 시작 시점 순자금운용 변화] (시작 = 1). "
        "모두 GNI 대비 %. 투자 등 = 저축 − 순자금운용 (고정투자 + 재고투자 + 비생산자산 − 순자본이전 수취). 음영 68% 구간(사례 부트스트랩 499회).")
crit = ("기준: 1년 변화, 국가별 상·하위 10%, 연도평균 미제거, 전체 국가, 시작 시점 순자금운용이 충격과 같은 방향인 사례만, "
        "같은 나라 4년 내 중복 제외, 시작 2020–22 제외. 가계 성향: 가계저축률 변화, 그해 성장률 ≥ 0. 교역조건: 교역조건(수출/수입 디플레이터) 변화, 영업잉여 비중 같은 방향.")
html = f"""<!doctype html><html><head><meta charset="utf-8">{style}</head><body><main style="max-width:900px">
<section id="A"><h2 style="font-size:18px">세부 변수 반응 (모든 패널 같은 세로축)</h2>{grid(cells)}
<p class="ci" style="font-size:12px">{crit} {note}</p></section>
<section id="S"><h2 style="font-size:18px">충격 변수 자체의 경로 (원단위)</h2>{grid(src)}
<p class="ci" style="font-size:12px">충격 방향으로 부호를 맞춘 평균. 교역조건 = WDI 국민계정 수출 디플레이터 / 수입 디플레이터(상품·서비스).</p></section>
<section id="B"><h2 style="font-size:18px">조건별 민감도: 자기 부문 순자금운용 (같은 세로축)</h2>{grid(sens)}
<p class="ci" style="font-size:12px">각 패널은 기본 기준에서 조건 하나만 바꾼 것. {note}</p></section>
</main></body></html>"""
(OUT / "ep_detail_figs.html").write_text(html)
with sync_playwright() as p:
    b = p.chromium.launch(executable_path="/opt/pw-browsers/chromium-1194/chrome-linux/chrome")
    pg = b.new_page(viewport={"width": 900, "height": 800}, device_scale_factor=2)
    pg.set_content(html)
    pg.wait_for_timeout(1500)
    for sec in ("A", "S", "B"):
        pg.locator(f"#{sec}").screenshot(path=str(OUT / f"ep_detail_fig{sec}.png"))
    b.close()
print("ok", YL, YS)
