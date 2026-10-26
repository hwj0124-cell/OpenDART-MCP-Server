"""저축 정규화 사양 비교(점추정치): 사례 선정 조건 × 문턱(상·하위 10% / 1σ).

공통: 새 원자료 전체 기간, GDP 대비, 국가·연도FE, 가중치 없음, 가계 = 소비 주도, 교역조건 = 상품·서비스(영업잉여 같은 방향).
정규화: 시작 연도 자기 부문 저축 반응(γ0) = 1. 선정 조건:
  자금잉여 같은 방향 — 시작 연도 자기 부문 순자금운용이 충격과 같은 방향(흑자 기여)
  저축 같은 방향     — 시작 연도 자기 부문 저축이 충격과 같은 방향
  조건 없음          — 충격 부호로만 정렬
→ output/ep_sv_specs.csv, ep_sv_specs_figA.png(사양 비교), ep_sv_specs_figB.png(기본 사양 세부)
"""
import numpy as np
import pandas as pd
from playwright.sync_api import sync_playwright

import ep_detail as ed
import ep_full_check as fc
import ep_lpiv as el
import lp_tools as lt
from make_report import HERE, OUT, svg_lines

K = ed.K
SPECS = {
    "10%, 자금잉여 같은 방향": dict(base="nl"),
    "1σ, 자금잉여 같은 방향": dict(base="nl", sd=1.0),
    "10%, 저축 같은 방향": dict(base="sv"),
    "1σ, 저축 같은 방향": dict(base="sv", sd=1.0),
    "10%, 조건 없음": dict(base="sv", align="shock"),
    "1σ, 조건 없음": dict(base="sv", align="shock", sd=1.0),
}
HHV = ["sv_hh", "nl_hh", "yd_hh", "cons_hh", "invtot_hh"]
COV = ["sv_corp", "nl_corp", "prof", "div_paid", "inv_corp", "invtot_corp"]
LAB = {"sv_hh": "가계 저축", "nl_hh": "가계 자금잉여", "yd_hh": "가처분소득", "cons_hh": "소비", "invtot_hh": "가계 투자 등",
       "sv_corp": "기업 저축", "nl_corp": "기업 자금잉여", "prof": "영업잉여", "div_paid": "배당 지급", "inv_corp": "고정투자",
       "invtot_corp": "투자 등"}
XL = "h (t−1 = 0, t = 시작)"


def main():
    x = ed.er.panel()
    fc.VARS["hh"], fc.VARS["corp"] = HHV, COV
    rows = []
    for name, ch in SPECS.items():
        P = ed.paths(x, {**el.SPEC, "tot": fc.TOT, "hh_def": "cons", **ch})
        d = fc.design(x, P)
        for s, vs in (("hh", HHV), ("corp", COV)):
            g0 = lt.ols(d, f"sv_{s}_h0", [f"z_{s}"], [], se=False)[0][0]
            for v in vs:
                for h in range(K + 1):
                    rows.append(dict(spec=name, sector=s, var=v, h=h, n=len(P[s]), g0=g0,
                                     ratio=lt.ols(d, f"{v}_h{h}", [f"z_{s}"], [], se=False)[0][0] / g0))
    r = pd.DataFrame(rows)
    r.to_csv(OUT / "ep_sv_specs.csv", index=False)
    q = r[r["var"].isin(["sv_hh", "sv_corp", "nl_hh", "nl_corp"]) & r.h.isin([0, 1, 2, 3, 4, 6])]
    print(q.pivot_table(index=["spec", "var"], columns="h", values="ratio", sort=False).round(2).to_string())
    print(r[r.h == 0].groupby(["spec", "sector"]).agg(n=("n", "first"), g0=("g0", "first")).round(2).to_string())
    render(r)


def render(r):
    lim = lambda a: (np.floor(np.nanmin(a) * 2) / 2, np.ceil(np.nanmax(a) * 2) / 2)
    YA = lim(r[r["var"].isin(["sv_hh", "sv_corp", "nl_hh", "nl_corp"])].ratio)
    n = r[r.h == 0].groupby(["spec", "sector"]).n.first()

    def ser(df, v, cls):
        q = df[df["var"] == v].sort_values("h")
        return (cls, "", q.h.values, q.ratio.values, None, None)

    def fig(title, series, legend, ylim):
        leg = "".join(f'<span><i class="sw {c}"></i>{t}</span>' for c, t in legend)
        return (f'<figure><div class="panel-title">{title}</div>{svg_lines(series, XL, title, w=420, h=220, ylim=ylim)}'
                f'<div class="legend">{leg}</div></figure>')

    cells = []
    for sp in SPECS:
        q = r[r.spec == sp]
        t = f"{sp} (가계 {n[(sp, 'hh')]}·기업 {n[(sp, 'corp')]}건)"
        cells.append(fig(f"{t}: 저축", [ser(q, "sv_hh", "s1"), ser(q, "sv_corp", "s2")], [("s1", "가계"), ("s2", "기업")], YA))
        cells.append(fig(f"{t}: 자금잉여", [ser(q, "nl_hh", "s1"), ser(q, "nl_corp", "s2")], [("s1", "가계"), ("s2", "기업")], YA))
    b = r[r.spec == list(SPECS)[0]]
    YB = lim(b.ratio)
    det = [fig("기업: 저축·자금잉여·영업잉여", [ser(b, "sv_corp", "s1"), ser(b, "nl_corp", "s2"), ser(b, "prof", "s3")],
               [("s1", "저축"), ("s2", "자금잉여"), ("s3", "영업잉여")], YB),
           fig("기업: 배당·고정투자·투자 등", [ser(b, "div_paid", "s1"), ser(b, "inv_corp", "s2"), ser(b, "invtot_corp", "s3")],
               [("s1", "배당 지급"), ("s2", "고정투자"), ("s3", "투자 등")], YB),
           fig("가계: 저축·자금잉여·투자 등", [ser(b, "sv_hh", "s1"), ser(b, "nl_hh", "s2"), ser(b, "invtot_hh", "s3")],
               [("s1", "저축"), ("s2", "자금잉여"), ("s3", "투자 등")], YB),
           fig("가계: 가처분소득·소비", [ser(b, "yd_hh", "s1"), ser(b, "cons_hh", "s2")], [("s1", "가처분소득"), ("s2", "소비")], YB)]
    style = (HERE / "report_template.html").read_text()
    style = style[:style.index("<main>")] + """<style>
svg .line.s1{stroke:#2459a8;fill:none} svg .dot.s1{fill:#2459a8}
svg .line.s2{stroke:#d1701f;fill:none;stroke-dasharray:6 4} svg .dot.s2{fill:#d1701f}
svg .line.s3{stroke:#3b8a4f;fill:none;stroke-dasharray:2 3} svg .dot.s3{fill:#3b8a4f}
.sw.s1{border-color:#2459a8} .sw.s2{border-color:#d1701f;border-top-style:dashed} .sw.s3{border-color:#3b8a4f;border-top-style:dotted}</style>"""
    note = ("점추정치. 시작 연도 자기 부문 저축 반응 = 1로 정규화. 가계 = 소비 주도 가계저축률 급변 사례, 기업 = 상품·서비스 교역조건 사례(영업잉여 같은 방향). "
            "10% = 국가별 상·하위 10%, 1σ = 국가별 평균 ± 1표준편차 밖. 자금잉여 = 저축 − 투자 등(순대출). "
            "새 원자료 전체 기간, GDP 대비, 국가·연도FE, 가중치 없음, 4년 내 중복·시작 2020–22·침체 연도 제외.")
    html = (f'<!doctype html><html><head><meta charset="utf-8">{style}</head><body><main style="max-width:900px">'
            f'<section id="A"><h2 style="font-size:18px">저축 정규화: 사례 선정 조건 × 문턱 비교 (모두 같은 세로축)</h2>'
            f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:10px">{"".join(cells)}</div><p class="ci" style="font-size:12px">{note}</p></section>'
            f'<section id="B"><h2 style="font-size:18px">세부: {list(SPECS)[0]} (같은 세로축)</h2>'
            f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:10px">{"".join(det)}</div><p class="ci" style="font-size:12px">{note}</p></section>'
            f'</main></body></html>')
    (OUT / "ep_sv_specs_figs.html").write_text(html)
    with sync_playwright() as p:
        br = p.chromium.launch(executable_path="/opt/pw-browsers/chromium-1194/chrome-linux/chrome")
        pg = br.new_page(viewport={"width": 900, "height": 800}, device_scale_factor=2)
        pg.set_content(html)
        pg.wait_for_timeout(1500)
        for sec in ("A", "B"):
            pg.locator(f"#{sec}").screenshot(path=str(OUT / f"ep_sv_specs_fig{sec}.png"))
        br.close()


if __name__ == "__main__":
    main()
