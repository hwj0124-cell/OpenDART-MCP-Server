"""가계 사례 재정의 점검(점추정치): 기본 사양(새 자료 전체, 상품·서비스 교역조건, 가중치 없음, 국가·연도FE).

정의: 기존(가계저축률 급변) / 소비 주도 / 기업 저축 비상쇄 / 둘 다 (ep_detail의 hh_def)
가계 사례는 가계 순자금운용 시작 반응으로, 교역조건 사례는 기업 순자금운용 시작 반응으로 정규화(시작 = 1).
→ output/ep_hh_redef.csv, output/ep_hh_redef_fig.png
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
DEFS = {"기존 (저축률 급변)": None, "소비 주도": "cons", "기업 저축 비상쇄": "nocorp", "소비 주도 + 기업 저축 비상쇄": "both"}
HHV = ["nl_hh", "sv_hh", "invtot_hh", "yd_hh", "cons_hh", "sv_corp", "nl_corp", "nl_gov", "ca", "src_hh"]
COV = ["nl_corp", "sv_corp", "invtot_corp", "nl_hh", "ca"]


def est(d, s, vs):
    g0 = lt.ols(d, f"nl_{s}_h0", [f"z_{s}"], [], se=False)[0][0]
    return {v: np.array([lt.ols(d, f"{v}_h{h}", [f"z_{s}"], [], se=False)[0][0] for h in range(K + 1)]) / (1 if v.startswith("src") else g0)
            for v in vs}


def main():
    x = ed.er.panel()
    fc.VARS["hh"] = HHV
    fc.VARS["corp"] = COV
    rows = []
    for name, hd in DEFS.items():
        P = ed.paths(x, {**el.SPEC, "tot": fc.TOT, "hh_def": hd})
        d = fc.design(x, P)
        for v, r in est(d, "hh", HHV).items():
            rows += [dict(defn=name, sector="hh", var=v, h=h, ratio=r[h], n=len(P["hh"])) for h in range(K + 1)]
        if hd is None:  # 교역조건 사례는 가계 정의와 무관
            for v, r in est(d, "corp", COV).items():
                rows += [dict(defn="교역조건", sector="corp", var=v, h=h, ratio=r[h], n=len(P["corp"])) for h in range(K + 1)]
    r = pd.DataFrame(rows)
    r.to_csv(OUT / "ep_hh_redef.csv", index=False)
    print(r[r.h.isin([0, 1, 2, 3, 4, 6])].pivot_table(index=["defn", "var"], columns="h", values="ratio", sort=False).round(2).to_string())
    print(r[r.h == 0].groupby("defn").n.first().to_dict())
    render(r)


def render(r):
    XL = "h (t−1 = 0, t = 시작)"
    q = r[~r["var"].str.startswith("src")]
    YL = (np.floor(q.ratio.min() * 2) / 2, np.ceil(q.ratio.max() * 2) / 2)
    n = r[r.h == 0].groupby("defn").n.first()
    co = r[r.defn == "교역조건"]

    def s(df, v, cls):
        x = df[df["var"] == v].sort_values("h")
        return (cls, "", x.h.values, x.ratio.values, None, None)

    cells = []
    for name in DEFS:
        hh = r[r.defn == name]
        for v, title in (("nl", "순자금운용"), ("sv", "저축")):
            t = f"{name} (가계 n {n[name]}) vs 교역조건 (기업 n {n['교역조건']}): {title}"
            cells.append(f'<figure><div class="panel-title">{t}</div>'
                         f'{svg_lines([s(hh, f"{v}_hh", "s1"), s(co, f"{v}_corp", "s2")], XL, t, w=420, h=230, ylim=YL)}'
                         f'<div class="legend"><span><i class="sw s1"></i>가계 {title}</span><span><i class="sw s2"></i>기업 {title}</span></div></figure>')
    style = (HERE / "report_template.html").read_text()
    style = style[:style.index("<main>")] + """<style>
svg .line.s1{stroke:#2459a8;fill:none} svg .dot.s1{fill:#2459a8}
svg .line.s2{stroke:#d1701f;fill:none;stroke-dasharray:6 4} svg .dot.s2{fill:#d1701f}
.sw.s1{border-color:#2459a8} .sw.s2{border-color:#d1701f;border-top-style:dashed}</style>"""
    note = ("점추정치, GDP 대비 %, 새 원자료 전체 기간. 가계 = 가계 성향 사례의 가계 부문(가계 순자금운용 시작 = 1), "
            "기업 = 상품·서비스 교역조건 사례의 기업 부문(기업 순자금운용 시작 = 1), 국가·연도FE, 가중치 없음. "
            "소비 주도: 부호 맞춘 −Δ소비 ≥ Δ가처분소득(GDP 대비). 기업 저축 비상쇄: 같은 해 부호 맞춘 Δ기업저축 ≥ 0. "
            "공통: 1년 변화 국가별 상·하위 10%, 순자금운용이 충격과 같은 방향, 4년 내 중복 제외, 시작 2020–22 제외, 그해 성장률 ≥ 0.")
    html = (f'<!doctype html><html><head><meta charset="utf-8">{style}</head><body><main style="max-width:900px">'
            f'<section id="A"><h2 style="font-size:18px">가계 사례 재정의: 가계 vs 기업 순자금운용·저축 (같은 세로축)</h2>'
            f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:10px">{"".join(cells)}</div>'
            f'<p class="ci" style="font-size:12px">{note}</p></section></main></body></html>')
    (OUT / "ep_hh_redef_figs.html").write_text(html)
    with sync_playwright() as p:
        b = p.chromium.launch(executable_path="/opt/pw-browsers/chromium-1194/chrome-linux/chrome")
        pg = b.new_page(viewport={"width": 900, "height": 800}, device_scale_factor=2)
        pg.set_content(html)
        pg.wait_for_timeout(1500)
        pg.locator("#A").screenshot(path=str(OUT / "ep_hh_redef_fig.png"))
        b.close()


if __name__ == "__main__":
    main()
