"""이벤트 더미 국소투영(정규화): 세부 변수. 기본 = 상품 교역조건, 아이슬란드 2003 제외.

y_{t+h} − y_{t−1} = γ_h · z_it + 국가FE + 연도FE,  정규화 = γ_h ÷ γ_0(자기 부문 순자금운용)
원천 변수(가계저축률 %p, 교역조건 누적 %)는 정규화 없이 원단위 γ_h.
구간: 국가 단위 부트스트랩 499회. 에피소드 기준은 ep_event와 같음.
→ output/ep_event_detail.csv, output/ep_event_detail_fig.png
"""
import numpy as np
import pandas as pd
from playwright.sync_api import sync_playwright

import ep_detail as ed
import ep_lpiv as el
import lp_tools as lt
import run_analysis as ra
from make_report import HERE, OUT, svg_lines

K = ed.K
CH = dict(tot="tot_goods_g", drop_eps=(("ISL", 2003),))
VARS = {"hh": ["nl_hh", "sv_hh", "invtot_hh", "inv_hh", "yd_hh", "cons_hh", "src_hh"],
        "corp": ["nl_corp", "sv_corp", "invtot_corp", "inv_corp", "prof", "div_paid", "int_paid", "tax_paid", "src_corp"]}
KEYS = [(s, v, h) for s in VARS for v in VARS[s] for h in range(K + 1)]


def design(x, P, tot):
    d = x.copy()
    g = lambda v: d.groupby("country")[v]
    d["src_hh"] = d.hh_rate
    d["src_corp"] = 100 * np.exp(np.log1p(d[tot] / 100).groupby(d.country).cumsum())  # 교역조건 수준(누적)
    for s, E in P.items():
        z = E.set_index(["country", "year"]).sign
        d[f"z_{s}"] = [z.get((c, y), 0.0) for c, y in zip(d.country, d.year)]
        for v in VARS[s]:
            for h in range(K + 1):
                if v == "src_corp":  # 누적 % 변화
                    d[f"{v}_h{h}"] = 100 * (g(v).shift(-h) / g(v).shift(1) - 1)
                else:
                    d[f"{v}_h{h}"] = g(v).shift(-h) - g(v).shift(1)
    return d


def gammas(d):
    return np.array([lt.ols(d, f"{v}_h{h}", [f"z_{s}"], [], se=False)[0][0] for s, v, h in KEYS])


def norm(g):
    g = pd.Series(g, index=pd.MultiIndex.from_tuples(KEYS))
    return np.array([g[k] if k[1].startswith("src") else g[k] / g[(k[0], f"nl_{k[0]}", 0)] for k in KEYS])


def render(r, n):
    XL = "h (t−1 = 0, t = 시작)"
    LAB = {"nl": "순자금운용", "sv": "저축", "invtot": "투자 등", "inv": "고정투자", "yd_hh": "가처분소득", "cons_hh": "소비",
           "prof": "영업잉여", "div_paid": "배당 지급", "int_paid": "이자 지급", "tax_paid": "법인세 등 지급"}
    lab = lambda v: LAB.get(v, LAB.get(v.rsplit("_", 1)[0], v))

    def ser(v, cls):
        x = r[r["var"] == v].sort_values("h")
        return (cls, "", x.h.values, x.ratio.values, x.lo68.values, x.hi68.values)

    def fig(title, vs, ylim):
        items = [("s1", lab(vs[0]))] + ([("s2", lab(vs[1]))] if len(vs) > 1 else [])
        leg = "".join(f'<span><i class="sw {c}"></i>{t}</span>' for c, t in items)
        return (f'<figure><div class="panel-title">{title}</div>'
                f'{svg_lines([ser(v, c) for v, (c, _) in zip(vs, items)], XL, title, w=420, h=240, ylim=ylim)}'
                f'<div class="legend">{leg}</div></figure>')

    lim = lambda a: (np.floor(np.nanmin(a) * 2) / 2, np.ceil(np.nanmax(a) * 2) / 2)
    YL = lim(r[~r["var"].str.startswith("src")][["ratio", "lo68", "hi68"]].to_numpy())
    rows = [(("nl_hh", "sv_hh"), ("nl_corp", "sv_corp")), (("invtot_hh", "inv_hh"), ("invtot_corp", "inv_corp")),
            (("yd_hh", "cons_hh"), ("prof", "div_paid")), (None, ("int_paid", "tax_paid"))]
    cells = []
    for hh, co in rows:
        cells.append(fig(f"가계 성향 (n {n['hh']}): {lab(hh[0])}·{lab(hh[1])}", hh, YL) if hh else
                     '<figure><p class="ci" style="font-size:12px;padding:12px">가계 저축 ≈ 가처분소득 − 소비.<br>'
                     '교역조건 쪽은 영업잉여가 배당·이자·법인세로 얼마나 나가는지 확인.</p></figure>')
        cells.append(fig(f"상품 교역조건 (n {n['corp']}): {lab(co[0])}·{lab(co[1])}", co, YL))
    for v, t in (("src_hh", "가계저축률 변화 (%p, 원단위)"), ("src_corp", "상품 교역조건 누적 변화 (%, 원단위)")):
        x = r[r["var"] == v]
        cells.append(f'<figure><div class="panel-title">{t}</div>'
                     f'{svg_lines([("s1", "", x.h.values, x.ratio.values, x.lo68.values, x.hi68.values)], XL, t, w=420, h=240, ylim=lim(x[["ratio", "lo68", "hi68"]].to_numpy()))}</figure>')
    style = (HERE / "report_template.html").read_text()
    style = style[:style.index("<main>")] + """<style>
svg .line.s1{stroke:#2459a8;fill:none} svg .band.s1{fill:#2459a8;opacity:.13} svg .dot.s1{fill:#2459a8}
svg .line.s2{stroke:#d1701f;fill:none;stroke-dasharray:6 4} svg .band.s2{fill:#d1701f;opacity:.13} svg .dot.s2{fill:#d1701f}
.sw.s1{border-color:#2459a8} .sw.s2{border-color:#d1701f;border-top-style:dashed}</style>"""
    note = ("y<sub>t+h</sub> − y<sub>t−1</sub> = γ<sub>h</sub>·z<sub>it</sub> + 국가FE + 연도FE. 정규화 = γ<sub>h</sub> ÷ γ<sub>0</sub>(자기 부문 순자금운용), "
            "원단위 패널 제외 모두 같은 세로축. 에피소드: 1년 변화 국가별 상·하위 10%, 순자금운용이 충격과 같은 방향, 4년 내 중복 제외, "
            "시작 2020–22 제외, 그해 성장률 ≥ 0. 교역조건 = OECD 상품 수출/수입 디플레이터, 영업잉여 같은 방향, 아이슬란드 2003 제외. "
            "모두 GNI 대비 %. 음영 68%(국가 부트스트랩 499회).")
    html = (f'<!doctype html><html><head><meta charset="utf-8">{style}</head><body><main style="max-width:900px">'
            f'<section id="A"><h2 style="font-size:18px">세부 변수: 가계 성향 vs 상품 교역조건(아이슬란드 2003 제외)</h2>'
            f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:10px">{"".join(cells)}</div>'
            f'<p class="ci" style="font-size:12px">{note}</p></section></main></body></html>')
    (OUT / "ep_event_detail_figs.html").write_text(html)
    with sync_playwright() as p:
        b = p.chromium.launch(executable_path="/opt/pw-browsers/chromium-1194/chrome-linux/chrome")
        pg = b.new_page(viewport={"width": 900, "height": 800}, device_scale_factor=2)
        pg.set_content(html)
        pg.wait_for_timeout(1500)
        pg.locator("#A").screenshot(path=str(OUT / "ep_event_detail_fig.png"))
        b.close()


def main():
    x = ed.er.panel()
    P = ed.paths(x, {**el.SPEC, **CH})
    d = design(x, P, CH["tot"])
    g = gammas(d)
    bs = lt.cluster_bootstrap(d, lambda b: norm(gammas(b)), reps=499)
    r = pd.DataFrame(KEYS, columns=["sector", "var", "h"]).assign(
        gamma=g, ratio=norm(g), lo68=np.nanpercentile(bs, 16, axis=0), hi68=np.nanpercentile(bs, 84, axis=0))
    r.to_csv(ra.OUT / "ep_event_detail.csv", index=False)
    print(r.pivot_table(index="var", columns="h", values="ratio", sort=False).round(2).to_string())
    render(r, {s: len(E) for s, E in P.items()})


if __name__ == "__main__":
    main()
