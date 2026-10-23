"""기본 사양 전 변수 점검(점추정치): 새 원자료 전체 기간, 상품·서비스 교역조건, 가중치 없음, 국가·연도FE.

y_{t+h} − y_{t−1} = γ_h · z_it + 국가FE + 연도FE,  정규화 = γ_h ÷ γ_0(원천 부문 순자금운용)
원천 변수(가계저축률 %p, 교역조건 누적 %)는 원단위 γ_h.
경상수지(국민경제 순대출) = 기업 + 금융 + 정부 + 가계 순자금운용 (금융은 미제공 시 잔차).
→ output/ep_full_check.csv, output/ep_full_check_fig.png
"""
import numpy as np
import pandas as pd
try:
    from playwright.sync_api import sync_playwright
except ImportError:  # 로컬 PC 등 playwright가 없으면 한글 HTML 미리보기 PNG만 건너뜀(분석 결과·논문 그림과 무관)
    sync_playwright = None

import ep_detail as ed
import ep_lpiv as el
import lp_tools as lt
from make_report import HERE, OUT, svg_lines

K = ed.K
TOT = "tot_na_g"
COMMON = ["ca", "nl_hh", "nl_corp", "nl_gov", "nl_fin"]
VARS = {"hh": COMMON + ["sv_hh", "invtot_hh", "inv_hh", "yd_hh", "cons_hh", "src_hh"],
        "corp": COMMON + ["sv_corp", "invtot_corp", "inv_corp", "prof", "div_paid", "int_paid", "tax_paid", "src_corp"]}
LAB = {"ca": "경상수지", "nl_hh": "가계 순자금운용", "nl_corp": "기업 순자금운용", "nl_gov": "정부 순자금운용",
       "nl_fin": "금융 순자금운용", "sv_hh": "가계 저축", "sv_corp": "기업 저축", "invtot_hh": "가계 투자 등",
       "invtot_corp": "기업 투자 등", "inv_hh": "가계 고정투자", "inv_corp": "기업 고정투자", "yd_hh": "가처분소득",
       "cons_hh": "소비", "prof": "영업잉여", "div_paid": "배당 지급", "int_paid": "이자 지급", "tax_paid": "법인세 등 지급",
       "src_hh": "가계저축률 (%p)", "src_corp": "교역조건 누적 (%)"}


def design(x, P):
    d = x.copy()
    d["src_hh"] = d.hh_rate
    d["src_corp"] = 100 * np.exp(np.log1p(d[TOT] / 100).groupby(d.country).cumsum())
    g = lambda v: d.groupby("country")[v]
    cols = {}
    for s, E in P.items():
        z = E.set_index(["country", "year"]).sign
        cols[f"z_{s}"] = pd.Series([z.get((c, y), 0.0) for c, y in zip(d.country, d.year)], index=d.index)
        for v in VARS[s]:
            for h in range(K + 1):
                key = f"{v}_h{h}"
                if key in cols:
                    continue
                cols[key] = 100 * (g(v).shift(-h) / g(v).shift(1) - 1) if v == "src_corp" else g(v).shift(-h) - g(v).shift(1)
    return pd.concat([d, pd.DataFrame(cols)], axis=1)


def main():
    x = ed.er.panel()
    P = ed.paths(x, {**el.SPEC, "tot": TOT})
    d = design(x, P)
    rows = []
    for s, vs in VARS.items():
        g0 = lt.ols(d, f"nl_{s}_h0", [f"z_{s}"], [], se=False)[0][0]
        for v in vs:
            for h in range(K + 1):
                g = lt.ols(d, f"{v}_h{h}", [f"z_{s}"], [], se=False)[0][0]
                rows.append(dict(sector=s, var=v, h=h, gamma=g, ratio=g if v.startswith("src") else g / g0, n=len(P[s])))
    r = pd.DataFrame(rows)
    r.to_csv(OUT / "ep_full_check.csv", index=False)
    for s in VARS:
        print(f"== {s} (n {len(P[s])}, γ0 = {r[(r.sector==s)&(r['var']==f'nl_{s}')&(r.h==0)].gamma.iloc[0]:.2f}%p)")
        print(r[r.sector == s].pivot_table(index="var", columns="h", values="ratio", sort=False).round(2).to_string())
    render(r, {s: len(E) for s, E in P.items()})


def render(r, n):
    XL = "h (t−1 = 0, t = 시작)"
    lim = lambda a: (np.floor(np.nanmin(a) * 2) / 2, np.ceil(np.nanmax(a) * 2) / 2)
    YL = lim(r[~r["var"].str.startswith("src")].ratio)

    def fig(s, title, vs, ylim):
        cls = ["s1", "s2", "s3"][:len(vs)]
        ser = [(c, "", r[(r.sector == s) & (r["var"] == v)].h.values, r[(r.sector == s) & (r["var"] == v)].ratio.values, None, None)
               for v, c in zip(vs, cls)]
        leg = "".join(f'<span><i class="sw {c}"></i>{LAB[v]}</span>' for v, c in zip(vs, cls))
        return (f'<figure><div class="panel-title">{title}</div>{svg_lines(ser, XL, title, w=420, h=230, ylim=ylim)}'
                f'<div class="legend">{leg}</div></figure>')

    H, C = f"가계 성향 (n {n['hh']})", f"상품·서비스 교역조건 (n {n['corp']})"
    cells = [fig("hh", f"{H}: 순자금운용·저축", ["nl_hh", "sv_hh"], YL), fig("corp", f"{C}: 순자금운용·저축", ["nl_corp", "sv_corp"], YL),
             fig("hh", f"{H}: 투자", ["invtot_hh", "inv_hh"], YL), fig("corp", f"{C}: 투자", ["invtot_corp", "inv_corp"], YL),
             fig("hh", f"{H}: 소득·소비", ["yd_hh", "cons_hh"], YL), fig("corp", f"{C}: 이익·배당", ["prof", "div_paid"], YL),
             '<figure><p class="ci" style="font-size:12px;padding:12px">가계 저축 ≈ 가처분소득 − 소비.</p></figure>',
             fig("corp", f"{C}: 이자·법인세", ["int_paid", "tax_paid"], YL),
             fig("hh", f"{H}: 경상수지·정부", ["ca", "nl_gov"], YL), fig("corp", f"{C}: 경상수지·정부", ["ca", "nl_gov"], YL),
             fig("hh", f"{H}: 다른 민간부문", ["nl_corp", "nl_fin"], YL), fig("corp", f"{C}: 다른 민간부문", ["nl_hh", "nl_fin"], YL),
             fig("hh", "원천: 가계저축률 변화 (%p, 원단위)", ["src_hh"], lim(r[r["var"] == "src_hh"].ratio)),
             fig("corp", "원천: 교역조건 누적 변화 (%, 원단위)", ["src_corp"], lim(r[r["var"] == "src_corp"].ratio))]
    style = (HERE / "report_template.html").read_text()
    style = style[:style.index("<main>")] + """<style>
svg .line.s1{stroke:#2459a8;fill:none} svg .dot.s1{fill:#2459a8}
svg .line.s2{stroke:#d1701f;fill:none;stroke-dasharray:6 4} svg .dot.s2{fill:#d1701f}
.sw.s1{border-color:#2459a8} .sw.s2{border-color:#d1701f;border-top-style:dashed}</style>"""
    note = ("점추정치. 새 원자료 전체 기간(불균형 패널), GDP 대비 %. y<sub>t+h</sub> − y<sub>t−1</sub> = γ<sub>h</sub>·z<sub>it</sub> + 국가FE + 연도FE, "
            "정규화 = γ<sub>h</sub> ÷ γ<sub>0</sub>(원천 부문 순자금운용), 원단위 패널 제외 모두 같은 세로축. 경상수지 = 국민경제 순대출 = 기업+금융+정부+가계. "
            "에피소드: 1년 변화 국가별 상·하위 10%, 순자금운용이 충격과 같은 방향, 4년 내 중복 제외, 시작 2020–22 제외, 그해 성장률 ≥ 0. "
            "교역조건 = 수출/수입 디플레이터(상품·서비스), 영업잉여 같은 방향.")
    html = (f'<!doctype html><html><head><meta charset="utf-8">{style}</head><body><main style="max-width:900px">'
            f'<section id="A"><h2 style="font-size:18px">기본 사양 전 변수 점검: 가계 성향 vs 상품·서비스 교역조건</h2>'
            f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:10px">{"".join(cells)}</div>'
            f'<p class="ci" style="font-size:12px">{note}</p></section></main></body></html>')
    (OUT / "ep_full_check_figs.html").write_text(html)
    with sync_playwright() as p:
        b = p.chromium.launch(executable_path="/opt/pw-browsers/chromium-1194/chrome-linux/chrome")
        pg = b.new_page(viewport={"width": 900, "height": 800}, device_scale_factor=2)
        pg.set_content(html)
        pg.wait_for_timeout(1500)
        pg.locator("#A").screenshot(path=str(OUT / "ep_full_check_fig.png"))
        b.close()


if __name__ == "__main__":
    main()
