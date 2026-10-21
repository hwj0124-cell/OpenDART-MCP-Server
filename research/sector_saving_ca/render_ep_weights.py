"""점추정치 비교: 가중치(없음/역분산) × 교역조건(상품/상품·서비스). GDP 대비 패널.

패널: 가계 순자금운용 / 기업 순자금운용·저축·투자 등·고정투자 — 각 패널에 네 경우를 겹쳐 그림, 모두 같은 세로축.
y_{t+h} − y_{t−1} = γ_h z_it + 국가FE + 연도FE, 정규화 = γ_h ÷ γ_0(자기 부문 순자금운용).
GNI 대비 패널(data/panel_gni.csv)로 같은 계산을 해 순자금운용만 비교용으로 함께 저장.
→ output/ep_weights.csv, output/ep_weights_fig.png
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
CASES = [("상품, 가중치 없음", "tot_goods_g", False, "c1"), ("상품, 역분산 가중", "tot_goods_g", True, "c2"),
         ("상품·서비스, 가중치 없음", "tot_na_g", False, "c3"), ("상품·서비스, 역분산 가중", "tot_na_g", True, "c4")]
VARS = {"hh": ["nl_hh"], "corp": ["nl_corp", "sv_corp", "invtot_corp", "inv_corp"]}
LAB = {"nl_hh": "가계 성향 → 가계 순자금운용", "nl_corp": "교역조건 → 기업 순자금운용", "sv_corp": "교역조건 → 기업 저축",
       "invtot_corp": "교역조건 → 기업 투자 등 (저축 − 순자금운용)", "inv_corp": "교역조건 → 기업 고정투자"}


def estimate(x, denom):
    rows = []
    for name, tot, wls, _ in CASES:
        P = ed.paths(x, {**el.SPEC, "tot": tot})
        d = el.design(x, P)
        for s, vs in VARS.items():
            d[f"w_{s}"] = 1 / d.groupby("country")[f"dx_{s}"].transform("var")
            w = f"w_{s}" if wls else None
            g0 = lt.ols(d, f"nl_{s}_h0", [f"z_{s}"], [], se=False, w=w)[0][0]
            for v in vs:
                for h in range(K + 1):
                    g = lt.ols(d, f"{v}_h{h}", [f"z_{s}"], [], se=False, w=w)[0][0]
                    rows.append(dict(denom=denom, case=name, sector=s, var=v, h=h, gamma=g, ratio=g / g0, n=len(P[s])))
    return pd.DataFrame(rows)


def main():
    r = pd.concat([estimate(ed.er.panel(), "GDP"),
                   estimate(ed.er.panel(ra.HERE / "data" / "panel_gni.csv"), "GNI")], ignore_index=True)
    r.to_csv(OUT / "ep_weights.csv", index=False)
    print(r[r["var"].str.startswith("nl")].pivot_table(index=["var", "case", "denom"], columns="h", values="ratio",
                                                        sort=False).round(2).to_string())
    print(r[(r.denom == "GDP") & ~r["var"].str.startswith("nl")].pivot_table(
        index=["var", "case"], columns="h", values="ratio", sort=False).round(2).to_string())
    g = r[r.denom == "GDP"]
    lo, hi = np.floor(g.ratio.min() * 2) / 2, np.ceil(g.ratio.max() * 2) / 2
    XL = "h (t−1 = 0, t = 시작)"
    leg = "".join(f'<span><i class="sw {c}"></i>{n}</span>' for n, _, _, c in CASES)
    cells = []
    for v in ("nl_hh", "nl_corp", "sv_corp", "invtot_corp", "inv_corp"):
        q = g[g["var"] == v]
        ser = [(c, n, q[q.case == n].h.values, q[q.case == n].ratio.values, None, None) for n, _, _, c in CASES]
        nn = q[q.h == 0].set_index("case").n
        sub = f"상품 n {nn[CASES[0][0]]} / 상품·서비스 n {nn[CASES[2][0]]}"
        cells.append(f'<figure><div class="panel-title">{LAB[v]} ({sub})</div>'
                     f'{svg_lines(ser, XL, LAB[v], w=640, h=280, ylim=(lo, hi))}<div class="legend">{leg}</div></figure>')
    style = (HERE / "report_template.html").read_text()
    style = style[:style.index("<main>")] + """<style>
svg .line.c1{stroke:#2459a8;fill:none} svg .dot.c1{fill:#2459a8}
svg .line.c2{stroke:#2459a8;fill:none;stroke-dasharray:6 4} svg .dot.c2{fill:#fff;stroke:#2459a8}
svg .line.c3{stroke:#d1701f;fill:none} svg .dot.c3{fill:#d1701f}
svg .line.c4{stroke:#d1701f;fill:none;stroke-dasharray:6 4} svg .dot.c4{fill:#fff;stroke:#d1701f}
.sw.c1{border-color:#2459a8} .sw.c2{border-color:#2459a8;border-top-style:dashed}
.sw.c3{border-color:#d1701f} .sw.c4{border-color:#d1701f;border-top-style:dashed}</style>"""
    note = ("점추정치(구간 없음). 모두 GDP 대비 %. y<sub>t+h</sub> − y<sub>t−1</sub> = γ<sub>h</sub>·z<sub>it</sub> + 국가FE + 연도FE, "
            "정규화 = γ<sub>h</sub> ÷ γ<sub>0</sub>(자기 부문 순자금운용). 역분산 가중 = 국가별 자기 부문 Δ순자금운용 분산(전 기간)의 역수. "
            "가계 사례는 교역조건 지표와 무관해 같은 가중치끼리 선이 겹칩니다. 투자 등 = 저축 − 순자금운용(고정투자 + 재고 + 비생산자산 − 순자본이전). "
            "에피소드: 1년 변화 국가별 상·하위 10%, 순자금운용이 충격과 같은 방향, 4년 내 중복 제외, 시작 2020–22 제외, 그해 성장률 ≥ 0. "
            "교역조건은 영업잉여 같은 방향.")
    html = (f'<!doctype html><html><head><meta charset="utf-8">{style}</head><body><main style="max-width:700px">'
            f'<section id="A"><h2 style="font-size:18px">가중치 × 교역조건 지표: 순자금운용·저축·투자 (GDP 대비)</h2>{"".join(cells)}'
            f'<p class="ci" style="font-size:12px">{note}</p></section></main></body></html>')
    (OUT / "ep_weights_figs.html").write_text(html)
    with sync_playwright() as p:
        b = p.chromium.launch(executable_path="/opt/pw-browsers/chromium-1194/chrome-linux/chrome")
        pg = b.new_page(viewport={"width": 700, "height": 800}, device_scale_factor=2)
        pg.set_content(html)
        pg.wait_for_timeout(1500)
        pg.locator("#A").screenshot(path=str(OUT / "ep_weights_fig.png"))
        b.close()


if __name__ == "__main__":
    main()
