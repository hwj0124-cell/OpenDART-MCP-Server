"""점추정치 비교: 가중치(없음/역분산) × 교역조건(상품/상품·서비스), 가계·기업 따로. 같은 세로축."""
import numpy as np
import pandas as pd
from playwright.sync_api import sync_playwright

import ep_detail as ed
import ep_lpiv as el
import lp_tools as lt
from make_report import HERE, OUT, svg_lines

K = ed.K
CASES = [("상품, 가중치 없음", "tot_goods_g", False, "c1"), ("상품, 역분산 가중", "tot_goods_g", True, "c2"),
         ("상품·서비스, 가중치 없음", "tot_na_g", False, "c3"), ("상품·서비스, 역분산 가중", "tot_na_g", True, "c4")]


def main():
    x = ed.er.panel()
    rows = []
    for name, tot, wls, _ in CASES:
        P = ed.paths(x, {**el.SPEC, "tot": tot})
        d = el.design(x, P)
        for s in ("hh", "corp"):
            d[f"w_{s}"] = 1 / d.groupby("country")[f"dx_{s}"].transform("var")
            g = np.array([lt.ols(d, f"nl_{s}_h{h}", [f"z_{s}"], [], se=False, w=f"w_{s}" if wls else None)[0][0]
                          for h in range(K + 1)])
            rows += [dict(case=name, sector=s, h=h, ratio=v, n=len(P[s])) for h, v in enumerate(g / g[0])]
    r = pd.DataFrame(rows)
    r.to_csv(OUT / "ep_weights.csv", index=False)
    print(r.pivot_table(index=["sector", "case"], columns="h", values="ratio", sort=False).round(2).to_string())
    lo, hi = np.floor(r.ratio.min() * 2) / 2, np.ceil(r.ratio.max() * 2) / 2
    XL = "h (t−1 = 0, t = 시작)"
    leg = "".join(f'<span><i class="sw {c}"></i>{n}</span>' for n, _, _, c in CASES)
    cells = []
    for s, title in (("hh", "가계 성향 → 가계 순자금운용"), ("corp", "교역조건 → 기업 순자금운용")):
        ser = [(c, n, r[(r.case == n) & (r.sector == s)].h.values, r[(r.case == n) & (r.sector == s)].ratio.values, None, None)
               for n, _, _, c in CASES]
        nn = r[(r.sector == s) & (r.h == 0)].set_index("case").n
        sub = " / ".join(f"{n.split(',')[0]} n {nn[n]}" for n in (CASES[0][0], CASES[2][0]))
        cells.append(f'<figure><div class="panel-title">{title} ({sub})</div>'
                     f'{svg_lines(ser, XL, title, w=640, h=300, ylim=(lo, hi))}<div class="legend">{leg}</div></figure>')
    style = (HERE / "report_template.html").read_text()
    style = style[:style.index("<main>")] + """<style>
svg .line.c1{stroke:#2459a8;fill:none} svg .dot.c1{fill:#2459a8}
svg .line.c2{stroke:#2459a8;fill:none;stroke-dasharray:6 4} svg .dot.c2{fill:#fff;stroke:#2459a8}
svg .line.c3{stroke:#d1701f;fill:none} svg .dot.c3{fill:#d1701f}
svg .line.c4{stroke:#d1701f;fill:none;stroke-dasharray:6 4} svg .dot.c4{fill:#fff;stroke:#d1701f}
.sw.c1{border-color:#2459a8} .sw.c2{border-color:#2459a8;border-top-style:dashed}
.sw.c3{border-color:#d1701f} .sw.c4{border-color:#d1701f;border-top-style:dashed}</style>"""
    note = ("점추정치(구간 없음). y<sub>t+h</sub> − y<sub>t−1</sub> = γ<sub>h</sub>·z<sub>it</sub> + 국가FE + 연도FE, 정규화 = γ<sub>h</sub> ÷ γ<sub>0</sub>. "
            "역분산 가중 = 국가별 자기 부문 Δ순자금운용 분산(전 기간)의 역수. 가계 사례는 교역조건 지표와 무관해 같은 가중치끼리 선이 겹칩니다. "
            "에피소드: 1년 변화 국가별 상·하위 10%, 순자금운용이 충격과 같은 방향, 4년 내 중복 제외, 시작 2020–22 제외, 그해 성장률 ≥ 0. "
            "교역조건은 영업잉여 같은 방향. 모두 GNI 대비 %.")
    html = (f'<!doctype html><html><head><meta charset="utf-8">{style}</head><body><main style="max-width:700px">'
            f'<section id="A"><h2 style="font-size:18px">가중치 × 교역조건 지표: 순자금운용 지속성</h2>{"".join(cells)}'
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
