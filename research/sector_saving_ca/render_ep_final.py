"""ep_final_ci 결과 → 한글 PNG (정의별 한 장). 모든 패널 같은 세로축, 음영 68%(국가 부트스트랩)."""
import numpy as np
import pandas as pd
from playwright.sync_api import sync_playwright

from make_report import HERE, OUT, svg_lines

TITLE = {"base": "가계 성향: 기존 정의 (저축률 급변)", "cons": "가계 성향: 소비 주도 정의"}
LAB = {"nl_hh": "가계 순자금운용", "nl_corp": "기업 순자금운용", "sv_hh": "가계 저축", "sv_corp": "기업 저축", "ca": "경상수지",
       "invtot_hh": "가계 투자 등", "inv_hh": "가계 고정투자", "invtot_corp": "기업 투자 등", "inv_corp": "기업 고정투자",
       "yd_hh": "가처분소득", "cons_hh": "소비", "prof": "영업잉여", "div_paid": "배당 지급", "nl_gov": "정부 순자금운용"}
XL = "h (t−1 = 0, t = 시작)"


def main():
    frames = {d: pd.read_csv(OUT / f"ep_final_ci_{d}.csv") for d in ("base", "cons")}
    allv = pd.concat(frames.values())
    YL = (np.floor(allv[["lo68"]].min().iloc[0] * 2) / 2, np.ceil(allv[["hi68"]].max().iloc[0] * 2) / 2)
    style = (HERE / "report_template.html").read_text()
    style = style[:style.index("<main>")] + """<style>
svg .line.s1{stroke:#2459a8;fill:none} svg .band.s1{fill:#2459a8;opacity:.13} svg .dot.s1{fill:#2459a8}
svg .line.s2{stroke:#d1701f;fill:none;stroke-dasharray:6 4} svg .band.s2{fill:#d1701f;opacity:.13} svg .dot.s2{fill:#d1701f}
.sw.s1{border-color:#2459a8} .sw.s2{border-color:#d1701f;border-top-style:dashed}</style>"""
    secs = []
    for defn, r in frames.items():
        nh, nc = int(r.n_hh.iloc[0]), int(r.n_corp.iloc[0])

        def ser(sec, v, cls):
            q = r[(r.sector == sec) & (r["var"] == v)].sort_values("h")
            return (cls, "", q.h.values, q.ratio.values, q.lo68.values, q.hi68.values)

        def fig(title, series, legend):
            leg = "".join(f'<span><i class="sw {c}"></i>{t}</span>' for c, t in legend)
            return (f'<figure><div class="panel-title">{title}</div>{svg_lines(series, XL, title, w=420, h=230, ylim=YL)}'
                    f'<div class="legend">{leg}</div></figure>')

        H, C = f"가계 사례(n {nh})", f"교역조건 사례(n {nc})"
        cells = [
            fig("순자금운용: 가계 vs 기업", [ser("hh", "nl_hh", "s1"), ser("corp", "nl_corp", "s2")],
                [("s1", f"{H} 가계 부문"), ("s2", f"{C} 기업 부문")]),
            fig("저축: 가계 vs 기업", [ser("hh", "sv_hh", "s1"), ser("corp", "sv_corp", "s2")],
                [("s1", f"{H} 가계 저축"), ("s2", f"{C} 기업 저축")]),
            fig("경상수지", [ser("hh", "ca", "s1"), ser("corp", "ca", "s2")], [("s1", H), ("s2", C)]),
            fig("차이(가계 − 교역조건): 순자금운용", [ser("diff", "nl", "s1")], [("s1", "차이")]),
            fig("차이(가계 − 교역조건): 저축", [ser("diff", "sv", "s1")], [("s1", "차이")]),
            fig("차이(가계 − 교역조건): 경상수지", [ser("diff", "ca", "s1")], [("s1", "차이")]),
            fig(f"{H}: 소득·소비", [ser("hh", "yd_hh", "s1"), ser("hh", "cons_hh", "s2")], [("s1", "가처분소득"), ("s2", "소비")]),
            fig(f"{C}: 투자", [ser("corp", "invtot_corp", "s1"), ser("corp", "inv_corp", "s2")], [("s1", "투자 등"), ("s2", "고정투자")]),
            fig(f"{H}: 기업·정부 순자금운용", [ser("hh", "nl_corp", "s1"), ser("hh", "nl_gov", "s2")], [("s1", "기업"), ("s2", "정부")]),
            fig(f"{C}: 가계·정부 순자금운용", [ser("corp", "nl_hh", "s1"), ser("corp", "nl_gov", "s2")], [("s1", "가계"), ("s2", "정부")]),
        ]
        grid = f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:10px">{"".join(cells)}</div>'
        secs.append(f'<section id="{defn}"><h2 style="font-size:18px">{TITLE[defn]} vs 상품·서비스 교역조건 (68% 구간)</h2>{grid}'
                    f'<p class="ci" style="font-size:12px">{NOTE}</p></section>')
    html = f'<!doctype html><html><head><meta charset="utf-8">{style}</head><body><main style="max-width:900px">{"".join(secs)}</main></body></html>'
    (OUT / "ep_final_figs.html").write_text(html)
    with sync_playwright() as p:
        b = p.chromium.launch(executable_path="/opt/pw-browsers/chromium-1194/chrome-linux/chrome")
        pg = b.new_page(viewport={"width": 900, "height": 800}, device_scale_factor=2)
        pg.set_content(html)
        pg.wait_for_timeout(1500)
        for d in frames:
            pg.locator(f"#{d}").screenshot(path=str(OUT / f"ep_final_fig_{d}.png"))
        b.close()


NOTE = ("새 원자료 전체 기간(불균형 패널), GDP 대비 %. y<sub>t+h</sub> − y<sub>t−1</sub> = γ<sub>h</sub>·z<sub>it</sub> + 국가FE + 연도FE, 가중치 없음. "
        "정규화 = γ<sub>h</sub> ÷ γ<sub>0</sub>(원천 부문 순자금운용): 가계 사례는 가계, 교역조건 사례는 기업 순자금운용 시작 = 1. "
        "에피소드: 1년 변화 국가별 상·하위 10%, 순자금운용이 충격과 같은 방향, 4년 내 중복 제외, 시작 2020–22 제외, 그해 성장률 ≥ 0. "
        "소비 주도: 부호 맞춘 −Δ소비 ≥ Δ가처분소득(GDP 대비). 교역조건 = 수출/수입 디플레이터(상품·서비스), 영업잉여 같은 방향. "
        "음영 68% 구간(국가 단위 부트스트랩 499회). 모든 패널 같은 세로축.")

if __name__ == "__main__":
    main()
