"""hh_tot_episodes 결과를 그림(PNG, 한글)으로: HTML(SVG) → 헤드리스 크롬 스크린샷."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from playwright.sync_api import sync_playwright

from make_report import HERE, OUT, svg_lines

Z68 = 1.0
q = int(sys.argv[1]) if len(sys.argv) > 1 else 10
o = pd.read_csv(OUT / f"hh_tot_episodes_q{q}.csv")
cs = pd.read_csv(OUT / f"hh_tot_episodes_cases_q{q}.csv").set_index(["type", "variant"]).n


def ser(t, var, kind, cls):
    x = o[(o.type == t) & (o.variant == var) & (o.kind == kind)].sort_values("k")
    return (cls, "", x.k.values, x["mean"].values, (x["mean"] - Z68 * x.se.fillna(0)).values,
            (x["mean"] + Z68 * x.se.fillna(0)).values)


KINDS = [("순자금운용", "ca"), ("저축", "hh"), ("투자", "corp"), ("기타(재고·자본이전 등)", "prof")]
VARS = [("기본", "ca"), ("코로나 제외", "hh"), ("코로나·금융위기 제외", "corp")]
leg_k = "".join(f'<span><i class="sw {c}"></i>{k}</span>' for k, c in KINDS)
leg_v = "".join(f'<span><i class="sw {c}"></i>{v}</span>' for v, c in VARS)
figA = "".join(
    f'<figure><div class="panel-title">{t} (사례 {cs[(t, "기본")]}개)</div>{svg_lines([ser(t, "기본", k, c) for k, c in KINDS], "시작 기준 연도 (t)", t)}'
    f'<div class="legend">{leg_k}</div></figure>' for t in ("가계 성향", "교역조건"))
figB = "".join(
    f'<figure><div class="panel-title">{t}: 순자금운용</div>{svg_lines([ser(t, v, "순자금운용", c) for v, c in VARS], "시작 기준 연도 (t)", t + "s")}'
    f'<div class="legend">{leg_v}</div><div class="ci" style="font-size:12px">사례: ' +
    " / ".join(f"{v} {cs[(t, v)]}" for v, _ in VARS) + '</div></figure>' for t in ("가계 성향", "교역조건"))
style = (HERE / "report_template.html").read_text()
style = style[:style.index("<main>")]
note = (f"41개국 OECD 부문계정. 원천 변화가 2년 변화 분포의 상·하위 {q}%인 해(증가·감소 통합). 같은 해 국가평균 제거, "
        "t−2 대비 변화를 시작 시점 자기 부문 순자금운용 변화로 나눔(시작 = 1). 선은 평균, 음영은 68% 구간(±1 표준오차).")
html = f"""<!doctype html><html><head><meta charset="utf-8">{style}</head><body><main style="max-width:1000px">
<section id="A"><h2 style="font-size:18px">순자금운용 = 저축 − 고정투자 + 기타 (상·하위 {q}%)</h2>
<div class="pair">{figA}</div><p class="ci" style="font-size:12px">{note}</p></section>
<section id="B"><h2 style="font-size:18px">시기 제외 민감도: 순자금운용 (상·하위 {q}%)</h2>
<div class="pair">{figB}</div><p class="ci" style="font-size:12px">코로나 제외: 시작 2019–2021 제외(이후 경로가 2020–21에 걸치는 사례). 코로나·금융위기 제외: 추가로 시작 2007–2009 제외. {note}</p></section>
</main></body></html>"""
(OUT / f"hh_tot_figs_q{q}.html").write_text(html)
with sync_playwright() as p:
    b = p.chromium.launch(executable_path="/opt/pw-browsers/chromium-1194/chrome-linux/chrome")
    pg = b.new_page(viewport={"width": 1000, "height": 800}, device_scale_factor=2)
    pg.set_content(html)
    pg.wait_for_timeout(1500)
    for sec in ("A", "B"):
        pg.locator(f"#{sec}").screenshot(path=str(OUT / f"hh_tot_fig{sec}_q{q}.png"))
    b.close()
print("ok")
