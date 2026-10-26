"""ep_robust 결과 → 한글 PNG (선마다 별도 패널, 68% 구간)."""
import pandas as pd
from playwright.sync_api import sync_playwright

from make_report import HERE, OUT, svg_lines

r = pd.read_csv(OUT / "ep_robust.csv")
BASE = "기본 (1년·전체국가·연도평균 미제거·10%)"
TYPES = [("가계 성향", "hh", "가계"), ("교역조건", "gov", "비금융법인")]
KINDS = ["순자금운용", "저축", "총투자", "고정투자"]
KLAB = {"총투자": "투자 등 (저축 − 순자금운용)", "고정투자": "고정투자"}


def ser(spec, t, kind, cls):
    x = r[(r.spec == spec) & (r.type == t) & (r.kind == kind)].sort_values("k")
    return [(cls, "", x.k.values, x.ratio.values, x.lo68.values, x.hi68.values)]


def panel(title, series, xl="t (t−1 = 0, t = 시작)"):
    return f'<figure><div class="panel-title">{title}</div>{svg_lines(series, xl, title, w=420, h=230)}</figure>'


def grid(cells, cols=2):
    return f'<div style="display:grid;grid-template-columns:repeat({cols},1fr);gap:10px">{"".join(cells)}</div>'


n = r[(r.k == 0) & (r.kind == "순자금운용")].set_index(["spec", "type"]).n_ep
figA = grid([panel(f"{t} → {sec} {KLAB.get(k, k)}", ser(BASE, t, k, c)) for k in KINDS for t, c, sec in TYPES])
figB = grid([panel(f"{t}: {spec} (n {int(n[(spec, t)])})", ser(spec, t, "순자금운용", c))
             for spec in r.spec.unique() for t, c, _ in TYPES])
style = (HERE / "report_template.html").read_text()
style = style[:style.index("<main>")]
note = ("41개국 OECD 부문계정. 순자금운용 = 저축 − 투자 등. 투자 등 = 총자본형성(고정투자+재고투자) + 비생산자산 순취득 − 순자본이전 수취. "
        "값 = 사례 평균[부호 맞춘 v<sub>t+k</sub> − v<sub>t−1</sub>] / 사례 평균|시작 시점 순자금운용 변화| (시작 = 1). "
        "음영 68% 구간(사례 부트스트랩).")
html = f"""<!doctype html><html><head><meta charset="utf-8">{style}</head><body><main style="max-width:900px">
<section id="A"><h2 style="font-size:18px">기본 사양: 순자금운용·저축·투자 (1년 변화 상·하위 10%, 증가·감소 통합, 전체 국가, 연도평균 미제거)</h2>{figA}
<p class="ci" style="font-size:12px">{note} 가계 성향: 가계저축률 1년 변화 상·하위 10%, 그해 성장률 ≥ 0, 2020–21 제외. 교역조건: 교역조건 1년 변화 상·하위 10%, 영업잉여 비중 같은 방향. 전체 국가, 연도평균 미제거.</p></section>
<section id="B"><h2 style="font-size:18px">사양 민감도: 자기 부문 순자금운용</h2>{figB}
<p class="ci" style="font-size:12px">연도평균 제거: 매년 41개국 평균을 빼 글로벌 공통요인 제거. IRL·LUX 제외: 교역조건 사례에서 두 나라 제외. 2년 정의: 2년 변화 기준(t−2 = 0). 코로나 제외: 시작 2019–21 제외. 상·하위 20%: 기준 완화. {note}</p></section>
</main></body></html>"""
(OUT / "ep_robust_figs.html").write_text(html)
with sync_playwright() as p:
    b = p.chromium.launch(executable_path="/opt/pw-browsers/chromium-1194/chrome-linux/chrome")
    pg = b.new_page(viewport={"width": 900, "height": 800}, device_scale_factor=2)
    pg.set_content(html)
    pg.wait_for_timeout(1500)
    for sec in ("A", "B"):
        pg.locator(f"#{sec}").screenshot(path=str(OUT / f"ep_robust_fig{sec}.png"))
    b.close()
print("ok")
