"""보고서용 최종 사양: 신뢰구간·차이 검정·그림 2장.

사양: 새 원자료 전체 기간(불균형 패널), GDP 대비, 이벤트 더미 국소투영(국가·연도FE, 가중치 없음).
  가계 = 소비 주도 가계저축률 급변(국가별 상·하위 10%), 기업 = 상품·서비스 교역조건 급변(국가별 상·하위 10%, 영업잉여 같은 방향)
  시작 연도 자기 부문 자금잉여(순대출)가 충격과 같은 방향인 사례, 4년 내 중복·시작 2020–22·침체 연도 제외
  정규화 = γ_h ÷ γ_0(자기 부문 저축)  → 시작 연도 저축 반응 = 1
구간: 국가 단위 부트스트랩 499회(비율·차이를 각 표본에서 다시 계산).
→ output/ep_report.csv, output/ep_report_fig1.png, output/ep_report_fig2.png
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
CH = dict(tot=fc.TOT, hh_def="cons", base="nl")
HHV = ["sv_hh", "nl_hh", "yd_hh", "cons_hh"]
COV = ["sv_corp", "nl_corp", "prof", "div_paid", "inv_corp"]
KEYS = [("hh", v, h) for v in HHV for h in range(K + 1)] + [("corp", v, h) for v in COV for h in range(K + 1)]
DIFFS = [("sv", "sv_hh", "sv_corp"), ("nl", "nl_hh", "nl_corp")]
XL = "h년 후 (t−1 = 0, t = 시작)"


def stats(d):
    g = {k: lt.ols(d, f"{k[1]}_h{k[2]}", [f"z_{k[0]}"], [], se=False)[0][0] for k in KEYS}
    r = {k: g[k] / g[(k[0], f"sv_{k[0]}", 0)] for k in KEYS}
    return np.array([r[k] for k in KEYS] + [r[("hh", a, h)] - r[("corp", b, h)] for _, a, b in DIFFS for h in range(K + 1)])


def estimate():
    x = ed.er.panel()
    fc.VARS["hh"], fc.VARS["corp"] = HHV, COV
    P = ed.paths(x, {**el.SPEC, **CH})
    d = fc.design(x, P)
    pt = stats(d)
    bs = lt.cluster_bootstrap(d, stats, reps=499)
    lab = [dict(kind="resp", sector=s, var=v, h=h) for s, v, h in KEYS] + \
          [dict(kind="diff", sector="diff", var=n, h=h) for n, _, _ in DIFFS for h in range(K + 1)]
    r = pd.DataFrame(lab).assign(
        ratio=pt, lo68=np.nanpercentile(bs, 16, axis=0), hi68=np.nanpercentile(bs, 84, axis=0),
        lo90=np.nanpercentile(bs, 5, axis=0), hi90=np.nanpercentile(bs, 95, axis=0),
        p=np.minimum(1, 2 * np.minimum((bs <= 0).mean(0), (bs >= 0).mean(0))), n_hh=len(P["hh"]), n_corp=len(P["corp"]))
    r.to_csv(OUT / "ep_report.csv", index=False)
    return r


def tight(*arrs, step=0.25):
    a = np.concatenate([np.asarray(v, float).ravel() for v in arrs])
    a = a[~np.isnan(a)]
    lo, hi = min(a.min(), 0), max(a.max(), 0)
    return np.floor(lo / step) * step, np.ceil(hi / step) * step


def render(r):
    n_hh, n_corp = int(r.n_hh.iloc[0]), int(r.n_corp.iloc[0])

    def ser(s, v, cls, band=True):
        q = r[(r.sector == s) & (r["var"] == v)].sort_values("h")
        return (cls, "", q.h.values, q.ratio.values, q.lo68.values if band else None, q.hi68.values if band else None)

    def fig(title, series, legend, w=430, h=290):
        ys = [np.r_[s[3], s[4] if s[4] is not None else [], s[5] if s[5] is not None else []] for s in series]
        leg = "".join(f'<span><i class="sw {c}"></i>{t}</span>' for c, t in legend)
        return (f'<figure><div class="panel-title">{title}</div>{svg_lines(series, XL, title, w=w, h=h, ylim=tight(*ys))}'
                f'<div class="legend">{leg}</div></figure>')

    H, C = f"가계(소비행태 변화, {n_hh}건)", f"기업(교역조건, {n_corp}건)"
    f1 = [fig("(a) 저축", [ser("hh", "sv_hh", "s1"), ser("corp", "sv_corp", "s2")], [("s1", H), ("s2", C)]),
          fig("(b) 자금잉여(저축 − 투자)", [ser("hh", "nl_hh", "s1"), ser("corp", "nl_corp", "s2")], [("s1", H), ("s2", C)])]
    f2 = [fig("(a) 기업: 영업잉여·배당·고정투자", [ser("corp", "prof", "s1", False), ser("corp", "div_paid", "s2", False),
                                            ser("corp", "inv_corp", "s3", False)],
              [("s1", "영업잉여"), ("s2", "배당 지급"), ("s3", "고정투자")]),
          fig("(b) 가계: 가처분소득·소비", [ser("hh", "yd_hh", "s1", False), ser("hh", "cons_hh", "s2", False)],
              [("s1", "가처분소득"), ("s2", "소비")])]
    style = (HERE / "report_template.html").read_text()
    style = style[:style.index("<main>")] + """<style>
svg .line.s1{stroke:#2459a8;fill:none;stroke-width:2.2} svg .band.s1{fill:#2459a8;opacity:.14} svg .dot.s1{fill:#2459a8}
svg .line.s2{stroke:#d1701f;fill:none;stroke-width:2.2;stroke-dasharray:6 4} svg .band.s2{fill:#d1701f;opacity:.14} svg .dot.s2{fill:#d1701f}
svg .line.s3{stroke:#3b8a4f;fill:none;stroke-width:2.2;stroke-dasharray:2 3} svg .dot.s3{fill:#3b8a4f}
.sw.s1{border-color:#2459a8} .sw.s2{border-color:#d1701f;border-top-style:dashed} .sw.s3{border-color:#3b8a4f;border-top-style:dotted}</style>"""
    note = ("주: 시작 연도 자기 부문 저축 반응을 1로 정규화한 누적 반응(GDP 대비). 가계 = 가계저축률 변화가 국가별 상·하위 10%이고 소비 비중 변화가 "
            "가처분소득 비중 변화보다 큰 해, 기업 = 교역조건(수출·수입 디플레이터 비율) 변화가 국가별 상·하위 10%이고 영업잉여가 같은 방향인 해. "
            "시작 연도 자기 부문 자금잉여가 충격과 같은 방향인 사례, 증가·감소는 부호를 맞춰 통합. 국가·연도 고정효과 국소투영")
    n1 = note + ", 음영은 68% 구간(국가 단위 부트스트랩 499회)."
    n2 = note + ", 점추정치."
    html = (f'<!doctype html><html><head><meta charset="utf-8">{style}</head><body><main style="max-width:900px">'
            f'<section id="F1"><h2 style="font-size:18px">그림 1. 원천별 저축·자금잉여의 지속성</h2>'
            f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:10px">{"".join(f1)}</div><p class="ci" style="font-size:12px">{n1}</p></section>'
            f'<section id="F2"><h2 style="font-size:18px">그림 2. 기업 흑자 소멸 경로와 가계 소비행태</h2>'
            f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:10px">{"".join(f2)}</div><p class="ci" style="font-size:12px">{n2}</p></section>'
            f'</main></body></html>')
    (OUT / "ep_report_figs.html").write_text(html)
    with sync_playwright() as p:
        b = p.chromium.launch(executable_path="/opt/pw-browsers/chromium-1194/chrome-linux/chrome")
        pg = b.new_page(viewport={"width": 900, "height": 800}, device_scale_factor=2)
        pg.set_content(html)
        pg.wait_for_timeout(1500)
        pg.locator("#F1").screenshot(path=str(OUT / "ep_report_fig1.png"))
        pg.locator("#F2").screenshot(path=str(OUT / "ep_report_fig2.png"))
        b.close()


if __name__ == "__main__":
    import sys
    r = pd.read_csv(OUT / "ep_report.csv") if "--render" in sys.argv else estimate()
    render(r)
    q = r[r.kind == "diff"]
    print(q.pivot_table(index="var", columns="h", values=["ratio", "p"]).round(2).to_string())
