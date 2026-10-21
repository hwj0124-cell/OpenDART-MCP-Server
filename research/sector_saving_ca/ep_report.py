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
HHV = ["sv_hh", "nl_hh", "yd_hh", "cons_hh", "invtot_hh", "inv_hh"]
COV = ["sv_corp", "nl_corp", "prof", "div_paid", "div_net", "invtot_corp", "inv_corp"]
KEYS = [("hh", v, h) for v in HHV for h in range(K + 1)] + [("corp", v, h) for v in COV for h in range(K + 1)]
DIFFS = [("sv", "sv_hh", "sv_corp"), ("nl", "nl_hh", "nl_corp")]
XL = "h년 후 (t−1 = 0, t = 시작)"
WSV = False  # True: 가중 추정(--wsv: 국가별 자기 부문 Δ저축 분산의 역수, --fgls: 1단계 잔차 분산의 역수)
FGLS = False
NORM = "sv"  # 정규화·가중치 기준 변수: "sv"(저축) 또는 "nl"(저축−투자 차, --nl)
HMAX = 6     # 그림에 보이는 마지막 시점(--h5 → 5)
LAGS = False  # True(--lags): 자기 부문 S−I 변화의 사건 전 2개 시차(Δy_{t−1}, Δy_{t−2})를 모든 식에 통제


def ctrl(s):
    return [f"dl1_{s}", f"dl2_{s}"] if LAGS else []
TAG = ""


def stats(d):
    g = {k: lt.ols(d, f"{k[1]}_h{k[2]}", [f"z_{k[0]}"], ctrl(k[0]), se=False, w=f"w_{k[0]}" if WSV else None)[0][0] for k in KEYS}
    r = {k: g[k] / g[(k[0], f"{NORM}_{k[0]}", 0)] for k in KEYS}
    return np.array([r[k] for k in KEYS] + [r[("hh", a, h)] - r[("corp", b, h)] for _, a, b in DIFFS for h in range(K + 1)])


def prepare():
    x = ed.er.panel()
    x["div_net"] = x.div_paid - x.div_recv_corp  # 순배당: 기업 간 배당(지주회사 등)을 상쇄
    fc.VARS["hh"], fc.VARS["corp"] = HHV, COV
    P = ed.paths(x, {**el.SPEC, **CH})
    d = fc.design(x, P)
    for s in ("hh", "corp"):
        gd = d.groupby("country")[f"nl_{s}"].diff().groupby(d.country)
        d[f"dl1_{s}"], d[f"dl2_{s}"] = gd.shift(1), gd.shift(2)
        if FGLS:  # 2단계 가중최소제곱: 시작 연도 자기 부문 저축 식(무가중)의 잔차 → 국가별 분산의 역수, 부문 내 모든 식에 공통
            y0 = f"{NORM}_{s}_h0"
            q = d[[y0, f"z_{s}", *ctrl(s), "country", "year"]].dropna()
            X = np.hstack([q[[f"z_{s}", *ctrl(s)]].to_numpy(float), lt._fe_matrix(q, ("country", "year"))])
            b = np.linalg.lstsq(X, q[y0].to_numpy(float), rcond=None)[0]
            e = pd.Series(q[y0].to_numpy(float) - X @ b, index=q.index)
            d[f"w_{s}"] = 1 / d.country.map(e.groupby(q.country).var())
        else:
            d[f"w_{s}"] = 1 / d.groupby("country")[f"sv_{s}"].transform(lambda v: v.diff().var())
    return P, d


def estimate(reps=499):
    P, d = prepare()
    pt = stats(d)
    bs = lt.cluster_bootstrap(d, stats, reps=reps) if reps else np.tile(pt, (2, 1))
    lab = [dict(kind="resp", sector=s, var=v, h=h) for s, v, h in KEYS] + \
          [dict(kind="diff", sector="diff", var=n, h=h) for n, _, _ in DIFFS for h in range(K + 1)]
    r = pd.DataFrame(lab).assign(
        ratio=pt, lo68=np.nanpercentile(bs, 16, axis=0), hi68=np.nanpercentile(bs, 84, axis=0),
        lo90=np.nanpercentile(bs, 5, axis=0), hi90=np.nanpercentile(bs, 95, axis=0),
        p=np.minimum(1, 2 * np.minimum((bs <= 0).mean(0), (bs >= 0).mean(0))), n_hh=len(P["hh"]), n_corp=len(P["corp"]))
    r.to_csv(OUT / f"ep_report{TAG}.csv", index=False)
    return r


def tight(*arrs, step=0.25):
    a = np.concatenate([np.asarray(v, float).ravel() for v in arrs])
    a = a[~np.isnan(a)]
    lo, hi = min(a.min(), 0), max(a.max(), 0)
    return np.floor(lo / step) * step, np.ceil(hi / step) * step


def render(r):
    n_hh, n_corp = int(r.n_hh.iloc[0]), int(r.n_corp.iloc[0])

    def ser(s, v, cls, band=True):
        q = r[(r.sector == s) & (r["var"] == v) & (r.h <= HMAX)].sort_values("h")
        return (cls, "", q.h.values, q.ratio.values, q.lo68.values if band else None, q.hi68.values if band else None)

    def fig(title, series, legend, w=430, h=290):
        ys = [np.r_[s[3], s[4] if s[4] is not None else [], s[5] if s[5] is not None else []] for s in series]
        leg = "".join(f'<span><i class="sw {c}"></i>{t}</span>' for c, t in legend)
        return (f'<figure><div class="panel-title">{title}</div>{svg_lines(series, XL, title, w=w, h=h, ylim=tight(*ys))}'
                f'<div class="legend">{leg}</div></figure>')

    H, C = f"가계(소비행태 변화, {n_hh}건)", f"기업(교역조건, {n_corp}건)"
    SI = "저축−투자 차(S−I)"
    p1 = [fig(f"(a) {SI}", [ser("hh", "nl_hh", "s1"), ser("corp", "nl_corp", "s2")], [("s1", H), ("s2", C)]),
          fig("(b) 저축", [ser("hh", "sv_hh", "s1"), ser("corp", "sv_corp", "s2")], [("s1", H), ("s2", C)])]
    if NORM == "sv":
        p1 = p1[::-1]
        p1 = [p1[0].replace("(b) 저축", "(a) 저축"), p1[1].replace(f"(a) {SI}", f"(b) {SI}")]
    f1 = p1
    f2 = [fig("(a) 기업: 영업잉여·순배당", [ser("corp", "prof", "s1"), ser("corp", "div_net", "s2")],
              [("s1", "영업잉여"), ("s2", "순배당 지급(지급 − 수취)")]),
          fig("(b) 기업: 투자", [ser("corp", "invtot_corp", "s1"), ser("corp", "inv_corp", "s2")],
              [("s1", "투자 등(재고·토지 포함)"), ("s2", "고정투자")]),
          fig("(c) 가계: 가처분소득·소비", [ser("hh", "yd_hh", "s1"), ser("hh", "cons_hh", "s2")],
              [("s1", "가처분소득"), ("s2", "소비")]),
          fig("(d) 가계: 투자", [ser("hh", "invtot_hh", "s1"), ser("hh", "inv_hh", "s2")],
              [("s1", "투자 등(재고·토지 포함)"), ("s2", "고정투자")])]
    style = (HERE / "report_template.html").read_text()
    style = style[:style.index("<main>")] + """<style>
svg .line.s1{stroke:#2459a8;fill:none;stroke-width:2.2} svg .band.s1{fill:#2459a8;opacity:.14} svg .dot.s1{fill:#2459a8}
svg .line.s2{stroke:#d1701f;fill:none;stroke-width:2.2;stroke-dasharray:6 4} svg .band.s2{fill:#d1701f;opacity:.14} svg .dot.s2{fill:#d1701f}
svg .line.s3{stroke:#3b8a4f;fill:none;stroke-width:2.2;stroke-dasharray:2 3} svg .dot.s3{fill:#3b8a4f}
.sw.s1{border-color:#2459a8} .sw.s2{border-color:#d1701f;border-top-style:dashed} .sw.s3{border-color:#3b8a4f;border-top-style:dotted}</style>"""
    base_lab = "저축−투자 차(S−I, 국민계정상 순대출)" if NORM == "nl" else "저축"
    if LAGS:
        base_lab += ", 사건 전 자기 부문 S−I 변화 2개 시차 통제"
    note = (f"주: 시작 연도 자기 부문 {base_lab} 반응을 1로 정규화한 누적 반응(GDP 대비). 가계 = 가계저축률 변화가 국가별 상·하위 10%이고 소비 비중 변화가 "
            "가처분소득 비중 변화보다 큰 해, 기업 = 교역조건(수출·수입 디플레이터 비율) 변화가 국가별 상·하위 10%이고 영업잉여가 같은 방향인 해. "
            "시작 연도 자기 부문 자금잉여가 충격과 같은 방향인 사례, 증가·감소는 부호를 맞춰 통합. 국가·연도 고정효과 국소투영")
    if FGLS:
        note += f"(2단계 가중최소제곱: 1단계 {'S−I' if NORM == 'nl' else '저축'} 식 잔차의 국가별 분산 역수로 부문별 공통 가중)"
    elif WSV:
        note += "(국가별 자기 부문 저축 변화 분산의 역수로 가중)"
    n1 = note + (", 음영은 68% 구간(국가 단위 부트스트랩 499회)." if r.lo68.ne(r.ratio).any() else ", 점추정치(구간 미산출).")
    n2 = n1
    html = (f'<!doctype html><html><head><meta charset="utf-8">{style}</head><body><main style="max-width:900px">'
            f'<section id="F1"><h2 style="font-size:18px">그림 1. 원천별 저축−투자 차와 저축의 지속성</h2>'
            f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:10px">{"".join(f1)}</div><p class="ci" style="font-size:12px">{n1}</p></section>'
            f'<section id="F2"><h2 style="font-size:18px">그림 2. 기업 흑자의 해소 경로(배당·투자)와 가계 소비행태</h2>'
            f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:10px">{"".join(f2)}</div><p class="ci" style="font-size:12px">{n2}</p></section>'
            f'</main></body></html>')
    (OUT / f"ep_report{TAG}_figs.html").write_text(html)
    with sync_playwright() as p:
        b = p.chromium.launch(executable_path="/opt/pw-browsers/chromium-1194/chrome-linux/chrome")
        pg = b.new_page(viewport={"width": 900, "height": 800}, device_scale_factor=2)
        pg.set_content(html)
        pg.wait_for_timeout(1500)
        pg.locator("#F1").screenshot(path=str(OUT / f"ep_report{TAG}_fig1.png"))
        pg.locator("#F2").screenshot(path=str(OUT / f"ep_report{TAG}_fig2.png"))
        b.close()


if __name__ == "__main__":
    import sys
    if "--wsv" in sys.argv:
        WSV, TAG = True, "_wsv"
    if "--fgls" in sys.argv:
        WSV, FGLS, TAG = True, True, "_fgls"
    if "--nl" in sys.argv:
        NORM, TAG = "nl", TAG + "_nl"
    if "--h5" in sys.argv:
        HMAX = 5
    if "--lags" in sys.argv:
        LAGS, TAG = True, TAG + "_lags"
    r = pd.read_csv(OUT / f"ep_report{TAG}.csv") if "--render" in sys.argv else \
        estimate(reps=0 if "--point" in sys.argv else 499)
    render(r)
    q = r[r.kind == "diff"]
    print(q.pivot_table(index="var", columns="h", values=["ratio", "p"]).round(2).to_string())
