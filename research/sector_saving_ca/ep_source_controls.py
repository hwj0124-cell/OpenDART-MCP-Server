"""원천 분해 LP(수준 사양)의 통제변수 민감도. 원칙: 정상 변수는 수준, 비정상 변수는 차분으로 넣는다.

(1) 정상성 판정: 연도 평균을 뺀 뒤 국가별 ADF(상수, 시차 0–2 AIC). 국가 과반이 10%에서 단위근을 기각하면 수준, 아니면 차분.
    Maddala–Wu 결합 검정(일부 국가만 정상이어도 기각)과 국가 FE 통합 AR(1) 계수도 참고로 보고.
    자기 S−I는 판정과 무관하게 수준(LP 시차 보강: 단위근이 있어도 추론이 견고).
(2) 기준 사양: 자기 S−I 수준 2시차, 네 성분·성장률 2시차, 같은 해 자기 부문 두 성분만, 국가·연도 FE, 2단계 WLS.
    통제를 하나씩 더하고(판정된 변환으로 2시차), 시차 수를 바꾸고, 참고 사양 두 개(다른 부문 같은 해 성분 포함, 자기 S−I 변화분)를 비교.
    표본은 4시차까지 관측되는 국가·연도로 고정. 교역조건·실질금리는 관측 국가가 적어 같은 부분표본의 기준 사양과 짝지어 비교.
→ output/ep_source_controls_unitroot.csv, output/ep_source_controls.csv, output/paper/fig_source_controls.(png|pdf)
"""
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats as st
from statsmodels.tsa.stattools import adfuller

import ep_source_grid as gr
import lp_tools as lt
from make_report import OUT

warnings.filterwarnings("ignore")
SRC, OWN, H = gr.SRC, gr.OWN, gr.H
CAND = ["nl_corp", "nl_hh", "ca", "prof", "sv_corp", "sv_hh", "hh_rate", "yd_hh", "real_interest_rate", "old_age_dependency",
        "gdp_g", "tot_goods_g"]


def unit_root(p):
    rows = []
    for v in CAND:
        s = p[["country", "year", v]].dropna()
        s[v] = s[v] - s.groupby("year")[v].transform("mean")
        pv = []
        for _, g in s.sort_values("year").groupby("country"):
            if len(g) >= 15:
                pv.append(adfuller(g[v].to_numpy(), maxlag=2, autolag="AIC", regression="c")[1])
        pv = np.clip(pv, 1e-10, 1)
        chi = -2 * np.log(pv).sum()
        x = s.assign(l1=s.groupby("country")[v].shift(1)).dropna()
        xd, yd = x.l1 - x.groupby("country").l1.transform("mean"), x[v] - x.groupby("country")[v].transform("mean")
        rows.append(dict(var=v, countries=len(pv), fisher_p=st.chi2.sf(chi, 2 * len(pv)), share_reject10=np.mean(np.array(pv) < 0.1),
                         ar1_fe=(xd * yd).sum() / (xd ** 2).sum()))
    r = pd.DataFrame(rows)
    # Fisher 결합 검정은 '일부 국가라도 정상'이면 기각되므로, 개별 국가 과반이 10%에서 기각할 때만 수준으로 둔다
    r["transform"] = np.where(r.share_reject10 >= 0.5, "level", "diff")
    return r


def data(tr):
    d, _ = gr.data()
    g = d.groupby("country")
    for v in CAND:
        base = v if tr[v] == "level" else f"D_{v}"
        if tr[v] == "diff":
            d[base] = g[v].diff()
        for l in (1, 2):
            d[f"L{l}c_{v}"] = d.groupby("country")[base].shift(l)
    _, q = gr.data()
    return q.join(d[[c for c in d.columns if c.startswith("L") and "c_" in c]])


def persistence(q, y, X, C):
    q = q.dropna(subset=X + C)
    M = np.hstack([q[X + C].to_numpy(float), lt._fe_matrix(q, ("country", "year"))])
    e = q[f"{y}_h0"].to_numpy(float) - M @ np.linalg.lstsq(M, q[f"{y}_h0"].to_numpy(float), rcond=None)[0]
    q = q.assign(w=1 / q.country.map(pd.Series(e, index=q.index).groupby(q.country).var()))
    B = np.array([lt.ols(q, f"{y}_h{h}", X, C, se=False, w="w")[0] for h in range(H)])
    return {x: (B[1:3, k].mean() / B[0, k], B[3:, k].mean() / B[0, k]) for k, x in enumerate(X) if x in OWN[y]}, len(q), q.country.nunique()


def specs(tr):
    lag = lambda vs, p=2: [f"L{l}_{v}" for v in vs for l in range(1, p + 1)]
    cl = lambda vs: [f"L{l}c_{v}" for v in vs for l in (1, 2)]
    t = lambda v: ("" if tr[v] == "level" else "Δ") + {"ca": "CA", "prof": "operating surplus", "hh_rate": "saving rate", "yd_hh": "disposable income",
                                                       "real_interest_rate": "real rate", "old_age_dependency": "old-age dependency",
                                                       "tot_goods_g": "ToT growth"}.get(v, v)
    base = lambda y, p=2: lag(SRC + ["gdp_g"], p) + lag([y], p)
    other = lambda y: "nl_hh" if y == "nl_corp" else "nl_corp"
    S = [("Baseline (own S–I in levels, 2 lags)", lambda y: (OWN[y], base(y)), None)]
    for p in (1, 3, 4):
        S.append((f"Baseline, {p} lag{'s' if p > 1 else ''}", lambda y, p=p: (OWN[y], base(y, p)), None))
    S += [(f"+ {t('ca')}", lambda y: (OWN[y], base(y) + cl(["ca"])), None),
          (f"+ other-sector S–I ({'level' if tr['nl_corp'] == 'level' else 'Δ'})", lambda y: (OWN[y], base(y) + cl([other(y)])), None),
          (f"+ {t('prof')}, {t('hh_rate')}, {t('yd_hh')}", lambda y: (OWN[y], base(y) + cl(["prof", "hh_rate", "yd_hh"])), None),
          (f"+ {t('old_age_dependency')}", lambda y: (OWN[y], base(y) + cl(["old_age_dependency"])), None),
          ("+ all of the above", lambda y: (OWN[y], base(y) + cl(["ca", other(y), "prof", "hh_rate", "yd_hh", "old_age_dependency"])), None),
          (f"+ {t('tot_goods_g')}", lambda y: (OWN[y], base(y) + cl(["tot_goods_g"])), "tot_goods_g"),
          (f"+ {t('real_interest_rate')}", lambda y: (OWN[y], base(y) + cl(["real_interest_rate"])), "real_interest_rate"),
          ("Ref.: state variables in levels", lambda y: (OWN[y], base(y) + [f"L1_{v}" for v in ["sv_corp", "sv_hh", "prof", "hh_rate"]]), None),
          ("Ref.: other sector's components at t", lambda y: (SRC, base(y)), None),
          ("Ref.: own S–I in changes (first draft)", lambda y: (OWN[y], lag(SRC + ["gdp_g"]) + lag([f"d_{y}"])), None)]
    return S


def run():
    p = lt.balance(pd.read_csv(OUT.parent / "data" / "panel.csv"))
    ur = unit_root(p)
    ur.to_csv(OUT / "ep_source_controls_unitroot.csv", index=False)
    print(ur.round(3).to_string(index=False))
    tr = dict(zip(ur["var"], ur["transform"]))
    q = data(tr)
    rows = []
    for name, fn, sub in specs(tr):
        for y in OWN:
            X, C = fn(y)
            qq = q.dropna(subset=[f"L1c_{sub}", f"L2c_{sub}"]) if sub else q
            if sub:
                Xb, Cb = specs(tr)[0][1](y)
                res, n, k = persistence(qq, y, Xb, Cb)
                rows += [dict(spec=f"Baseline on {name[2:]} sample", group=name, source=x, short=s, medium=m, n=n, countries=k) for x, (s, m) in res.items()]
            res, n, k = persistence(qq, y, X, C)
            rows += [dict(spec=name, group=name, source=x, short=s, medium=m, n=n, countries=k) for x, (s, m) in res.items()]
    r = pd.DataFrame(rows)
    r.to_csv(OUT / "ep_source_controls.csv", index=False)
    return r


def figure(r):
    import fig_paper as fp
    import ep_source_fig as sf
    plt = fp.plt
    col = {"x_P": "#c98a1b", "x_Y": "#c98a1b", "x_R": "#2b6c8f", "x_H": "#2b6c8f"}
    mk = {"x_P": "o", "x_R": "D", "x_Y": "o", "x_H": "D"}
    fill = {"x_P": True, "x_R": True, "x_Y": False, "x_H": False}
    specs_ = list(dict.fromkeys(r.spec))
    lab = {s: s + (f"  (N = {int(r[r.spec == s].n.min())}, {int(r[r.spec == s].countries.min())})" if r[r.spec == s].n.min() < r.n.max() else "") for s in specs_}
    y = np.arange(len(specs_))[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 0.27 * len(specs_) + 1.4), sharey=True)
    for ax, hz, title in ((axes[0], "short", "(a) Years 1–2"), (axes[1], "medium", "(b) Years 3–6")):
        base = r[r.spec == specs_[0]].set_index("source")[hz]
        for x in base.index:
            ax.axvline(base[x], color=col[x], lw=0.6, ls=":" if not fill[x] else "-", alpha=0.6)
        for j, x in enumerate(["x_P", "x_R", "x_Y", "x_H"]):
            v = r[r.source == x].set_index("spec").loc[specs_, hz].to_numpy()
            ax.scatter(v, y + (1.5 - j) * 0.16, s=20, marker=mk[x], label=sf.LAB[x], zorder=3,
                       facecolor=col[x] if fill[x] else "white", edgecolor=col[x], lw=1.1)
        ax.axvline(0, color=".3", lw=.6)
        ax.set_title(title, loc="left", fontsize=9.5)
        ax.set_xlabel("Average response (impact = 1)")
        ax.grid(axis="y", color=".92", lw=.6)
        ax.set_xlim(-0.35, 1.25 if hz == "short" else 0.75)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([lab[s] for s in specs_], fontsize=7.6)
    h_, l_ = axes[1].get_legend_handles_labels()
    fig.legend(h_, l_, loc="lower center", bbox_to_anchor=(0.55, -0.06), ncol=4, fontsize=7.5, frameon=False)
    fig.suptitle("Sensitivity of persistence to control variables", x=0.0, ha="left", fontweight="bold", y=1.0)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.text(0.0, -0.07, "Notes: Point estimates. Baseline: own-sector balance in levels, the four components and GDP growth, two lags each; the own sector's two "
             "components at t; country and year fixed effects; two-step weighted least squares with weights re-estimated for each specification. Added controls enter "
             "with two lags, in levels if country-by-country ADF tests reject a unit root at 10 percent for a majority of economies and in first differences "
             "otherwise (only GDP growth, ToT growth and the real rate pass). Common sample of "
             f"{int(r.n.max())} country-years unless noted; ToT and the real rate are compared with the baseline on the same subsample. Vertical lines mark the baseline.",
             fontsize=6.8, wrap=True, va="top")
    fp.save(fig, "fig_source_controls", sfx="")


if __name__ == "__main__":
    r = pd.read_csv(OUT / "ep_source_controls.csv") if "--render" in sys.argv else run()
    figure(r)
    t = r.pivot_table(index="spec", columns="source", values=["short", "medium"], sort=False)
    print(t.reindex(columns=pd.MultiIndex.from_product([["short", "medium"], ["x_P", "x_R", "x_Y", "x_H"]])).round(2).to_string())
