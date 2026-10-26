"""(1) 통제 민감도에서 가계 소득 성분이 가장 높게 나온 사양('+ all of the above')의 경로를 기준 사양과 비교.
    '+ all'은 노년부양비 때문에 표본이 줄어(715) 같은 표본의 기준 사양도 함께 그림.
(2) 종속변수(부문 순대출)의 단위근 여부: 변화분 사양(Δn 시차)에 n_{t−1}을 더했을 때 그 계수(시점별, 국가 군집 90%).
    단위근이면 0, 평균회귀면 음수. 좌변 n_{t+h} − n_{t−1}은 Δn_{t+j} LP 계수의 누적합과 정확히 같음도 확인.
→ output/ep_source_allctrl.csv, output/ep_source_ecm.csv, output/paper/fig_source_allctrl.(png|pdf), fig_source_ecm.(png|pdf)
"""
import sys

import numpy as np
import pandas as pd

import ep_source_controls as ec
import ep_source_fig as sf
import lp_tools as lt
from make_report import OUT

SRC, OWN, H = ec.SRC, ec.OWN, ec.H


def setup():
    ur = pd.read_csv(OUT / "ep_source_controls_unitroot.csv")
    tr = dict(zip(ur["var"], ur["transform"]))
    q = ec.data(tr)
    S = {name: fn for name, fn, _ in ec.specs(tr)}
    allc = S["+ all of the above"]
    small = q.dropna(subset=sum((allc(y)[1] for y in OWN), []))
    return q, small, S


def weights(q, y, X, C):
    M = np.hstack([q[X + C].to_numpy(float), lt._fe_matrix(q, ("country", "year"))])
    e = q[f"{y}_h0"].to_numpy(float) - M @ np.linalg.lstsq(M, q[f"{y}_h0"].to_numpy(float), rcond=None)[0]
    return 1 / q.country.map(pd.Series(e, index=q.index).groupby(q.country).var())


CASES = [("Baseline (N = {n})", "full", "Baseline (own S–I in levels, 2 lags)"),
         ("Baseline on the smaller sample (N = {n})", "small", "Baseline (own S–I in levels, 2 lags)"),
         ("+ all added controls (N = {n})", "small", "+ all of the above")]


def paths(qs, S):
    out = []
    for _, smp, spec in CASES:
        q = qs[smp]
        for y in OWN:
            X, C = S[spec](y)
            B = np.array([lt.ols(q, f"{y}_h{h}", X, C, se=False, w=f"w_{smp}_{spec[:4]}_{y}")[0] for h in range(H)])
            out += [B[:, k] / B[0, k] for k, x in enumerate(X)]
    return np.concatenate(out)


def run_paths(reps=499):
    q, small, S = setup()
    qs = {"full": q.copy(), "small": small.copy()}
    for _, smp, spec in CASES:
        for y in OWN:
            X, C = S[spec](y)
            for k in qs:
                pass
            qs[smp][f"w_{smp}_{spec[:4]}_{y}"] = weights(qs[smp], y, X, C)
    for col in [c for c in qs["full"].columns if c.startswith("w_full")]:
        pass
    # 부트스트랩은 표본 둘을 따로 뽑지 않도록 작은 표본 행에 표시만 하고 큰 표본에서 함께 재추출
    big = qs["full"].join(qs["small"][[c for c in qs["small"].columns if c.startswith("w_small")]])
    split = lambda d: {"full": d, "small": d.dropna(subset=[c for c in d.columns if c.startswith("w_small")])}
    pt = paths(split(big), S)
    bs = lt.cluster_bootstrap(big, lambda d: paths(split(d), S), reps=reps)
    keys = [(lab.format(n=len(qs[smp])), x) for lab, smp, spec in CASES for y in OWN for x in S[spec](y)[0]]
    r = pd.DataFrame([dict(case=c, source=x, h=h) for c, x in keys for h in range(H)]).assign(
        ratio=pt, lo68=np.nanpercentile(bs, 16, axis=0), hi68=np.nanpercentile(bs, 84, axis=0))
    r.to_csv(OUT / "ep_source_allctrl.csv", index=False)
    return r


def ecm():
    q, _, S = setup()
    rows = []
    for y in OWN:
        X = OWN[y]
        C = [f"L{l}_{v}" for v in SRC + ["gdp_g", f"d_{y}"] for l in (1, 2)] + [f"L1_{y}"]
        q = q.assign(w=weights(q, y, X, C))
        cum = np.zeros(len(X))
        for h in range(H):
            b, V, n = lt.ols(q, f"{y}_h{h}", X + [f"L1_{y}"], [c for c in C if c != f"L1_{y}"], w="w")
            rows.append(dict(sector=y, h=h, b=b[-1], se=np.sqrt(V[-1, -1]), n=n))
            # 누적 확인: Δn_{t+h} = (n_{t+h} − n_{t−1}) − (n_{t+h−1} − n_{t−1})
            dy = q[f"{y}_h{h}"] - (q[f"{y}_h{h - 1}"] if h else 0)
            bd = lt.ols(q.assign(_d=dy), "_d", X + [f"L1_{y}"], [c for c in C if c != f"L1_{y}"], se=False, w="w")[0]
            cum += bd[:len(X)]
            rows[-1].update({f"long_{x}": b[k] for k, x in enumerate(X)}, **{f"cum_{x}": cum[k] for k, x in enumerate(X)})
    r = pd.DataFrame(rows)
    r.to_csv(OUT / "ep_source_ecm.csv", index=False)
    return r


def fig_paths(r):
    import fig_paper as fp
    plt = fp.plt
    INC, BEH = "#c98a1b", "#2b6c8f"
    col = {"x_P": INC, "x_Y": INC, "x_R": BEH, "x_H": BEH}
    cases = list(dict.fromkeys(r.case))
    sty = {cases[0]: ("-", 1.8, True), cases[1]: (":", 1.3, False), cases[2]: ("--", 1.8, True)}
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.6), sharex=True)
    for ax, x in zip(axes.flat, ["x_P", "x_Y", "x_R", "x_H"]):
        for c in cases:
            ls, lw, band = sty[c]
            q = r[(r.case == c) & (r.source == x)].sort_values("h")
            if band:
                ax.fill_between(q.h, q.lo68, q.hi68, color=col[x] if c == cases[0] else "0.5", alpha=0.15, lw=0)
            ax.plot(q.h, q.ratio, color=col[x] if c != cases[2] else "0.25", ls=ls, lw=lw, marker="o" if c == cases[0] else None, ms=3, label=c)
        fp.finish(ax, "Own-sector S–I\n(impact = 1)" if x in ("x_P", "x_R") else "", h0=0)
        ax.set_xlabel("Years after the saving change ($h$)" if x in ("x_R", "x_H") else "")
        ax.set_title(sf.LAB[x], loc="left", fontsize=9.5)
        ax.set_ylim(-0.6, 1.9)
    h_, l_ = axes[0, 1].get_legend_handles_labels()
    fig.legend(h_, l_, loc="lower center", bbox_to_anchor=(0.5, -0.04), ncol=3, fontsize=7.6, frameon=False)
    fig.suptitle("The specification with all added controls, and the baseline on the same sample", x=0.0, ha="left", fontweight="bold", y=1.0)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.text(0.0, -0.05, "Notes: Baseline: own-sector balance in levels, the four components and GDP growth (two lags each), the own sector's two components at t, "
             "country and year fixed effects, two-step weighted least squares. All added controls: two lags each of ΔCA, the change in the other sector's balance, "
             "Δoperating surplus, Δhousehold saving rate, Δdisposable income and Δold-age dependency; the old-age dependency ratio is missing for some years, which "
             "shrinks the sample. Shaded areas: 68 percent country-cluster bootstrap bands (499 replications).", fontsize=7, wrap=True, va="top")
    fp.save(fig, "fig_source_allctrl", sfx="")


def fig_ecm(e):
    import fig_paper as fp
    plt = fp.plt
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    for y, cl, lab in (("nl_corp", fp.CO, "Corporations"), ("nl_hh", fp.HH, "Households")):
        q = e[e.sector == y]
        ax.fill_between(q.h, q.b - 1.645 * q.se, q.b + 1.645 * q.se, color=cl, alpha=0.15, lw=0)
        ax.plot(q.h, q.b, color=cl, lw=1.8, marker="o", ms=3.5, label=lab)
    fp.finish(ax, "Coefficient on $n_{t-1}$", h0=0)
    ax.set_xlabel("Horizon ($h$)")
    ax.legend(loc="lower left", fontsize=8, frameon=False)
    ax.set_title("Mean reversion of sector balances", loc="left", fontsize=9.5)
    fig.tight_layout()
    fig.text(0.0, -0.06, "Notes: Coefficient on the lagged level of the own-sector balance in the specification with lagged changes "
             "($n_{t+h}-n_{t-1}$ on the own sector's components, two lags of the four components, GDP growth and $\\Delta n$, country and year fixed effects). "
             "Zero at every horizon under a unit root. Shaded areas: 90 percent country-clustered intervals.", fontsize=6.8, wrap=True, va="top")
    fp.save(fig, "fig_source_ecm", sfx="")


if __name__ == "__main__":
    render = "--render" in sys.argv
    r = pd.read_csv(OUT / "ep_source_allctrl.csv") if render else run_paths()
    fig_paths(r)
    e = pd.read_csv(OUT / "ep_source_ecm.csv") if render else ecm()
    fig_ecm(e)
    a = r.assign(per=np.where(r.h.between(1, 2), "1-2", np.where(r.h >= 3, "3-6", "0")))
    print(a[a.per != "0"].pivot_table(index=["case", "source"], columns="per", values="ratio", sort=False).round(2).to_string())
    print(e.round(3).to_string(index=False))
