"""원천 분해의 패널 VAR 대안: (a) VAR 충격반응(반복 계산, 매끄러움), (b) VAR로 식별한 촐레스키 충격을 LP에 넣기.

VAR: Z_t = [x_P, x_R, x_H, x_Y, 성장률, 기업 S−I, 가계 S−I] (S−I는 수준), 2시차, 국가·연도 FE, 식별 OLS(무가중).
촐레스키 순서는 LP 기준 사양을 그대로 옮김: 충격 성분 앞에 같은 부문의 다른 성분(같은 해 통제),
뒤에 다른 부문 성분·성장률·S−I(같은 해 반응 허용). 예) x_P 충격: [x_R, x_P, x_H, x_Y, g, n_C, n_H].
충격반응은 자기 부문 S−I의 시작 연도 반응으로 정규화(LP와 같은 단위).
(b) 촐레스키 충격 ε_t로 n_{t+h} − n_{t−1}을 국가·연도 FE만 두고 LP(시작 연도 ≤ 2019).
VAR 추정 표본: LP 표본 38개국, 현재·2시차가 있는 모든 연도(2025년까지). 구간: 국가 군집 부트스트랩 68%(499회).
→ output/ep_source_var.csv, output/paper/fig_source_var.(png|pdf)
"""
import sys

import numpy as np
import pandas as pd

import ep_source_fig as sf
import lp_tools as lt
import source_shocks as ss
from make_report import OUT

H, P = 7, 2
Z = ["x_P", "x_R", "x_H", "x_Y", "gdp_g", "nl_corp", "nl_hh"]
ORDERING = {"x_P": ["x_R", "x_P", "x_H", "x_Y", "gdp_g", "nl_corp", "nl_hh"],
            "x_R": ["x_P", "x_R", "x_H", "x_Y", "gdp_g", "nl_corp", "nl_hh"],
            "x_H": ["x_Y", "x_H", "x_P", "x_R", "gdp_g", "nl_hh", "nl_corp"],
            "x_Y": ["x_H", "x_Y", "x_P", "x_R", "gdp_g", "nl_hh", "nl_corp"]}
OWN = {"x_P": "nl_corp", "x_R": "nl_corp", "x_H": "nl_hh", "x_Y": "nl_hh"}
SHOCKS = list(ORDERING)


def data():
    d = ss.prepare(pd.read_csv(OUT.parent / "data" / "panel.csv"))
    q, _ = sf.sample()
    d = d[d.country.isin(q.country.unique())].copy()
    g = d.groupby("country")
    lags = [f"V{l}_{v}" for l in range(1, P + 1) for v in Z]
    for l in range(1, P + 1):
        for v in Z:
            d[f"V{l}_{v}"] = g[v].shift(l)
    est = d.dropna(subset=Z + lags).copy()
    return est, lags


def var_fit(est, lags):
    X = np.hstack([est[lags].to_numpy(float), lt._fe_matrix(est, ("country", "year"))])
    Y = est[Z].to_numpy(float)
    B = np.linalg.lstsq(X, Y, rcond=None)[0]
    U = Y - X @ B
    k = len(Z)
    A = [B[l * k:(l + 1) * k].T for l in range(P)]          # A_l: k×k
    S = U.T @ U / (len(U) - X.shape[1])
    return A, S, U


def irf(A, S):
    k = len(Z)
    F = np.zeros((k * P, k * P))
    F[:k] = np.hstack(A)
    F[k:, :-k] = np.eye(k * (P - 1))
    out, eps_map = {}, {}
    for s, order in ORDERING.items():
        idx = [Z.index(v) for v in order]
        L = np.linalg.cholesky(S[np.ix_(idx, idx)])
        b = np.zeros(k)
        b[idx] = L[:, 1]                                     # 순서상 두 번째 = 관심 충격
        state, path = np.concatenate([b, np.zeros(k * (P - 1))]), []
        for h in range(H):
            path.append(state[Z.index(OWN[s])])
            state = F @ state
        out[s] = np.array(path) / path[0]
        eps_map[s] = (idx, L)
    return out, eps_map


def shock_series(U, eps_map):
    E = {}
    for s, (idx, L) in eps_map.items():
        E[s] = np.linalg.solve(L, U[:, idx].T).T[:, 1]
    return E


def stats(est, lags):
    A, S, U = var_fit(est, lags)
    v, eps_map = irf(A, S)
    E = shock_series(U, eps_map)
    e = est.assign(**{f"eps_{s}": E[s] for s in SHOCKS})
    e = e[e.year <= 2019]
    lp = {}
    for s in SHOCKS:
        y = OWN[s]
        b = np.array([lt.ols(e, f"{y}_h{h}", [f"eps_{s}"], [], se=False)[0][0] for h in range(H)])
        lp[s] = b / b[0]
    return np.concatenate([v[s] for s in SHOCKS] + [lp[s] for s in SHOCKS])


def run(reps=499):
    est, lags = data()
    pt = stats(est, lags)
    bs = lt.cluster_bootstrap(est, lambda d: stats(d, lags), reps=reps)
    keys = [(m, s) for m in ("VAR", "LP on VAR shocks") for s in SHOCKS]
    r = pd.DataFrame([dict(method=m, source=s, h=h) for m, s in keys for h in range(H)]).assign(
        ratio=pt, lo68=np.nanpercentile(bs, 16, axis=0), hi68=np.nanpercentile(bs, 84, axis=0),
        n=len(est), countries=est.country.nunique(), first=est.year.min(), last=est.year.max())
    r.to_csv(OUT / "ep_source_var.csv", index=False)
    return r


def figure(r):
    import fig_paper as fp
    plt = fp.plt
    col = {"x_P": "#c98a1b", "x_Y": "#c98a1b", "x_R": "#2b6c8f", "x_H": "#2b6c8f"}
    lp = pd.read_csv(OUT / "ep_source_robust.csv")
    lb = pd.read_csv(OUT / "ep_source_robust_bands.csv")
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.6), sharex=True)
    for ax, x in zip(axes.flat, ["x_P", "x_Y", "x_R", "x_H"]):
        b = lb[lb.source == x].sort_values("h")
        ax.fill_between(b.h, b.lo68, b.hi68, color=col[x], alpha=0.18, lw=0)
        g = lp[(lp.spec == "Baseline") & (lp.source == x)].sort_values("h")
        ax.plot(g.h, g.ratio, color=col[x], lw=2.0, marker="o", ms=3.5, label="LP baseline (weighted), 68% band")
        g = lp[(lp.spec == "Unweighted") & (lp.source == x)].sort_values("h")
        ax.plot(g.h, g.ratio, color=col[x], lw=1.1, ls=":", label="LP baseline, unweighted")
        v = r[(r.method == "VAR") & (r.source == x)].sort_values("h")
        ax.plot(v.h, v.lo68, color="0.2", lw=0.7, ls="--")
        ax.plot(v.h, v.hi68, color="0.2", lw=0.7, ls="--")
        ax.plot(v.h, v.ratio, color="0.1", lw=1.9, marker="s", ms=3, label="Panel VAR (unweighted), 68% band dashed")
        w = r[(r.method == "LP on VAR shocks") & (r.source == x)].sort_values("h")
        ax.plot(w.h, w.ratio, color="0.5", lw=1.3, ls="-.", label="LP on Cholesky shocks from the VAR")
        fp.finish(ax, "Own-sector S–I\n(impact = 1)" if x in ("x_P", "x_R") else "", h0=0)
        ax.set_xlabel("Years after the saving change ($h$)" if x in ("x_R", "x_H") else "")
        ax.set_title(sf.LAB[x], loc="left", fontsize=9.5)
        ax.set_ylim(-1.0, 2.4) if x == "x_Y" else ax.set_ylim(-1.0, 1.8)
    h_, l_ = axes[0, 0].get_legend_handles_labels()
    fig.legend(h_, l_, loc="lower center", bbox_to_anchor=(0.5, -0.05), ncol=2, fontsize=7.6, frameon=False)
    fig.suptitle("Local projections and a panel VAR", x=0.0, ha="left", fontweight="bold", y=1.0)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    n, k, a, z = (int(r[c].iloc[0]) for c in ("n", "countries", "first", "last"))
    fig.text(0.0, -0.06, f"Notes: Panel VAR in the four components, GDP growth and the two sector balances (levels), two lags, country and year fixed "
             f"effects, estimated on {n} country-years ({k} economies, {a}–{z}). Cholesky ordering mirrors the LP: the same sector's other component "
             "comes first (held fixed at t), the other sector's components, growth and the balances come after (free to respond). Responses are "
             "normalized by the impact response of the own-sector balance. 'LP on Cholesky shocks' regresses the balance on the identified shock with "
             "country and year fixed effects only. Country-cluster bootstrap (499 replications).", fontsize=6.9, wrap=True, va="top")
    fp.save(fig, "fig_source_var", sfx="")


if __name__ == "__main__":
    r = pd.read_csv(OUT / "ep_source_var.csv") if "--render" in sys.argv else run()
    figure(r)
    print(r.pivot_table(index=["method", "source"], columns="h", values="ratio", sort=False).round(2).to_string())
    print(r[r.method == "VAR"].pivot_table(index="source", columns="h", values=["lo68", "hi68"]).round(2).to_string())
