"""제안 사양의 네 성분 반응(부트스트랩 구간)과 통제변수 민감도 그림.

제안 사양: 네 성분(이익·유보율·저축률·소득)을 함께 넣고, 네 성분·성장률·자기 부문 S−I 변화의 2개 시차, 국가·연도 FE,
2단계 가중최소제곱(부문별 시작 연도 식 잔차의 국가별 분산 역수), 모든 시점 고정 표본, 국가 단위 부트스트랩 499회.
→ output/ep_source_fig.csv, output/paper/fig_source_components.(png|pdf), fig_source_sensitivity.(png|pdf)
"""
import numpy as np
import pandas as pd

import ep_source_check as c
import lp_tools as lt
from make_report import OUT

H = c.H
LAB = {"x_P": "Corporate: profit-driven", "x_R": "Corporate: retention-driven",
       "x_H": "Household: saving-rate-driven", "x_Y": "Household: income-driven"}


def sample():
    d, q = c.prepare()
    for y, s in (("nl_corp", "c"), ("nl_hh", "h")):
        dy = d.groupby("country")[y].diff()
        q[f"dl1_{s}"], q[f"dl2_{s}"] = dy.groupby(d.country).shift(1), dy.groupby(d.country).shift(2)
    ctrl = {y: c.LAG(c.SRC + ["gdp_g"]) + [f"dl1_{s}", f"dl2_{s}"] for y, s in (("nl_corp", "c"), ("nl_hh", "h"))}
    q = q.dropna(subset=sum(ctrl.values(), [])).copy()
    for y in ctrl:
        M = np.hstack([q[c.SRC + ctrl[y]].to_numpy(float), lt._fe_matrix(q, ("country", "year"))])
        e = q[f"{y}_h0"].to_numpy(float) - M @ np.linalg.lstsq(M, q[f"{y}_h0"].to_numpy(float), rcond=None)[0]
        q[f"w_{y}"] = 1 / q.country.map(pd.Series(e, index=q.index).groupby(q.country).var())
    return q.dropna(subset=["w_nl_corp", "w_nl_hh"]), ctrl


def stats(q, ctrl):
    out = []
    for y in ("nl_corp", "nl_hh"):
        B = np.array([lt.ols(q, f"{y}_h{h}", c.SRC, ctrl[y], se=False, w=f"w_{y}")[0] for h in range(H)])
        for k, x in enumerate(c.SRC):
            if c.OWN[x] == y:
                out.append(B[:, k] / B[0, k])
    return np.concatenate(out)


def run(reps=499):
    q, ctrl = sample()
    pt = stats(q, ctrl)
    bs = lt.cluster_bootstrap(q, lambda b: stats(b, ctrl), reps=reps)
    keys = [x for y in ("nl_corp", "nl_hh") for x in c.SRC if c.OWN[x] == y]
    r = pd.DataFrame([dict(source=x, h=h) for x in keys for h in range(H)]).assign(
        ratio=pt, lo68=np.nanpercentile(bs, 16, axis=0), hi68=np.nanpercentile(bs, 84, axis=0), n=len(q), countries=q.country.nunique())
    r.to_csv(OUT / "ep_source_fig.csv", index=False)
    return r


def figures(r):
    import fig_paper as fp
    plt = fp.plt
    col = {"x_P": fp.CO, "x_R": fp.CO, "x_H": fp.HH, "x_Y": fp.HH}   # 색 = 부문, 패널 = 성분 유형
    ls = {"x_P": "-", "x_R": "-", "x_H": "--", "x_Y": "--"}
    n, k = int(r.n.iloc[0]), int(r.countries.iloc[0])
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.3), sharey=True)
    for ax, srcs, title in ((axes[0], ["x_P", "x_Y"], "(a) Saving from higher income or profits"),
                            (axes[1], ["x_R", "x_H"], "(b) Saving from a change in behavior")):
        for x in srcs:
            q = r[r.source == x].sort_values("h")
            ax.fill_between(q.h, q.lo68, q.hi68, color=col[x], alpha=0.15, lw=0)
            ax.plot(q.h, q.ratio, color=col[x], ls=ls[x], lw=1.6, marker="o", ms=3, label=LAB[x])
        fp.finish(ax, "Own-sector saving–investment balance\n(impact response = 1)" if srcs[0] == "x_P" else "", h0=0)
        ax.set_xlabel("Years after the saving change ($h$)")
        ax.set_title(title, loc="left", fontsize=9.5)
        ax.legend(loc="upper right", fontsize=7.5)
    axes[0].set_ylim(-0.4, 2.2)
    fig.suptitle("Persistence of sector balances by source of the saving change", x=0.0, ha="left", fontweight="bold", y=1.03)
    fig.tight_layout()
    fig.text(0.0, -0.05, f"Notes: Local projections on the four components of the change in sector saving (N = {n}, {k} economies; same country-years at every horizon). "
             "Controls: two lags of the four components, GDP growth and the own-sector balance; country and year fixed effects; two-step weighted least squares. "
             "Shaded areas are 68 percent bands from a country-cluster bootstrap (499 replications).", fontsize=7, wrap=True, va="top")
    fp.save(fig, "fig_source_components", sfx="")

    s = pd.read_csv(OUT / "ep_source_check.csv")
    specs = list(dict.fromkeys(s.spec))
    names = {"no controls": "Fixed effects only", "lags of sources": "+ lags of components",
             "+ lags of CA change, growth": "+ lags of CA change and growth",
             "+ lagged levels (main)": "+ lagged levels (previous main)",
             "main, own S-I lags instead of CA": "Levels, own-sector lags",
             "lags of sources, growth, own S-I (no levels)": "Proposed: own-sector lags, no levels"}
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2), sharey=True)
    y = np.arange(len(specs))[::-1]
    for ax, horizon, title in ((axes[0], "short", "(a) Years 1–2"), (axes[1], "medium", "(b) Years 3–6")):
        for j, x in enumerate(c.SRC):
            v = s[s.source == x].set_index("spec").loc[specs, horizon].to_numpy()
            ax.scatter(v, y + (j - 1.5) * 0.15, color=col[x], s=18, marker="o" if ls[x] == "-" else "D", label=LAB[x], zorder=3)
        ax.axvline(0, color=".3", lw=.6)
        ax.set_title(title, loc="left", fontsize=9.5)
        ax.set_xlabel("Average response (impact = 1)")
        ax.grid(axis="y", color=".9", lw=.6)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([names[p] for p in specs], fontsize=8)
    h_, l_ = axes[1].get_legend_handles_labels()
    fig.legend(h_, l_, loc="lower center", bbox_to_anchor=(0.55, -0.1), ncol=4, fontsize=7.5)
    fig.suptitle("Sensitivity to control variables", x=0.0, ha="left", fontweight="bold", y=1.03)
    fig.tight_layout()
    fig.text(0.0, -0.13, "Notes: Point estimates; same sample and weighting as the proposed specification, with weights re-estimated for each set of controls.",
             fontsize=7, va="top")
    fp.save(fig, "fig_source_sensitivity", sfx="")


if __name__ == "__main__":
    import sys
    r = pd.read_csv(OUT / "ep_source_fig.csv") if "--render" in sys.argv else run()
    figures(r)
    print(r.pivot_table(index="source", columns="h", values="ratio").round(2).to_string())
