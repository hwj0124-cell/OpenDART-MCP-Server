"""원천 분해 LP: 자기 부문 S−I 시차를 변화분(현행)으로 넣을 때와 수준(표준 LP)으로 넣을 때 비교.

공통: 네 성분, 네 성분·성장률 두 시차, 국가·연도 FE, 같은 표본(시작 연도 ≤ 2019).
  현행: + Δn_{t−1}, Δn_{t−2}   (수준 계수 0 = 단위근 가정)
  수준: + n_{t−1}, n_{t−2}     (n_{t−1}을 통제하므로 좌변을 수준으로 써도 계수 동일)
가중치는 사양마다 시작 연도 식으로 다시 계산. 구간: 국가 군집 부트스트랩 68%(가중치 고정).
→ output/ep_source_levels.csv, output/paper/fig_source_levels.(png|pdf)
"""
import sys

import numpy as np
import pandas as pd

import ep_source_check as c
import ep_source_fig as sf
import lp_tools as lt
from make_report import OUT

H = c.H
SPECS = ("changes", "levels")
SEC = ("nl_corp", "nl_hh")


def sample():
    q, ctrl = sf.sample()
    p = lt.balance(pd.read_csv(OUT.parent / "data" / "panel.csv"))
    for y in SEC:
        for l in (1, 2):
            p[f"lv{l}_{y}"] = p.groupby("country")[y].shift(l)
    q = q.merge(p[["country", "year"] + [f"lv{l}_{y}" for y in SEC for l in (1, 2)]], on=["country", "year"], how="left")
    C = {("changes", y): ctrl[y] for y in SEC}
    C.update({("levels", y): c.LAG(c.SRC + ["gdp_g"]) + [f"lv1_{y}", f"lv2_{y}"] for y in SEC})
    for (s, y), cc in C.items():
        M = np.hstack([q[c.SRC + cc].to_numpy(float), lt._fe_matrix(q, ("country", "year"))])
        e = q[f"{y}_h0"].to_numpy(float) - M @ np.linalg.lstsq(M, q[f"{y}_h0"].to_numpy(float), rcond=None)[0]
        q[f"w_{s}_{y}"] = 1 / q.country.map(pd.Series(e, index=q.index).groupby(q.country).var())
    return q, C


def stats(q, C):
    out = []
    for s in SPECS:
        for y in SEC:
            B = np.array([lt.ols(q, f"{y}_h{h}", c.SRC, C[s, y], se=False, w=f"w_{s}_{y}")[0] for h in range(H)])
            out += [B[:, k] / B[0, k] for k, x in enumerate(c.SRC) if c.OWN[x] == y]
    return np.concatenate(out)


def run(reps=499):
    q, C = sample()
    pt = stats(q, C)
    bs = lt.cluster_bootstrap(q, lambda b: stats(b, C), reps=reps)
    keys = [(s, x) for s in SPECS for y in SEC for x in c.SRC if c.OWN[x] == y]
    r = pd.DataFrame([dict(spec=s, source=x, h=h) for s, x in keys for h in range(H)]).assign(
        ratio=pt, lo68=np.nanpercentile(bs, 16, axis=0), hi68=np.nanpercentile(bs, 84, axis=0), n=len(q), countries=q.country.nunique())
    r.to_csv(OUT / "ep_source_levels.csv", index=False)
    return r


def figure(r):
    import fig_paper as fp
    plt = fp.plt
    col = {"x_P": fp.CO, "x_R": fp.CO, "x_H": fp.HH, "x_Y": fp.HH}
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.6), sharex=True, sharey=True)
    for ax, x in zip(axes.flat, ["x_P", "x_Y", "x_R", "x_H"]):
        for s, ls, mk, lab in (("levels", "-", "o", "Own-sector balance in levels ($n_{t-1}, n_{t-2}$)"),
                               ("changes", "--", "s", "Own-sector balance in changes ($\\Delta n_{t-1}, \\Delta n_{t-2}$)")):
            q = r[(r.spec == s) & (r.source == x)].sort_values("h")
            if s == "levels":
                ax.fill_between(q.h, q.lo68, q.hi68, color=col[x], alpha=0.16, lw=0)
            else:
                ax.plot(q.h, q.lo68, color=col[x], ls=":", lw=0.8)
                ax.plot(q.h, q.hi68, color=col[x], ls=":", lw=0.8)
            ax.plot(q.h, q.ratio, color=col[x], ls=ls, lw=1.6, marker=mk, ms=3, label=lab)
        fp.finish(ax, "Own-sector S–I\n(impact = 1)" if x in ("x_P", "x_R") else "", h0=0)
        ax.set_title(sf.LAB[x], loc="left", fontsize=9.5)
        ax.set_xlabel("Years after the saving change ($h$)" if x in ("x_R", "x_H") else "")
    axes[0, 0].set_ylim(-0.7, 2.2)
    h_, l_ = axes[0, 0].get_legend_handles_labels()
    fig.legend(h_, l_, loc="lower center", bbox_to_anchor=(0.5, -0.03), ncol=2, fontsize=8)
    fig.suptitle("Persistence by source: own-sector balance controlled in levels or in changes", x=0.0, ha="left", fontweight="bold", y=1.0)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    n, k = int(r.n.iloc[0]), int(r.countries.iloc[0])
    fig.text(0.0, -0.04, f"Notes: N = {n}, {k} economies; same country-years at every horizon. Both specifications include the four components, two lags of the four "
             "components and GDP growth, and country and year fixed effects, and differ only in how two lags of the own-sector balance enter. Two-step weighted least "
             "squares with weights re-estimated for each specification. Shaded areas (levels) and dotted lines (changes) are 68 percent bands from a country-cluster "
             "bootstrap (499 replications).", fontsize=7, wrap=True, va="top")
    fp.save(fig, "fig_source_levels", sfx="")


if __name__ == "__main__":
    r = pd.read_csv(OUT / "ep_source_levels.csv") if "--render" in sys.argv else run()
    figure(r)
    print(r.pivot_table(index=["spec", "source"], columns="h", values="ratio", sort=False).round(2).to_string())
    a = r.assign(per=np.where(r.h.between(1, 2), "1-2", np.where(r.h >= 3, "3-6", "0")))
    print(a[a.per != "0"].pivot_table(index=["spec", "source"], columns="per", values="ratio", sort=False).round(2).to_string())
