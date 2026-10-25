"""수준 시차 사양: 부문 안에서 이익·소득 원천과 행태 원천 비교 + 차이(행태 − 이익·소득)의 부트스트랩 구간.

위: 부문별 두 성분 경로(축은 부문별). 아래: 차이와 68·90% 구간.
→ output/ep_source_levels_fig.csv, output/paper/fig_source_levels_contrast.(png|pdf)
"""
import sys

import numpy as np
import pandas as pd

import ep_source_check as c
import ep_source_levels as sl
import lp_tools as lt
from make_report import OUT

H = c.H
PAIRS = {"nl_corp": ("x_P", "x_R"), "nl_hh": ("x_Y", "x_H")}   # (이익·소득, 행태)


def stats(q, C):
    out = []
    for y, (a, b) in PAIRS.items():
        B = np.array([lt.ols(q, f"{y}_h{h}", c.SRC, C["levels", y], se=False, w=f"w_levels_{y}")[0] for h in range(H)])
        ra, rb = B[:, c.SRC.index(a)] / B[0, c.SRC.index(a)], B[:, c.SRC.index(b)] / B[0, c.SRC.index(b)]
        out += [ra, rb, rb - ra]
    return np.concatenate(out)


def run(reps=499):
    q, C = sl.sample()
    pt = stats(q, C)
    bs = lt.cluster_bootstrap(q, lambda d: stats(d, C), reps=reps)
    keys = [(y, s) for y, (a, b) in PAIRS.items() for s in (a, b, "diff")]
    pc = lambda p: np.nanpercentile(bs, p, axis=0)
    r = pd.DataFrame([dict(sector=y, series=s, h=h) for y, s in keys for h in range(H)]).assign(
        ratio=pt, lo68=pc(16), hi68=pc(84), lo90=pc(5), hi90=pc(95), n=len(q), countries=q.country.nunique())
    r.to_csv(OUT / "ep_source_levels_fig.csv", index=False)
    return r


def figure(r):
    import fig_paper as fp
    plt = fp.plt
    INC, BEH = "#c98a1b", "#2b6c8f"
    lab = {"x_P": "Profit-driven", "x_R": "Retention-driven", "x_Y": "Income-driven", "x_H": "Saving-rate-driven"}
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.4), sharex=True, gridspec_kw=dict(height_ratios=[1.35, 1]))
    for j, (y, (a, b)) in enumerate(PAIRS.items()):
        ax = axes[0, j]
        for s, cl, ls, mk in ((a, INC, "-", "o"), (b, BEH, "-", "D")):
            q = r[(r.sector == y) & (r.series == s)].sort_values("h")
            ax.fill_between(q.h, q.lo68, q.hi68, color=cl, alpha=0.18, lw=0)
            ax.plot(q.h, q.ratio, color=cl, ls=ls, lw=1.8, marker=mk, ms=3.5, label=lab[s])
        fp.finish(ax, "Own-sector S–I (impact = 1)" if j == 0 else "", h0=0)
        ax.set_title(("(a) Corporations" if j == 0 else "(b) Households"), loc="left", fontsize=9.5)
        ax.legend(loc="upper right", fontsize=7.8, frameon=False)
        ax.set_ylim(-0.45, 1.1) if j == 0 else ax.set_ylim(-0.6, 1.75)
        ax.set_xlabel("")

        ax = axes[1, j]
        q = r[(r.sector == y) & (r.series == "diff")].sort_values("h")
        ax.fill_between(q.h, q.lo90, q.hi90, color="0.55", alpha=0.18, lw=0, label="90% band")
        ax.fill_between(q.h, q.lo68, q.hi68, color="0.35", alpha=0.25, lw=0, label="68% band")
        ax.plot(q.h, q.ratio, color="0.1", lw=1.8, marker="o", ms=3.5, label=f"{lab[b]} minus {lab[a].lower()}")
        fp.finish(ax, "Difference" if j == 0 else "", h0=0)
        ax.set_xlabel("Years after the saving change ($h$)")
        ax.set_title(("(c)" if j == 0 else "(d)") + " Behavior minus " + ("profits" if j == 0 else "income"), loc="left", fontsize=9.5)
        ax.legend(loc="upper left", fontsize=7.3, frameon=False)
    fig.suptitle("Profit-driven corporate saving fades faster than retention-driven saving", x=0.0, ha="left", fontweight="bold", y=1.0)
    fig.tight_layout()
    n, k = int(r.n.iloc[0]), int(r.countries.iloc[0])
    fig.text(0.0, -0.02, f"Notes: N = {n}, {k} economies; same country-years at every horizon. Local projections of the own-sector saving–investment balance on the four "
             "components of the change in sector saving, with two lags of the four components, GDP growth and the own-sector balance (in levels), and country and "
             "year fixed effects; two-step weighted least squares. Responses are normalized by the impact response. Bands from a country-cluster bootstrap "
             "(499 replications).", fontsize=7, wrap=True, va="top")
    fp.save(fig, "fig_source_levels_contrast", sfx="")


if __name__ == "__main__":
    r = pd.read_csv(OUT / "ep_source_levels_fig.csv") if "--render" in sys.argv else run()
    figure(r)
    d = r[r.series == "diff"]
    print(d.pivot_table(index="sector", columns="h", values="ratio").round(2).to_string())
    print(d[["sector", "h", "lo90", "hi90"]].round(2).to_string(index=False))
    a = r.assign(per=np.where(r.h.between(1, 2), "1-2", np.where(r.h >= 3, "3-6", "0")))
    print(a[a.per != "0"].pivot_table(index=["sector", "series"], columns="per", values="ratio", sort=False).round(2).to_string())
