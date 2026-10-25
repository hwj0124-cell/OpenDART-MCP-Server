"""부문별 저축 '성향'의 역사적 안정성: 기업 유보율(총저축/영업잉여), 가계 저축률(총저축/처분가능소득), 정부 저축률(총저축/처분가능소득).

(a) 한국 1975–2025, (b) OECD 패널 연도별 중앙값과 10–90 백분위.
→ output/ep_ratio_history.csv, output/paper/fig_ratio_history.(png|pdf)
"""
import numpy as np
import pandas as pd

import ep_wb_gov as eg
import source_shocks as ss
from make_report import OUT

V = {"ret_pct": "Corporate retention ratio", "hh_rate": "Household saving rate", "gov_rate": "Government saving rate"}


def data():
    d = ss.prepare(pd.read_csv(OUT.parent / "data" / "panel.csv")).merge(eg.gov_series(), on=["country", "year"], how="left")
    d["gov_rate"] = 100 * d.sv_gov / d.yd_gov
    d["ret_pct"] = 100 * d.ret
    d = d[["country", "year"] + list(V)].replace([np.inf, -np.inf], np.nan)
    return d


def stats(d):
    rows = {}
    for v in V:
        x = d[["country", "year", v]].dropna()
        ch = x.groupby("country")[v].diff()
        lag = x.groupby("country")[v].shift(1)
        rows[V[v]] = {"country-years": len(x), "median": x[v].median(), "p10": x[v].quantile(.1), "p90": x[v].quantile(.9),
                      "share <= 0": (x[v] <= 0).mean(), "sd of annual change": ch.std(),
                      "median |change| / |level|": (ch.abs() / lag.abs()).median()}
    return pd.DataFrame(rows).T


def figure(d):
    import fig_paper as fp
    plt = fp.plt
    col = {"ret_pct": fp.CO, "hh_rate": fp.HH, "gov_rate": "#6a8f3c"}
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))
    ax = axes[0]
    k = d[d.country == "KOR"].set_index("year").loc[1975:2025]
    for v in V:
        ax.plot(k.index, k[v], color=col[v], lw=1.8, label=V[v])
    ax.axhline(0, color=".3", lw=.6)
    ax.set_ylabel("Percent")
    ax.set_ylim(-30, 80)
    ax.set_title("(a) Korea", loc="left", fontsize=9.3)
    ax.legend(loc="upper left", fontsize=7, frameon=False)
    ax = axes[1]
    p = d[d.year.between(1975, 2025)].groupby("year")
    for v in V:
        q = p[v].quantile([.1, .5, .9]).unstack()
        q = q[p[v].count() >= 10]
        ax.fill_between(q.index, q[0.1], q[0.9], color=col[v], alpha=0.15, lw=0)
        ax.plot(q.index, q[0.5], color=col[v], lw=1.8)
    ax.axhline(0, color=".3", lw=.6)
    ax.set_ylim(-30, 80)
    ax.set_title("(b) OECD panel: median and 10th–90th percentiles", loc="left", fontsize=9.3)
    fig.suptitle("Across economies, government saving ratios swing far more than corporate and household ones", x=0.0, ha="left",
                 fontweight="bold", y=1.03)
    fig.tight_layout()
    s = stats(d)
    fig.text(0.0, -0.02,
             "Notes: Gross saving of each sector divided by its operating surplus (corporations) or disposable income (households, government), "
             f"percent. Annual OECD sector accounts. Across all country-years, the share at or below zero is {100 * s.iloc[0, 4]:.1f}, "
             f"{100 * s.iloc[1, 4]:.1f} and {100 * s.iloc[2, 4]:.1f} percent, and the median absolute annual change relative to the previous level "
             f"is {100 * s.iloc[0, 6]:.0f}, {100 * s.iloc[1, 6]:.0f} and {100 * s.iloc[2, 6]:.0f} percent, for the corporate, household and government "
             "ratios. (b) Years with at least ten economies. Source: OECD; authors' calculations.", fontsize=6.9, wrap=True, va="top")
    fp.save(fig, "fig_ratio_history", sfx="")


if __name__ == "__main__":
    d = data()
    s = stats(d)
    s.to_csv(OUT / "ep_ratio_history.csv")
    print(s.round(3).to_string())
    figure(d)
