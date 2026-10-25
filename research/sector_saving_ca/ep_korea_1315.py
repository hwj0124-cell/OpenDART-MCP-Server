"""한국 2013–15년 흑자의 원천과 이후 경로(OECD 부문계정, GNI 대비 %).

(a) 경상수지와 부문 순대출, 2005–2025
(b) 가계: 총저축률(처분가능소득 대비, 오른쪽 축), 총저축, 총자본형성, 순대출
(c) 비금융법인: 총저축, 총자본형성, 순대출
(d) 원천별 누적 기여(2012→2015, 2023→2025): 가계 저축률·소득 성분, 가계 투자(−), 기업 이익·유보율 성분, 기업 투자(−),
    정부 순대출 변화, 기타(경상수지 변화와의 차이: 금융법인·통계 불일치·자본이전 등)
→ output/ep_korea_1315.csv, output/paper/fig_korea_1315.(png|pdf)
"""
import numpy as np
import pandas as pd

import source_shocks as ss
from make_report import OUT


def data():
    d = ss.prepare(pd.read_csv(OUT.parent / "data" / "panel.csv"))
    return d[(d.country == "KOR") & d.year.between(2005, 2025)].set_index("year")


def contributions(k, a, b):
    yrs = range(a + 1, b + 1)
    c = {"Household saving rate": k.loc[yrs, "x_H"].sum(), "Household income": k.loc[yrs, "x_Y"].sum(),
         "Household investment (−)": -(k.loc[b, "gcf_hh"] - k.loc[a, "gcf_hh"]),
         "Corporate profits": k.loc[yrs, "x_P"].sum(), "Corporate retention": k.loc[yrs, "x_R"].sum(),
         "Corporate investment (−)": -(k.loc[b, "gcf_corp"] - k.loc[a, "gcf_corp"]),
         "Government net lending": k.loc[b, "nl_gov"] - k.loc[a, "nl_gov"]}
    dca = k.loc[b, "ca"] - k.loc[a, "ca"]
    c["Other"] = dca - sum(c.values())
    return pd.Series(c), dca


def figure(k):
    import fig_paper as fp
    plt = fp.plt
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.6))
    shade = lambda ax: ax.axvspan(2012.5, 2015.5, color="0.9", lw=0, zorder=0)
    ax = axes[0, 0]
    shade(ax)
    for v, cl, lab in (("ca", "0.1", "Current account"), ("nl_hh", fp.HH, "Households"), ("nl_corp", fp.CO, "Nonfinancial corporations"),
                       ("nl_gov", "#6a8f3c", "Government")):
        ax.plot(k.index, k[v], color=cl, lw=2.0 if v == "ca" else 1.4, label=lab)
    ax.axhline(0, color=".3", lw=.6)
    ax.set_title("(a) Current account and net lending", loc="left", fontsize=9.2)
    ax.set_ylabel("Percent of GNI")
    ax.legend(loc="upper left", fontsize=6.8, frameon=False, ncol=2)
    ax.set_ylim(-8.5, 12.5)
    ax = axes[0, 1]
    shade(ax)
    for v, cl, ls, lab in (("sv_hh", fp.HH, "-", "Gross saving"), ("gcf_hh", "#2b6c8f", "--", "Gross capital formation"), ("nl_hh", "0.2", ":", "Net lending")):
        ax.plot(k.index, k[v], color=cl, ls=ls, lw=1.6, label=lab)
    ax.set_ylabel("Percent of GNI")
    ax2 = ax.twinx()
    ax2.plot(k.index, k.hh_rate, color="#c98a1b", lw=1.8, marker="o", ms=2.5, label="Saving rate (right, % of disposable income)")
    ax2.set_ylim(0, 20)
    ax2.spines["right"].set_visible(True)
    ax.set_title("(b) Households", loc="left", fontsize=9.2)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=6.6, frameon=False)
    ax.set_ylim(0, 13)
    ax = axes[1, 0]
    shade(ax)
    for v, cl, ls, lab in (("sv_corp", fp.CO, "-", "Gross saving"), ("gcf_corp", "#2b6c8f", "--", "Gross capital formation"), ("nl_corp", "0.2", ":", "Net lending")):
        ax.plot(k.index, k[v], color=cl, ls=ls, lw=1.6, label=lab)
    ax.axhline(0, color=".3", lw=.6)
    ax.set_ylabel("Percent of GNI")
    ax.set_title("(c) Nonfinancial corporations", loc="left", fontsize=9.2)
    ax.legend(loc="center left", fontsize=7, frameon=False)
    ax = axes[1, 1]
    c1, d1 = contributions(k, 2012, 2015)
    c2, d2 = contributions(k, 2023, 2025)
    pal = {"Household saving rate": fp.HH, "Household income": "#7f9cc9", "Household investment (−)": "#9db6d9", "Corporate profits": fp.CO,
           "Corporate retention": "#e0956a", "Corporate investment (−)": "#f2c3a5", "Government net lending": "#6a8f3c", "Other": "0.75"}
    for x, (c, dca) in enumerate(((c1, d1), (c2, d2))):
        pos = neg = 0.0
        for name, v in c.items():
            ax.bar(x, v, bottom=pos if v >= 0 else neg, color=pal[name], width=0.55, label=name if x == 0 else None, edgecolor="white", lw=0.3)
            pos, neg = (pos + v, neg) if v >= 0 else (pos, neg + v)
        ax.scatter(x, dca, color="k", zorder=5, s=22, label="Change in current account" if x == 0 else None)
    ax.axhline(0, color=".3", lw=.6)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["2012→2015", "2023→2025"])
    ax.set_xlim(-0.6, 2.6)
    ax.set_ylabel("Percentage points of GNI")
    ax.set_title("(d) Sources of the change in the current account", loc="left", fontsize=9.2)
    ax.legend(loc="center right", fontsize=6.5, frameon=False)
    for ax in axes.flat[:3]:
        ax.set_xlim(2004.5, 2025.5)
        ax.set_xticks(range(2005, 2026, 5))
    fig.suptitle("Korea's 2013–15 surplus: a higher household saving rate and weak corporate investment", x=0.0, ha="left", fontweight="bold", y=1.0)
    fig.tight_layout()
    fig.text(0.0, -0.03, "Notes: Annual, percent of gross national income; shading marks 2013–15. Households include NPISH. (d) Cumulative sums of the "
             "saving components over the period (saving-rate and income components for households; profit and retention components for corporations), "
             "minus the change in each sector's gross capital formation, plus the change in government net lending. 'Other' is the remainder of the "
             "change in the current account (financial corporations, capital transfers, discrepancies). Annual sector accounts end in 2025. Source: OECD "
             "sector accounts (Table 14); authors' calculations.", fontsize=6.9, wrap=True, va="top")
    fp.save(fig, "fig_korea_1315", sfx="")
    return c1, d1, c2, d2


if __name__ == "__main__":
    k = data()
    c1, d1, c2, d2 = figure(k)
    out = pd.DataFrame({"2012-2015": c1, "2023-2025": c2})
    out.loc["Change in CA"] = [d1, d2]
    out.to_csv(OUT / "ep_korea_1315.csv")
    print(out.round(2).to_string())
