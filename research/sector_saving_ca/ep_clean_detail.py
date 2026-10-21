"""깨끗한 대조군(LP-DiD식) 세부 변수 점추정치와 영문 그림.

사양: ep_clean_controls와 같음(시차 통제, 2단계 가중최소제곱, 국가·연도FE, S−I 정규화).
→ output/ep_clean_detail.csv, output/paper/fig_clean_detail.(png|pdf)
"""
import numpy as np
import pandas as pd

import ep_clean_controls as cc
import ep_report as rp
import fig_paper as fp
import lp_tools as lt
from make_report import OUT

K = rp.K
VARS = {"hh": ["nl_hh", "sv_hh", "yd_hh", "cons_hh", "gcf_hh"],
        "corp": ["nl_corp", "sv_corp", "prof", "div_net", "gcf_corp", "inv_corp"]}


def main():
    rp.FGLS = rp.WSV = rp.LAGS = True
    rp.NORM = "nl"
    rp.HHV, rp.COV = VARS["hh"], VARS["corp"]
    P, d = rp.prepare()
    m = cc.clean_masks(d)
    rows = []
    for s, vs in VARS.items():
        c, w = rp.ctrl(s), f"w_{s}"
        g0 = lt.ols(d[m[(s, 0)]], f"nl_{s}_h0", [f"z_{s}"], c, se=False, w=w)[0][0]
        for v in vs:
            for h in range(K + 1):
                rows.append(dict(sector=s, var=v, h=h,
                                 ratio=lt.ols(d[m[(s, h)]], f"{v}_h{h}", [f"z_{s}"], c, se=False, w=w)[0][0] / g0))
    r = pd.DataFrame(rows)
    r.to_csv(OUT / "ep_clean_detail.csv", index=False)
    print(r.pivot_table(index=["sector", "var"], columns="h", values="ratio", sort=False).round(2).to_string())
    plot(r)


def plot(r):
    plt = fp.plt
    panels = [("(a) Corporations: saving and operating surplus", "corp", [("sv_corp", "Gross saving", "-"), ("prof", "Gross operating surplus", "--")]),
              ("(b) Corporations: dividends and investment", "corp", [("div_net", "Net dividends paid", "-"), ("gcf_corp", "Gross capital formation", "--"),
                                                                       ("inv_corp", "Fixed capital formation", ":")]),
              ("(c) Households: saving, income and consumption", "hh", [("sv_hh", "Gross saving", "-"), ("yd_hh", "Disposable income", "--"),
                                                                         ("cons_hh", "Final consumption", ":")]),
              ("(d) Households: investment", "hh", [("gcf_hh", "Gross capital formation", "-")])]
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.6))
    cols = ["#1f3b73", "#b5541c", "#5f8f3e"]
    for ax, (title, s, series) in zip(axes.ravel(), panels):
        for (v, lab, ls), col in zip(series, cols):
            q = r[(r.sector == s) & (r["var"] == v)].sort_values("h")
            ax.plot(q.h, q.ratio, color=col, ls=ls, lw=1.5, marker="o", ms=2.8, label=lab)
        fp.finish(ax, "Response (impact own-sector\nsaving–investment balance = 1)", h0=0)
        ax.set_title(title, loc="left")
        ax.legend(loc="best", fontsize=7)
    fig.suptitle("Clean-control estimates: components of sectoral saving–investment balances", x=0.0, ha="left",
                 fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.text(0.0, -0.03, "Notes: Point estimates, percent of GDP, normalized by the impact response of the own-sector saving–investment "
             "balance. Clean controls exclude country-years with a large shock of the same type in $[t-3, t+h]$. "
             "Same episodes and estimator as Figure 1 (lag-controlled).", fontsize=7, wrap=True, va="top")
    fp.save(fig, "fig_clean_detail", sfx="")


if __name__ == "__main__":
    main()
