"""Journal-style English figures (matplotlib) for the final specification.

Inputs (from ep_report.py --fgls --nl, render_ep_bars.py, ep_report_extra.py):
  output/ep_report_fgls_nl.csv, output/ep_report_bars.csv, output/ep_report_extra.csv
Outputs: output/paper/fig1_si_persistence*.(png|pdf), fig2_corporate_decomposition.(png|pdf),
         fig3_household_saving_consumption.(png|pdf)   (suffix _lags for the lag-controlled specification)
All responses are y_{t+h} − y_{t−1}; h = −1 is the reference year and equals zero by construction.
"""
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import sys

OUT = Path(__file__).resolve().parent / "output"
DST = OUT / "paper"
LAGS = "--lags" in sys.argv          # 사건 전 자기 부문 S−I 변화 2개 시차를 통제한 사양(본문 추천)
SFX = "_lags" if LAGS else ""
H0 = 0 if LAGS else -1               # 시차 통제판은 h = 0부터
HMAX = 6
HH, CO = "#1f3b73", "#b5541c"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix",
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9, "legend.fontsize": 8, "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.7,
    "xtick.major.width": 0.7, "ytick.major.width": 0.7, "legend.frameon": False, "savefig.dpi": 300,
})
XLAB = "Years after episode onset ($h$)"
LAB_HH = "Households: consumption-driven saving shifts"
LAB_CO = "Corporations: terms-of-trade-driven profit shifts"
SPEC = ("Episodes: country-specific top/bottom decile changes in the household saving rate with the consumption-share change exceeding "
        "the disposable-income-share change (households), or in the terms of trade with operating surplus moving in the same direction "
        "(corporations); increases and decreases pooled with signs aligned. Local projections with country and year fixed effects, "
        "estimated by two-step weighted least squares; 42 economies, 1970–2025."
        + (" All regressions control for two lags of the change in the own-sector saving–investment balance." if "--lags" in sys.argv else ""))


def with_ref(q, cols=("ratio", "lo68", "hi68")):
    """Prepend h = −1 (reference year, zero by construction)."""
    q = q[q.h <= HMAX].sort_values("h")
    ref = pd.DataFrame({"h": [-1], **{c: [0.0] for c in cols}})
    return pd.concat([ref, q[["h", *cols]]], ignore_index=True)


def line(ax, q, color, label, ls="-", band=True, ref=None):
    ref = H0 < 0 if ref is None else ref
    q = with_ref(q) if ref else q[q.h <= HMAX].sort_values("h")
    if band:
        ax.fill_between(q.h, q.lo68, q.hi68, color=color, alpha=0.14, lw=0)
    ax.plot(q.h, q.ratio, color=color, ls=ls, lw=1.6, marker="o", ms=3, label=label)


def finish(ax, ylab, h0=None):
    h0 = H0 if h0 is None else h0
    ax.axhline(0, color="0.3", lw=0.6)
    if h0 < 0:
        ax.axvline(-1, color="0.75", lw=0.5, ls=":")
    ax.set_xticks(range(h0, HMAX + 1))
    ax.set_xlabel(XLAB)
    ax.set_ylabel(ylab)


def save(fig, name):
    DST.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(DST / f"{name}{SFX}.{ext}", bbox_inches="tight")
    plt.close(fig)


def fig1():
    r = pd.read_csv(OUT / "ep_report_fgls_nl.csv".replace(".csv", SFX + ".csv"))
    r = r[r.kind == "resp"]
    n_hh, n_co = int(r.n_hh.iloc[0]), int(r.n_corp.iloc[0])
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    line(ax, r[(r.sector == "hh") & (r["var"] == "nl_hh")], HH, f"{LAB_HH} (N = {n_hh})")
    line(ax, r[(r.sector == "corp") & (r["var"] == "nl_corp")], CO, f"{LAB_CO} (N = {n_co})", ls="--")
    finish(ax, "Saving–investment balance\n(impact response = 1)")
    ax.legend(loc="lower left")
    ax.set_title("Figure 1. Persistence of sectoral saving–investment balances", loc="left", fontweight="bold")
    fig.text(0.0, -0.2, "Notes: Cumulative responses of the own-sector saving–investment balance (national-accounts net lending, "
             "percent of GDP), normalized by the impact response. Shaded areas are 68 percent confidence bands from a country-cluster "
             "bootstrap (499 replications). Responses are measured relative to year $-1$. " + SPEC, fontsize=7, wrap=True, va="top",
             transform=ax.transAxes)
    save(fig, "fig1_si_persistence")


def fig2():
    r = pd.read_csv(OUT / "ep_report_bars.csv".replace(".csv", SFX + ".csv"))
    c = r[(r.sector == "corp") & (r.h <= HMAX)]
    g = lambda v: c[c["var"] == v].sort_values("h").ratio.values
    nl = g("nl_corp")
    comps = [("Gross operating surplus", g("prof"), "#1f3b73"), ("Net dividends paid (−)", -g("div_net"), "#7a6aa8"),
             ("Gross capital formation (−)", -g("gcf_corp"), "#b5541c")]
    comps.append(("Other", nl - sum(v for _, v, _ in comps), "#a6a6a6"))
    hs = np.arange(len(nl))
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    up, dn = np.zeros(len(hs)), np.zeros(len(hs))
    for lab, v, col in comps:
        base = np.where(v >= 0, up, dn)
        ax.bar(hs, v, bottom=base, width=0.6, color=col, alpha=0.85, label=lab, edgecolor="white", linewidth=0.4)
        up, dn = up + np.clip(v, 0, None), dn + np.clip(v, None, 0)
    xs, ys = (np.r_[-1, hs], np.r_[0, nl]) if H0 < 0 else (hs, nl)
    ax.plot(xs, ys, color="black", lw=1.5, marker="o", ms=3.5, mfc="white", label="Saving–investment balance")
    finish(ax, "Contribution\n(impact saving–investment balance = 1)")
    ax.set_xlim(H0 - 0.5, HMAX + 0.5)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, columnspacing=1.2)
    ax.set_title("Figure 2. Decomposition of the corporate saving–investment balance", loc="left", fontweight="bold")
    fig.text(0.0, -0.36, "Notes: Point estimates for terms-of-trade episodes. Bars sum to the corporate saving–investment balance: "
             "gross operating surplus minus net dividends paid (paid minus received) minus gross capital formation (fixed capital and "
             "inventories) plus other items (interest, taxes, other property income, non-produced assets and capital transfers). " + SPEC,
             fontsize=7, wrap=True, va="top", transform=ax.transAxes)
    save(fig, "fig2_corporate_decomposition")


def fig3():
    """Figure 3: (a) household saving rate, (b) household consumption — both episode types, raw units."""
    p = OUT / "ep_report_extra.csv".replace(".csv", SFX + ".csv")
    if not p.exists():
        return
    r = pd.read_csv(p)
    if "cons_hh" not in set(r["var"]):
        return
    n_hh, n_co = int(r.n_hh.iloc[0]), int(r.n_corp.iloc[0])
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1))
    for ax, v, ylab, title in ((axes[0], "src_hh", "Percentage points of\ndisposable income", "(a) Household saving rate"),
                               (axes[1], "cons_hh", "Percentage points of GDP", "(b) Household consumption")):
        line(ax, r[(r.sector == "hh") & (r["var"] == v)], HH, f"Household episodes (N = {n_hh})")
        line(ax, r[(r.sector == "corp") & (r["var"] == v)], CO, f"Terms-of-trade episodes (N = {n_co})", ls="--")
        finish(ax, ylab)
        ax.set_title(title, loc="left")
    axes[0].legend(loc="center", bbox_to_anchor=(0.55, 0.47))
    fig.suptitle("Figure 3. Household saving rate and consumption", x=0.0, ha="left", fontweight="bold", y=1.03)
    fig.tight_layout()
    fig.text(0.0, -0.06, "Notes: Cumulative change per episode relative to year $-1$, not normalized. (a) Household gross saving over "
             "gross disposable income. (b) Household final consumption expenditure, percent of nominal GDP. Shaded areas are 68 percent "
             "confidence bands (country-cluster bootstrap, 499 replications). Same episodes and estimator as Figure 1"
             + (", including two lags of the change in the own-sector saving–investment balance." if LAGS else "."),
             fontsize=7, wrap=True, va="top")
    save(fig, "fig3_household_saving_consumption")


def fig1_variants():
    """그림 1 두 가지: 사전 시점 포함(h = −3…6, h = −1 기준 0) / h = 0부터."""
    p = OUT / f"ep_report_leads{SFX}.csv"
    if not p.exists():
        return
    r = pd.read_csv(p)
    n_hh, n_co = int(r.n_hh.iloc[0]), int(r.n_corp.iloc[0])
    hmin = int(r.h.min())
    for name, h0 in (("fig1_si_persistence_leads", hmin), ("fig1_si_persistence_h0", 0)):
        fig, ax = plt.subplots(figsize=(5.2, 3.4))
        for s, col, lab, ls in (("hh", HH, f"{LAB_HH} (N = {n_hh})", "-"), ("corp", CO, f"{LAB_CO} (N = {n_co})", "--")):
            q = r[r.sector == s]
            if h0 < 0:  # 기준 시점 h = −1(0)을 끼워 넣음
                q = pd.concat([q, pd.DataFrame({"h": [-1], "ratio": [0.0], "lo68": [0.0], "hi68": [0.0]})])
            line(ax, q[q.h >= h0], col, lab, ls=ls, ref=False)
        finish(ax, "Saving–investment balance\n(impact response = 1)", h0=h0)
        if h0 < 0:
            ax.axvspan(h0 - 0.4, -0.5, color="0.93", lw=0, zorder=0)
        ax.legend(loc="lower left")
        ax.set_title("Figure 1. Persistence of sectoral saving–investment balances", loc="left", fontweight="bold")
        pre = (f" Years ${hmin}$ to $-2$ are pre-episode responses ($y_{{t+h}} - y_{{t-1}}$, shaded area); year $-1$ is the reference year."
               if h0 < 0 else " Responses are measured relative to year $-1$.")
        if h0 < 0 and LAGS:
            pre += (" With two lags of the own-sector change controlled for, years $-3$ and $-2$ are zero by construction; "
                    "years $-5$ and $-4$ test for remaining pre-episode movement.")
        fig.text(0.0, -0.2, "Notes: Cumulative responses of the own-sector saving–investment balance (national-accounts net lending, "
                 "percent of GDP), normalized by the impact response. Shaded bands are 68 percent confidence bands from a "
                 "country-cluster bootstrap (499 replications)." + pre + " " + SPEC, fontsize=7, wrap=True, va="top",
                 transform=ax.transAxes)
        save(fig, name)


if __name__ == "__main__":
    fig1()
    fig2()
    fig3()
    fig1_variants()
