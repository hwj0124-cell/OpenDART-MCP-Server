"""Journal-style English figures (matplotlib) for the final specification.

Inputs (from ep_report.py --fgls --nl, render_ep_bars.py, ep_report_extra.py):
  output/ep_report_fgls_nl.csv, output/ep_report_bars.csv, output/ep_report_extra.csv
Outputs: output/paper/fig1_si_persistence.(png|pdf), fig2_corporate_decomposition.(png|pdf),
         fig3_saving_rate_current_account.(png|pdf)
All responses are y_{t+h} − y_{t−1}; h = −1 is the reference year and equals zero by construction.
"""
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path(__file__).resolve().parent / "output"
DST = OUT / "paper"
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
        "estimated by two-step weighted least squares; 42 economies, 1970–2025.")


def with_ref(q, cols=("ratio", "lo68", "hi68")):
    """Prepend h = −1 (reference year, zero by construction)."""
    q = q[q.h <= HMAX].sort_values("h")
    ref = pd.DataFrame({"h": [-1], **{c: [0.0] for c in cols}})
    return pd.concat([ref, q[["h", *cols]]], ignore_index=True)


def line(ax, q, color, label, ls="-", band=True):
    q = with_ref(q)
    if band:
        ax.fill_between(q.h, q.lo68, q.hi68, color=color, alpha=0.14, lw=0)
    ax.plot(q.h, q.ratio, color=color, ls=ls, lw=1.6, marker="o", ms=3, label=label)


def finish(ax, ylab):
    ax.axhline(0, color="0.3", lw=0.6)
    ax.axvline(-1, color="0.75", lw=0.5, ls=":")
    ax.set_xticks(range(-1, HMAX + 1))
    ax.set_xlabel(XLAB)
    ax.set_ylabel(ylab)


def save(fig, name):
    DST.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(DST / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)


def fig1():
    r = pd.read_csv(OUT / "ep_report_fgls_nl.csv")
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
             "bootstrap (499 replications). Year $-1$ is the reference year. " + SPEC, fontsize=7, wrap=True, va="top",
             transform=ax.transAxes)
    save(fig, "fig1_si_persistence")


def fig2():
    r = pd.read_csv(OUT / "ep_report_bars.csv")
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
    ax.plot(np.r_[-1, hs], np.r_[0, nl], color="black", lw=1.5, marker="o", ms=3.5, mfc="white", label="Saving–investment balance")
    finish(ax, "Contribution\n(impact saving–investment balance = 1)")
    ax.set_xlim(-1.5, HMAX + 0.5)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, columnspacing=1.2)
    ax.set_title("Figure 2. Decomposition of the corporate saving–investment balance", loc="left", fontweight="bold")
    fig.text(0.0, -0.36, "Notes: Point estimates for terms-of-trade episodes. Bars sum to the corporate saving–investment balance: "
             "gross operating surplus minus net dividends paid (paid minus received) minus gross capital formation (fixed capital and "
             "inventories) plus other items (interest, taxes, other property income, non-produced assets and capital transfers). " + SPEC,
             fontsize=7, wrap=True, va="top", transform=ax.transAxes)
    save(fig, "fig2_corporate_decomposition")


def fig3():
    p = OUT / "ep_report_extra.csv"
    if not p.exists():
        return
    r = pd.read_csv(p)
    n_hh, n_co = int(r.n_hh.iloc[0]), int(r.n_corp.iloc[0])
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1))
    for ax, v, ylab, title in ((axes[0], "src_hh", "Percentage points of\ndisposable income", "(a) Household saving rate"),
                               (axes[1], "ca", "Current account\n(impact own-sector balance = 1)", "(b) Current account")):
        line(ax, r[(r.sector == "hh") & (r["var"] == v)], HH, f"Household episodes (N = {n_hh})")
        line(ax, r[(r.sector == "corp") & (r["var"] == v)], CO, f"Terms-of-trade episodes (N = {n_co})", ls="--")
        finish(ax, ylab)
        ax.set_title(title, loc="left")
    axes[0].legend(loc="center", bbox_to_anchor=(0.55, 0.47))
    fig.suptitle("Figure 3. Household saving rate and current account responses", x=0.0, ha="left", fontweight="bold", y=1.03)
    fig.tight_layout()
    fig.text(0.0, -0.06, "Notes: (a) Cumulative change in the household gross saving rate per episode, percentage points of gross "
             "disposable income (not normalized). (b) Cumulative response of national net lending (percent of GDP), normalized by the "
             "impact response of the originating sector's saving–investment balance. Shaded areas are 68 percent confidence bands "
             "(country-cluster bootstrap, 499 replications). Same episodes and estimator as Figure 1.", fontsize=7, wrap=True, va="top")
    save(fig, "fig3_saving_rate_current_account")


if __name__ == "__main__":
    fig1()
    fig2()
    fig3()
