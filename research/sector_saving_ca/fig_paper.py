"""Journal-style English figures (matplotlib) for the final specification.

Inputs (from ep_report.py --fgls --nl, render_ep_bars.py, ep_report_extra.py):
  output/ep_report_fgls_nl.csv, output/ep_report_bars.csv, output/ep_report_extra.csv
Outputs: output/paper/fig1_si_persistence*.(png|pdf), fig2_corporate_decomposition.(png|pdf),
         fig3_household_decomposition, fig4_saving_propensities, fig5_diagnostics, fig6_consumption_growth   (suffix _lags for the lag-controlled specification)
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


def save(fig, name, sfx=None):
    DST.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(DST / f"{name}{SFX if sfx is None else sfx}.{ext}", bbox_inches="tight")
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


def stack(ax, comps, nl, ylim=None):
    """Stacked contributions (bars) and the saving–investment balance (line)."""
    hs = np.arange(len(nl))
    up, dn = np.zeros(len(hs)), np.zeros(len(hs))
    for lab, v, col in comps:
        base = np.where(v >= 0, up, dn)
        ax.bar(hs, v, bottom=base, width=0.6, color=col, alpha=0.85, label=lab, edgecolor="white", linewidth=0.4)
        up, dn = up + np.clip(v, 0, None), dn + np.clip(v, None, 0)
    xs, ys = (np.r_[-1, hs], np.r_[0, nl]) if H0 < 0 else (hs, nl)
    ax.plot(xs, ys, color="black", lw=1.5, marker="o", ms=3.5, mfc="white", label="Saving–investment balance")
    finish(ax, "Contribution\n(impact saving–investment balance = 1)")
    ax.set_xlim(H0 - 0.5, HMAX + 0.5)
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, columnspacing=1.2)
    return up, dn


def comps_corp():
    r = pd.read_csv(OUT / "ep_report_bars.csv".replace(".csv", SFX + ".csv"))
    c = r[(r.sector == "corp") & (r.h <= HMAX)]
    g = lambda v: c[c["var"] == v].sort_values("h").ratio.values
    nl = g("nl_corp")
    comps = [("Gross operating surplus", g("prof"), "#1f3b73"), ("Net dividends paid (−)", -g("div_net"), "#7a6aa8"),
             ("Gross capital formation (−)", -g("gcf_corp"), "#b5541c")]
    comps.append(("Other", nl - sum(v for _, v, _ in comps), "#a6a6a6"))
    return comps, nl


def comps_hh():
    r = pd.read_csv(OUT / f"ep_report_diag{SFX}.csv")
    c = r[(r.sector == "hh") & (r.h <= HMAX)]
    g = lambda v: c[c["var"] == v].sort_values("h").ratio.values
    nl = g("nl_hh")
    comps = [("Final consumption (−)", -g("cons_hh"), "#7a6aa8"), ("Gross disposable income", g("yd_hh"), "#1f3b73"),
             ("Gross capital formation (−)", -g("gcf_hh"), "#b5541c")]  # 소비를 맨 아래에: 지속되는 소비 감소가 바닥에 보이도록
    comps.append(("Other", nl - sum(v for _, v, _ in comps), "#a6a6a6"))
    return comps, nl


def common_ylim():
    """Same vertical scale for Figures 2 and 3 so the two decompositions can be compared directly."""
    lo, hi = 0.0, 1.0
    for comps, nl in (comps_corp(), comps_hh()):
        vs = np.array([v for _, v, _ in comps])
        lo, hi = min(lo, np.clip(vs, None, 0).sum(0).min(), nl.min()), max(hi, np.clip(vs, 0, None).sum(0).max(), nl.max())
    pad = 0.06 * (hi - lo)
    return lo - pad, hi + pad


def fig2():
    comps, nl = comps_corp()
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    stack(ax, comps, nl, common_ylim())
    ax.set_title("Figure 2. Decomposition of the corporate saving–investment balance", loc="left", fontweight="bold")
    fig.text(0.0, -0.36, "Notes: Point estimates for terms-of-trade episodes. Bars sum to the corporate saving–investment balance: "
             "gross operating surplus minus net dividends paid (paid minus received) minus gross capital formation (fixed capital and "
             "inventories) plus other items (interest, taxes, other property income, non-produced assets and capital transfers). "
             "Vertical scale as in Figure 3. " + SPEC, fontsize=7, wrap=True, va="top", transform=ax.transAxes)
    save(fig, "fig2_corporate_decomposition")


def fig3():
    """Figure 3: household counterpart of Figure 2."""
    comps, nl = comps_hh()
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    stack(ax, comps, nl, common_ylim())
    ax.set_title("Figure 3. Decomposition of the household saving–investment balance", loc="left", fontweight="bold")
    fig.text(0.0, -0.36, "Notes: Point estimates for household episodes. Bars sum to the household saving–investment balance: "
             "gross disposable income minus final consumption minus gross capital formation (mainly dwellings) plus other items "
             "(pension-entitlement adjustment, non-produced assets and capital transfers). Vertical scale as in Figure 2. " + SPEC,
             fontsize=7, wrap=True, va="top", transform=ax.transAxes)
    save(fig, "fig3_household_decomposition")


def fig4():
    """Figure 4: behavioural ratios — household saving rate (household episodes) vs corporate reinvestment and payout
    ratios (terms-of-trade episodes), raw percentage points."""
    p, pc = OUT / "ep_report_extra.csv".replace(".csv", SFX + ".csv"), OUT / f"ep_corp_ratios{SFX}.csv"
    if not (p.exists() and pc.exists()):
        return
    r, c = pd.read_csv(p), pd.read_csv(pc)
    n_hh, n_co = int(r.n_hh.iloc[0]), int(c.n_corp.iloc[0])
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2))
    line(axes[0], r[(r.sector == "hh") & (r["var"] == "src_hh")], HH, "Household saving rate")
    finish(axes[0], "Percentage points of\ngross disposable income")
    axes[0].set_title(f"(a) Households: household episodes (N = {n_hh})", loc="left")
    line(axes[1], c[c["var"] == "reinv_corp"], CO, "Reinvestment ratio (capital formation / surplus)")
    line(axes[1], c[c["var"] == "pay_corp"], "#7a6aa8", "Net payout ratio (net dividends / surplus)", ls="--")
    finish(axes[1], "Percentage points of\ngross operating surplus")
    axes[1].set_title(f"(b) Corporations: terms-of-trade episodes (N = {n_co})", loc="left")
    for ax in axes:
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi + 0.3 * (hi - lo))
        ax.legend(loc="upper right", fontsize=7)
    fig.suptitle("Figure 4. Household saving propensity and corporate spending propensities", x=0.0, ha="left",
                 fontweight="bold", y=0.99)
    fig.tight_layout()
    fig.text(0.0, -0.06, "Notes: Cumulative change per episode relative to year $-1$, not normalized. Panel (a): household gross saving over "
             "gross disposable income. Panel (b): non-financial corporations' gross capital formation and net dividends paid (paid minus "
             "received), each over gross operating surplus. Shaded areas are 68 percent confidence bands "
             "(country-cluster bootstrap, 499 replications). Same episodes and estimator as Figure 1.",
             fontsize=7, wrap=True, va="top")
    save(fig, "fig4_saving_propensities")


def fig5_diagnostics():
    """Figure 5 (diagnostics, 2×2, point estimates): components behind Figures 1–2, main specification."""
    p = OUT / f"ep_report_diag{SFX}.csv"
    if not p.exists():
        return
    r = pd.read_csv(p)
    panels = [("(a) Corporations: saving and operating surplus", "corp",
               [("sv_corp", "Gross saving", "-"), ("prof", "Gross operating surplus", "--")]),
              ("(b) Corporations: dividends and investment", "corp",
               [("div_net", "Net dividends paid", "-"), ("gcf_corp", "Gross capital formation", "--"), ("inv_corp", "Fixed capital formation", ":")]),
              ("(c) Households: saving, income and consumption", "hh",
               [("sv_hh", "Gross saving", "-"), ("yd_hh", "Disposable income", "--"), ("cons_hh", "Final consumption", ":")]),
              ("(d) Households: investment", "hh", [("gcf_hh", "Gross capital formation", "-")])]
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.6))
    cols = ["#1f3b73", "#b5541c", "#5f8f3e"]
    for ax, (title, s, series) in zip(axes.ravel(), panels):
        for (v, lab, ls), col in zip(series, cols):
            q = r[(r.sector == s) & (r["var"] == v) & (r.h <= HMAX)].sort_values("h")
            ax.plot(q.h, q.ratio, color=col, ls=ls, lw=1.5, marker="o", ms=2.8, label=lab)
        finish(ax, "Response (impact own-sector\nsaving–investment balance = 1)", h0=0)
        ax.set_title(title, loc="left")
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi + 0.45 * (hi - lo))  # 범례 자리 확보
        ax.legend(loc="upper left", fontsize=7, ncol=1)
    fig.suptitle("Figure 5. Diagnostics: components of sectoral saving–investment balances", x=0.0, ha="left",
                 fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.text(0.0, -0.03, "Notes: Point estimates, percent of GDP, normalized by the impact response of the own-sector saving–investment "
             "balance. Panels (a)–(b): terms-of-trade episodes; (c)–(d): household episodes. Same episodes and estimator as Figure 1.",
             fontsize=7, wrap=True, va="top")
    save(fig, "fig5_diagnostics")


def fig6_consumption():
    """Figure 6: nominal growth of household consumption, disposable income and GDP in the two episode types."""
    p = OUT / f"ep_report_growth{SFX}.csv"
    if not p.exists():
        return
    r = pd.read_csv(p)
    n_hh, n_co = int(r.n_hh.iloc[0]), int(r.n_corp.iloc[0])
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2), sharey=True)
    series = [("lcons_hh", "Household consumption", "#1f3b73", "-"), ("lyd_hh", "Household disposable income", "#5f8f3e", ":"),
              ("lgdp", "Nominal GDP", "#b5541c", "--")]
    for ax, s, title in ((axes[0], "hh", f"(a) Household episodes (N = {n_hh})"), (axes[1], "corp", f"(b) Terms-of-trade episodes (N = {n_co})")):
        for v, lab, col, ls in series:
            line(ax, r[(r.sector == s) & (r["var"] == v)], col, lab, ls=ls)
        finish(ax, "Cumulative nominal growth\n(percent, relative to year $-1$)" if s == "hh" else "")
        ax.set_title(title, loc="left")
    axes[0].legend(loc="lower left", fontsize=7)
    fig.suptitle("Figure 6. Household consumption, income and GDP in the two episode types", x=0.0, ha="left", fontweight="bold", y=1.03)
    fig.tight_layout()
    fig.text(0.0, -0.06, "Notes: Cumulative change in the log of nominal household final consumption, household gross disposable income and "
             "nominal GDP (×100) per episode, not normalized; common nominal growth is absorbed by year fixed effects. "
             "Shaded areas are 68 percent confidence bands (country-cluster bootstrap, 499 replications). Same episodes and estimator as Figure 1.",
             fontsize=7, wrap=True, va="top")
    save(fig, "fig6_consumption_growth")


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


def fig_clean():
    """Clean-control (LP-DiD) versions of Figures 1 and 2 from output/ep_clean_controls.csv."""
    p = OUT / "ep_clean_controls.csv"
    if not p.exists():
        return
    r = pd.read_csv(p)
    band = "lo68" in r
    note_cc = (" Clean controls: for each horizon $h$, control observations are restricted to country-years with no large shock "
               "of the same type (top/bottom decile of the episode-defining variable) in the same country during $[t-3, t+h]$.")
    q = lambda s, v: r[(r.sector == s) & (r["var"] == v)].assign(**({} if band else {"lo68": np.nan, "hi68": np.nan}))
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    line(ax, q("hh", "nl_hh"), HH, LAB_HH, band=band, ref=False)
    line(ax, q("corp", "nl_corp"), CO, LAB_CO, ls="--", band=band, ref=False)
    finish(ax, "Saving–investment balance\n(impact response = 1)", h0=0)
    ax.legend(loc="lower left")
    ax.set_title("Figure 1 (clean controls). Persistence of sectoral saving–investment balances", loc="left", fontweight="bold")
    fig.text(0.0, -0.2, "Notes: " + ("Shaded areas are 68 percent confidence bands (country-cluster bootstrap, 499 replications). "
             if band else "Point estimates. ") + "Responses relative to year $-1$, normalized by the impact response." + note_cc
             + " " + SPEC, fontsize=7, wrap=True, va="top", transform=ax.transAxes)
    save(fig, "fig1_si_persistence_clean", sfx="")
    g = lambda v: q("corp", v).sort_values("h").ratio.values[:HMAX + 1]
    nl = g("nl_corp")
    comps = [("Gross operating surplus", g("prof"), "#1f3b73"), ("Net dividends paid (−)", -g("div_net"), "#7a6aa8"),
             ("Gross capital formation (−)", -g("gcf_corp"), "#b5541c")]
    comps.append(("Other", nl - sum(v for _, v, _ in comps), "#a6a6a6"))
    hs = np.arange(len(nl))
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    up, dn = np.zeros(len(hs)), np.zeros(len(hs))
    for lab, v, col in comps:
        ax.bar(hs, v, bottom=np.where(v >= 0, up, dn), width=0.6, color=col, alpha=0.85, label=lab, edgecolor="white", linewidth=0.4)
        up, dn = up + np.clip(v, 0, None), dn + np.clip(v, None, 0)
    ax.plot(hs, nl, color="black", lw=1.5, marker="o", ms=3.5, mfc="white", label="Saving–investment balance")
    finish(ax, "Contribution\n(impact saving–investment balance = 1)", h0=0)
    ax.set_xlim(-0.5, HMAX + 0.5)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, columnspacing=1.2)
    ax.set_title("Figure 2 (clean controls). Decomposition of the corporate saving–investment balance", loc="left", fontweight="bold")
    fig.text(0.0, -0.36, "Notes: Point estimates for terms-of-trade episodes; bars sum to the corporate saving–investment balance "
             "(gross operating surplus − net dividends paid − gross capital formation + other)." + note_cc + " " + SPEC,
             fontsize=7, wrap=True, va="top", transform=ax.transAxes)
    save(fig, "fig2_corporate_decomposition_clean", sfx="")


if __name__ == "__main__":
    fig1()
    fig2()
    fig3()
    fig4()
    fig5_diagnostics()
    fig6_consumption()
    fig1_variants()
    if LAGS:
        fig_clean()
