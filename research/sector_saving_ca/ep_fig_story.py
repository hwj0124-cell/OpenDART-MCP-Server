"""본문용 3패널: (a) 한국의 두 저축 증가 에피소드의 원천, (b) OECD 패널 순대출 반응(가계 저축률 변화 vs 기업 이익 횡재),
(c) 이익 횡재 후 기업 순대출의 조정 경로(기여도).

(a) 2012→2015: OECD 연간 부문계정(ep_korea_sources.annual). H1 2025→H1 2026: 한국은행 분기 자료, 기준안
    (가계 외 부문의 실물–금융 불일치 D를 2025 상반기 GNI 비중으로 고정, 나머지는 잔차; ep_korea_2026_alloc 'none').
(b)(c) 기준 사양(표본 B + 정부 순대출 통제, 999회 국가 군집 부트스트랩) 결과 output/ep_final.csv.
→ output/paper/fig_story.(png|pdf)
"""
import numpy as np
import pandas as pd

import ep_final as ef
import ep_korea_2026 as k
import ep_korea_2026_alloc as al
import ep_korea_sources as ks
import source_shocks as ss
from make_report import OUT


def episodes():
    past, dpast = ks.annual(2012, 2015)
    q, a, fa = k.load()
    b, n, e, dIN, dD, dD_hh = al.pieces(q, a)
    c, dS, dnow, _, _ = al.decompose(b, n, e, dIN, al.allocate("none", dD - dD_hh, None), dD_hh)
    cat = ["Household saving rate", "Household income", "Corporate profits", "Corporate retention", "Government revenue",
           "Government spending", "Financial corporations and other", "Residual"]
    p = past.rename({"Other": "Financial corporations and other"}).reindex(cat).fillna(0.0)
    now = c.rename({"Financial corporations": "Financial corporations and other", "Unallocated": "Residual"}).reindex(cat).fillna(0.0)
    return cat, p, dpast, now, dnow


def figure():
    import fig_paper as fp
    plt = fp.plt
    cat, p, dp, now, dn = episodes()
    r = pd.read_csv(OUT / "ep_final.csv")
    g = lambda key: r[r.key == key].sort_values("h")
    H = ef.H
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.4), gridspec_kw=dict(width_ratios=[1.05, 1, 1]))

    ax = axes[0]
    pal = {"Household saving rate": fp.HH, "Household income": "#7f9cc9", "Corporate profits": fp.CO, "Corporate retention": "#7a5195",
           "Government revenue": "#6a8f3c", "Government spending": "#b3cf8f", "Financial corporations and other": "0.6", "Residual": "0.88"}
    lab = {"Household saving rate": "Household saving rate", "Household income": "Household income", "Corporate profits": "Corporate profits",
           "Corporate retention": "Corporate retention", "Government revenue": "Government disposable income",
           "Government spending": "Government consumption (−)", "Financial corporations and other": "Financial corporations, other",
           "Residual": "Residual (real–financial discrepancy)"}
    est = {"Corporate profits", "Corporate retention", "Government revenue", "Financial corporations and other"}
    for x, (c, d) in enumerate(((p, dp), (now, dn))):
        pos = neg = 0.0
        for kk in cat:
            v = c[kk]
            if abs(v) < 1e-9:
                continue
            hatch = "xx" if kk == "Residual" else ("////" if (x == 1 and kk in est) else None)
            ax.bar(x, v, bottom=pos if v >= 0 else neg, color=pal[kk], width=0.55, edgecolor="white", lw=0.3, hatch=hatch, label=lab[kk])
            pos, neg = (pos + v, neg) if v >= 0 else (pos, neg + v)
        ax.scatter(x, d, color="k", s=24, zorder=5, label="Change in national saving" if x == 0 else None)
        ax.text(x + 0.31, d, f"{d:+.1f}", ha="left", va="center", fontsize=7.5)
    ax.axhline(0, color=".3", lw=.6)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["2012→2015\n(annual)", "H1 2025→H1 2026\n(half-year, estimated)"], fontsize=8)
    ax.set_xlim(-0.5, 2.6)
    ax.set_ylim(-3, 12.5)
    ax.set_ylabel("Percentage points of GNI")
    h_, l_ = ax.get_legend_handles_labels()
    seen = {}
    for hh, ll in zip(h_, l_):
        seen.setdefault(ll, hh)
    ax.legend(seen.values(), seen.keys(), loc="center right", fontsize=6.2, frameon=False)
    ax.set_title("(a) Korea: sources of the rise in national saving", loc="left", fontsize=9.2)

    ax = axes[1]
    for s, cl, l in (("hh", fp.HH, "Households: shift in the saving rate"), ("corp", fp.CO, "Corporations: profit windfall")):
        q = g(s)
        ax.fill_between(q.h, q.lo90, q.hi90, color=cl, alpha=0.08, lw=0)
        ax.fill_between(q.h, q.lo68, q.hi68, color=cl, alpha=0.18, lw=0)
        ax.plot(q.h, q.b, color=cl, lw=2.0, marker="o", ms=3.4, label=l)
    fp.finish(ax, "Net lending (impact = 1)", h0=0)
    ax.set_xlabel("Years after the saving change ($h$)")
    ax.set_xlim(-0.3, H - 0.7)
    ax.set_xticks(range(H))
    ax.set_ylim(-0.75, 1.1)
    ax.legend(loc="upper right", fontsize=7, frameon=False)
    ax.set_title("(b) OECD panel: response of sector net lending", loc="left", fontsize=9.2)

    ax = axes[2]
    ef.bars(ax, g, "corp")
    fp.finish(ax, "Contribution (impact net lending = 1)", h0=0)
    ax.set_xlabel("Years after the profit windfall ($h$)")
    ax.set_xlim(-0.5, H - 0.5)
    ax.set_xticks(range(H))
    ax.legend(loc="upper right", fontsize=6.6, frameon=False, ncol=2)
    ax.set_ylim(-3.2, 3.2)
    ax.set_title("(c) OECD panel: how corporate net lending adjusts", loc="left", fontsize=9.2)

    fig.suptitle("Korea's saving surge then and now, and what the panel says about persistence", x=0.0, ha="left", fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.text(0.0, -0.01,
             "Notes: (a) Gross saving, percent of GNI. Corporate saving splits into a profit component (change in operating surplus at the earlier "
             "retention ratio) and a retention component; household saving into a saving-rate and an income component; government saving into "
             "disposable income and minus final consumption. 2012→2015: OECD annual sector accounts, cumulated over the years. H1 2026: Bank of "
             "Korea quarterly accounts; national, household and government-consumption figures are published, hatched components are estimated: "
             "sector saving = investment + flow-of-funds net lending + the gap between the two accounts held at its H1 2025 share of GNI (households: "
             "published saving); the part of the observed widening of that gap not attributed to households is shown as a residual. "
             "(b)–(c) " + ef.note(r) + " Local projections with lags of own net lending, both saving components, GDP growth and government net "
             "lending, and country and year fixed effects; normalized by the impact response. (b) 68 and 90 percent bands, country-cluster "
             "bootstrap (999 replications). (c) Cumulative changes in the items of the corporate account; outlays enter with a negative sign; "
             "'Other' is the remainder. Sources: OECD; Bank of Korea; authors' calculations.", fontsize=6.7, wrap=True, va="top")
    fp.save(fig, "fig_story", sfx="")


def figure2():
    """2×2: (a) 에피소드 (b) 순대출 반응 + 반감기 (c) 누적 순대출 (d) 기업 조정. 누적·반감기는 ep_cum.csv."""
    import fig_paper as fp
    plt = fp.plt
    cat, p, dp, now, dn = episodes()
    r = pd.read_csv(OUT / "ep_final.csv")
    cu = pd.read_csv(OUT / "ep_cum.csv")
    g = lambda key: r[r.key == key].sort_values("h")
    H = ef.H
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 8.0))
    axa, axb, axc, axd = axes.flat
    # (a)
    pal = {"Household saving rate": fp.HH, "Household income": "#7f9cc9", "Corporate profits": fp.CO, "Corporate retention": "#7a5195",
           "Government revenue": "#6a8f3c", "Government spending": "#b3cf8f", "Financial corporations and other": "0.6", "Residual": "0.88"}
    lab = {"Household saving rate": "Household saving rate", "Household income": "Household income", "Corporate profits": "Corporate profits",
           "Corporate retention": "Corporate retention", "Government revenue": "Government disposable income",
           "Government spending": "Government consumption (−)", "Financial corporations and other": "Financial corporations, other",
           "Residual": "Residual (real–financial discrepancy)"}
    est = {"Corporate profits", "Corporate retention", "Government revenue", "Financial corporations and other"}
    for x, (c, d) in enumerate(((p, dp), (now, dn))):
        pos = neg = 0.0
        for kk in cat:
            v = c[kk]
            if abs(v) < 1e-9:
                continue
            hatch = "xx" if kk == "Residual" else ("////" if (x == 1 and kk in est) else None)
            axa.bar(x, v, bottom=pos if v >= 0 else neg, color=pal[kk], width=0.55, edgecolor="white", lw=0.3, hatch=hatch, label=lab[kk])
            pos, neg = (pos + v, neg) if v >= 0 else (pos, neg + v)
        axa.scatter(x, d, color="k", s=24, zorder=5, label="Change in national saving" if x == 0 else None)
        axa.text(x + 0.31, d, f"{d:+.1f}", ha="left", va="center", fontsize=7.5)
    axa.axhline(0, color=".3", lw=.6)
    axa.set_xticks([0, 1])
    axa.set_xticklabels(["2012→2015\n(annual)", "H1 2025→H1 2026\n(half-year, estimated)"], fontsize=8)
    axa.set_xlim(-0.5, 2.7)
    axa.set_ylim(-3, 12.5)
    axa.set_ylabel("Percentage points of GNI")
    seen = {}
    for hh, ll in zip(*axa.get_legend_handles_labels()):
        seen.setdefault(ll, hh)
    axa.legend(seen.values(), seen.keys(), loc="center right", fontsize=6.2, frameon=False)
    axa.set_title("(a) Korea: sources of the rise in national saving", loc="left", fontsize=9.2)
    # (b)
    hl = cu[cu.stat == "half_life"].set_index("sector")
    for s, cl, l in (("hh", fp.HH, "Households: shift in the saving rate"), ("corp", fp.CO, "Corporations: profit windfall")):
        q = g(s)
        axb.fill_between(q.h, q.lo90, q.hi90, color=cl, alpha=0.08, lw=0)
        axb.fill_between(q.h, q.lo68, q.hi68, color=cl, alpha=0.18, lw=0)
        axb.plot(q.h, q.b, color=cl, lw=2.0, marker="o", ms=3.4, label=l)
        b_, lo, hi = hl.loc[s, ["b", "lo90", "hi90"]]
        axb.errorbar(b_, 0.5, xerr=[[b_ - lo], [hi - b_]], fmt="D", color=cl, ms=5, capsize=3, lw=1.2, zorder=6)
        axb.text(b_ + (0.12 if s == "hh" else -0.1), 0.56 if s == "hh" else 0.42, f"half-life {b_:.1f} yr", color=cl, fontsize=7.4,
                 ha="left" if s == "hh" else "right")
    axb.axhline(0.5, color="0.6", lw=0.6, ls=":")
    fp.finish(axb, "Net lending (impact = 1)", h0=0)
    axb.set_xlabel("Years after the saving change ($h$)")
    axb.set_xlim(-0.3, H - 0.7)
    axb.set_xticks(range(H))
    axb.set_ylim(-0.75, 1.1)
    axb.legend(loc="upper right", fontsize=7, frameon=False)
    axb.set_title("(b) OECD panel: response of sector net lending", loc="left", fontsize=9.2)
    # (c)
    for s, cl, l in (("hh", fp.HH, "Households: shift in the saving rate"), ("corp", fp.CO, "Corporations: profit windfall")):
        q = cu[(cu.sector == s) & cu.stat.str.startswith("cum")].copy()
        q["h"] = q.stat.str[3:].astype(int)
        axc.fill_between(q.h, q.lo90, q.hi90, color=cl, alpha=0.08, lw=0)
        axc.fill_between(q.h, q.lo68, q.hi68, color=cl, alpha=0.18, lw=0)
        axc.plot(q.h, q.b, color=cl, lw=2.0, marker="o", ms=3.4, label=l)
        axc.text(H - 1 + 0.12, q.b.iloc[-1], f"{q.b.iloc[-1]:.1f}", color=cl, fontsize=8, va="center")
    fp.finish(axc, "Cumulative net lending (impact = 1)", h0=0)
    axc.set_xlabel("Years after the saving change ($h$)")
    axc.set_xlim(-0.3, H - 0.4)
    axc.set_xticks(range(H))
    axc.set_ylim(-0.8, 4.0)
    axc.legend(loc="upper left", fontsize=7, frameon=False)
    axc.set_title("(c) OECD panel: net lending accumulated to year $h$", loc="left", fontsize=9.2)
    # (d)
    ef.bars(axd, g, "corp")
    fp.finish(axd, "Contribution (impact net lending = 1)", h0=0)
    axd.set_xlabel("Years after the profit windfall ($h$)")
    axd.set_xlim(-0.5, H - 0.5)
    axd.set_xticks(range(H))
    axd.legend(loc="upper right", fontsize=6.6, frameon=False, ncol=2)
    axd.set_ylim(-3.2, 3.2)
    axd.set_title("(d) OECD panel: how corporate net lending adjusts", loc="left", fontsize=9.2)
    fig.suptitle("Korea's saving surge then and now, and what the panel says about persistence", x=0.0, ha="left", fontweight="bold", y=1.0)
    fig.tight_layout()
    fig.text(0.0, -0.005,
             "Notes: (a) Gross saving, percent of GNI. Corporate saving splits into a profit component (change in operating surplus at the earlier "
             "retention ratio) and a retention component; household saving into a saving-rate and an income component; government saving into "
             "disposable income and minus final consumption. 2012→2015: OECD annual sector accounts, cumulated over the years. H1 2026: Bank of "
             "Korea quarterly accounts; hatched components are estimated (sector saving = investment + flow-of-funds net lending + the gap between "
             "the two accounts at its H1 2025 share of GNI; households: published saving); the unattributed widening of that gap is a residual. "
             "(b)–(d) " + ef.note(r) + " Local projections with lags of own net lending, both saving components, GDP growth and government net "
             "lending, and country and year fixed effects; normalized by the impact response. Bands: 68 and 90 percent, country-cluster bootstrap "
             "(999 replications). (b) Diamonds: half-life, the first year the response falls below one half (linear interpolation), with 90 "
             "percent intervals. (c) Sum of the responses from year 0 to year h: the net lending, and hence the current-account surplus, that a "
             "unit saving change generates over that span. (d) Cumulative changes in the items of the corporate account; outlays enter with a "
             "negative sign; 'Other' is the remainder. Sources: OECD; Bank of Korea; authors' calculations.", fontsize=6.7, wrap=True, va="top")
    fp.save(fig, "fig_story2", sfx="")


def figure3(hmax=5, name="fig_story3"):
    """2×2: (a) 에피소드 (b) 순대출 반응 (c) 기업 분해 (d) 가계 분해, h = 0…hmax."""
    import fig_paper as fp
    plt = fp.plt
    cat, p, dp, now, dn = episodes()
    r = pd.read_csv(OUT / "ep_final.csv")
    r = r[r.h <= hmax]
    g = lambda key: r[r.key == key].sort_values("h")
    H = hmax + 1
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 8.0))
    axa, axb, axc, axd = axes.flat
    pal = {"Household saving rate": fp.HH, "Household income": "#7f9cc9", "Corporate profits": fp.CO, "Corporate retention": "#7a5195",
           "Government revenue": "#6a8f3c", "Government spending": "#b3cf8f", "Financial corporations and other": "0.6", "Residual": "0.88"}
    lab = {"Household saving rate": "Household saving rate", "Household income": "Household income", "Corporate profits": "Corporate profits",
           "Corporate retention": "Corporate retention", "Government revenue": "Government disposable income",
           "Government spending": "Government consumption (−)", "Financial corporations and other": "Financial corporations, other",
           "Residual": "Residual (real–financial discrepancy)"}
    est = {"Corporate profits", "Corporate retention", "Government revenue", "Financial corporations and other"}
    for x, (c, d) in enumerate(((p, dp), (now, dn))):
        pos = neg = 0.0
        for kk in cat:
            v = c[kk]
            if abs(v) < 1e-9:
                continue
            hatch = "xx" if kk == "Residual" else ("////" if (x == 1 and kk in est) else None)
            axa.bar(x, v, bottom=pos if v >= 0 else neg, color=pal[kk], width=0.55, edgecolor="white", lw=0.3, hatch=hatch, label=lab[kk])
            pos, neg = (pos + v, neg) if v >= 0 else (pos, neg + v)
        axa.scatter(x, d, color="k", s=24, zorder=5, label="Change in national saving" if x == 0 else None)
        axa.text(x + 0.31, d, f"{d:+.1f}", ha="left", va="center", fontsize=7.5)
    axa.axhline(0, color=".3", lw=.6)
    axa.set_xticks([0, 1])
    axa.set_xticklabels(["2012→2015\n(annual)", "H1 2025→H1 2026\n(half-year, estimated)"], fontsize=8)
    axa.set_xlim(-0.5, 2.7)
    axa.set_ylim(-3, 12.5)
    axa.set_ylabel("Percentage points of GNI")
    seen = {}
    for hh, ll in zip(*axa.get_legend_handles_labels()):
        seen.setdefault(ll, hh)
    axa.legend(seen.values(), seen.keys(), loc="center right", fontsize=6.2, frameon=False)
    axa.set_title("(a) Korea: sources of the rise in national saving", loc="left", fontsize=9.2)

    for s, cl, l in (("hh", fp.HH, "Households: shift in the saving rate"), ("corp", fp.CO, "Corporations: profit windfall")):
        q = g(s)
        axb.fill_between(q.h, q.lo90, q.hi90, color=cl, alpha=0.08, lw=0)
        axb.fill_between(q.h, q.lo68, q.hi68, color=cl, alpha=0.18, lw=0)
        axb.plot(q.h, q.b, color=cl, lw=2.0, marker="o", ms=3.4, label=l)
    fp.finish(axb, "Net lending (impact = 1)", h0=0)
    axb.set_xlabel("Years after the saving change ($h$)")
    axb.set_xlim(-0.3, H - 0.7)
    axb.set_xticks(range(H))
    axb.set_ylim(-0.6, 1.1)
    axb.legend(loc="upper right", fontsize=7, frameon=False)
    axb.set_title("(b) OECD panel: response of sector net lending", loc="left", fontsize=9.2)

    def bars(ax, s):
        hs = np.arange(H)
        pos, neg = np.zeros(H), np.zeros(H)
        for v in ef.ITEMS[s] + ["other"]:
            b = g(f"{s}_{v}").b.to_numpy()
            ax.bar(hs, b, bottom=np.where(b >= 0, pos, neg), color=ef.PAL[v], width=0.7, label=ef.LAB[v], edgecolor="white", lw=0.3)
            pos, neg = pos + np.clip(b, 0, None), neg + np.clip(b, None, 0)
        ax.plot(hs, g(f"{s}_nl").b, color="k", lw=1.8, marker="o", ms=3.4, label="Net lending")
    bars(axc, "corp")
    fp.finish(axc, "Contribution (impact net lending = 1)", h0=0)
    axc.set_xlabel("Years after the profit windfall ($h$)")
    axc.set_ylim(-3.2, 3.2)
    axc.legend(loc="upper right", fontsize=6.6, frameon=False, ncol=2)
    axc.set_title("(c) OECD panel: corporate net lending after a profit windfall", loc="left", fontsize=9.2)
    bars(axd, "hh")
    fp.finish(axd, "Contribution (impact net lending = 1)", h0=0)
    axd.set_xlabel("Years after the shift in the saving rate ($h$)")
    axd.set_ylim(-0.6, 1.4)
    for ax in (axc, axd):
        ax.set_xlim(-0.5, H - 0.5)
        ax.set_xticks(range(H))
    axd.legend(loc="upper right", fontsize=6.6, frameon=False, ncol=2)
    axd.set_title("(d) OECD panel: household net lending after a saving-rate shift", loc="left", fontsize=9.2)
    fig.suptitle("Korea's saving surge then and now, and how each kind of saving fades", x=0.0, ha="left", fontweight="bold", y=1.0)
    fig.tight_layout()
    fig.text(0.0, -0.005,
             "Notes: (a) Gross saving, percent of GNI. Corporate saving splits into a profit component (change in operating surplus at the earlier "
             "retention ratio) and a retention component; household saving into a saving-rate and an income component; government saving into "
             "disposable income and minus final consumption. 2012→2015: OECD annual sector accounts, cumulated over the years. H1 2026: Bank of "
             "Korea quarterly accounts; hatched components are estimated (sector saving = investment + flow-of-funds net lending + the gap between "
             "the two accounts at its H1 2025 share of GNI; households: published saving); the unattributed widening of that gap is a residual. "
             "(b)–(d) " + ef.note(r) + " Local projections with lags of own net lending, both saving components, GDP growth and government net "
             "lending, and country and year fixed effects; normalized by the impact response of net lending. (b) 68 and 90 percent bands, "
             "country-cluster bootstrap (999 replications). (c)–(d) Cumulative changes in the items of each sector's account, percent of GNI; "
             "outlays (dividends, taxes, interest, consumption, investment) enter with a negative sign, so a fall in consumption is a positive "
             "contribution; 'Other' is the remainder. Sources: OECD; Bank of Korea; authors' calculations.", fontsize=6.7, wrap=True, va="top")
    if name:
        fp.save(fig, name, sfx="")


def figure4(hmax=5):
    """fig_story3과 같되 (c)(d)를 항목별 선(수준 변화, 자연 부호, 68% 구간)으로."""
    import fig_paper as fp
    plt = fp.plt
    figure3(hmax, name=None)
    fig = plt.gcf()
    axc, axd = fig.axes[2], fig.axes[3]
    r = pd.read_csv(OUT / "ep_final.csv")
    r = r[r.h <= hmax]
    g = lambda key: r[r.key == key].sort_values("h")
    H = hmax + 1
    for ax, s, items, ttl, xl in ((axc, "corp", ["prof", "gcf_corp", "div_net"], "(c) OECD panel: corporations after a profit windfall",
                                   "Years after the profit windfall ($h$)"),
                                  (axd, "hh", ["yd_hh", "cons_hh", "gcf_hh"], "(d) OECD panel: households after a saving-rate shift",
                                   "Years after the shift in the saving rate ($h$)")):
        ax.clear()
        for v in items:
            q = g(f"{s}_{v}")
            sg = ef.SIGN[v]
            lo, hi = sorted([sg * q.lo68.to_numpy(), sg * q.hi68.to_numpy()], key=lambda a: a.mean())
            ax.fill_between(q.h, np.minimum(sg * q.lo68, sg * q.hi68), np.maximum(sg * q.lo68, sg * q.hi68), color=ef.PAL[v], alpha=0.16, lw=0)
            ax.plot(q.h, sg * q.b, color=ef.PAL[v], lw=1.9, marker="o", ms=3.2, label=ef.LAB[v].replace(" (−)", ""))
        q = g(f"{s}_nl")
        ax.plot(q.h, q.b, color="k", lw=1.6, ls="--", marker="s", ms=3, label="Net lending")
        fp.finish(ax, "Cumulative change, percent of GNI\n(impact net lending = 1)" if s == "corp" else "", h0=0)
        ax.set_xlim(-0.3, H - 0.7)
        ax.set_xticks(range(H))
        ax.set_xlabel(xl)
        ax.set_title(ttl, loc="left", fontsize=9.2)
        ax.legend(loc="upper right", fontsize=7, frameon=False)
    axc.set_ylim(-0.6, 2.4)
    axd.set_ylim(-0.8, 1.15)
    fig.texts[-1].set_text(fig.texts[-1].get_text().replace(
        "(c)–(d) Cumulative changes in the items of each sector's account, percent of GNI; outlays (dividends, taxes, interest, consumption, "
        "investment) enter with a negative sign, so a fall in consumption is a positive contribution; 'Other' is the remainder.",
        "(c)–(d) Cumulative changes from year −1 in items of each sector's account, percent of GNI, scaled by the impact response of the "
        "sector's net lending; shaded: 68 percent bands. Corporate taxes and interest, small, are not shown."))
    fp.save(fig, "fig_story4", sfx="")


CAT_G = ["Household saving rate", "Household income", "Corporate profits", "Corporate retention", "Government saving",
         "Financial corporations and other", "Residual"]


def episodes_gdp():
    """GDP 대비, 정부는 저축 합계. 2012→2015: OECD 패널(이미 GDP 대비), H1 2026: 한국은행 분기, 분모를 명목 GDP로."""
    d = ss.prepare(pd.read_csv(OUT.parent / "data" / "panel.csv"))
    kr = d[d.country == "KOR"].set_index("year").sort_index()
    a, b = 2012, 2015
    yrs = range(a + 1, b + 1)
    past = {"Corporate profits": kr.loc[yrs, "x_P"].sum(), "Corporate retention": kr.loc[yrs, "x_R"].sum(),
            "Household saving rate": kr.loc[yrs, "x_H"].sum(), "Household income": kr.loc[yrs, "x_Y"].sum(),
            "Government saving": kr.loc[b, "sv_gov"] - kr.loc[a, "sv_gov"]}
    dpast = kr.loc[b, "sv_nat"] - kr.loc[a, "sv_nat"]
    past["Financial corporations and other"] = dpast - sum(past.values())
    q, a_, fa = k.load()
    g = pd.read_csv(k.RAW / "bok_gdp_exp.csv", comment="#").set_index("period")
    qs = [x for x in q.index if x in g.index]
    q.loc[qs, "gni"] = g.loc[qs, "gdp"]              # 분모를 명목 GDP로 교체(열 이름만 gni)
    bq, n, e, dIN, dD, dD_hh = al.pieces(q, a_)
    c, dS, dnow, _, _ = al.decompose(bq, n, e, dIN, al.allocate("none", dD - dD_hh, None), dD_hh)
    now = {"Corporate profits": c["Corporate profits"], "Corporate retention": c["Corporate retention"],
           "Household saving rate": c["Household saving rate"], "Household income": c["Household income"],
           "Government saving": c["Government revenue"] + c["Government spending"],
           "Financial corporations and other": c["Financial corporations"], "Residual": c["Unallocated"]}
    return pd.Series(past).reindex(CAT_G).fillna(0.0), dpast, pd.Series(now).reindex(CAT_G), dnow


def figure5(hmax=5):
    """fig_story4의 GDP판: (a) 정부 합계, 모든 수치 GDP 대비."""
    import fig_paper as fp
    plt = fp.plt
    p, dp, now, dn = episodes_gdp()
    r = pd.read_csv(OUT / "ep_final.csv")
    r = r[r.h <= hmax]
    g = lambda key: r[r.key == key].sort_values("h")
    H = hmax + 1
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 8.0))
    axa, axb, axc, axd = axes.flat
    pal = {"Household saving rate": fp.HH, "Household income": "#7f9cc9", "Corporate profits": fp.CO, "Corporate retention": "#7a5195",
           "Government saving": "#6a8f3c", "Financial corporations and other": "0.6", "Residual": "0.88"}
    lab = {"Financial corporations and other": "Financial corporations, other", "Residual": "Residual (real–financial discrepancy)"}
    est = {"Corporate profits", "Corporate retention", "Government saving", "Financial corporations and other"}
    for x, (c, d) in enumerate(((p, dp), (now, dn))):
        pos = neg = 0.0
        for kk_ in CAT_G:
            v = c[kk_]
            if abs(v) < 1e-9:
                continue
            hatch = "xx" if kk_ == "Residual" else ("////" if (x == 1 and kk_ in est) else None)
            axa.bar(x, v, bottom=pos if v >= 0 else neg, color=pal[kk_], width=0.55, edgecolor="white", lw=0.3, hatch=hatch,
                    label=lab.get(kk_, kk_))
            pos, neg = (pos + v, neg) if v >= 0 else (pos, neg + v)
        axa.scatter(x, d, color="k", s=24, zorder=5, label="Change in national saving" if x == 0 else None)
        axa.text(x + 0.31, d, f"{d:+.1f}", ha="left", va="center", fontsize=7.5)
    axa.axhline(0, color=".3", lw=.6)
    axa.set_xticks([0, 1])
    axa.set_xticklabels(["2012→2015\n(annual)", "H1 2025→H1 2026\n(half-year, estimated)"], fontsize=8)
    axa.set_xlim(-0.5, 2.7)
    axa.set_ylim(-3, 12.5)
    axa.set_ylabel("Percentage points of GDP")
    seen = {}
    for hh, ll in zip(*axa.get_legend_handles_labels()):
        seen.setdefault(ll, hh)
    axa.legend(seen.values(), seen.keys(), loc="center right", fontsize=6.4, frameon=False)
    axa.set_title("(a) Korea: sources of the rise in national saving", loc="left", fontsize=9.2)
    for s, cl, l in (("hh", fp.HH, "Households: shift in the saving rate"), ("corp", fp.CO, "Corporations: profit windfall")):
        q = g(s)
        axb.fill_between(q.h, q.lo90, q.hi90, color=cl, alpha=0.08, lw=0)
        axb.fill_between(q.h, q.lo68, q.hi68, color=cl, alpha=0.18, lw=0)
        axb.plot(q.h, q.b, color=cl, lw=2.0, marker="o", ms=3.4, label=l)
    fp.finish(axb, "Net lending (impact = 1)", h0=0)
    axb.set_xlabel("Years after the saving change ($h$)")
    axb.set_xlim(-0.3, H - 0.7)
    axb.set_xticks(range(H))
    axb.set_ylim(-0.6, 1.1)
    axb.legend(loc="upper right", fontsize=7, frameon=False)
    axb.set_title("(b) OECD panel: response of sector net lending", loc="left", fontsize=9.2)
    for ax, s, items, ttl, xl in ((axc, "corp", ["prof", "gcf_corp", "div_net"], "(c) OECD panel: corporations after a profit windfall",
                                   "Years after the profit windfall ($h$)"),
                                  (axd, "hh", ["yd_hh", "cons_hh", "gcf_hh"], "(d) OECD panel: households after a saving-rate shift",
                                   "Years after the shift in the saving rate ($h$)")):
        for v in items:
            q = g(f"{s}_{v}")
            sg = ef.SIGN[v]
            ax.fill_between(q.h, np.minimum(sg * q.lo68, sg * q.hi68), np.maximum(sg * q.lo68, sg * q.hi68), color=ef.PAL[v], alpha=0.16, lw=0)
            ax.plot(q.h, sg * q.b, color=ef.PAL[v], lw=1.9, marker="o", ms=3.2, label=ef.LAB[v].replace(" (−)", ""))
        q = g(f"{s}_nl")
        ax.plot(q.h, q.b, color="k", lw=1.6, ls="--", marker="s", ms=3, label="Net lending")
        fp.finish(ax, "Cumulative change, percent of GDP\n(impact net lending = 1)" if s == "corp" else "", h0=0)
        ax.set_xlim(-0.3, H - 0.7)
        ax.set_xticks(range(H))
        ax.set_xlabel(xl)
        ax.set_title(ttl, loc="left", fontsize=9.2)
        ax.legend(loc="upper right", fontsize=7, frameon=False)
    axc.set_ylim(-0.6, 2.4)
    axd.set_ylim(-0.8, 1.15)
    fig.suptitle("Korea's saving surge then and now, and how each kind of saving fades", x=0.0, ha="left", fontweight="bold", y=1.0)
    fig.tight_layout()
    fig.text(0.0, -0.005,
             "Notes: All shares are percent of nominal GDP. (a) Corporate saving splits into a profit component (change in operating surplus at the "
             "earlier retention ratio) and a retention component, household saving into a saving-rate and an income component; government saving "
             "is shown as a total. 2012→2015: OECD annual sector accounts, cumulated over the years. H1 2026: Bank of Korea quarterly accounts; "
             "hatched components are estimated (sector saving = investment + flow-of-funds net lending + the gap between the two accounts at its "
             "H1 2025 share of GDP; households: published saving); the unattributed widening of that gap is a residual. (b)–(d) " + ef.note(r) +
             " Local projections with lags of own net lending, both saving components, GDP growth and government net lending, and country and "
             "year fixed effects; normalized by the impact response of the sector's net lending. (b) 68 and 90 percent bands, country-cluster "
             "bootstrap (999 replications). (c)–(d) Cumulative changes from year −1 in items of each sector's account; shaded: 68 percent bands. "
             "Corporate taxes and interest, small, are not shown. Sources: OECD; Bank of Korea; authors' calculations.",
             fontsize=6.7, wrap=True, va="top")
    fp.save(fig, "fig_story5", sfx="")
    return p, dp, now, dn


if __name__ == "__main__":
    cat, p, dp, now, dn = episodes()
    print(pd.DataFrame({"2012-2015": p, "H1 2026": now}).round(2).to_string(), "\nnational", round(dp, 2), round(dn, 2))
    figure()
    figure2()
    figure3()
    figure4()
    p5, dp5, n5, dn5 = figure5()
    print(pd.DataFrame({'2012-2015 (GDP)': p5, 'H1 2026 (GDP)': n5}).round(2).to_string(), '\nnational', round(dp5, 2), round(dn5, 2))
