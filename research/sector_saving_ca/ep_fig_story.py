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


if __name__ == "__main__":
    cat, p, dp, now, dn = episodes()
    print(pd.DataFrame({"2012-2015": p, "H1 2026": now}).round(2).to_string(), "\nnational", round(dp, 2), round(dn, 2))
    figure()
    figure2()
