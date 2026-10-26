"""현재 기준 사양: 이익 횡재(기업) vs 저축 행태 변화(가계), h = 0…5.

  n^j_{t+h} − n^j_{t−1} = β_h x^j_t + γ_h z^j_t + Σ_{l=1,2}[n^j_{t−l}, x^j_{t−l}, z^j_{t−l}, g_{t−l}] + 국가·연도 FE
  기업: x = x_P(이익 횡재), z = x_R / 가계: x = x_H(저축률), z = x_Y. 2단계 WLS. 정부·다른 부문 통제 없음.
표본: 두 부문 순대출이 h = 0…5 모두 관측, 시작 연도 ≤ 2019. 구간: 국가 군집 델타 방법.
그림 1: 두 부문 순대출 경로와 차이. 그림 2: 기업 순대출 분해(영업잉여 − 순배당 − 법인세 − 이자 − 총자본형성 + 기타).
그림 3: 하위 표본(1998년 이후 시작, 2020년 이전에 끝나는 창, 1998–2007 시작, 2008–2019 시작).
→ output/ep_baseline.csv, output/ep_baseline_sub.csv, output/paper/fig_base_persistence, fig_base_decomp, fig_base_subsamples (png|pdf)
"""
import sys

import numpy as np
import pandas as pd

import ep_windfall_behavior as wb

wb.H = 6
import ep_wb_extra as ex  # noqa: E402

ex.H = 6
from make_report import OUT  # noqa: E402

H, SEC = 6, wb.SEC
COMPS = ["prof", "div_net", "tax_paid", "int_paid", "gcf_corp"]
SIGN = wb.SIGN
SUBS = {"Full sample": lambda q: q, "Start years 1998 onward": lambda q: q[q.year >= 1998],
        "Windows ending before 2020": lambda q: q[q.year + H - 1 <= 2019],
        "Start years 1998–2007": lambda q: q[q.year.between(1998, 2007)], "Start years 2008–2019": lambda q: q[q.year >= 2008]}


def ycols(v):
    return [f"{v}_h{h}" for h in range(H)]


def estimate(q, decomp=True):
    q = wb.weights(q.copy())
    D = ex.Delta()
    for s, m in SEC.items():
        D.add(s, *ex.design(q, [m["x"], m["z"]], wb.ctrl(s), f"w_{s}", ycols(m["y"])))
    if decomp:
        D.add("dc", *ex.design(q, [SEC["corp"]["x"], SEC["corp"]["z"]], wb.ctrl("corp"), "w_corp",
                               sum((ycols(v) for v in ["nl_corp"] + COMPS), [])))
    D.finish()
    rows = []

    def emit(key, f):
        for h in range(H):
            b, se = D.fn(lambda d, h=h: f(d, h))
            rows.append(dict(key=key, h=h, b=b, se=se, lo68=b - se, hi68=b + se, lo90=b - 1.645 * se, hi90=b + 1.645 * se))
    for s, m in SEC.items():
        emit(s, lambda d, h, s=s, y=m["y"]: d[f"{s}|{y}_h{h}"] / d[f"{s}|{y}_h0"])
    emit("diff", lambda d, h: d[f"hh|nl_hh_h{h}"] / d["hh|nl_hh_h0"] - d[f"corp|nl_corp_h{h}"] / d["corp|nl_corp_h0"])
    if decomp:
        emit("dc_nl_corp", lambda d, h: d[f"dc|nl_corp_h{h}"] / d["dc|nl_corp_h0"])
        for v in COMPS:
            emit(f"dc_{v}", lambda d, h, v=v: SIGN[v] * d[f"dc|{v}_h{h}"] / d["dc|nl_corp_h0"])
        emit("dc_other", lambda d, h: d[f"dc|nl_corp_h{h}"] / d["dc|nl_corp_h0"]
             - sum(SIGN[v] * d[f"dc|{v}_h{h}"] for v in COMPS) / d["dc|nl_corp_h0"])
    avg = []
    for key, f in (("corp", lambda d, lo, hi: np.mean([d[f"corp|nl_corp_h{h}"] for h in range(lo, hi)]) / d["corp|nl_corp_h0"]),
                   ("hh", lambda d, lo, hi: np.mean([d[f"hh|nl_hh_h{h}"] for h in range(lo, hi)]) / d["hh|nl_hh_h0"]),
                   ("diff", lambda d, lo, hi: np.mean([d[f"hh|nl_hh_h{h}"] / d["hh|nl_hh_h0"] - d[f"corp|nl_corp_h{h}"] / d["corp|nl_corp_h0"]
                                                       for h in range(lo, hi)]))):
        for lab, lo, hi in (("1-2", 1, 3), ("3-5", 3, H)):
            b, se = D.fn(lambda d: f(d, lo, hi))
            avg.append(dict(key=key, years=lab, b=b, se=se, lo90=b - 1.645 * se, hi90=b + 1.645 * se))
    return pd.DataFrame(rows), pd.DataFrame(avg), len(q), q.country.nunique(), q.year.min(), q.year.max()


def run():
    _, q = wb.data()
    r, a, n, k, y0, y1 = estimate(q)
    r = r.assign(n=n, countries=k, first=y0, last=y1)
    r.to_csv(OUT / "ep_baseline.csv", index=False)
    a.to_csv(OUT / "ep_baseline_avg.csv", index=False)
    print(f"full sample: N = {n}, {k} economies, start years {y0}–{y1}")
    print(a.round(2).to_string(index=False))
    subs = []
    for name, f in SUBS.items():
        rs, asub, n, k, y0, y1 = estimate(f(q), decomp=False)
        subs.append(rs.assign(sub=name, n=n, countries=k, first=y0, last=y1))
        print(name, n, k, y0, y1, asub.set_index(["key", "years"]).b.round(2).to_dict())
    s = pd.concat(subs)
    s.to_csv(OUT / "ep_baseline_sub.csv", index=False)
    return r, s


def figs(r, s):
    import fig_paper as fp
    plt = fp.plt
    g = lambda k: r[r.key == k].sort_values("h")
    n, k, y0, y1 = (int(r[c].iloc[0]) for c in ("n", "countries", "first", "last"))
    xl = "Years after the saving change ($h$)"

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.3))
    ax = axes[0]
    for key, cl, lab in (("hh", fp.HH, "Household S–I after a shift in the saving rate"), ("corp", fp.CO, "Corporate S–I after a profit windfall")):
        q = g(key)
        ax.fill_between(q.h, q.lo68, q.hi68, color=cl, alpha=0.16, lw=0)
        ax.plot(q.h, q.b, color=cl, lw=1.9, marker="o", ms=3.4, label=lab)
    fp.finish(ax, "Net lending (impact = 1)", h0=0)
    ax.set_xlabel(xl)
    ax.set_title("(a) Net lending", loc="left", fontsize=9.5)
    ax.set_ylim(-0.5, 1.1)
    ax.legend(loc="upper right", fontsize=7.2, frameon=False)
    ax = axes[1]
    q = g("diff")
    ax.fill_between(q.h, q.lo90, q.hi90, color="0.6", alpha=0.2, lw=0, label="90% band")
    ax.fill_between(q.h, q.lo68, q.hi68, color="0.35", alpha=0.25, lw=0, label="68% band")
    ax.plot(q.h, q.b, color="0.1", lw=1.9, marker="o", ms=3.4, label="Household minus corporate")
    fp.finish(ax, "Difference", h0=0)
    ax.set_xlabel(xl)
    ax.set_title("(b) Difference", loc="left", fontsize=9.5)
    ax.set_ylim(-0.5, 1.1)
    ax.legend(loc="upper left", fontsize=7.2, frameon=False)
    fig.suptitle("A profit windfall fades faster than a shift in household saving behavior", x=0.0, ha="left", fontweight="bold", y=1.03)
    fig.tight_layout()
    fig.text(0.0, -0.05, f"Notes: {n} country-years, {k} economies, start years {y0}–{y1}. Local projections of each sector's net lending on the profit-windfall "
             "(corporations) or saving-rate (households) component of its saving change and the remaining component, with two lags of the sector's net "
             "lending (levels), both components and GDP growth, and country and year fixed effects; two-step weighted least squares. Normalized by the "
             "impact response. Bands: country-cluster delta method.", fontsize=7, wrap=True, va="top")
    for ax in axes:
        ax.set_xlim(-0.3, H - 0.7); ax.set_xticks(range(H))
    fp.save(fig, "fig_base_persistence", sfx="")

    pal = {"prof": "#c98a1b", "div_net": "#7a5195", "tax_paid": "#bc5090", "int_paid": "#ef8e9c", "gcf_corp": "#2b6c8f", "other": "0.75"}
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.5))
    ax = axes[0]
    hs = np.arange(H)
    pos, neg = np.zeros(H), np.zeros(H)
    for v in COMPS + ["other"]:
        b = g(f"dc_{v}").b.to_numpy()
        ax.bar(hs, b, bottom=np.where(b >= 0, pos, neg), color=pal[v], width=0.7, label=wb.LAB[v], edgecolor="white", lw=0.3)
        pos, neg = pos + np.clip(b, 0, None), neg + np.clip(b, None, 0)
    ax.plot(hs, g("dc_nl_corp").b, color="k", lw=1.8, marker="o", ms=3.5, label="Net lending")
    fp.finish(ax, "Cumulative change (impact net lending = 1)", h0=0)
    ax.set_xlabel(xl)
    ax.set_title("(a) Contributions to corporate net lending", loc="left", fontsize=9.3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), fontsize=6.8, frameon=False, ncol=3)
    ax = axes[1]
    for v in ("prof", "gcf_corp", "div_net"):
        q = g(f"dc_{v}")
        sg = SIGN[v]
        ax.fill_between(q.h, sg * q.lo68, sg * q.hi68, color=pal[v], alpha=0.16, lw=0)
        ax.plot(q.h, sg * q.b, color=pal[v], lw=1.8, marker="o", ms=3.2, label=wb.LAB[v].replace(" (−)", ""))
    fp.finish(ax, "Cumulative change in level", h0=0)
    ax.set_xlabel(xl)
    ax.set_title("(b) Profits, investment and dividends", loc="left", fontsize=9.3)
    ax.legend(loc="upper right", fontsize=7.2, frameon=False)
    fig.suptitle("Firms spend a profit windfall on investment and dividends", x=0.0, ha="left", fontweight="bold", y=1.03)
    fig.tight_layout()
    fig.text(0.0, -0.1, "Notes: Same local projections as the persistence figure, with each item of the corporate account as the outcome (cumulative change "
             "from year −1, percent of GNI), scaled by the impact response of corporate net lending; country-years with all items observed. (a) Outlays enter "
             "with a negative sign; 'Other' is the remainder (property income received other than dividends, capital transfers, discrepancies). (b) Levels "
             "of selected items with 68 percent bands (country-cluster delta method).", fontsize=7, wrap=True, va="top")
    for ax in axes:
        ax.set_xlim(-0.5, H - 0.5); ax.set_xticks(range(H))
    fp.save(fig, "fig_base_decomp", sfx="")

    subs = list(dict.fromkeys(s["sub"]))
    sty = dict(zip(subs, ["-", "--", "-.", ":", (0, (5, 1, 1, 1, 1, 1))]))
    gry = dict(zip(subs, [None, "0.15", "0.45", "0.3", "0.6"]))
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.4), sharey=True)
    for ax, key, cl, title in ((axes[0], "corp", fp.CO, "(a) Corporate S–I, profit windfall"), (axes[1], "hh", fp.HH, "(b) Household S–I, saving-rate shift")):
        for sub in subs:
            q = s[(s["sub"] == sub) & (s.key == key)].sort_values("h")
            if sub == "Full sample":
                ax.fill_between(q.h, q.lo68, q.hi68, color=cl, alpha=0.15, lw=0)
            ax.plot(q.h, q.b, color=cl if sub == "Full sample" else gry[sub], ls=sty[sub], lw=1.9 if sub == "Full sample" else 1.2,
                    marker="o" if sub == "Full sample" else None, ms=3.2, label=f"{sub} (N = {int(q.n.iloc[0])})")
        fp.finish(ax, "Net lending (impact = 1)" if key == "corp" else "", h0=0)
        ax.set_xlabel(xl)
        ax.set_title(title, loc="left", fontsize=9.3)
        ax.set_ylim(-0.6, 1.1)
    h_, l_ = axes[0].get_legend_handles_labels()
    fig.legend(h_, l_, loc="lower center", bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize=7, frameon=False)
    fig.suptitle("Subsamples", x=0.0, ha="left", fontweight="bold", y=1.03)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.text(0.0, -0.13, "Notes: Baseline specification re-estimated on each subsample (weights re-estimated). Before 1998 only five economies (France, Sweden, "
             "the United States, Finland, Korea) report sector accounts. 'Windows ending before 2020': start years up to 2014, so that year t+5 precedes "
             "COVID-19. Shaded: 68 percent band for the full sample. Short subsamples have few years per economy, which biases persistence "
             "downward when lagged levels and country fixed effects are both included.", fontsize=7, wrap=True, va="top")
    for ax in axes:
        ax.set_xlim(-0.3, H - 0.7); ax.set_xticks(range(H))
    fp.save(fig, "fig_base_subsamples", sfx="")


if __name__ == "__main__":
    if "--render" in sys.argv:
        r, s = pd.read_csv(OUT / "ep_baseline.csv"), pd.read_csv(OUT / "ep_baseline_sub.csv")
    else:
        r, s = run()
    figs(r, s)
    print(r.pivot_table(index="key", columns="h", values="b", sort=False).round(2).to_string())
