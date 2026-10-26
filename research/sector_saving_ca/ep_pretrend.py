"""사전 추세 점검: 충격 이전 시점의 '반응' n_{t+h} − n_{t−1}, h = −5…−2 (기준 사양 표본 B + 정부 순대출 통제).

기준 사양은 n_{t−1}, n_{t−2}를 통제하므로 h = −1(정의상 0)과 h = −2(통제변수의 선형결합)는 기계적으로 0.
의미 있는 점검은 통제 창 밖인 h ≤ −3. 비교용으로 자기 부문 순대출 시차를 뺀 사양도 추정(모든 사전 시점이 정보를 가짐).
구간: 국가 군집 부트스트랩 999회(fastboot), 시작 연도 반응으로 정규화.
→ output/ep_pretrend.csv, output/paper/fig_pretrend.(png|pdf)
"""
import sys

import numpy as np
import pandas as pd

import ep_final as ef
import ep_specs as sp
import ep_windfall_behavior as wb
import fastboot as fb
from make_report import OUT

SEC = wb.SEC
HS = list(range(-5, 7))
SPECS = {"base": "Baseline (with lags of own net lending)", "nolag": "Without lags of own net lending"}


def data():
    q = sp.data()["B"]
    p = pd.read_csv(OUT.parent / "data" / "panel.csv")
    import lp_tools as lt
    p = lt.balance(p)
    g = p.groupby("country")
    new = {}
    for y in ("nl_corp", "nl_hh"):
        for h in range(-5, -1):
            new[f"{y}_h{h}"] = g[y].shift(-h) - g[y].shift(1)
    k = pd.concat([p[["country", "year"]], pd.DataFrame(new, index=p.index)], axis=1)
    return q.drop(columns=[c for c in new if c in q.columns]).merge(k, on=["country", "year"], how="left")


def ctrl(s, spec):
    c = ef.ctrl(s)
    return c if spec == "base" else [x for x in c if x not in (f"L1_{SEC[s]['y']}", f"L2_{SEC[s]['y']}")]


def run(reps=999):
    q = data()
    D = {}
    for spec in SPECS:
        for s, m in SEC.items():
            w = f"w_{s}_{spec}"
            q[w] = sp.weights(q, s, ctrl(s, spec))
            for h in HS:
                if h == -1:
                    continue
                D[f"{spec}_{s}_{h}"] = fb.Design(q, [m["x"], m["z"]], ctrl(s, spec), w, [f"{m['y']}_h{h}"])

    def stat(b):
        out = []
        for spec in SPECS:
            for s, m in SEC.items():
                b0 = b[f"{spec}_{s}_0"].iloc[0]
                out += [0.0 if h == -1 else b[f"{spec}_{s}_{h}"].iloc[0] / b0 for h in HS]
        return np.array(out)
    pt, bs = fb.bootstrap(D, stat, reps=reps)
    pc = lambda p_: np.nanpercentile(bs, p_, axis=0)
    keys = [(spec, s, h) for spec in SPECS for s in SEC for h in HS]
    r = pd.DataFrame(keys, columns=["spec", "sector", "h"]).assign(b=pt, lo68=pc(16), hi68=pc(84), lo90=pc(5), hi90=pc(95))
    r["n"] = [D[f"{sp_}_{s}_{h}"].n if h != -1 else np.nan for sp_, s, h in keys]
    r.to_csv(OUT / "ep_pretrend.csv", index=False)
    return r


def figure(r):
    import fig_paper as fp
    plt = fp.plt
    col = {"corp": fp.CO, "hh": fp.HH}
    title = {"corp": "(a) Corporate net lending, profit windfall", "hh": "(b) Household net lending, saving-rate shift"}
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.5), sharey=True)
    for ax, s in zip(axes, ("corp", "hh")):
        ax.axvspan(-5.4, -0.5, color="0.94", lw=0)
        for spec, ls, mk in (("base", "-", "o"), ("nolag", "--", "s")):
            q = r[(r.spec == spec) & (r.sector == s)].sort_values("h")
            if spec == "base":
                ax.fill_between(q.h, q.lo90, q.hi90, color=col[s], alpha=0.09, lw=0)
                ax.fill_between(q.h, q.lo68, q.hi68, color=col[s], alpha=0.18, lw=0)
            else:
                ax.plot(q.h, q.lo90, color="0.35", lw=0.7, ls=":")
                ax.plot(q.h, q.hi90, color="0.35", lw=0.7, ls=":")
            ax.plot(q.h, q.b, color=col[s] if spec == "base" else "0.25", lw=1.9 if spec == "base" else 1.3, ls=ls, marker=mk, ms=3.2,
                    label=SPECS[spec])
        fp.finish(ax, "Net lending relative to $t-1$\n(impact = 1)" if s == "corp" else "", h0=0)
        ax.axvline(-0.5, color="0.5", lw=0.6, ls=":")
        ax.set_xlim(-5.4, 6.4)
        ax.set_xticks(HS)
        ax.set_xlabel("Years relative to the saving change ($h$)")
        ax.set_title(title[s], loc="left", fontsize=9.3)
        ax.set_ylim(-1.2, 1.4)
        ax.text(-5.2, 1.27, "before", fontsize=7, color="0.4")
        ax.text(0.0, 1.27, "after", fontsize=7, color="0.4")
    h_, l_ = axes[0].get_legend_handles_labels()
    fig.legend(h_, l_, loc="lower center", bbox_to_anchor=(0.5, -0.06), ncol=2, fontsize=7.4, frameon=False)
    fig.suptitle("Net lending before and after the saving change", x=0.0, ha="left", fontweight="bold", y=1.02)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.text(0.0, -0.08, "Notes: Annual data for 38 economies, 1973–2025 (unbalanced). Coefficients on the windfall or saving-rate component in local "
             "projections of n(t+h) − n(t−1); for h < 0 the outcome is the change from year t−1 back to t+h. With two lags of own net lending in the "
             "controls, h = −1 and h = −2 are zero by construction, so h ≤ −3 is the informative pre-period test. Dashed: the same controls without lags "
             "of own net lending (90 percent band dotted). Shaded: 68 and 90 percent bands, country-cluster bootstrap (999 replications).",
             fontsize=6.9, wrap=True, va="top")
    fp.save(fig, "fig_pretrend", sfx="")


if __name__ == "__main__":
    r = pd.read_csv(OUT / "ep_pretrend.csv") if "--render" in sys.argv else run()
    figure(r)
    print(r.pivot_table(index=["spec", "sector"], columns="h", values="b", sort=False).round(2).to_string())
    print(r[r.h < -1].pivot_table(index=["spec", "sector"], columns="h", values=["lo90", "hi90"], sort=False).round(2).to_string())
