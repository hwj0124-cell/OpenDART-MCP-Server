"""기준 사양(표본 B + 정부 순대출 통제): 두 부문 순대출 반응과 기업 순대출 분해, 국가 군집 부트스트랩(fastboot).

표본 B: 전체 기간(1973–2025), 2020년 이후 시작 포함, 시점별 비균형.
통제: 자기 부문 순대출 수준, 자기 부문 두 성분, 성장률, 정부 순대출 수준(각 2시차) + 국가·연도 FE, 2단계 WLS(가중치는 반복에서 고정).
분해(누적 변화, 시작 연도 순대출 반응 = 1):
  기업 순대출 = 영업잉여 − 순배당 − 법인세 − 이자 지급 − 총자본형성 + 기타
  가계 순대출 = 가처분소득 − 소비 − 총자본형성 + 기타
  시점마다 모든 항목이 관측되는 행으로 추정해 '기타'가 같은 표본의 잔차가 되도록 함.
구간: 국가 군집 부트스트랩 999회, 백분위 68·90%.
→ output/ep_final.csv, output/paper/fig_final.(png|pdf) [왼쪽 순대출 반응, 오른쪽 기업 분해],
  fig_final_items.(png|pdf) [기업·가계 세부 항목]
"""
import sys

import numpy as np
import pandas as pd

import ep_specs as sp
import ep_windfall_behavior as wb
import fastboot as fb
from make_report import OUT

H, SEC = sp.H, wb.SEC
ITEMS = {"corp": ["prof", "div_net", "tax_paid", "int_paid", "gcf_corp"], "hh": ["yd_hh", "cons_hh", "gcf_hh"]}
SIGN, LAB = wb.SIGN, wb.LAB


def ctrl(s):
    return wb.ctrl(s) + sp.GOV


def designs(q):
    D = {}
    for s, m in SEC.items():
        q[f"w_{s}"] = sp.weights(q, s, ctrl(s))
        X = [m["x"], m["z"]]
        for h in range(H):
            D[f"{s}_{h}"] = fb.Design(q, X, ctrl(s), f"w_{s}", [f"{m['y']}_h{h}"])
            D[f"{s}_dc_{h}"] = fb.Design(q, X, ctrl(s), f"w_{s}", [f"{v}_h{h}" for v in [m["y"]] + ITEMS[s]])
    return D


def keys():
    out = []
    for s in SEC:
        out += [(s, h) for h in range(H)] + [(f"{s}_nl", h) for h in range(H)]
        out += [(f"{s}_{v}", h) for v in ITEMS[s] + ["other"] for h in range(H)]
    return out


def stat(b):
    out = []
    for s, m in SEC.items():
        y = m["y"]
        b0 = b[f"{s}_0"][f"{y}_h0"]
        out += [b[f"{s}_{h}"][f"{y}_h{h}"] / b0 for h in range(H)]
        d0 = b[f"{s}_dc_0"][f"{y}_h0"]
        nl = [b[f"{s}_dc_{h}"][f"{y}_h{h}"] / d0 for h in range(H)]
        out += nl
        items = {v: [SIGN[v] * b[f"{s}_dc_{h}"][f"{v}_h{h}"] / d0 for h in range(H)] for v in ITEMS[s]}
        for v in ITEMS[s]:
            out += items[v]
        out += [nl[h] - sum(items[v][h] for v in ITEMS[s]) for h in range(H)]
    return np.array(out)


def run(reps=999):
    q = sp.data()["B"]
    D = designs(q)
    pt, bs = fb.bootstrap(D, stat, reps=reps)
    pc = lambda p: np.nanpercentile(bs, p, axis=0)
    r = pd.DataFrame(keys(), columns=["key", "h"]).assign(b=pt, lo68=pc(16), hi68=pc(84), lo90=pc(5), hi90=pc(95))
    m = SEC["corp"]["y"]
    r = r.assign(n0=int(q[f"{m}_h0"].notna().sum()), n6=int(q[f"{m}_h6"].notna().sum()), countries=q.country.nunique(),
                 first=int(q.year.min()), last=int(q.year.max()))
    r.to_csv(OUT / "ep_final.csv", index=False)
    return r


PAL = {"prof": "#c98a1b", "div_net": "#7a5195", "tax_paid": "#bc5090", "int_paid": "#ef8e9c", "gcf_corp": "#2b6c8f",
       "yd_hh": "#c98a1b", "cons_hh": "#7a5195", "gcf_hh": "#2b6c8f", "other": "0.75"}


def bars(ax, g, s):
    hs = np.arange(H)
    pos, neg = np.zeros(H), np.zeros(H)
    for v in ITEMS[s] + ["other"]:
        b = g(f"{s}_{v}").b.to_numpy()
        ax.bar(hs, b, bottom=np.where(b >= 0, pos, neg), color=PAL[v], width=0.7, label=LAB[v], edgecolor="white", lw=0.3)
        pos, neg = pos + np.clip(b, 0, None), neg + np.clip(b, None, 0)
    ax.plot(hs, g(f"{s}_nl").b, color="k", lw=1.8, marker="o", ms=3.4, label="Net lending")


def note(r):
    k, a, z = int(r.countries.iloc[0]), int(r["first"].iloc[0]), int(r["last"].iloc[0])
    return f"Annual data for {k} economies, {a}–{z} (unbalanced)."


def figs(r):
    import fig_paper as fp
    plt = fp.plt
    g = lambda k: r[r.key == k].sort_values("h")
    xl = "Years after the saving change ($h$)"

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.6))
    ax = axes[0]
    for s, cl, lab in (("hh", fp.HH, "Households: shift in the saving rate"), ("corp", fp.CO, "Corporations: profit windfall")):
        q = g(s)
        ax.fill_between(q.h, q.lo90, q.hi90, color=cl, alpha=0.08, lw=0)
        ax.fill_between(q.h, q.lo68, q.hi68, color=cl, alpha=0.18, lw=0)
        ax.plot(q.h, q.b, color=cl, lw=2.0, marker="o", ms=3.4, label=lab)
    fp.finish(ax, "Net lending (impact = 1)", h0=0)
    ax.set_xlabel(xl)
    ax.set_xlim(-0.3, H - 0.7)
    ax.set_xticks(range(H))
    ax.set_ylim(-0.75, 1.1)
    ax.legend(loc="upper right", fontsize=7.2, frameon=False)
    ax.set_title("(a) Response of sector net lending", loc="left", fontsize=9.3)
    ax = axes[1]
    bars(ax, g, "corp")
    fp.finish(ax, "Contribution (impact net lending = 1)", h0=0)
    ax.set_xlabel(xl)
    ax.set_xlim(-0.5, H - 0.5)
    ax.set_xticks(range(H))
    ax.set_title("(b) Corporate net lending after a profit windfall", loc="left", fontsize=9.3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), fontsize=6.8, frameon=False, ncol=3)
    fig.suptitle("A profit windfall fades; a shift in household saving behavior persists", x=0.0, ha="left", fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.text(0.0, 0.0, "Notes: " + note(r) + " Local projections of each sector's net lending on the profit-windfall or saving-rate component of its "
             "saving change, controlling for the remaining component, lags of own net lending, both components, GDP growth and government net lending, "
             "and country and year fixed effects. Normalized by the impact response. (a) 68 and 90 percent bands from a country-cluster bootstrap "
             "(999 replications). (b) Outlays enter with a negative sign; 'Other' is the remainder.", fontsize=6.9, wrap=True, va="top")
    fp.save(fig, "fig_final", sfx="")

    fig, axes = plt.subplots(2, 2, figsize=(7.8, 6.8), gridspec_kw=dict(height_ratios=[1.1, 1]))
    titles = {"corp": "Corporations after a profit windfall", "hh": "Households after a shift in the saving rate"}
    for j, s in enumerate(("corp", "hh")):
        ax = axes[0, j]
        bars(ax, g, s)
        fp.finish(ax, "Contribution to net lending\n(impact net lending = 1)" if j == 0 else "", h0=0)
        ax.set_xlabel("")
        ax.set_xlim(-0.5, H - 0.5)
        ax.set_xticks(range(H))
        ax.set_title(("(a) " if j == 0 else "(b) ") + titles[s], loc="left", fontsize=9.2)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), fontsize=6.8, frameon=False, ncol=3)
        ax = axes[1, j]
        for v in ITEMS[s]:
            if v in ("tax_paid", "int_paid"):
                continue
            q = g(f"{s}_{v}")
            sg = SIGN[v]
            ax.fill_between(q.h, sg * q.lo68, sg * q.hi68, color=PAL[v], alpha=0.16, lw=0)
            ax.plot(q.h, sg * q.b, color=PAL[v], lw=1.8, marker="o", ms=3, label=LAB[v].replace(" (−)", ""))
        fp.finish(ax, "Cumulative change in level\n(impact net lending = 1)" if j == 0 else "", h0=0)
        ax.set_xlabel(xl)
        ax.set_xlim(-0.3, H - 0.7)
        ax.set_xticks(range(H))
        ax.set_title(("(c) " if j == 0 else "(d) ") + ("Profits, investment and dividends" if s == "corp" else "Income, consumption and investment"),
                     loc="left", fontsize=9.2)
        ax.legend(loc="upper right" if s == "corp" else "lower right", fontsize=7, frameon=False)
    fig.suptitle("Where the extra saving goes", x=0.0, ha="left", fontweight="bold", y=1.0)
    fig.tight_layout()
    fig.text(0.0, -0.02, "Notes: " + note(r) + " Same local projections as in the main figure, with each item of the sector account (cumulative change "
             "from year −1, percent of GNI) as the outcome, scaled by the impact response of the sector's net lending. (a)–(b) Outlays enter with a "
             "negative sign; 'Other' is the remainder. (c)–(d) 68 percent bands from a country-cluster bootstrap (999 replications).",
             fontsize=6.8, wrap=True, va="top")
    fp.save(fig, "fig_final_items", sfx="")


if __name__ == "__main__":
    import time
    t = time.time()
    r = pd.read_csv(OUT / "ep_final.csv") if "--render" in sys.argv else run()
    print(f"{time.time() - t:.1f} s")
    figs(r)
    print(r[r.key.isin(["corp", "hh"])].pivot_table(index="key", columns="h", values=["b", "lo90", "hi90"]).round(2).to_string())
