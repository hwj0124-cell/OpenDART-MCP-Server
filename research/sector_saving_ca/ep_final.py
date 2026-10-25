"""기준 사양(표본 B + 정부 순대출 통제): 두 부문 순대출 반응을 겹쳐 그리고, 세부 항목 반응을 기업(왼쪽)·가계(오른쪽)로.

표본 B: 전체 기간, 2020년 이후 시작 포함, 시점별 비균형(ep_specs와 같음).
통제: 자기 부문 순대출 수준, 자기 부문 두 성분, 성장률, 정부 순대출 수준(각 2시차) + 국가·연도 FE, 2단계 WLS.
세부 항목(누적 변화, 시작 연도 순대출 반응 = 1):
  기업 순대출 = 영업잉여 − 순배당 − 법인세 − 이자 지급 − 총자본형성 + 기타
  가계 순대출 = 가처분소득 − 소비 − 총자본형성 + 기타
  분해는 시점마다 모든 항목이 관측되는 행으로 추정해 '기타'가 같은 표본의 잔차가 되도록 함.
구간: 국가 군집 델타 방법.
→ output/ep_final.csv, output/paper/fig_final_persistence.(png|pdf), fig_final_items.(png|pdf)
"""
import sys

import numpy as np
import pandas as pd

import ep_specs as sp
import ep_wb_extra as ex
import ep_windfall_behavior as wb
from make_report import OUT

H, SEC = sp.H, wb.SEC
ITEMS = {"corp": ["prof", "div_net", "tax_paid", "int_paid", "gcf_corp"], "hh": ["yd_hh", "cons_hh", "gcf_hh"]}
SIGN, LAB = wb.SIGN, wb.LAB


def ctrl(s):
    return wb.ctrl(s) + sp.GOV


def run():
    q = sp.data()["B"]
    D = ex.Delta()
    for s, m in SEC.items():
        q[f"w_{s}"] = sp.weights(q, s, ctrl(s))
        D.add(s, *sp.design_h(q, [m["x"], m["z"]], ctrl(s), f"w_{s}", m["y"]))
        bs, ps = {}, []
        for h in range(H):
            ys = [f"{v}_h{h}" for v in [m["y"]] + ITEMS[s]]
            b, psi = ex.design(q, [m["x"], m["z"]], ctrl(s), f"w_{s}", ys)
            bs.update(b.to_dict())
            ps.append(psi)
        D.add(f"{s}_dc", pd.Series(bs), pd.concat(ps, axis=1))
    D.finish()
    rows = []

    def emit(key, f):
        for h in range(H):
            b, se = D.fn(lambda d, h=h: f(d, h))
            rows.append(dict(key=key, h=h, b=b, se=se, lo68=b - se, hi68=b + se, lo90=b - 1.645 * se, hi90=b + 1.645 * se))
    for s, m in SEC.items():
        y = m["y"]
        emit(s, lambda d, h, s=s, y=y: d[f"{s}|{y}_h{h}"] / d[f"{s}|{y}_h0"])
        t = f"{s}_dc"
        emit(f"{s}_nl", lambda d, h, t=t, y=y: d[f"{t}|{y}_h{h}"] / d[f"{t}|{y}_h0"])
        for v in ITEMS[s]:
            emit(f"{s}_{v}", lambda d, h, t=t, y=y, v=v: SIGN[v] * d[f"{t}|{v}_h{h}"] / d[f"{t}|{y}_h0"])
        emit(f"{s}_other", lambda d, h, t=t, y=y, s=s: (d[f"{t}|{y}_h{h}"] - sum(SIGN[v] * d[f"{t}|{v}_h{h}"] for v in ITEMS[s])) / d[f"{t}|{y}_h0"])
    n0 = int(q[f"{SEC['corp']['y']}_h0"].notna().sum())
    n6 = int(q[f"{SEC['corp']['y']}_h6"].notna().sum())
    r = pd.DataFrame(rows).assign(n0=n0, n6=n6, countries=q.country.nunique(), first=int(q.year.min()), last=int(q.year.max()))
    r.to_csv(OUT / "ep_final.csv", index=False)
    return r


def figs(r):
    import fig_paper as fp
    plt = fp.plt
    g = lambda k: r[r.key == k].sort_values("h")
    xl = "Years after the saving change ($h$)"
    n0, n6, k, a, z = (int(r[c].iloc[0]) for c in ("n0", "n6", "countries", "first", "last"))
    base_note = (f"{k} economies, start years {a}–{z}; {n0} country-years at h = 0 and {n6} at h = 6 (every horizon uses all start years with the outcome "
                 "observed). Local projections of each sector's net lending on the profit-windfall (corporations) or saving-rate (households) component and "
                 "the remaining component, with two lags of the sector's net lending (levels), both components, GDP growth and government net lending, and "
                 "country and year fixed effects; two-step weighted least squares.")

    fig, ax = plt.subplots(figsize=(5.2, 3.5))
    for s, cl, lab in (("hh", fp.HH, "Household net lending after a shift in the saving rate"), ("corp", fp.CO, "Corporate net lending after a profit windfall")):
        q = g(s)
        ax.fill_between(q.h, q.lo90, q.hi90, color=cl, alpha=0.08, lw=0)
        ax.fill_between(q.h, q.lo68, q.hi68, color=cl, alpha=0.18, lw=0)
        ax.plot(q.h, q.b, color=cl, lw=2.0, marker="o", ms=3.5, label=lab)
    fp.finish(ax, "Net lending (impact = 1)", h0=0)
    ax.set_xlabel(xl)
    ax.set_xlim(-0.3, H - 0.7)
    ax.set_xticks(range(H))
    ax.set_ylim(-0.75, 1.1)
    ax.legend(loc="upper right", fontsize=7.4, frameon=False)
    ax.set_title("A profit windfall fades; a shift in household saving behavior persists", loc="left", fontsize=9.5, fontweight="bold")
    fig.tight_layout()
    fig.text(0.0, -0.03, "Notes: " + base_note + " Normalized by the impact response. Bands: 68 and 90 percent, country-cluster delta method.",
             fontsize=6.8, wrap=True, va="top")
    fp.save(fig, "fig_final_persistence", sfx="")

    pal = {"prof": "#c98a1b", "div_net": "#7a5195", "tax_paid": "#bc5090", "int_paid": "#ef8e9c", "gcf_corp": "#2b6c8f",
           "yd_hh": "#c98a1b", "cons_hh": "#7a5195", "gcf_hh": "#2b6c8f", "other": "0.75"}
    fig, axes = plt.subplots(2, 2, figsize=(7.8, 6.8), gridspec_kw=dict(height_ratios=[1.1, 1]))
    titles = {"corp": "Corporations after a profit windfall", "hh": "Households after a shift in the saving rate"}
    for j, s in enumerate(("corp", "hh")):
        ax = axes[0, j]
        hs = np.arange(H)
        pos, neg = np.zeros(H), np.zeros(H)
        for v in ITEMS[s] + ["other"]:
            b = g(f"{s}_{v}").b.to_numpy()
            ax.bar(hs, b, bottom=np.where(b >= 0, pos, neg), color=pal[v], width=0.7, label=LAB[v], edgecolor="white", lw=0.3)
            pos, neg = pos + np.clip(b, 0, None), neg + np.clip(b, None, 0)
        ax.plot(hs, g(f"{s}_nl").b, color="k", lw=1.8, marker="o", ms=3.4, label="Net lending")
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
            ax.fill_between(q.h, sg * q.lo68, sg * q.hi68, color=pal[v], alpha=0.16, lw=0)
            ax.plot(q.h, sg * q.b, color=pal[v], lw=1.8, marker="o", ms=3, label=LAB[v].replace(" (−)", ""))
        fp.finish(ax, "Cumulative change in level\n(impact net lending = 1)" if j == 0 else "", h0=0)
        ax.set_xlabel(xl)
        ax.set_xlim(-0.3, H - 0.7)
        ax.set_xticks(range(H))
        ax.set_title(("(c) " if j == 0 else "(d) ") + ("Profits, investment and dividends" if s == "corp" else "Income, consumption and investment"),
                     loc="left", fontsize=9.2)
        ax.legend(loc="upper right" if s == "corp" else "lower right", fontsize=7, frameon=False)
    fig.suptitle("Where the extra saving goes", x=0.0, ha="left", fontweight="bold", y=1.0)
    fig.tight_layout()
    fig.text(0.0, -0.02, "Notes: " + base_note + " Each item of the sector account is the outcome (cumulative change from year −1, percent of GNI), scaled "
             "by the impact response of the sector's net lending, on country-years with all items observed. (a)–(b): outlays enter with a negative sign; "
             "'Other' is the remainder. (c)–(d): levels with 68 percent bands (country-cluster delta method).", fontsize=6.8, wrap=True, va="top")
    fp.save(fig, "fig_final_items", sfx="")


if __name__ == "__main__":
    r = pd.read_csv(OUT / "ep_final.csv") if "--render" in sys.argv else run()
    figs(r)
    print(r.pivot_table(index="key", columns="h", values="b", sort=False).round(2).to_string())
