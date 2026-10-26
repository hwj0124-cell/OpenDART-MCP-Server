"""본문 후보 그림: 이익 원천 기업저축(x_P) vs 저축률 원천 가계저축(x_H)의 자기 부문 S−I 지속성과 그 차이.

기준 사양(자기 S−I 수준 2시차, 네 성분·성장률 2시차, 같은 해 자기 부문 두 성분, 국가·연도 FE, 2단계 WLS).
네 성분은 모두 추정에 들어가고, 그림은 두 성분만. 패널 VAR 경로(ep_source_var.csv)를 가는 선으로 겹침.
부문저축 변화의 분산 분해(각 성분 분산 + 공분산 몫)도 출력.
→ output/ep_source_main.csv, output/paper/fig_source_main.(png|pdf)
"""
import sys

import numpy as np
import pandas as pd

import ep_source_robust as rb
import lp_tools as lt
from make_report import OUT

H = rb.H


def stats(q, base):
    pc = rb.path(q, "nl_corp", *base("nl_corp"))[0]["x_P"]
    ph = rb.path(q, "nl_hh", *base("nl_hh"))[0]["x_H"]
    return np.concatenate([pc, ph, ph - pc])


def run(reps=499):
    q, S = rb.setup()
    base = S["Baseline (own S–I in levels, 2 lags)"][0]
    pt = stats(q, base)
    bs = lt.cluster_bootstrap(q, lambda d: stats(d, base), reps=reps)
    pc = lambda p: np.nanpercentile(bs, p, axis=0)
    r = pd.DataFrame([dict(series=s, h=h) for s in ("x_P", "x_H", "diff") for h in range(H)]).assign(
        ratio=pt, lo68=pc(16), hi68=pc(84), lo90=pc(5), hi90=pc(95), n=len(q), countries=q.country.nunique())
    avg = lambda a, lo, hi: a[:, lo:hi].mean(axis=1)
    d = bs[:, 2 * H:]
    for lab, lo, hi in (("1-2", 1, 3), ("3-6", 3, 7)):
        print(f"difference, years {lab}: {pt[2 * H + lo:2 * H + hi].mean():.2f}  "
              f"90% [{np.percentile(avg(d, lo, hi), 5):.2f}, {np.percentile(avg(d, lo, hi), 95):.2f}]  share of draws <= 0: {np.mean(avg(d, lo, hi) <= 0):.3f}")
    for s, sec, comps in (("corporate", "sv_corp", ("x_P", "x_R")), ("household", "sv_hh", ("x_H", "x_Y"))):
        a, b = q[comps[0]], q[comps[1]]
        tot = (a + b).var()
        print(f"{s}: var share {comps[0]} {a.var() / tot:.2f}, {comps[1]} {b.var() / tot:.2f}, 2cov {2 * a.cov(b) / tot:.2f}; "
              f"sd {a.std():.2f} / {b.std():.2f}")
    r.to_csv(OUT / "ep_source_main.csv", index=False)
    return r


def figure(r):
    import fig_paper as fp
    plt = fp.plt
    v = pd.read_csv(OUT / "ep_source_var.csv")
    v = v[v.method == "VAR"]
    col = {"x_P": fp.CO, "x_H": fp.HH}
    lab = {"x_P": "Corporate saving from higher profits", "x_H": "Household saving from a higher saving rate"}
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.3))
    ax = axes[0]
    for x in ("x_P", "x_H"):
        g = r[r.series == x]
        ax.fill_between(g.h, g.lo68, g.hi68, color=col[x], alpha=0.16, lw=0)
        ax.plot(g.h, g.ratio, color=col[x], lw=1.9, marker="o", ms=3.5, label=lab[x])
        w = v[v.source == x].sort_values("h")
        ax.plot(w.h, w.ratio, color=col[x], lw=0.9, ls="--", label="Panel VAR (dashed)" if x == "x_H" else None)
    fp.finish(ax, "Own-sector S–I (impact = 1)", h0=0)
    ax.set_xlabel("Years after the saving change ($h$)")
    ax.set_title("(a) Response of the own-sector balance", loc="left", fontsize=9.5)
    ax.set_ylim(-0.55, 1.1)
    ax.legend(loc="upper right", fontsize=7.2, frameon=False)
    ax = axes[1]
    g = r[r.series == "diff"]
    ax.fill_between(g.h, g.lo90, g.hi90, color="0.6", alpha=0.2, lw=0, label="90% band")
    ax.fill_between(g.h, g.lo68, g.hi68, color="0.35", alpha=0.25, lw=0, label="68% band")
    ax.plot(g.h, g.ratio, color="0.1", lw=1.9, marker="o", ms=3.5, label="Household minus corporate")
    fp.finish(ax, "Difference", h0=0)
    ax.set_xlabel("Years after the saving change ($h$)")
    ax.set_title("(b) Difference", loc="left", fontsize=9.5)
    ax.set_ylim(-0.55, 1.1)
    ax.legend(loc="upper left", fontsize=7.2, frameon=False)
    fig.suptitle("A profit windfall fades faster than a shift in household saving behavior", x=0.0, ha="left", fontweight="bold", y=1.03)
    fig.tight_layout()
    n, k = int(r.n.iloc[0]), int(r.countries.iloc[0])
    fig.text(0.0, -0.05, f"Notes: N = {n}, {k} economies. Local projections of each sector's saving–investment balance on the components of the change in its "
             "saving, with two lags of the own-sector balance (levels), of all four components and of GDP growth, and country and year fixed effects; two-step "
             "weighted least squares. Responses are normalized by the impact response. Bands from a country-cluster bootstrap (499 replications). "
             "Dashed lines: panel VAR with the same variables and a recursive ordering that mirrors the local projections.", fontsize=7, wrap=True, va="top")
    fp.save(fig, "fig_source_main", sfx="")


if __name__ == "__main__":
    r = pd.read_csv(OUT / "ep_source_main.csv") if "--render" in sys.argv else run()
    figure(r)
    print(r.pivot_table(index="series", columns="h", values="ratio").round(2).to_string())
