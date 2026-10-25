"""원천 분해 LP: 제안 사양 vs 성분별 AR(2) 잔차(사전 백색화) 비교 그림.

AR 잔차: 각 성분을 자기 두 시차와 국가·연도 FE에 회귀한 잔차. 네 잔차를 함께 넣고 다른 통제는 없음.
제안 사양: 네 성분 + 네 성분·성장률 두 시차 + 자기 부문 S−I 변화 두 시차 + 국가·연도 FE.
같은 표본·가중치(제안 사양의 2단계 가중치). 구간: 국가 군집 부트스트랩 68%(AR 잔차도 매 반복 재추정).
→ output/ep_source_ar.csv, output/paper/fig_source_ar.(png|pdf)
"""
import sys

import numpy as np
import pandas as pd

import ep_source_check as c
import ep_source_fig as sf
import lp_tools as lt
from make_report import OUT

H = c.H
U = [f"u_{x}" for x in c.SRC]


def ar_resid(q):
    q = q.copy()
    F = lt._fe_matrix(q, ("country", "year"))
    for x in c.SRC:
        M = np.hstack([q[[f"L1_{x}", f"L2_{x}"]].to_numpy(float), F])
        v = q[x].to_numpy(float)
        q[f"u_{x}"] = v - M @ np.linalg.lstsq(M, v, rcond=None)[0]
    return q


def stats(q, ctrl):
    q = ar_resid(q)
    out = []
    for X, C in ((c.SRC, ctrl), (U, None)):
        for y in ("nl_corp", "nl_hh"):
            B = np.array([lt.ols(q, f"{y}_h{h}", X, C[y] if C else [], se=False, w=f"w_{y}")[0] for h in range(H)])
            out += [B[:, k] / B[0, k] for k, x in enumerate(c.SRC) if c.OWN[x] == y]
    return np.concatenate(out)


def run(reps=499):
    q, ctrl = sf.sample()
    pt = stats(q, ctrl)
    bs = lt.cluster_bootstrap(q, lambda b: stats(b, ctrl), reps=reps)
    keys = [(s, x) for s in ("proposed", "AR(2) residuals") for y in ("nl_corp", "nl_hh") for x in c.SRC if c.OWN[x] == y]
    r = pd.DataFrame([dict(spec=s, source=x, h=h) for s, x in keys for h in range(H)]).assign(
        ratio=pt, lo68=np.nanpercentile(bs, 16, axis=0), hi68=np.nanpercentile(bs, 84, axis=0), n=len(q), countries=q.country.nunique())
    r.to_csv(OUT / "ep_source_ar.csv", index=False)
    return r


def figure(r):
    import fig_paper as fp
    plt = fp.plt
    col = {"x_P": fp.CO, "x_R": fp.CO, "x_H": fp.HH, "x_Y": fp.HH}
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.6), sharex=True, sharey=True)
    for ax, x in zip(axes.flat, ["x_P", "x_Y", "x_R", "x_H"]):
        for s, ls, a, lab in (("proposed", "-", 0.16, "Proposed specification"), ("AR(2) residuals", "--", 0.0, "AR(2) residuals, no other controls")):
            q = r[(r.spec == s) & (r.source == x)].sort_values("h")
            if a:
                ax.fill_between(q.h, q.lo68, q.hi68, color=col[x], alpha=a, lw=0)
            else:
                ax.plot(q.h, q.lo68, color=col[x], ls=":", lw=0.8)
                ax.plot(q.h, q.hi68, color=col[x], ls=":", lw=0.8)
            ax.plot(q.h, q.ratio, color=col[x], ls=ls, lw=1.6, marker="o" if s == "proposed" else "s", ms=3, label=lab)
        fp.finish(ax, "Own-sector S–I\n(impact = 1)" if x in ("x_P", "x_R") else "", h0=0)
        ax.set_title(sf.LAB[x], loc="left", fontsize=9.5)
        ax.set_xlabel("Years after the saving change ($h$)" if x in ("x_R", "x_H") else "")
    axes[0, 0].set_ylim(-0.6, 2.4)
    h_, l_ = axes[0, 0].get_legend_handles_labels()
    fig.legend(h_, l_, loc="lower center", bbox_to_anchor=(0.5, -0.03), ncol=2, fontsize=8)
    fig.suptitle("Proposed specification and AR(2) pre-whitened components", x=0.0, ha="left", fontweight="bold", y=1.0)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    n, k = int(r.n.iloc[0]), int(r.countries.iloc[0])
    fig.text(0.0, -0.04, f"Notes: N = {n}, {k} economies; same country-years and weights in both specifications. AR(2) residuals: each component regressed on its own two lags "
             "and country and year fixed effects; the four residuals enter jointly without other controls. Shaded areas (proposed) and dotted lines (AR) are 68 percent "
             "bands from a country-cluster bootstrap (499 replications), re-estimating the residuals in each replication.", fontsize=7, wrap=True, va="top")
    fp.save(fig, "fig_source_ar", sfx="")


if __name__ == "__main__":
    r = pd.read_csv(OUT / "ep_source_ar.csv") if "--render" in sys.argv else run()
    figure(r)
    print(r.pivot_table(index=["spec", "source"], columns="h", values="ratio", sort=False).round(2).to_string())
