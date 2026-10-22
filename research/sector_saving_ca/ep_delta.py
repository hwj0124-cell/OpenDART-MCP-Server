"""델타법 vs 부트스트랩: 주 사양(FGLS, S−I 정규화, 시차 통제)의 Figure 1 비율 γ_h/γ_0 구간 비교.

델타법: 각 식의 계수를 FWL로 영향함수 ψ_i = w_i z̃_i u_i / Σ w z̃² 로 쓰고 국가별 합으로 (γ_h, γ_0)의 클러스터 공분산을 구함
(두 식을 쌓아 국가 클러스터로 추정한 것과 같음, 소표본 보정 G/(G−1)).
Var(γ_h/γ_0) = V_h/γ_0² + γ_h² V_0/γ_0⁴ − 2 γ_h C/γ_0³, 68% 구간 = ±0.994·se.
→ output/ep_delta_lags.csv, output/paper/figA_delta_vs_bootstrap_lags.(png|pdf)
"""
import sys

import numpy as np
import pandas as pd

import ep_report as rp
import lp_tools as lt
from make_report import OUT

Z68 = 0.9945


def infl(d, y, s):
    """계수 γ와 관측별 영향함수(국가 인덱스 포함)."""
    w = f"w_{s}"
    q = d[[y, f"z_{s}", *rp.ctrl(s), w, "country", "year"]].dropna()
    sw = np.sqrt(q[w].to_numpy(float))
    C = np.hstack([q[rp.ctrl(s)].to_numpy(float), lt._fe_matrix(q, ("country", "year"))]) * sw[:, None]
    Z, Y = q[f"z_{s}"].to_numpy(float) * sw, q[y].to_numpy(float) * sw
    zt = Z - C @ np.linalg.lstsq(C, Z, rcond=None)[0]       # FWL: 통제·고정효과를 걸러낸 z
    g = zt @ Y / (zt @ zt)
    u = Y - C @ np.linalg.lstsq(C, Y - g * Z, rcond=None)[0] - g * Z
    return g, pd.Series(zt * u / (zt @ zt), index=q.index).groupby(q.country).sum()


def main():
    rp.FGLS = rp.WSV = rp.LAGS = True
    rp.NORM = "nl"
    P, d = rp.prepare()
    bs = pd.read_csv(OUT / "ep_report_fgls_nl_lags.csv")
    rows = []
    for s in ("hh", "corp"):
        g0, p0 = infl(d, f"nl_{s}_h0", s)
        for h in range(rp.K + 1):
            gh, ph = infl(d, f"nl_{s}_h{h}", s)
            a = pd.concat([ph, p0], axis=1).fillna(0).to_numpy()
            G = len(a)
            V = a.T @ a * G / (G - 1)
            r = gh / g0
            se = np.sqrt(V[0, 0] / g0**2 + gh**2 * V[1, 1] / g0**4 - 2 * gh * V[0, 1] / g0**3)
            b = bs[(bs.kind == "resp") & (bs.sector == s) & (bs["var"] == f"nl_{s}") & (bs.h == h)].iloc[0]
            assert abs(b.ratio - r) < 1e-8, (s, h, b.ratio, r)
            rows.append(dict(sector=s, h=h, ratio=r, d_lo=r - Z68 * se, d_hi=r + Z68 * se, b_lo=b.lo68, b_hi=b.hi68))
    r = pd.DataFrame(rows)
    r.to_csv(OUT / "ep_delta_lags.csv", index=False)
    print(r.round(2).to_string(index=False))
    return r, len(P["hh"]), len(P["corp"])


def plot(r, n_hh, n_co):
    import fig_paper as fp
    plt = fp.plt
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2), sharey=True)
    for ax, s, col, title in ((axes[0], "hh", fp.HH, f"(a) Households (N = {n_hh})"), (axes[1], "corp", fp.CO, f"(b) Corporations (N = {n_co})")):
        q = r[r.sector == s]
        ax.fill_between(q.h, q.b_lo, q.b_hi, color=col, alpha=0.16, lw=0, label="Bootstrap (percentile)")
        ax.plot(q.h, q.ratio, color=col, lw=1.6, marker="o", ms=3, label="Point estimate")
        ax.plot(q.h, q.d_lo, color="0.15", lw=1.0, ls="--", label="Delta method")
        ax.plot(q.h, q.d_hi, color="0.15", lw=1.0, ls="--")
        fp.finish(ax, "Saving–investment balance\n(impact response = 1)" if s == "hh" else "", h0=0)
        ax.set_title(title, loc="left")
    axes[0].legend(loc="lower left", fontsize=7)
    fig.suptitle("Figure A. 68 percent confidence bands: bootstrap vs delta method", x=0.0, ha="left", fontweight="bold", y=1.0)
    fig.tight_layout()
    fig.text(0.0, -0.04, "Notes: Same specification as Figure 1. Shaded areas: percentile intervals from a country-cluster bootstrap "
             "(499 replications). Dashed lines: delta-method intervals (±0.994 s.e.) using the country-clustered joint covariance of the "
             "impact-year and horizon-$h$ coefficients.", fontsize=7, wrap=True, va="top")
    fp.save(fig, "figA_delta_vs_bootstrap")


if __name__ == "__main__":
    sys.argv.append("--lags")  # fig_paper의 _lags 접미사
    plot(*main())
