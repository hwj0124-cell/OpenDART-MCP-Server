"""원천 분해 국소투영(사례 선정 없음): 이익 원천 기업저축 vs 성향 원천 가계저축 이후 자기 부문 S−I의 지속성.

  Δs^C = ρ_{t−1}Δπ + π_tΔρ  → x_P(이익 원천), x_R      ρ = 기업저축/영업잉여
  Δs^H = y_{t−1}Δσ/100 + σ_tΔy/100 → x_H(성향 원천), x_Y   σ = 가계저축률(%), y = 가처분소득/GDP

  n^j_{t+h} − n^j_{t−1} = b_P x_P + b_H x_H + 통제(source_shocks.controls) + 국가FE + 연도FE
  보고: θ^C_h = b_P,h / b_P,0 (기업 S−I),  θ^H_h = b_H,h / b_H,0 (가계 S−I)

표본: h = 0…6 모든 시점에 두 부문 S−I가 관측되는 국가·연도(시점 간 표본 고정).
가중치: 시작 연도 식(무가중) 잔차의 국가별 분산의 역수, 부문별로 따로, 모든 시점 공통(2단계 가중최소제곱).
구간: 국가 단위 부트스트랩 499회(가중치 고정).
→ output/ep_source_lp.csv, output/paper/fig_source_lp.(png|pdf)
"""
import sys

import numpy as np
import pandas as pd

import lp_tools as lt
import source_shocks as ss
from make_report import OUT

H = 7
X = ["x_P", "x_H"]
SEC = {"corp": ("nl_corp", 0), "hh": ("nl_hh", 1)}  # 부문 → (결과 변수, 해당 원천의 계수 위치)


def sample(d, last_start=2019):
    cols = [f"{y}_h{h}" for y, _ in SEC.values() for h in range(H)] + X + ss.controls()
    q = d.dropna(subset=cols)
    q = q[q.year <= last_start].copy()
    for s, (y, _) in SEC.items():
        M = np.hstack([q[X + ss.controls()].to_numpy(float), lt._fe_matrix(q, ("country", "year"))])
        e = q[f"{y}_h0"].to_numpy(float) - M @ np.linalg.lstsq(M, q[f"{y}_h0"].to_numpy(float), rcond=None)[0]
        q[f"w_{s}"] = 1 / q.country.map(pd.Series(e, index=q.index).groupby(q.country).var())
    return q.dropna(subset=["w_corp", "w_hh"])


def stats(q):
    out = []
    for s, (y, k) in SEC.items():
        b = np.array([lt.ols(q, f"{y}_h{h}", X, ss.controls(), se=False, w=f"w_{s}")[0][k] for h in range(H)])
        out.append(b / b[0])
    r = np.concatenate(out)
    avg = lambda a, lo, hi: a[lo:hi].mean()
    c, hh = out
    return np.concatenate([r, [avg(c, 1, 3), avg(c, 3, 7), avg(hh, 1, 3), avg(hh, 3, 7),
                               avg(c, 1, 3) - avg(hh, 1, 3), avg(c, 3, 7) - avg(hh, 3, 7)]])


LAB = [("path", s, h) for s in SEC for h in range(H)] + \
      [("avg", "corp", "h1-2"), ("avg", "corp", "h3-6"), ("avg", "hh", "h1-2"), ("avg", "hh", "h3-6"),
       ("diff", "corp-hh", "h1-2"), ("diff", "corp-hh", "h3-6")]


def run(reps=499):
    d = ss.prepare(pd.read_csv(OUT.parent / "data" / "panel.csv"))
    rows = []
    for name, last in (("main", 2019), ("pre-COVID windows", 2013)):
        q = sample(d, last)
        pt = stats(q)
        bs = lt.cluster_bootstrap(q, stats, reps=reps)
        for (kind, s, h), p, col in zip(LAB, pt, bs.T):
            rows.append(dict(spec=name, kind=kind, sector=s, h=h, ratio=p,
                             lo68=np.nanpercentile(col, 16), hi68=np.nanpercentile(col, 84),
                             lo90=np.nanpercentile(col, 5), hi90=np.nanpercentile(col, 95),
                             n=len(q), countries=q.country.nunique()))
    r = pd.DataFrame(rows)
    r.to_csv(OUT / "ep_source_lp.csv", index=False)
    return r


def figure(r):
    import fig_paper as fp
    plt = fp.plt
    HH, CO = fp.HH, fp.CO
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.3), sharey=True)
    for ax, spec, title in ((axes[0], "main", "(a) All windows"), (axes[1], "pre-COVID windows", "(b) Windows ending before 2020")):
        p = r[(r.spec == spec) & (r.kind == "path")]
        n, k = int(p.n.iloc[0]), int(p.countries.iloc[0])
        for s, col, ls, lab in (("hh", HH, "-", "Household S–I after a saving-rate-driven rise"),
                                ("corp", CO, "--", "Corporate S–I after a profit-driven rise")):
            q = p[p.sector == s].assign(h=lambda x: x.h.astype(int)).sort_values("h")
            ax.fill_between(q.h, q.lo68, q.hi68, color=col, alpha=0.14, lw=0)
            ax.plot(q.h, q.ratio, color=col, ls=ls, lw=1.6, marker="o", ms=3, label=lab)
        fp.finish(ax, "Saving–investment balance\n(impact response = 1)" if spec == "main" else "", h0=0)
        ax.set_xlabel("Years after the saving change ($h$)")
        ax.set_title(f"{title}, N = {n}, {k} economies", loc="left", fontsize=9)
    axes[0].set_ylim(-0.75, 1.45)
    axes[0].legend(loc="upper right", fontsize=7.3)
    fig.suptitle("Persistence of sector balances by source of the saving change", x=0.0, ha="left", fontweight="bold", y=1.03)
    fig.tight_layout()
    fig.text(0.0, -0.05, "Notes: Local projections of the own-sector saving–investment balance (net lending, percent of GDP) on profit-driven corporate saving and "
             "saving-rate-driven household saving, entered jointly with the remaining components of sector saving, their lags and lagged levels, and "
             "country and year fixed effects; two-step weighted least squares. Responses are relative to year −1 and normalized by the impact response. "
             "The same country-years are used at every horizon. Shaded areas are 68 percent bands from a country-cluster bootstrap (499 replications).",
             fontsize=7, wrap=True, va="top")
    fp.save(fig, "fig_source_lp", sfx="")


if __name__ == "__main__":
    r = pd.read_csv(OUT / "ep_source_lp.csv") if "--render" in sys.argv else run()
    figure(r)
    print(r[r.kind != "path"].round(2).to_string(index=False))
    print(r[r.kind == "path"].pivot_table(index=["spec", "sector"], columns="h", values="ratio").round(2).to_string())
