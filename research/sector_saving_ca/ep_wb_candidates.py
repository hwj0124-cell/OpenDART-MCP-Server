"""이익 횡재 vs 저축 행태 변화: 추가 통제 후보를 하나씩 점검(기준 = 자기 블록 + 성장률 + 정부 두 성분 시차).

후보(각 2시차, 정상성 판정에 따라 수준/차분): 물가상승률, 실업률, 민간신용/GDP, 실질금리, 상품 교역조건 증가율,
  노년부양비, 경상수지, 다른 민간 부문 두 성분.
판정: 연도 평균 제거 후 국가별 ADF, 과반이 10%에서 단위근 기각 → 수준, 아니면 차분(ep_source_controls와 같은 규칙).
점검: (i) 후보 시차가 충격(x_P, x_H)과 결과(h0, h3)를 예측하는가(군집 강건 F), 둘 다 예측해야 누락 시 편의 가능.
      (ii) 후보를 넣었을 때 지속성(1–2년, 3–6년 평균)이 바뀌는가 — 같은 부분표본의 기준 사양과 비교.
→ output/ep_wb_candidates.csv, output/paper/fig_wb_candidates.(png|pdf)
"""
import sys
import warnings

import numpy as np
import pandas as pd

import ep_iv_industry as e
import ep_source_controls as ec
import ep_wb_extra as ex
import ep_wb_gov as eg
import ep_wb_three as e3
import ep_windfall_behavior as wb
import lp_tools as lt
from make_report import OUT

warnings.filterwarnings("ignore")
H, SEC = wb.H, wb.SEC
CAND = {"infl": "Inflation", "unemp": "Unemployment rate", "credit": "Private credit / GDP", "real_interest_rate": "Real interest rate",
        "tot_goods_g": "Goods ToT growth", "old_age_dependency": "Old-age dependency", "ca": "Current account"}


def panel():
    p = pd.read_csv(OUT.parent / "data" / "panel.csv").merge(pd.read_csv(OUT.parent / "data" / "raw" / "wb_controls.csv"), on=["country", "year"], how="left")
    p["infl"] = p.infl.clip(-10, 50)
    return lt.balance(p)


def transforms(p):
    old = ec.CAND
    ec.CAND = list(CAND)
    ur = ec.unit_root(p)
    ec.CAND = old
    return dict(zip(ur["var"], ur["transform"])), ur


def data():
    q, _ = e3.data()
    q = q.dropna(subset=eg.GL + eg.GN + ["g_Y", "g_C"]).copy()
    p = panel()
    tr, ur = transforms(p)
    g = p.groupby("country")
    new = {}
    for v in CAND:
        s = p[v] if tr[v] == "level" else g[v].diff()
        for l in (1, 2):
            new[f"K{l}_{v}"] = s.groupby(p.country).shift(l)
    k = pd.concat([p[["country", "year"]], pd.DataFrame(new, index=p.index)], axis=1)
    q = q.drop(columns=[c for c in new if c in q.columns]).merge(k, on=["country", "year"], how="left")
    return q, tr, ur


def other_lags(s):
    o = "hh" if s == "corp" else "corp"
    return [f"L{l}_{v}" for v in (SEC[o]["x"], SEC[o]["z"]) for l in (1, 2)]


def run():
    q, tr, ur = data()
    print(ur[["var", "share_reject10", "ar1_fe", "transform"]].round(2).to_string(index=False))
    base = lambda s: wb.ctrl(s) + eg.GL
    specs = [(f"{'' if tr[v] == 'level' else 'Δ'}{lab}", lambda s, v=v: [f"K{l}_{v}" for l in (1, 2)]) for v, lab in CAND.items()]
    specs += [("Other private sector's components", other_lags)]
    rows = []
    for name, fn in specs:
        for s, m in SEC.items():
            Z = fn(s)
            qq = wb.weights(q.dropna(subset=Z).copy())
            F = {}
            for dep in (m["x"], f"{m['y']}_h0", f"{m['y']}_h3"):
                C = base(s) + ([m["z"]] if dep == m["x"] else [m["x"], m["z"]])
                F[dep] = e.first_stage(qq.assign(w_nl_corp=qq[f"w_{s}"]), Z, C, dep)[0]
            D = ex.Delta()
            D.add("b", *ex.design(qq, [m["x"], m["z"]], base(s), f"w_{s}", ex.ycols(m["y"])))
            D.add("c", *ex.design(qq, [m["x"], m["z"]], base(s) + Z, f"w_{s}", ex.ycols(m["y"])))
            D.finish()
            y = m["y"]
            for lab, lo, hi in (("1-2", 1, 3), ("3-6", 3, 7)):
                avg = lambda d, t, lo=lo, hi=hi: np.mean([d[f"{t}|{y}_h{h}"] for h in range(lo, hi)]) / d[f"{t}|{y}_h0"]
                bb, sb = D.fn(lambda d: avg(d, "b"))
                bc, sc = D.fn(lambda d: avg(d, "c"))
                bd, sd = D.fn(lambda d: avg(d, "c") - avg(d, "b"))
                rows.append(dict(candidate=name, sector=s, years=lab, base=bb, base_se=sb, added=bc, added_se=sc, change=bd, change_se=sd,
                                 F_shock=F[m["x"]], F_h0=F[f"{y}_h0"], F_h3=F[f"{y}_h3"], n=len(qq), countries=qq.country.nunique()))
    r = pd.DataFrame(rows)
    r.to_csv(OUT / "ep_wb_candidates.csv", index=False)
    return r


def figure(r):
    import fig_paper as fp
    plt = fp.plt
    cands = list(dict.fromkeys(r.candidate))
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 0.42 * len(cands) + 1.6), sharey=True)
    y = np.arange(len(cands))[::-1]
    for ax, s, cl, title in ((axes[0], "corp", fp.CO, "(a) Profit windfall, corporate S–I"), (axes[1], "hh", fp.HH, "(b) Saving-rate shift, household S–I")):
        for years, mk, dy in (("1-2", "o", 0.14), ("3-6", "D", -0.14)):
            t = r[(r.sector == s) & (r.years == years)].set_index("candidate").loc[cands]
            ax.errorbar(t.added, y + dy, xerr=1.645 * t.added_se, fmt=mk, color=cl, ms=4, lw=0.8, capsize=0,
                        label=f"Years {years}, with candidate (90% CI)")
            ax.scatter(t.base, y + dy, marker="|", s=90, color="0.2", zorder=4, label="Baseline on the same sample (ticks)" if years == "1-2" else None)
        ax.axvline(0, color=".3", lw=.6)
        ax.set_title(title, loc="left", fontsize=9.3)
        ax.set_xlabel("Average response (impact = 1)")
        ax.grid(axis="y", color=".92", lw=.6)
    lab = []
    for c in cands:
        t = r[(r.candidate == c)]
        fc = t[t.sector == "corp"].iloc[0]
        fh = t[t.sector == "hh"].iloc[0]
        lab.append(f"{c}\nF shock {fc.F_shock:.1f} | {fh.F_shock:.1f};  N = {int(fc.n)}")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(lab, fontsize=7)
    h_, l_ = axes[1].get_legend_handles_labels()
    fig.legend(h_, l_, loc="lower center", bbox_to_anchor=(0.55, -0.07), ncol=3, fontsize=7, frameon=False)
    fig.suptitle("Additional control candidates", x=0.0, ha="left", fontweight="bold", y=1.0)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.text(0.0, -0.1, "Notes: Baseline: own-sector balance (levels), the sector's two components, GDP growth and the two government components, two lags each, "
             "country and year fixed effects, two-step weighted least squares. Each row adds two lags of one candidate (levels if a majority of country ADF tests "
             "reject a unit root, first differences otherwise); markers with 90 percent intervals (country-cluster delta method), vertical ticks show the "
             "baseline on the same sample. 'F shock': cluster-robust F of the candidate's lags in predicting the corporate | household shock.",
             fontsize=6.9, wrap=True, va="top")
    fp.save(fig, "fig_wb_candidates", sfx="")


if __name__ == "__main__":
    r = pd.read_csv(OUT / "ep_wb_candidates.csv") if "--render" in sys.argv else run()
    figure(r)
    pd.set_option("display.width", 220)
    print(r.round(2).to_string(index=False))
