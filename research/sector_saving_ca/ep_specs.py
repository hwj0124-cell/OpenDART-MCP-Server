"""이익 횡재(기업 순대출) vs 저축 행태 변화(가계 순대출): 표본 × 통제 사양별 반응, h = 0…6, 시점별 비균형 표본.

기본 통제: 자기 부문 순대출 수준, 자기 부문 두 성분, 성장률(각 2시차) + 국가·연도 FE, 2단계 WLS(시작 연도 식 잔차의 국가별 분산 역수).
정부 통제(선택): 정부 순대출 수준 2시차(정부를 나누지 않고 통째로).
표본(시점마다 관측되는 행을 모두 사용 → 시점별로 표본이 다름):
  A 전체 기간, 시작 연도 ≤ 2019        B 전체 기간, 2020년 이후 시작 포함
  C 공통 구간(1998년부터, 1998년에 이미 관측되는 국가만), ≤ 2019     D 공통 구간, 2020년 이후 포함
  두 통제 사양이 같은 행을 쓰도록 정부 시차가 없는 행은 처음부터 제외.
구간: 국가 군집 델타 방법(시점마다 다른 행이어도 국가 단위 영향함수로 공분산 계산), 68·90%.
→ output/ep_specs.csv, output/paper/fig_spec_A…D.(png|pdf), fig_spec_overview.(png|pdf)
"""
import sys

import numpy as np
import pandas as pd

import ep_wb_extra as ex
import ep_windfall_behavior as wb
import lp_tools as lt
from make_report import OUT

H, SEC = 7, wb.SEC
GOV = ["L1_nl_gov", "L2_nl_gov"]
SAMPLES = {"A": "All years, start years up to 2019", "B": "All years, including start years 2020–24",
           "C": "Common period (1998–), start years up to 2019", "D": "Common period (1998–), including start years 2020–24"}
CTRL = {"base": "Own sector and growth", "gov": "+ government net lending (two lags)"}


def data():
    d, q = wb.data()
    g = d.groupby("country")
    d["L1_nl_gov"], d["L2_nl_gov"] = g.nl_gov.shift(1), g.nl_gov.shift(2)
    need = sorted({c for s in SEC for c in [SEC[s]["x"], SEC[s]["z"]] + wb.ctrl(s)} | set(GOV))
    d = d[d.country.isin(q.country.unique())].dropna(subset=need).copy()
    first = d.groupby("country").year.min()
    common = first[first <= 1998].index
    S = {"A": d[d.year <= 2019], "B": d,
         "C": d[(d.year >= 1998) & (d.year <= 2019) & d.country.isin(common)], "D": d[(d.year >= 1998) & d.country.isin(common)]}
    return {k: v.copy() for k, v in S.items()}


def design_h(q, X, C, w, y):
    """시점마다 관측되는 행으로 따로 추정. 반환: 첫 계수(시점별), 국가별 영향함수."""
    bs, ps = {}, []
    for h in range(H):
        col = f"{y}_h{h}"
        b, psi = ex.design(q, X, C, w, [col])
        bs[col] = b[col]
        ps.append(psi)
    return pd.Series(bs), pd.concat(ps, axis=1)


def weights(q, s, C):
    m = SEC[s]
    r = q.dropna(subset=[f"{m['y']}_h0"])
    X = [m["x"], m["z"]]
    M = np.hstack([r[X + C].to_numpy(float), lt._fe_matrix(r, ("country", "year"))])
    e = r[f"{m['y']}_h0"].to_numpy(float) - M @ np.linalg.lstsq(M, r[f"{m['y']}_h0"].to_numpy(float), rcond=None)[0]
    v = pd.Series(e, index=r.index).groupby(r.country).var()
    return 1 / q.country.map(v)


def run():
    S = data()
    rows = []
    for sk, q in S.items():
        for ck in CTRL:
            D = ex.Delta()
            ns = {}
            for s, m in SEC.items():
                C = wb.ctrl(s) + (GOV if ck == "gov" else [])
                q[f"w_{s}"] = weights(q, s, C)
                D.add(s, *design_h(q, [m["x"], m["z"]], C, f"w_{s}", m["y"]))
                ns[s] = [int(q[f"{m['y']}_h{h}"].notna().sum()) for h in range(H)]
            D.finish()
            for s, m in SEC.items():
                y = m["y"]
                for h in range(H):
                    b, se = D.fn(lambda d, h=h, s=s, y=y: d[f"{s}|{y}_h{h}"] / d[f"{s}|{y}_h0"])
                    rows.append(dict(sample=sk, ctrl=ck, sector=s, h=h, b=b, se=se, lo68=b - se, hi68=b + se, lo90=b - 1.645 * se,
                                     hi90=b + 1.645 * se, n=ns[s][h], countries=q.country.nunique(), first=int(q.year.min()), last=int(q.year.max())))
    r = pd.DataFrame(rows)
    r.to_csv(OUT / "ep_specs.csv", index=False)
    return r


def figs(r):
    import fig_paper as fp
    plt = fp.plt
    col = {"corp": fp.CO, "hh": fp.HH}
    title = {"corp": "Corporate net lending after a profit windfall", "hh": "Household net lending after a shift in the saving rate"}
    xl = "Years after the saving change ($h$)"
    for sk, sname in SAMPLES.items():
        fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.3), sharey=True)
        rs = r[r["sample"] == sk]
        for ax, s in zip(axes, ("corp", "hh")):
            b = rs[(rs.ctrl == "base") & (rs.sector == s)].sort_values("h")
            ax.fill_between(b.h, b.lo90, b.hi90, color=col[s], alpha=0.09, lw=0)
            ax.fill_between(b.h, b.lo68, b.hi68, color=col[s], alpha=0.18, lw=0)
            ax.plot(b.h, b.b, color=col[s], lw=1.9, marker="o", ms=3.3, label=CTRL["base"] + " (68/90% bands)")
            g = rs[(rs.ctrl == "gov") & (rs.sector == s)].sort_values("h")
            ax.plot(g.h, g.b, color="0.15", lw=1.3, ls="--", marker="s", ms=2.8, label=CTRL["gov"])
            fp.finish(ax, "Net lending (impact = 1)" if s == "corp" else "", h0=0)
            ax.set_xlabel(xl)
            ax.set_xlim(-0.3, H - 0.7)
            ax.set_xticks(range(H))
            ax.set_title(title[s], loc="left", fontsize=8.8)
            ax.set_ylim(-0.9, 1.15)
            ax.legend(loc="upper right", fontsize=6.8, frameon=False)
            ax.text(0.02, 0.03, f"N = {int(b.n.iloc[0])} at h = 0, {int(b.n.iloc[-1])} at h = 6", transform=ax.transAxes, fontsize=6.8, color="0.3")
        k, a, z = int(rs.countries.iloc[0]), int(rs["first"].iloc[0]), int(rs["last"].iloc[0])
        fig.suptitle(f"Sample {sk}: {sname}", x=0.0, ha="left", fontweight="bold", y=1.03)
        fig.tight_layout()
        fig.text(0.0, -0.05, f"Notes: {k} economies, start years {a}–{z}; every horizon uses all start years with the outcome observed, so the sample shrinks "
                 "with the horizon. Local projections of each sector's net lending on the profit-windfall (corporations) or saving-rate (households) component "
                 "and the remaining component, with two lags of the sector's net lending (levels), both components and GDP growth, and country and year fixed "
                 "effects; two-step weighted least squares. Dashed: adding two lags of government net lending. Normalized by the impact response. Bands: "
                 "country-cluster delta method." + (" 'Common period': economies already observed in 1998." if sk in "CD" else ""),
                 fontsize=6.9, wrap=True, va="top")
        fp.save(fig, f"fig_spec_{sk}", sfx="")

    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.4), sharey=True)
    sty = {"A": ("-", 1.9, "o"), "B": ("--", 1.4, None), "C": ("-.", 1.4, None), "D": (":", 1.6, None)}
    for ax, s in zip(axes, ("corp", "hh")):
        for sk in SAMPLES:
            b = r[(r["sample"] == sk) & (r.ctrl == "base") & (r.sector == s)].sort_values("h")
            if sk == "A":
                ax.fill_between(b.h, b.lo68, b.hi68, color=col[s], alpha=0.16, lw=0)
            ls, lw, mk = sty[sk]
            ax.plot(b.h, b.b, color=col[s] if sk == "A" else "0.2", ls=ls, lw=lw, marker=mk, ms=3.2, label=f"{sk}: {SAMPLES[sk]}")
        fp.finish(ax, "Net lending (impact = 1)" if s == "corp" else "", h0=0)
        ax.set_xlabel(xl)
        ax.set_xlim(-0.3, H - 0.7)
        ax.set_xticks(range(H))
        ax.set_title(title[s], loc="left", fontsize=8.8)
        ax.set_ylim(-0.9, 1.15)
    h_, l_ = axes[0].get_legend_handles_labels()
    fig.legend(h_, l_, loc="lower center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=7, frameon=False)
    fig.suptitle("Four samples, baseline controls", x=0.0, ha="left", fontweight="bold", y=1.03)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.text(0.0, -0.15, "Notes: Baseline controls (own sector and growth). Shaded: 68 percent band for sample A. Horizon-specific (unbalanced) samples.",
             fontsize=7, wrap=True, va="top")
    fp.save(fig, "fig_spec_overview", sfx="")


if __name__ == "__main__":
    r = pd.read_csv(OUT / "ep_specs.csv") if "--render" in sys.argv else run()
    figs(r)
    pd.set_option("display.width", 220)
    t = r.pivot_table(index=["sample", "ctrl", "sector"], columns="h", values="b", sort=False).round(2)
    print(t.to_string())
    a = r.assign(per=np.where(r.h.between(1, 2), "1-2", np.where(r.h >= 3, "3-6", "0")))
    print(a[a.per != "0"].pivot_table(index=["sample", "ctrl", "sector"], columns="per", values="b", sort=False).round(2).to_string())
    print(r[r.h.isin([0, 6])].pivot_table(index=["sample", "sector"], columns="h", values="n", aggfunc="first").to_string())
    print(r[(r.h == 5)][["sample", "ctrl", "sector", "b", "lo90", "hi90"]].round(2).to_string(index=False))
