"""세 부문: 기업(이익 횡재), 가계(저축 행태), 정부(수입 원천 vs 지출 원천).

정부 저축 분해(가법): Δs^G ≈ g_Y + g_C,  g_Y = Δ처분가능소득(수입 원천),  g_C = −Δ최종소비(지출 원천)   (GNI 대비 %)
  참고로 곱셈 분해(저축률 ρ = s/YD): Δs^G = ρ_{t−1}ΔYD + YD_tΔρ — 적자국에서 ρ < 0이라 수입 증가가 음의 기여로 잡힘.
(1) 부문별 두 성분의 분산 비중: var(a)/var(a+b), var(b)/var(a+b), 2cov/var(a+b).
(2) LP(델타 방법 구간):
  기업·가계: 기준 사양 + 정부 두 성분의 시차(t−1, t−2)를 통제에 추가(새 기준). 기존 기준과 비교.
  정부: n^G_{t+h} − n^G_{t−1} ~ g_Y + g_C, 시차 2개[n^G 수준, g_Y, g_C, 성장률], 국가·연도 FE, 2단계 WLS. 성분별 정규화.
→ output/ep_wb_three.csv, output/ep_wb_three_var.csv, output/paper/fig_wb_three.(png|pdf)
"""
import sys

import numpy as np
import pandas as pd

import ep_wb_extra as ex
import ep_wb_gov as eg
import ep_windfall_behavior as wb
import lp_tools as lt
from make_report import OUT

H, SEC = wb.H, wb.SEC
GOVC = [f"L{l}_{v}" for v in ("nl_gov", "g_Y", "g_C", "gdp_g") for l in (1, 2)]


def data():
    q = eg.data()
    p = eg.gov_series().merge(lt.balance(pd.read_csv(OUT.parent / "data" / "panel.csv"))[["country", "year", "sv_gov"]], on=["country", "year"], how="right")
    p = lt.balance(p.sort_values(["country", "year"]))
    g = p.groupby("country")
    p["g_Y"], p["g_C"] = g.yd_gov.diff(), -g.cons_gov.diff()
    p["d_sv_gov"] = g.sv_gov.diff()
    rho = p.sv_gov / p.yd_gov
    p["m_Y"] = rho.groupby(p.country).shift(1) * g.yd_gov.diff()
    p["m_R"] = p.yd_gov * rho.groupby(p.country).diff()
    new = {f"L{l}_{v}": p.groupby("country")[v].shift(l) for v in ("g_Y", "g_C") for l in (1, 2)}
    p = pd.concat([p, pd.DataFrame(new, index=p.index)], axis=1)
    cols = ["g_Y", "g_C", "d_sv_gov", "m_Y", "m_R"] + list(new)
    q = q.drop(columns=[c for c in cols if c in q.columns]).merge(p[["country", "year"] + cols], on=["country", "year"], how="left")
    return q, p


def var_shares(df, a, b, label):
    x = df[[a, b]].dropna()
    tot = (x[a] + x[b]).var()
    return dict(sector=label, comp_a=a, comp_b=b, share_a=x[a].var() / tot, share_b=x[b].var() / tot, share_2cov=2 * x[a].cov(x[b]) / tot,
                sd_a=x[a].std(), sd_b=x[b].std(), n=len(x))


def run():
    q, p = data()
    full = lt.balance(pd.read_csv(OUT.parent / "data" / "panel.csv"))
    import source_shocks as ss
    full = ss.prepare(full)
    vs = [var_shares(full, "x_P", "x_R", "corporate (panel)"), var_shares(full, "x_H", "x_Y", "household (panel)"),
          var_shares(p, "g_Y", "g_C", "government, additive (panel)"), var_shares(p, "m_Y", "m_R", "government, saving-rate (panel)")]
    q = q.dropna(subset=eg.GL + eg.GN + [f"L{l}_{v}" for v in ("g_Y", "g_C") for l in (1, 2)] + ["g_Y", "g_C"]).copy()
    vs += [var_shares(q, "x_P", "x_R", "corporate (LP sample)"), var_shares(q, "x_H", "x_Y", "household (LP sample)"),
           var_shares(q, "g_Y", "g_C", "government, additive (LP sample)"), var_shares(q, "m_Y", "m_R", "government, saving-rate (LP sample)")]
    chk = p[["d_sv_gov", "g_Y", "g_C"]].dropna()
    print(f"government identity check: corr(Δs, g_Y + g_C) = {chk.d_sv_gov.corr(chk.g_Y + chk.g_C):.3f}, "
          f"sd of residual {(chk.d_sv_gov - chk.g_Y - chk.g_C).std():.2f} vs sd Δs {chk.d_sv_gov.std():.2f}")
    v = pd.DataFrame(vs)
    v.to_csv(OUT / "ep_wb_three_var.csv", index=False)
    print(v.round(2).to_string(index=False))

    q = wb.weights(q)
    X, C = ["g_C", "g_Y"], GOVC
    M = np.hstack([q[X + C].to_numpy(float), lt._fe_matrix(q, ("country", "year"))])
    e = q["nl_gov_h0"].to_numpy(float) - M @ np.linalg.lstsq(M, q["nl_gov_h0"].to_numpy(float), rcond=None)[0]
    q["w_gov"] = 1 / q.country.map(pd.Series(e, index=q.index).groupby(q.country).var())

    D = ex.Delta()
    for s, m in SEC.items():
        D.add(f"{s}_base", *ex.design(q, [m["x"], m["z"]], wb.ctrl(s), f"w_{s}", ex.ycols(m["y"])))
        D.add(f"{s}_gov", *ex.design(q, [m["x"], m["z"]], wb.ctrl(s) + eg.GL, f"w_{s}", ex.ycols(m["y"])))
    D.add("gov_C", *ex.design(q, ["g_C", "g_Y"], GOVC, "w_gov", ex.ycols("nl_gov")))
    D.add("gov_Y", *ex.design(q, ["g_Y", "g_C"], GOVC, "w_gov", ex.ycols("nl_gov")))
    D.finish()
    rows = []

    def emit(key, f):
        for h in range(H):
            b, se = D.fn(lambda d, h=h: f(d, h))
            rows.append(dict(key=key, h=h, b=b, se=se, lo68=b - se, hi68=b + se, lo90=b - 1.645 * se, hi90=b + 1.645 * se))
    for s, m in SEC.items():
        for lab in ("base", "gov"):
            emit(f"{s}_{lab}", lambda d, h, t=f"{s}_{lab}", y=m["y"]: d[f"{t}|{y}_h{h}"] / d[f"{t}|{y}_h0"])
    for t in ("gov_C", "gov_Y"):
        emit(t, lambda d, h, t=t: d[f"{t}|nl_gov_h{h}"] / d[f"{t}|nl_gov_h0"])
    emit("diff_hh_corp_gov", lambda d, h: d[f"hh_gov|nl_hh_h{h}"] / d["hh_gov|nl_hh_h0"] - d[f"corp_gov|nl_corp_h{h}"] / d["corp_gov|nl_corp_h0"])
    imp = {t: D.b[f"{t}|nl_gov_h0"] for t in ("gov_C", "gov_Y")}
    print("government impact responses (per 1pp):", {k: round(v, 2) for k, v in imp.items()})
    avg = []
    for key, f in (("diff_hh_corp_gov", lambda d, lo, hi: np.mean([d[f"hh_gov|nl_hh_h{h}"] / d["hh_gov|nl_hh_h0"]
                                                                  - d[f"corp_gov|nl_corp_h{h}"] / d["corp_gov|nl_corp_h0"] for h in range(lo, hi)])),
                   ("gov_C", lambda d, lo, hi: np.mean([d[f"gov_C|nl_gov_h{h}"] for h in range(lo, hi)]) / d["gov_C|nl_gov_h0"]),
                   ("gov_Y", lambda d, lo, hi: np.mean([d[f"gov_Y|nl_gov_h{h}"] for h in range(lo, hi)]) / d["gov_Y|nl_gov_h0"])):
        for lab, lo, hi in (("1-2", 1, 3), ("3-6", 3, 7)):
            b, se = D.fn(lambda d: f(d, lo, hi))
            avg.append(dict(key=key, years=lab, b=b, se=se, lo90=b - 1.645 * se, hi90=b + 1.645 * se))
    print(pd.DataFrame(avg).round(2).to_string(index=False))
    r = pd.DataFrame(rows).assign(n=len(q), countries=q.country.nunique())
    r.to_csv(OUT / "ep_wb_three.csv", index=False)
    return r, v


def figure(r, v):
    import fig_paper as fp
    plt = fp.plt
    g = lambda k: r[r.key == k].sort_values("h")
    GV = "#6a8f3c"
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.4), gridspec_kw=dict(width_ratios=[1, 1, 0.85]))
    ax = axes[0]
    for s, cl, lab in (("corp", fp.CO, "Profit windfall (corporate S–I)"), ("hh", fp.HH, "Saving-rate shift (household S–I)")):
        b = g(f"{s}_gov")
        ax.fill_between(b.h, b.lo68, b.hi68, color=cl, alpha=0.15, lw=0)
        ax.plot(b.h, b.b, color=cl, lw=1.9, marker="o", ms=3.2, label=lab)
        c = g(f"{s}_base")
        ax.plot(c.h, c.b, color=cl, lw=1.0, ls=":", label="Without government lags" if s == "hh" else None)
    fp.finish(ax, "Own-sector S–I (impact = 1)", h0=0)
    ax.set_xlabel("Years after the saving change ($h$)")
    ax.set_title("(a) Corporations and households", loc="left", fontsize=9.3)
    ax.set_ylim(-0.6, 1.1)
    ax.legend(loc="upper right", fontsize=6.8, frameon=False)
    ax = axes[1]
    for t, ls, lab in (("gov_C", "-", "Spending-driven (lower consumption)"), ("gov_Y", "--", "Revenue-driven (higher disposable income)")):
        b = g(t)
        ax.fill_between(b.h, b.lo68, b.hi68, color=GV, alpha=0.15 if t == "gov_C" else 0.08, lw=0)
        ax.plot(b.h, b.b, color=GV, lw=1.9, ls=ls, marker="o" if t == "gov_C" else "s", ms=3.2, label=lab)
    fp.finish(ax, "Government S–I (impact = 1)", h0=0)
    ax.set_xlabel("Years after the saving change ($h$)")
    ax.set_title("(b) Government", loc="left", fontsize=9.3)
    ax.set_ylim(-0.6, 1.75)
    ax.legend(loc="lower left", fontsize=6.8, frameon=False)
    ax = axes[2]
    vv = v[v.sector.str.contains("LP sample")].copy()
    hh = vv.sector.str.startswith("household")
    vv.loc[hh, ["share_a", "share_b"]] = vv.loc[hh, ["share_b", "share_a"]].to_numpy()   # 소득형 먼저
    names = {"corporate (LP sample)": "Corporate\nprofit | retention", "household (LP sample)": "Household\nincome | saving rate",
             "government, additive (LP sample)": "Government\nrevenue | spending", "government, saving-rate (LP sample)": "Government, rate split\nrevenue | saving rate"}
    y = np.arange(len(vv))[::-1]
    a, b, c = vv.share_a.to_numpy(), vv.share_b.to_numpy(), vv.share_2cov.to_numpy()
    ax.barh(y, a, color="#c98a1b", label="Income-type (profit, income, revenue)")
    ax.barh(y, b, left=a, color="#2b6c8f", label="Behavior-type (retention, saving rate, spending)")
    ax.barh(y, np.abs(c), left=np.where(c >= 0, a + b, a + b - np.abs(c)), color="0.75", label="2 × covariance")
    for yi, ai, bi in zip(y, a, b):
        if ai > 0.1:
            ax.text(ai / 2, yi, f"{ai:.0%}", va="center", ha="center", fontsize=7, color="white")
        if bi > 0.1:
            ax.text(ai + bi / 2, yi, f"{bi:.0%}", va="center", ha="center", fontsize=7, color="white")
    ax.set_yticks(y)
    ax.set_yticklabels([names[s] for s in vv.sector], fontsize=7.2)
    ax.set_xlim(0, 1.15)
    ax.set_xlabel("Share of variance of the change in saving")
    ax.set_title("(c) Variance shares", loc="left", fontsize=9.3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.3, -0.2), fontsize=6.8, frameon=False, ncol=1)
    fig.suptitle("Three domestic sectors", x=0.0, ha="left", fontweight="bold", y=1.03)
    fig.tight_layout()
    n, k = int(r.n.iloc[0]), int(r.countries.iloc[0])
    fig.text(0.0, -0.22, f"Notes: N = {n}, {k} economies. (a) Each sector's balance on its two components, with two lags of the sector's balance (levels), both "
             "components, GDP growth and the two government components, and country and year fixed effects; dotted lines omit the government lags. "
             "(b) Government balance on the change in its disposable income and the negative change in its final consumption, with the same structure of "
             "controls. 68 percent bands, country-cluster delta method. (c) Variance of each component relative to the variance of their sum; the "
             "covariance term can be negative. 'Saving-rate split' applies the household-style decomposition to government saving, whose saving rate "
             "is negative in deficit years.", fontsize=7, wrap=True, va="top")
    fp.save(fig, "fig_wb_three", sfx="")


if __name__ == "__main__":
    if "--render" in sys.argv:
        r, v = pd.read_csv(OUT / "ep_wb_three.csv"), pd.read_csv(OUT / "ep_wb_three_var.csv")
    else:
        r, v = run()
    figure(r, v)
    a = r.assign(per=np.where(r.h.between(1, 2), "1-2", np.where(r.h >= 3, "3-6", "0")))
    print(r.pivot_table(index="key", columns="h", values="b", sort=False).round(2).to_string())
    print(a[a.per != "0"].pivot_table(index="key", columns="per", values="b", sort=False).round(2).to_string())
