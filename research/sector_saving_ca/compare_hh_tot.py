"""가계 성향 vs 교역조건: 자기 부문 순자금운용의 지속성 (에피소드 + LP).

LP(에피소드 교차항):
  nl_s(t+h) − nl_s(t−2) = β_h·[Δ2 nl_s,t × D_t] + γ_h·[Δ2 nl_s,t × (1−D_t)] + δ·D_t
                          + 통제(t−3 이전 Δ2 순자금운용·성장률·수준) + 국가·연도 FE
  → β_h 는 '원천 충격이 있던 해의 순자금운용 변화'가 h년 뒤 남는 비율 (h=0 에서 1)
  가계: D = 가계저축률 2년 변화 상·하위 q (성장률 ≥ 0, 2020–21 제외), s = 가계
  교역조건: D = 교역조건 2년 변화 상·하위 q 이고 이익이 같은 방향, s = 비금융법인 (IRL·LUX 제외)
참고 LP-IV(평균 충격): nl_hh(t+h) − nl_hh(t−1) 를 nl_hh 당해 변화에 회귀, 성향 충격 x_H 로 도구화
→ output/compare_hh_tot_lp.csv
"""
import numpy as np
import pandas as pd

import run_analysis as ra
import sector_persistence2 as sp2
from lp_tools import add_lp_columns, iv, lagged, ols

H = ra.H


def build():
    d = sp2.panel()
    g = d.groupby("country")
    for s in ("hh", "corp"):
        v = f"nl_{s}"
        d[f"d2_{v}"] = d[v] - g[v].shift(2)
        for h in range(H + 1):
            d[f"{v}_x{h}"] = g[v].shift(-h) - g[v].shift(2)
        d[f"L3_d2_{v}"] = g[f"d2_{v}"].shift(3)
        d[f"L3_{v}"] = g[v].shift(3)
    d["L3_gdp_g"] = g["gdp_g"].shift(3)
    d["L3_ca"] = g["ca"].shift(3)
    return d


def dummies(d, q):
    lo, hi = d.d_hh.quantile(q), d.d_hh.quantile(1 - q)
    Dh = ((d.d_hh >= hi) | (d.d_hh <= lo)) & (d.g2min >= 0) & ~d.year.isin([2020, 2021])
    lo, hi = d.tot2.quantile(q), d.tot2.quantile(1 - q)
    Dt = (((d.tot2 >= hi) & (d.d_prof > 0)) | ((d.tot2 <= lo) & (d.d_prof < 0))) & ~d.country.isin(sp2.EXCL)
    return Dh.fillna(False).astype(float), Dt.fillna(False).astype(float)


def interaction_lp(d, D, s, mask=None):
    v = f"nl_{s}"
    dd = d.copy() if mask is None else d[mask].copy()
    dd["D"] = D.loc[dd.index]
    dd["xD"] = dd[f"d2_{v}"] * dd.D
    dd["xN"] = dd[f"d2_{v}"] * (1 - dd.D)
    ctrl = ["D", "L3_d2_nl_hh", "L3_d2_nl_corp", "L3_nl_hh", "L3_nl_corp", "L3_gdp_g", "L3_ca"]
    out = []
    for h in range(H + 1):
        b, V, n = ols(dd, f"{v}_x{h}", ["xD", "xN"], ctrl)
        out.append(dict(h=h, b=b[0], se=np.sqrt(V[0, 0]), b_other=b[1], se_other=np.sqrt(V[1, 1]), n=n,
                        n_events=int(dd.loc[dd[f"{v}_x{h}"].notna(), "D"].sum())))
    return pd.DataFrame(out)


def hh_iv():
    import persistence as P
    d = P.prepare(pd.read_csv(ra.HERE / "data" / "panel.csv"))
    g = d.groupby("country")
    d["d_nl_corp"], d["d_nl_hh"] = g.nl_corp.diff(), g.nl_hh.diff()
    d = pd.concat([d, lagged(d, ["d_nl_corp", "d_nl_hh"], 2)], axis=1)
    for h in range(H + 1):
        d[f"nl_hh_h{h}"] = add_lp_columns(d, "nl_hh", h)
    base = ["L1_d_nl_corp", "L2_d_nl_corp", "L1_d_nl_hh", "L2_d_nl_hh", "L1_d_ca", "L2_d_ca", "L1_gdp_g", "L2_gdp_g",
            "L1_sv_corp", "L1_sv_hh", "L1_ca", "x_P", "L1_x_H", "L2_x_H"]
    rows = []
    for h in range(H + 1):
        b, se, n, F = iv(d, f"nl_hh_h{h}", "nl_hh_h0", "x_H", base)
        rows.append(dict(h=h, b=b, se=se, n=n, F=F))
    return pd.DataFrame(rows)


def main():
    d = build()
    res = []
    for q in (0.10, 0.20):
        Dh, Dt = dummies(d, q)
        for lab, D, s in (("가계 성향", Dh, "hh"), ("교역조건", Dt, "corp")):
            r = interaction_lp(d, D, s).assign(type=lab, q=int(q * 100), spec="에피소드 교차항 LP")
            res.append(r)
    iv_ = hh_iv().assign(type="가계 성향", q=0, spec="LP-IV 평균 충격")
    out = pd.concat(res + [iv_], ignore_index=True)
    out.to_csv(ra.OUT / "compare_hh_tot_lp.csv", index=False)
    for (spec, t, q), g in out.groupby(["spec", "type", "q"], sort=False):
        print(f"{spec} {t} q{q}:", " ".join(f"{r.b:5.2f}({r.se:.2f})" for r in g.itertuples()),
              f"events h0={g.n_events.iloc[0] if 'n_events' in g and g.n_events.notna().any() else '-'}")


if __name__ == "__main__":
    main()
