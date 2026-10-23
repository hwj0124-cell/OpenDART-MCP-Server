"""가계저축 급등과 재정의 관계: 정부 흡수인가, 재정이 원인인가.

1) 비침체기 가계저축률 급등 46개를 시작 구간(t−2→t)의 정부 순대출 변화로 양분해 경로 비교
2) LP: 성향 충격(x_H)의 경상수지 반응을 당해 Δ정부순대출 통제 전후로 비교
3) LP: 재정 충격(Δ정부저축 −1%p)에 대한 가계 가처분소득·저축·저축률·경상수지 반응
→ output/fiscal_offset_*.csv
"""
import numpy as np
import pandas as pd

import run_analysis as ra
import source_shocks as ss
from lp_tools import add_lp_columns, balance, lagged, ols

H = ra.H


def episode_split():
    pan = balance(pd.read_csv(ra.HERE / "data" / "panel.csv"))
    V = ["hh_rate", "sv_hh", "cons_hh", "nl_hh", "nl_gov", "ca", "yd_hh", "inv_hh", "sv_corp"]
    for v in V:
        pan[v + "_dm"] = pan[v] - pan.groupby("year")[v].transform("mean")
    pan = pan.set_index(["country", "year"])
    he = pd.read_csv(ra.OUT / "hh_episodes.csv")
    g = he[he.type == "비침체"].copy()
    g["dgov"] = [pan["nl_gov_dm"].get((r.country, r.year), np.nan) - pan["nl_gov_dm"].get((r.country, r.year - 2), np.nan)
                 for r in g.itertuples()]
    g["grp"] = np.where(g.dgov >= 0, "재정 개선·유지", "재정 악화")
    rows = []
    for grp, gg in g.dropna(subset=["dgov"]).groupby("grp"):
        for v in V:
            M = [[pan[v + "_dm"].get((r.country, r.year + k), np.nan) - pan[v + "_dm"].get((r.country, r.year - 2), np.nan)
                  for k in range(-2, H + 1)] for r in gg.itertuples()]
            P = pd.DataFrame(M, columns=range(-2, H + 1))
            for k in P:
                rows.append(dict(grp=grp, n=len(gg), var=v, k=k, mean=P[k].mean(),
                                 se=P[k].std() / np.sqrt(P[k].notna().sum())))
    pd.DataFrame(rows).to_csv(ra.OUT / "fiscal_offset_episodes.csv", index=False)


def lp():
    d = ss.prepare(pd.read_csv(ra.HERE / "data" / "panel.csv"))
    g = d.groupby("country")
    d["d_nl_gov"], d["d_sv_gov"] = g.nl_gov.diff(), g.sv_gov.diff()
    d = pd.concat([d, lagged(d, ["d_nl_gov", "d_sv_gov"], 2)], axis=1)
    for y in ["sv_gov", "nl_gov", "nl_hh", "hh_rate", "yd_hh", "cons_hh"]:
        for h in range(H + 1):
            d[f"{y}_h{h}"] = add_lp_columns(d, y, h)
    rows = []
    for lab, extra in [("통제 없음", []), ("재정 통제", ["d_nl_gov", "L1_d_nl_gov", "L2_d_nl_gov"])]:
        for y in ["ca", "nl_gov", "nl_hh", "sv_hh"]:
            for h in range(H + 1):
                b, V, n = ols(d, f"{y}_h{h}", ["x_P", "x_H"], ss.controls() + extra)
                rows.append(dict(spec="성향 충격 " + lab, y=y, h=h, b=b[1], se=np.sqrt(V[1, 1]), n=n))
    ctrl = ["L1_d_sv_gov", "L2_d_sv_gov", "L1_d_ca", "L2_d_ca", "L1_gdp_g", "L2_gdp_g", "L1_sv_corp", "L1_sv_hh", "L1_ca"]
    for y in ["sv_gov", "sv_hh", "hh_rate", "yd_hh", "cons_hh", "nl_hh", "ca"]:
        for h in range(H + 1):
            b, V, n = ols(d, f"{y}_h{h}", ["d_sv_gov"], ctrl)
            rows.append(dict(spec="재정 충격(정부저축 −1%p)", y=y, h=h, b=-b[0], se=np.sqrt(V[0, 0]), n=n))
    pd.DataFrame(rows).to_csv(ra.OUT / "fiscal_offset_lp.csv", index=False)


if __name__ == "__main__":
    episode_split()
    lp()
    print("saved")
