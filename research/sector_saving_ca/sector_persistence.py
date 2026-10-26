"""부문 자체의 지속성 비교: 기업이익 주도 에피소드 vs 비침체기 가계저축률 급등 (경상수지 제외).

기업 에피소드 정의(시작 구간 t−2→t, IRL·LUX 제외, 같은 국가 4년 내 중복 제외):
  D1 이익 주도 기업저축: Δsv_corp ≥ 2, Σx_P(t−1,t) ≥ 0.5·Δsv_corp, Δprof ≥ 1.5   (GNI %p)
  D2 교역조건 횡재:      Σtot_g(t−1,t) ≥ 10%, Δprof ≥ 1.5
  D3 이익비중 급등:      Δprof ≥ 3
가계: hh_episodes.py 의 비침체 유형
지표: 연도별 국가평균 제거 후 t−2 대비 변화를 시작 시점 변화로 나눈 잔존 비율
  기업: 영업잉여 비중(이익), 기업저축, 기업 순자금운용, 기업투자  ← 분모: 기업저축 변화(이익은 이익 변화)
  가계: 가계저축률, 가계저축, 가계 순자금운용, 가계소비          ← 분모: 가계저축 변화(저축률은 저축률 변화)
→ output/sector_persistence.csv
"""
import numpy as np
import pandas as pd

import ca_episodes as ce
import run_analysis as ra

K = 6
EXCL = {"IRL", "LUX"}
VARS = ["prof", "sv_corp", "nl_corp", "inv_corp", "hh_rate", "sv_hh", "nl_hh", "cons_hh", "sv_nat"]


def panel():
    d = ce.prepare()
    for v in VARS:
        d[v + "_dm"] = d[v] - d.groupby("year")[v].transform("mean")
    return d


def corp_episodes(d, rule, gap=4):
    rows = []
    for c, g in d[~d.country.isin(EXCL)].groupby("country"):
        g = g.set_index("year")
        last = -99
        for t in g.index:
            if t - 2 not in g.index or t - last <= gap:
                continue
            dsv = g.at[t, "sv_corp"] - g.at[t - 2, "sv_corp"]
            dpr = g.at[t, "prof"] - g.at[t - 2, "prof"]
            xp = g.loc[t - 1:t, "x_P"].sum()
            tot = g.loc[t - 1:t, "tot_g"].sum(min_count=2)
            ok = {"D1": dsv >= 2 and xp >= 0.5 * dsv and dpr >= 1.5,
                  "D2": (tot >= 10) if pd.notna(tot) else False,
                  "D3": dpr >= 3}[rule]
            if rule == "D2":
                ok = ok and dpr >= 1.5
            if ok:
                last = t
                rows.append(dict(country=c, year=t, dsv=dsv, dpr=dpr, tot=tot))
    return pd.DataFrame(rows)


def paths(d, ep, var, base):
    """var 의 (t+k − t−2) / base 의 (t − t−2), 연도평균 제거."""
    p = d.set_index(["country", "year"])
    M = []
    for r in ep.itertuples():
        b0 = p[base + "_dm"].get((r.country, r.year), np.nan) - p[base + "_dm"].get((r.country, r.year - 2), np.nan)
        if not (abs(b0) > 0.3):
            continue
        v0 = p[var + "_dm"].get((r.country, r.year - 2), np.nan)
        M.append([(p[var + "_dm"].get((r.country, r.year + k), np.nan) - v0) / b0 for k in range(-2, K + 1)])
    return pd.DataFrame(M, columns=range(-2, K + 1))


def main():
    d = panel()
    he = pd.read_csv(ra.OUT / "hh_episodes.csv")
    hh = he[he.type == "비침체"][["country", "year"]]
    groups = {"가계: 비침체 저축률 급등": (hh, [("hh_rate", "hh_rate"), ("sv_hh", "sv_hh"), ("nl_hh", "sv_hh"), ("cons_hh", "sv_hh"), ("sv_nat", "sv_hh")])}
    for rule, lab in (("D1", "기업: 이익 주도 기업저축 급증"), ("D2", "기업: 교역조건 횡재"), ("D3", "기업: 이익비중 급등")):
        ep = corp_episodes(d, rule)
        groups[lab] = (ep, [("prof", "prof"), ("sv_corp", "sv_corp"), ("nl_corp", "sv_corp"), ("inv_corp", "sv_corp"), ("sv_nat", "sv_corp")])
        print(lab, len(ep), ", ".join(f"{r.country}{r.year}" for r in ep.itertuples()))
    rows = []
    for g, (ep, specs) in groups.items():
        for var, base in specs:
            P = paths(d, ep, var, base)
            for k in P:
                x = P[k].dropna()
                rows.append(dict(group=g, var=var, base=base, k=k, n=len(x), mean=x.mean(), median=x.median(),
                                 se=x.std() / np.sqrt(len(x)) if len(x) > 1 else np.nan))
    out = pd.DataFrame(rows)
    out.to_csv(ra.OUT / "sector_persistence.csv", index=False)
    piv = out[out.k.isin([0, 1, 2, 3, 6])].pivot_table(index=["group", "var"], columns="k", values="median").round(2)
    nn = out[out.k == 6].set_index(["group", "var"]).n
    print(piv.assign(n6=nn).to_string())
    print(out[out.k.isin([1, 3, 6])].pivot_table(index=["group", "var"], columns="k", values="mean").round(2).to_string())


if __name__ == "__main__":
    main()
