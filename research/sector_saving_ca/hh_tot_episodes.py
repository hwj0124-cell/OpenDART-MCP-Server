"""가계 성향 vs 교역조건 에피소드: 순자금운용·저축·투자 경로 (68% 구간), 시기 제외 민감도.

에피소드(sector_persistence2.flags 와 동일, 상·하위 Q, 증가·감소 통합, 국가별 4년 내 중복 제외)
  가계 성향: 가계저축률 2년 변화 상·하위 Q, 시작 구간 성장률 ≥ 0, 시작 2020–21 제외 → 가계 부문
  교역조건 : 교역조건 2년 누적 변화 상·하위 Q, 영업잉여 비중이 같은 방향, IRL·LUX 제외 → 비금융법인
경로: 같은 해 국가평균 제거 후 (v_{t+k} − v_{t−2}) / (nl_t − nl_{t−2}),  v ∈ {순자금운용, 저축, 고정투자, 기타}
      기타 = 순자금운용 − (저축 − 고정투자) = 재고증감·자본이전·비생산자산 등
      → 순자금운용 = 저축 − 투자 이므로 세 경로가 같은 분모로 맞춰짐 (시작 시점 순자금운용 = 1)
민감도: 기본 / 코로나 제외(시작 2019–2021 제외: 이후 경로가 2020–21에 걸침) / 코로나·금융위기 제외(추가로 2007–2009)
→ output/hh_tot_episodes.csv, output/hh_tot_episodes.png
"""
import argparse

import numpy as np
import pandas as pd

import run_analysis as ra
import sector_persistence2 as sp2

K = 6
SECT = {"가계 성향": ("nl_hh", "sv_hh", "inv_hh"), "교역조건": ("nl_corp", "sv_corp", "inv_corp")}
VARIANTS = {"기본": set(), "코로나 제외": {2019, 2020, 2021},
            "코로나·금융위기 제외": {2007, 2008, 2009, 2019, 2020, 2021}}


def main(q):
    d = sp2.panel()
    for v in ("inv_hh", "inv_corp"):
        d[v + "_dm"] = d[v] - d.groupby("year")[v].transform("mean")
    dp = d.set_index(["country", "year"])
    fl = sp2.flags(d, q)
    rows, cases = [], []
    for name, (nl, sv, inv) in SECT.items():
        ep = pd.concat([sp2.dedupe(d, fl[(name, "증가")]), sp2.dedupe(d, fl[(name, "감소")])])
        for var_lab, drop in VARIANTS.items():
            e = ep[~ep.year.isin(drop)]
            M = {v: [] for v in (nl, sv, inv)}
            used = []
            for r in e.itertuples():
                den = dp[nl + "_dm"].get((r.country, r.year), np.nan) - dp[nl + "_dm"].get((r.country, r.year - 2), np.nan)
                if not abs(den) > 0.3:
                    continue
                used.append(f"{r.country}{r.year}")
                for v in M:
                    b = dp[v + "_dm"].get((r.country, r.year - 2), np.nan)
                    M[v].append([(dp[v + "_dm"].get((r.country, r.year + k), np.nan) - b) / den for k in range(-2, K + 1)])
            cases.append(dict(type=name, variant=var_lab, n=len(used), cases=", ".join(used)))
            # 기타 = 순자금운용 − (저축 − 고정투자): 재고증감·자본이전 등 (사례별로 계산해 합이 맞도록)
            M["other"] = [[a - (b - c) if not (np.isnan(a) or np.isnan(b) or np.isnan(c)) else np.nan
                           for a, b, c in zip(rn, rs, ri)] for rn, rs, ri in zip(M[nl], M[sv], M[inv])]
            for v, mat in M.items():
                P = pd.DataFrame(mat, columns=range(-2, K + 1))
                kind = {nl: "순자금운용", sv: "저축", inv: "투자", "other": "기타(재고·자본이전 등)"}[v]
                for k in P:
                    x = P[k].dropna()
                    rows.append(dict(type=name, variant=var_lab, kind=kind, k=k, n=len(x), mean=x.mean(),
                                     median=x.median(), se=x.std() / np.sqrt(len(x)) if len(x) > 1 else np.nan))
    out = pd.DataFrame(rows)
    out.to_csv(ra.OUT / f"hh_tot_episodes_q{int(q * 100)}.csv", index=False)
    pd.DataFrame(cases).to_csv(ra.OUT / f"hh_tot_episodes_cases_q{int(q * 100)}.csv", index=False)
    print(pd.DataFrame(cases)[["type", "variant", "n"]].to_string())
    print(out[out.k.isin([0, 1, 2, 3, 4, 6])].pivot_table(index=["variant", "type", "kind"], columns="k", values="mean").round(2).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", type=float, default=0.10)
    main(ap.parse_args().q)
