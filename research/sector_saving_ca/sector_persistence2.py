"""부문 지속성(대칭·상대 기준): 증가와 감소, 저축과 순자금운용 기준.

에피소드(시작 구간 t−2→t, 2년 변화의 국가 전체 분포 기준 상·하위 Q, 같은 국가·같은 방향 4년 내 중복 제외):
  이익 충격    Δprof 상/하위 Q, 그리고 |Σx_P| ≥ 0.5|Δsv_corp| (기업저축 변화의 절반 이상이 이익 원천, 같은 부호)
  교역조건     Σtot_g 상/하위 Q, Δprof 같은 부호
  가계 성향    Δhh_rate 상/하위 Q, 시작 구간 실질성장률 ≥ 0, 2020–21 제외
  (기업 유형은 IRL·LUX 제외)
지속성: 같은 해 국가평균 제거 후 (v_{t+k} − v_{t−2}) / (v_t − v_{t−2}), v = 자기 부문 저축 또는 순자금운용
        감소 에피소드도 같은 식(비율이라 부호가 자동 정렬)
→ output/sector_persistence2.csv
"""
import argparse

import numpy as np
import pandas as pd

import ca_episodes as ce
import run_analysis as ra

K = 6
EXCL = {"IRL", "LUX"}
SECTOR = {"이익 충격": ("sv_corp", "nl_corp"), "교역조건": ("sv_corp", "nl_corp"), "가계 성향": ("sv_hh", "nl_hh")}


def panel():
    d = ce.prepare()
    g = d.groupby("country")
    d["d_prof"] = d["prof"] - g["prof"].shift(2)
    d["d_hh"] = d["hh_rate"] - g["hh_rate"].shift(2)
    d["d_svc"] = d["sv_corp"] - g["sv_corp"].shift(2)
    d["xP2"] = d["x_P"] + g["x_P"].shift(1)
    d["tot2"] = d["tot_g"] + g["tot_g"].shift(1)
    d["g2min"] = np.minimum(d["gdp_g"], g["gdp_g"].shift(1))
    for v in ("sv_corp", "nl_corp", "sv_hh", "nl_hh", "prof", "sv_nat"):
        d[v + "_dm"] = d[v] - d.groupby("year")[v].transform("mean")
    return d


def flags(d, q):
    out = {}
    corp = ~d.country.isin(EXCL)
    for name, x in (("이익 충격", "d_prof"), ("교역조건", "tot2"), ("가계 성향", "d_hh")):
        lo, hi = d[x].quantile(q), d[x].quantile(1 - q)
        for sgn, cond in ((1, d[x] >= hi), (-1, d[x] <= lo)):
            f = cond.copy()
            if name == "이익 충격":
                f &= corp & (np.sign(d.xP2) == sgn) & (d.xP2.abs() >= 0.5 * d.d_svc.abs()) & (np.sign(d.d_svc) == sgn)
            elif name == "교역조건":
                f &= corp & (np.sign(d.d_prof) == sgn)
            else:
                f &= (d.g2min >= 0) & ~d.year.isin([2020, 2021])
            out[(name, "증가" if sgn > 0 else "감소")] = f.fillna(False)
    return out


def dedupe(d, f, gap=4):
    keep, last = [], {}
    for i, (c, y) in zip(d.index[f], zip(d.country[f], d.year[f])):
        if y - last.get(c, -99) > gap:
            keep.append(i)
            last[c] = y
    return d.loc[keep, ["country", "year"]]


def paths(dp, ep, v):
    M = []
    for r in ep.itertuples():
        b = dp[v + "_dm"].get((r.country, r.year - 2), np.nan)
        den = dp[v + "_dm"].get((r.country, r.year), np.nan) - b
        if not abs(den) > 0.3:
            continue
        M.append([(dp[v + "_dm"].get((r.country, r.year + k), np.nan) - b) / den for k in range(0, K + 1)])
    return pd.DataFrame(M, columns=range(0, K + 1))


def main(q):
    d = panel()
    dp = d.set_index(["country", "year"])
    rows, cases = [], []
    fl = flags(d, q)
    eps = {key: dedupe(d, f) for key, f in fl.items()}
    for name in SECTOR:  # 증가·감소 통합(비율이라 부호 정렬됨)
        eps[(name, "전체")] = pd.concat([eps[(name, "증가")], eps[(name, "감소")]])
    for (name, dirn), ep in eps.items():
        cases.append(dict(type=name, dir=dirn, n=len(ep), cases=", ".join(f"{a}{b}" for a, b in ep.values)))
        sv, nl = SECTOR[name]
        for basis, v in (("저축", sv), ("순자금운용", nl)):
            P = paths(dp, ep, v)
            for k in P:
                x = P[k].dropna()
                rows.append(dict(type=name, dir=dirn, basis=basis, k=k, n=len(x), mean=x.mean(), median=x.median(),
                                 se=x.std() / np.sqrt(len(x)) if len(x) > 1 else np.nan))
    out = pd.DataFrame(rows)
    out.to_csv(ra.OUT / f"sector_persistence2_q{int(q * 100)}.csv", index=False)
    pd.DataFrame(cases).to_csv(ra.OUT / f"sector_persistence2_cases_q{int(q * 100)}.csv", index=False)
    print(pd.DataFrame(cases)[["type", "dir", "n"]].to_string())
    t = out[out.k.isin([1, 3, 6])].pivot_table(index=["basis", "type", "dir"], columns="k", values="median").round(2)
    t["n6"] = out[out.k == 6].set_index(["basis", "type", "dir"]).n
    print(t.to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", type=float, default=0.10)
    main(ap.parse_args().q)
