"""가계저축률 급등 에피소드의 지속성 (41개국 전체, 서술적 분석).

에피소드: 가계저축률(B8G/B6G, %) 2년 변화 ≥ THR %p 인 첫 해 t (같은 국가 GAP년 내 중복 제외)
지속성:   변수 v 의 (v_{t+k} − v_{t−2}) / (hh_rate_t − hh_rate_{t−2})  ← 가계저축률은 자기 자신 기준(=1)
          가계저축·순자금운용·경상수지(GNI %)는 시작 시점 가계저축(GNI %) 증가분 기준
유형(시작 구간 t−2→t, 우선순위 순):
  코로나    t ∈ {2020, 2021}
  부동산    가계 투자(P51G, GNI %)가 1%p 이상 감소
  경기침체  t−1 또는 t 의 실질성장률 < 0
  비침체    그 밖(행태 변화에 가까운 경우)
비교: 기업 영업잉여 비중(GNI %) 2년 변화 ≥ THR %p 인 에피소드의 같은 지표
→ output/hh_episodes.csv
"""
import argparse

import numpy as np
import pandas as pd

import run_analysis as ra
from lp_tools import balance

K = 6


def find(d, var, thr, gap):
    rows = []
    for c, g in d.groupby("country"):
        g = g.set_index("year")
        last = -99
        for t in g.index:
            if t - 2 not in g.index or t - last <= gap:
                continue
            ch = g.at[t, var] - g.at[t - 2, var]
            if np.isnan(ch) or ch < thr:
                continue
            last = t
            e = dict(country=c, year=t, size=ch)
            base_sv = g.at[t, "sv_hh"] - g.at[t - 2, "sv_hh"] if var == "hh_rate" else g.at[t, "sv_corp"] - g.at[t - 2, "sv_corp"]
            e["size_gni"] = base_sv
            for v in ("hh_rate", "sv_hh", "nl_hh", "ca", "sv_corp", "prof", "nl_gov", "inv_hh", "cons_hh"):
                b0 = g[v].get(t - 2, np.nan)
                den = ch if v == var else base_sv
                for k in range(-2, K + 1):
                    e[f"{v}_{k}"] = (g[v].get(t + k, np.nan) - b0) / den if den and abs(den) > 0.3 else np.nan
            e["d_inv_hh"] = g.at[t, "inv_hh"] - g.at[t - 2, "inv_hh"]
            e["recession"] = (g["gdp_g"].get(t - 1, np.nan) < 0) or (g["gdp_g"].get(t, np.nan) < 0)
            rows.append(e)
    ep = pd.DataFrame(rows)
    ep["type"] = np.select([ep.year.isin([2020, 2021]), ep.d_inv_hh <= -1, ep.recession],
                           ["코로나", "부동산 붕괴", "경기침체"], "비침체")
    return ep


def summarize(ep, var, label):
    print(f"\n== {label}: {len(ep)}개")
    cols = {f"{var}_{k}": f"{var} t+{k}" for k in (3, 5, 6)}
    cols.update({f"nl_hh_{k}": f"nl_hh t+{k}" for k in (3, 6)} if var == "hh_rate" else {})
    cols.update({f"ca_{k}": f"CA t+{k}" for k in (0, 3, 6)})
    g = ep.groupby("type") if var == "hh_rate" else ep.assign(type="전체").groupby("type")
    out = g[list(cols)].median().rename(columns=cols).round(2)
    out.insert(0, "n", g.size())
    print(out.to_string())
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--thr", type=float, default=3.0)
    ap.add_argument("--gap", type=int, default=4)
    a = ap.parse_args()
    d = balance(pd.read_csv(ra.HERE / "data" / "panel.csv"))
    hh = find(d, "hh_rate", a.thr, a.gap)
    hh.to_csv(ra.OUT / "hh_episodes.csv", index=False)
    summarize(hh, "hh_rate", "가계저축률 급등 (중위수)")
    summarize(hh.assign(type="전체"), "hh_rate", "가계저축률 급등 전체")
    pr = find(d, "prof", a.thr, a.gap)
    pr.to_csv(ra.OUT / "prof_episodes.csv", index=False)
    summarize(pr, "prof", "비교: 기업 영업잉여 비중 급등")
    print(hh.sort_values("size", ascending=False).head(25)[["country", "year", "type", "size", "size_gni",
          "hh_rate_3", "hh_rate_6", "nl_hh_6", "ca_0", "ca_6"]].round(2).to_string())
