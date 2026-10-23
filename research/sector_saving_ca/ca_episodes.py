"""경상수지 에피소드의 지속성과 원천 (41개국 전체, 서술적 분석).

에피소드: 2년간 경상수지(국민경제 순대출, GNI 대비) 변화 |ca_t − ca_{t−2}| ≥ THR %p 인 첫 해 t
          (같은 국가에서 이전 에피소드 시작 후 GAP년 이내는 제외). 흑자 확대(+)와 적자 확대(−) 모두.
          적자 확대는 부호를 뒤집어 '크기'로 통일.
지속성:   R_k = (ca_{t+k} − ca_{t−2}) / (ca_t − ca_{t−2}),  k = 3, 5
원천 분해(시작 구간 t−2 → t, 모두 GNI %p, 합계 = ΔCA):
   부문 순자금운용: Δnl_corp + Δnl_hh + Δnl_gov + Δnl_fin
   저축 원천:      이익 원천(Σ ret_{-1}·Δprof) + 유보율 변화 + 성향 원천(Σ yd_{-1}·Δrate) + 가계소득 비중 변화
                   + Δ정부저축 + 기타(금융법인 저축·자본이전) − Δ투자
주도 부문: 기업 순자금운용 기여 ≥ 50% → 기업 주도, 가계 기여 ≥ 50% → 가계 주도

    python ca_episodes.py [--thr 3] [--gap 4]
"""
import argparse

import numpy as np
import pandas as pd

import run_analysis as ra
from lp_tools import balance

K = 6


def prepare(path=None):
    d = balance(pd.read_csv(path or ra.HERE / "data" / "panel.csv"))
    g = d.groupby("country")
    d["ret"] = d["sv_corp"] / d["prof"]
    d["x_P"] = g["ret"].shift(1) * g["prof"].diff()
    d["x_R"] = d["prof"] * d.groupby("country")["ret"].diff()
    d["x_H"] = g["yd_hh"].shift(1) * g["hh_rate"].diff() / 100
    d["x_Y"] = d["hh_rate"] * g["yd_hh"].diff() / 100
    d["inv_tot"] = d[["inv_corp", "inv_hh", "inv_gov"]].sum(axis=1, min_count=3)
    return d


def find(d, thr, gap):
    eps = []
    for c, gg in d.groupby("country"):
        gg = gg.set_index("year")
        last = -99
        for t in gg.index:
            if t - 2 not in gg.index or t - last <= gap:
                continue
            dca = gg.at[t, "ca"] - gg.at[t - 2, "ca"]
            if np.isnan(dca) or abs(dca) < thr:
                continue
            last = t
            s = np.sign(dca)
            e = dict(country=c, year=t, sign="흑자 확대" if s > 0 else "적자 확대", size=abs(dca))
            base = gg.at[t - 2, "ca"]
            for k in range(-3, K + 1):
                v = gg["ca"].get(t + k, np.nan)
                e[f"ca_{k}"] = s * (v - base) / abs(dca)  # 시작 변화 = 1 정규화
            for k in (3, 5):
                e[f"R{k}"] = e[f"ca_{k}"]
            w = lambda v: s * (gg.at[t, v] - gg.at[t - 2, v]) / abs(dca) if v in gg else np.nan
            for v in ("nl_corp", "nl_hh", "nl_gov", "nl_fin", "sv_corp", "sv_hh", "sv_gov", "sv_nat",
                      "inv_corp", "inv_hh", "inv_gov", "inv_tot", "prof", "hh_rate"):
                e[f"c_{v}"] = w(v)
            for v in ("x_P", "x_R", "x_H", "x_Y"):
                e[f"c_{v}"] = s * gg.loc[t - 1:t, v].sum() / abs(dca)
            # 이후 경로(정규화): 이익·부문저축·순자금운용
            for v in ("prof", "sv_corp", "sv_hh", "sv_nat", "nl_corp", "nl_hh", "nl_gov", "inv_tot"):
                b0 = gg[v].get(t - 2, np.nan)
                for k in range(-3, K + 1):
                    e[f"{v}_{k}"] = s * (gg[v].get(t + k, np.nan) - b0) / abs(dca)
            eps.append(e)
    ep = pd.DataFrame(eps)
    ep["driver"] = np.select([ep.c_nl_corp >= 0.5, ep.c_nl_hh >= 0.5, ep.c_nl_gov >= 0.5],
                             ["기업 주도", "가계 주도", "정부 주도"], "혼합")
    ep["persist"] = np.select([ep.R3 < 0.5, ep.R5 >= 0.5], ["단명", "지속"], "중간")
    ep.loc[ep.R3.isna() | ep.R5.isna(), "persist"] = "미확정"
    return ep


def summarize(ep):
    full = ep[ep.persist != "미확정"]
    print(f"에피소드 {len(ep)}개 (지속성 판정 가능 {len(full)}개)")
    print("\n[주도 부문별 지속성] R3, R5 = 시작 변화 중 3·5년 뒤 남은 비율")
    print(full.groupby("driver")[["R3", "R5"]].agg(["mean", "median", "count"]).round(2).to_string())
    print("\n[주도 부문 × 지속성 분포]")
    print(pd.crosstab(full.driver, full.persist, margins=True).to_string())
    print("\n[지속성별 시작 변화의 원천 구성, 평균]")
    cols = ["c_nl_corp", "c_nl_hh", "c_nl_gov", "c_nl_fin", "c_x_P", "c_x_R", "c_x_H", "c_x_Y", "c_sv_gov", "c_inv_tot"]
    print(full.groupby("persist")[cols].mean().round(2).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--thr", type=float, default=3.0)
    ap.add_argument("--gap", type=int, default=4)
    a = ap.parse_args()
    ep = find(prepare(), a.thr, a.gap)
    ep.to_csv(ra.OUT / f"ca_episodes_thr{a.thr:g}.csv", index=False)
    summarize(ep)
