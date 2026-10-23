"""경상수지 반응의 지속성: 충격 시점 경상수지 +1%p 로 정규화한 원천별 비교.

LP-IV:  y_{t+h} − y_{t−1} = β_h · (ca_t − ca_{t−1}) + 통제 + 국가·연도 FE,
        (ca_t − ca_{t−1}) 을 원천 충격(x_P 또는 x_H)으로 도구화.
→ β_h(ca) 는 h=0 에서 1. 이후 경로가 '그 원천에서 생긴 경상수지 1단위가 얼마나 남는가'.
   같은 정규화로 이익·부문저축·국민저축 경로도 본다(이익은 남는데 경상수지는 사라지는가).

대형충격: x_s 를 x_s·Big_s 와 x_s·(1−Big_s) 로 나눠(Big = |x_s| 상위 10%) 큰 충격만으로 식별.

    python persistence.py [--reps 299]
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import robustness_samples as rs
import run_analysis as ra
import source_shocks as ss
from lp_tools import cluster_bootstrap, iv, lagged

H = ra.H
OUTS = ["ca", "prof", "sv_corp", "sv_hh", "hh_rate", "sv_nat", "nl_corp", "nl_hh", "nl_gov", "inv_corp", "inv_hh"]
SRC = {"P": "x_P", "H": "x_H"}


def prepare(panel):
    d = ss.prepare(panel)
    extra = lagged(d, ["x_P", "x_H", "x_R", "x_Y", "d_ca", "gdp_g"], 4)
    extra = extra[[c for c in extra if c[:2] in ("L3", "L4")]]
    more = lagged(d, ["tot_g", "real_interest_rate", "old_age_dependency", "nl_gov"], 1)
    d = pd.concat([d, extra, more], axis=1)
    for s, x in SRC.items():
        a = d[x].abs()
        big = (a >= a.quantile(0.9)).astype(float).where(d[x].notna())
        d[f"{x}_big"] = d[x] * big
        d[f"{x}_small"] = d[x] * (1 - big)
    return d


def controls(variant):
    lags = 4 if variant == "lags4" else 2
    c = ([f"L{l}_{v}" for v in ss.SRC + ss.OTHER + ["d_ca", "gdp_g"] for l in range(1, lags + 1)]
         + [f"L1_{v}" for v in ["sv_corp", "sv_hh", "ca", "prof", "hh_rate"]] + ss.OTHER)
    if variant == "growth":
        c += ["gdp_g"]
    if variant == "tot":
        c += ["tot_g", "L1_tot_g"]
    if variant == "macro":
        c += ["L1_real_interest_rate", "L1_old_age_dependency", "L1_nl_gov"]
    return c


def irf(d, s, variant, y, se=True):
    """원천 s 충격, 충격시점 CA 1단위 정규화 반응 (h=0..H)."""
    other = SRC["H" if s == "P" else "P"]
    c = controls(variant) + [other]
    z = SRC[s]
    if variant == "big":  # 큰 충격만으로 식별, 작은 충격은 통제
        z = f"{SRC[s]}_big"
        c = c + [f"{SRC[s]}_small"]
    rows = []
    for h in range(H + 1):
        b, e, n, F = iv(d, f"{y}_h{h}", "ca_h0", z, c, se=se)
        rows.append((b, e, n, F))
    return rows


def stats(d, variant, extra=False):
    out = {}
    for s in SRC:
        ca = np.array([r[0] for r in irf(d, s, variant, "ca", se=False)])
        out[f"{s}_ca_13"] = ca[1:4].mean()    # 1~3년 평균 잔존
        out[f"{s}_ca_46"] = ca[4:7].mean()    # 4~6년 평균 잔존
        out[f"{s}_ca_cum"] = ca.sum()         # 0~6년 누적
        if extra:  # 점추정에만: 이익·국민저축의 4~6년 잔존
            for y in ("prof", "sv_nat"):
                r = np.array([q[0] for q in irf(d, s, variant, y, se=False)])
                out[f"{s}_{y}_46"] = r[4:7].mean()
    for k in ("ca_13", "ca_46", "ca_cum"):
        out[f"diff_{k}"] = out[f"H_{k}"] - out[f"P_{k}"]  # 문서 주장: > 0 (가계 성향 쪽이 더 오래)
    return out


VARIANTS = {"base": "기본 (2시차·직전 수준·국가/연도 FE)", "growth": "+ 당해 실질성장률",
            "tot": "+ 교역조건 변화(당해·전년)", "macro": "+ 실질금리·고령화·재정수지(전년)",
            "lags4": "시차 4년", "big": "대형충격(상위 10%)만으로 식별"}


def main(reps, part=None):
    panel = pd.read_csv(ra.HERE / "data" / "panel.csv")
    d = prepare(panel)
    smp = rs.samples(panel)
    keep = {k: v for k, v in smp.items() if k.split(" (")[0] in ("전체", "MNE 허브 제외", "체제전환국 제외", "선진 핵심국만")}
    if part is not None:  # 병렬 실행: 표본 하나만
        keep = dict([list(keep.items())[part]])
    rows, paths = [], []
    for name, drop in keep.items():
        ds = ra.sample(d, (), drop)
        for v, vlab in VARIANTS.items():
            for s in SRC:
                for y in OUTS:
                    for h, (b, e, n, F) in enumerate(irf(ds, s, v, y)):
                        paths.append(dict(sample=name, variant=v, shock=s, outcome=y, h=h, b=b, se=e, n=n, F=F))
            point = stats(ds, v)
            keys = list(point)
            point |= stats(ds, v, extra=True)
            need = list(dict.fromkeys(["country", "year", "ca_h0"] + list(SRC.values())
                                      + [f"{x}_{k}" for x in SRC.values() for k in ("big", "small")]
                                      + controls(v) + [f"ca_h{h}" for h in range(H + 1)]))
            draws = cluster_bootstrap(ds[need], lambda bd: list(stats(bd, v).values()), reps=reps)
            F0 = {s: next(p["F"] for p in paths[::-1] if p["sample"] == name and p["variant"] == v
                          and p["shock"] == s and p["outcome"] == "ca" and p["h"] == 1) for s in SRC}
            n0 = next(p["n"] for p in paths[::-1] if p["sample"] == name and p["variant"] == v and p["h"] == 0)
            for k in point:
                i = keys.index(k) if k in keys else None
                rows.append(dict(sample=name, variant=v, key=k, point=point[k], n=n0, F_P=F0["P"], F_H=F0["H"],
                                 lo90=np.nanpercentile(draws[:, i], 5) if i is not None else np.nan,
                                 hi90=np.nanpercentile(draws[:, i], 95) if i is not None else np.nan))
            print(name.split(" (")[0], v, f"n={n0} F_P={F0['P']:.0f} F_H={F0['H']:.0f}",
                  " ".join(f"{k}={point[k]:.2f}" for k in ("P_ca_13", "H_ca_13", "P_ca_46", "H_ca_46")), flush=True)
    suf = "" if part is None else f"_part{part}"
    pd.DataFrame(rows).to_csv(ra.OUT / f"persistence{suf}.csv", index=False)
    pd.DataFrame(paths).to_csv(ra.OUT / f"persistence_paths{suf}.csv", index=False)


def merge(n=4):
    for stem in ("persistence", "persistence_paths"):
        parts = [ra.OUT / f"{stem}_part{i}.csv" for i in range(n)]
        pd.concat([pd.read_csv(p) for p in parts]).to_csv(ra.OUT / f"{stem}.csv", index=False)
        for p in parts:
            p.unlink()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=299)
    ap.add_argument("--part", type=int)
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()
    merge() if a.merge else main(a.reps, a.part)
