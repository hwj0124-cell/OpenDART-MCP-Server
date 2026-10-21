"""원천별 저축 충격: '기업이익이 원천인 저축' vs '가계 성향이 원천인 저축'.

부문저축 변화를 원천별로 정확히 분해(모두 GNI 대비 %p):
  Δsv_corp = ret_{t-1}·Δprof_t          (x_P: 이익 원천)   + prof_t·Δret_t        (x_R: 유보율 변화)
  Δsv_hh   = yd_{t-1}·Δrate_t / 100     (x_H: 성향 원천)   + rate_t·Δyd_t / 100   (x_Y: 가처분소득 변화)
  ret = 기업저축/영업잉여, rate = 가계저축률(%), yd = 가계 가처분소득/GNI(%)

x_P 와 x_H 를 동시에 넣고 x_R, x_Y 는 같은 시점 통제로 둔 LP. 계수는 '해당 원천으로
그해 저축이 1%p 늘 때' h년 뒤 경상수지 등의 변화이므로 두 원천을 같은 단위로 비교할 수 있다.

    python source_shocks.py [--reps 499]
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import robustness_samples as rs
import run_analysis as ra
from lp_tools import add_lp_columns, balance, cluster_bootstrap, lagged, ols

H, LAGS = ra.H, ra.LAGS
SHORT, MEDIUM = (0, 1), (4, 5, 6)
SRC = ["x_P", "x_H"]
OTHER = ["x_R", "x_Y"]
OUTCOMES = ["ca", "sv_nat", "sv_corp", "sv_hh", "sv_gov", "nl_corp", "nl_hh", "nl_gov",
            "prof", "hh_rate", "div_paid", "tax_paid", "inv_corp", "inv_hh", "cons_hh"]


def prepare(panel):
    d = balance(panel)
    g = d.groupby("country")
    d["ret"] = d["sv_corp"] / d["prof"]
    lag = lambda v: g[v].shift(1)
    d["x_P"] = lag("ret") * g["prof"].diff()
    d["x_R"] = d["prof"] * d.groupby("country")["ret"].diff()
    d["x_H"] = lag("yd_hh") * g["hh_rate"].diff() / 100
    d["x_Y"] = d["hh_rate"] * g["yd_hh"].diff() / 100
    d["d_ca"] = g["ca"].diff()
    d["recession"] = d["gdp_g"] < 0
    levels = ["sv_corp", "sv_hh", "ca", "prof", "hh_rate"]
    d = pd.concat([d, lagged(d, SRC + OTHER + ["d_ca", "gdp_g"], LAGS), lagged(d, levels, 1)], axis=1)
    lp = {f"{y}_h{h}": add_lp_columns(d, y, h) for y in OUTCOMES for h in range(H + 1)}
    return pd.concat([d, pd.DataFrame(lp, index=d.index)], axis=1)


def controls(extra=()):
    return ([f"L{l}_{v}" for v in SRC + OTHER + ["d_ca", "gdp_g"] for l in range(1, LAGS + 1)]
            + [f"L1_{v}" for v in ["sv_corp", "sv_hh", "ca", "prof", "hh_rate"]] + OTHER + list(extra))


def irf(d, y, extra=(), se=True):
    rows = []
    for h in range(H + 1):
        b, V, n = ols(d, f"{y}_h{h}", SRC, controls(extra), se=se)
        r = dict(outcome=y, h=h, n=n, b_P=b[0], b_H=b[1])
        if se:
            r.update(se_P=np.sqrt(V[0, 0]), se_H=np.sqrt(V[1, 1]),
                     se_diff=np.sqrt(V[0, 0] + V[1, 1] - 2 * V[0, 1]))
        rows.append(r)
    return pd.DataFrame(rows)


def contrasts(d, extra=()):
    out = {}
    for y in ("ca", "sv_nat"):
        r = irf(d, y, extra, se=False).set_index("h")
        for s in ("P", "H"):
            sh, md = r.loc[list(SHORT), f"b_{s}"].mean(), r.loc[list(MEDIUM), f"b_{s}"].mean()
            out[f"{y}_{s}_short"], out[f"{y}_{s}_medium"] = sh, md
            out[f"{y}_{s}_decline"] = sh - md
        out[f"{y}_dd"] = out[f"{y}_P_decline"] - out[f"{y}_H_decline"]       # 문서 주장: > 0
        out[f"{y}_medium_diff"] = out[f"{y}_H_medium"] - out[f"{y}_P_medium"]  # 문서 주장: > 0
        out[f"_{y}"] = r
    # 누적 전가율: 0~6년 누적 경상수지 / 누적 국민저축 (늘어난 국민저축 중 국외로 나간 몫)
    ca, sn = out.pop("_ca"), out.pop("_sv_nat")
    for s in ("P", "H"):
        out[f"ptc_{s}"] = ca[f"b_{s}"].sum() / sn[f"b_{s}"].sum()
    out["ptc_diff"] = out["ptc_H"] - out["ptc_P"]  # 문서 주장: > 0
    return out


def boot(d, reps, extra=()):
    # 부트스트랩 메모리·속도: 필요한 열만 남긴다
    need = (["country", "year"] + SRC + controls(extra)
            + [f"{y}_h{h}" for y in ("ca", "sv_nat") for h in range(H + 1)])
    d = d[list(dict.fromkeys(need))]
    keys = list(contrasts(d, extra))
    point = contrasts(d, extra)
    draws = cluster_bootstrap(d, lambda bd: [contrasts(bd, extra)[k] for k in keys], reps=reps)
    return {k: dict(point=point[k], lo90=float(np.nanpercentile(draws[:, i], 5)),
                    hi90=float(np.nanpercentile(draws[:, i], 95)),
                    p_le0=float(np.mean(draws[:, i] <= 0))) for i, k in enumerate(keys)}


def episodes(d, q=0.9, gap=3):
    """이익 원천 저축이 상위 10%이고 성향 원천은 중위수 이하(그 반대)인 해."""
    P, Hh = d["x_P"], d["x_H"]
    flags = {"P": (P >= P.quantile(q)) & (Hh <= Hh.median()),
             "H": (Hh >= Hh.quantile(q)) & (P <= P.median())}
    ep = {}
    for s, f in flags.items():
        f = f.fillna(False).to_numpy()
        e, last = np.zeros(len(d)), {}
        for i, (c, yr) in enumerate(zip(d["country"], d["year"])):
            if f[i] and yr - last.get(c, -99) > gap:
                e[i], last[c] = 1, yr
        ep[s] = pd.Series(e, index=d.index)
    return ep


def main(reps):
    panel = pd.read_csv(ra.HERE / "data" / "panel.csv")
    d = prepare(panel)
    smp = rs.samples(panel)
    keep = {k: v for k, v in smp.items() if k.split(" (")[0] in
            ("전체", "MNE 허브 제외", "체제전환국 제외", "선진 핵심국만")}
    variants = [(name, drop, (), False) for name, drop in keep.items()]
    variants += [("전체, 경기침체 연도 제외", [], (), True),
                 ("전체, 당해 성장률 통제", [], ("gdp_g",), False)]
    res, irfs = [], []
    for name, drop, extra, norec in variants:
        ds = ra.sample(d, (), drop)
        if norec:
            ds = ds[~ds["recession"].fillna(False)]
        for y in OUTCOMES:
            irfs.append(irf(ds, y, extra).assign(sample=name))
        c = boot(ds, reps, extra)
        res.append(dict(sample=name, countries=int(ds.dropna(subset=["ca_h0", "x_P", "x_H"])["country"].nunique()),
                        **{f"{k}{suf}": v[f] for k, v in c.items()
                           for suf, f in (("", "point"), ("_lo", "lo90"), ("_hi", "hi90"))}))
        print(name, f"ca_dd={c['ca_dd']['point']:.2f} [{c['ca_dd']['lo90']:.2f},{c['ca_dd']['hi90']:.2f}]",
              f"ca_mdiff={c['ca_medium_diff']['point']:.2f} [{c['ca_medium_diff']['lo90']:.2f},{c['ca_medium_diff']['hi90']:.2f}]",
              f"ptc P={c['ptc_P']['point']:.2f} H={c['ptc_H']['point']:.2f} diff={c['ptc_diff']['point']:.2f} [{c['ptc_diff']['lo90']:.2f},{c['ptc_diff']['hi90']:.2f}]",
              flush=True)
    pd.DataFrame(res).to_csv(ra.OUT / "source_contrasts.csv", index=False)
    pd.concat(irfs).to_csv(ra.OUT / "source_irf.csv", index=False)

    # 원천별 에피소드
    ev = []
    for name, drop in keep.items():
        ds = ra.sample(d, (), drop)
        ep = episodes(ds)
        es = ra.event_study(ds, {"corp": ep["P"], "hh": ep["H"]}, ["ca", "sv_nat", "sv_corp", "sv_hh"])
        ev.append(es.assign(sample=name, n_P=int(ep["P"].sum()), n_H=int(ep["H"].sum())))
    pd.concat(ev).to_csv(ra.OUT / "source_episodes.csv", index=False)

    # 그림: 경상수지·국민저축 반응
    fig, axes = plt.subplots(2, len(variants), figsize=(3.3 * len(variants), 6), squeeze=False)
    allirf = pd.concat(irfs)
    for j, (name, *_) in enumerate(variants):
        for i, y in enumerate(("ca", "sv_nat")):
            r = allirf[(allirf["sample"] == name) & (allirf.outcome == y)]
            ax = axes[i, j]
            for s, col in (("P", ra.C["corp"]), ("H", ra.C["hh"])):
                ra.plot_band(ax, r.h, r[f"b_{s}"], r[f"se_{s}"], col, s)
            ax.axhline(0, color="k", lw=.6)
            ax.set_title(f"{y} | {j}", fontsize=8)
    fig.tight_layout()
    fig.savefig(ra.OUT / "source_irf.png", dpi=130)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=499)
    main(ap.parse_args().reps)
