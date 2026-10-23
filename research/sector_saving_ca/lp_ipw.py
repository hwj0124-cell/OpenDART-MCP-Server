"""LP-AIPW (Jordà & Taylor, 2016 'The Time for Austerity'): 원천별 저축 급증의 평균처치효과.

처치: 이익 원천 저축 급증(D_P) 또는 성향 원천 저축 급증(D_H) 에피소드 시작 연도
      (source_shocks.episodes 정의: 해당 원천이 상위 10%, 다른 원천은 중위수 이하).
대조군: 어느 원천도 급증하지 않은 관측치.
1) 성향점수: 사전결정 변수(t−1 이전)로 로짓 → p̂(X)
2) 결과: Δ_h y = y_{t+h} − y_{t−1}, 연도·국가 평균을 빼 글로벌 충격과 국가별 추세 제거
3) AIPW(가중치 정규화): ATE_h = E[ D(y−m̂1)/p̂ + m̂1 ] − E[ (1−D)(y−m̂0)/(1−p̂) + m̂0 ]
   m̂1, m̂0 는 처치군·대조군 각각의 OLS 회귀조정. p̂ 는 [0.01, 0.99]로 절단(overlap)
4) 국가 블록 부트스트랩(로짓까지 재추정)

    python lp_ipw.py [--reps 499]
"""
import argparse

import numpy as np
import pandas as pd

import robustness_samples as rs
import run_analysis as ra
import source_shocks as ss
from lp_tools import cluster_bootstrap

H = ra.H
OUTS = ["ca", "sv_nat", "sv_corp", "sv_hh", "nl_corp", "nl_hh", "nl_gov"]
X = ["L1_x_P", "L1_x_H", "L1_d_ca", "L1_gdp_g", "L1_sv_corp", "L1_sv_hh", "L1_ca", "L1_prof", "L1_hh_rate"]
TRIM = 0.01


def logit(Xm, y, ridge=1e-3, it=50):
    Xm = np.column_stack([np.ones(len(Xm)), Xm])
    b = np.zeros(Xm.shape[1])
    for _ in range(it):
        p = 1 / (1 + np.exp(-Xm @ b))
        W = p * (1 - p)
        g = Xm.T @ (y - p) - ridge * b
        Hs = (Xm * W[:, None]).T @ Xm + ridge * np.eye(len(b))
        step = np.linalg.solve(Hs, g)
        b += step
        if np.abs(step).max() < 1e-8:
            break
    return 1 / (1 + np.exp(-Xm @ b))


def prepare(panel):
    d = ss.prepare(panel)
    for y in OUTS:  # 이원 고정효과 제거(국가별 평균 추세, 연도별 글로벌 충격)
        for h in range(H + 1):
            c = f"{y}_h{h}"
            d[c] = d[c] - d.groupby("year")[c].transform("mean")
            d[c] = d[c] - d.groupby("country")[c].transform("mean")
    return d


STRICT = True


def treatments(d):
    ep = ss.episodes(d, strict=STRICT)
    P, Hh = d["x_P"], d["x_H"]
    anyflag = ((P >= P.quantile(0.9)) | (Hh >= Hh.quantile(0.9))).fillna(False)
    return ep, anyflag


def aipw(d, treat, ctrl_mask, y):
    """한 처치에 대해 h=0..H ATE 벡터."""
    z = d[[treat] + X].copy()
    z["D"] = d[treat]
    keep = (z["D"] == 1) | ctrl_mask
    z = z[keep].dropna()
    Xm = z[X].to_numpy(float)
    mu, sd = Xm.mean(0), Xm.std(0) + 1e-9
    Xs = (Xm - mu) / sd
    D = z["D"].to_numpy(float)
    p = np.clip(logit(Xs, D), TRIM, 1 - TRIM)
    out = []
    for h in range(H + 1):
        yy = d.loc[z.index, f"{y}_h{h}"].to_numpy(float)
        ok = ~np.isnan(yy)
        Dk, pk, Xk, yk = D[ok], p[ok], np.column_stack([np.ones(ok.sum()), Xs[ok]]), yy[ok]
        m = {}
        for g in (0, 1):
            sel = Dk == g
            b = np.linalg.lstsq(Xk[sel], yk[sel], rcond=None)[0]
            m[g] = Xk @ b
        # 정규화(Hájek) 가중치로 극단적 가중의 영향 완화
        w1, w0 = Dk / pk, (1 - Dk) / (1 - pk)
        out.append(np.sum(w1 * (yk - m[1])) / w1.sum() + m[1].mean()
                   - np.sum(w0 * (yk - m[0])) / w0.sum() - m[0].mean())
    return np.array(out), dict(n_treat=int(D.sum()), n_ctrl=int((1 - D).sum()),
                               p_treat_med=float(np.median(p[D == 1])), p_ctrl_med=float(np.median(p[D == 0])),
                               trimmed=float(np.mean((p <= TRIM) | (p >= 1 - TRIM))))


def stats(d):
    ep, anyflag = treatments(d)
    d = d.assign(D_P=ep["P"], D_H=ep["H"])
    ctrl = ~anyflag
    res = {}
    for t, own in (("D_P", "x_P"), ("D_H", "x_H")):
        for y in ("ca", "sv_nat"):
            res[f"{t}_{y}"], diag = aipw(d, t, ctrl, y)
        # 충격 크기: 처치 시점 원천 저축(AIPW, 같은 방식)
        d[f"{own}_h0"] = d[own] - d.groupby("year")[own].transform("mean")
        res[f"{t}_size"] = aipw(d.assign(**{f"{own}_h{h}": d[f"{own}_h0"] for h in range(1, H + 1)}),
                                t, ctrl, own)[0][0]
        res[f"{t}_diag"] = diag
    v = {}
    for t in ("D_P", "D_H"):
        ca, sn, size = res[f"{t}_ca"], res[f"{t}_sv_nat"], res[f"{t}_size"]
        for h in range(H + 1):
            v[f"{t}_ca_h{h}"] = ca[h] / size  # 원천 저축 1%p 당
            v[f"{t}_sv_nat_h{h}"] = sn[h] / size
        v[f"{t}_ca_short"] = ca[:2].mean() / size
        v[f"{t}_ca_medium"] = ca[4:].mean() / size
        v[f"{t}_ptc"] = ca.sum() / sn.sum()
    v["medium_diff"] = v["D_H_ca_medium"] - v["D_P_ca_medium"]  # 문서 주장: > 0
    v["ptc_diff"] = v["D_H_ptc"] - v["D_P_ptc"]                 # 문서 주장: > 0
    return v, {t: res[f"{t}_diag"] | {"size": res[f"{t}_size"]} for t in ("D_P", "D_H")}


def main(reps):
    panel = pd.read_csv(ra.HERE / "data" / "panel.csv")
    d = prepare(panel)
    smp = rs.samples(panel)
    keep = {k: v for k, v in smp.items() if k.split(" (")[0] in ("전체", "MNE 허브 제외", "체제전환국 제외", "선진 핵심국만")}
    need = list(dict.fromkeys(["country", "year", "x_P", "x_H", "x_R", "x_Y"] + X
                              + [f"{y}_h{h}" for y in OUTS for h in range(H + 1)]))
    rows = []
    for name, drop in keep.items():
        ds = ra.sample(d, (), drop)[need]
        point, diag = stats(ds)
        keys = list(point)
        draws = cluster_bootstrap(ds, lambda bd: list(stats(bd)[0].values()), reps=reps)
        for i, k in enumerate(keys):
            rows.append(dict(sample=name, key=k, point=point[k],
                             se=np.nanstd(draws[:, i]),
                             lo90=np.nanpercentile(draws[:, i], 5), hi90=np.nanpercentile(draws[:, i], 95)))
        print(name, diag, flush=True)
        print(f"  ptc P={point['D_P_ptc']:.2f} H={point['D_H_ptc']:.2f} "
              f"medium P={point['D_P_ca_medium']:.2f} H={point['D_H_ca_medium']:.2f}", flush=True)
    pd.DataFrame(rows).to_csv(ra.OUT / "lp_ipw.csv", index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=499)
    main(ap.parse_args().reps)
