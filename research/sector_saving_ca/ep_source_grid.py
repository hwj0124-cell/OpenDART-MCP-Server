"""원천 분해 LP의 사양 격자(미리 정한 선택지 전부, 점추정). 결과를 보고 사양을 고르지 않기 위한 표.

차원
  own   : 자기 부문 S−I 시차를 변화분(Δn) 또는 수준(n)으로
  p     : 시차 수 1–4 (네 성분·성장률·자기 S−I 공통)
  cross : 같은 해 다른 부문 두 성분을 넣음(joint) / 뺌(own only; 다른 부문 성분은 시차만)
  extra : 없음 / + 경상수지·다른 부문 S−I 수준(t−1) / + 상태 수준(기업저축·가계저축·영업잉여·저축률, t−1)
표본: p = 4까지 모든 시차와 h = 0…6 결과가 있는 국가·연도(시작 연도 ≤ 2019)로 고정. 가중치는 사양마다 재추정.
시차 선택 참고: n_t를 시차 p개(자기 S−I 수준·네 성분·성장률)와 FE에 회귀한 BIC.
→ output/ep_source_grid.csv
"""
import itertools

import numpy as np
import pandas as pd

import lp_tools as lt
import source_shocks as ss
from make_report import OUT

H, P = 7, 4
SRC = ["x_P", "x_R", "x_H", "x_Y"]
OWN = {"nl_corp": ["x_P", "x_R"], "nl_hh": ["x_H", "x_Y"]}
OTHER = {"nl_corp": "nl_hh", "nl_hh": "nl_corp"}
STATE = ["sv_corp", "sv_hh", "prof", "hh_rate"]


def data():
    d = ss.prepare(pd.read_csv(OUT.parent / "data" / "panel.csv"))
    g = d.groupby("country")
    for y in OWN:
        d[f"d_{y}"] = g[y].diff()
    d = d.copy()
    g = d.groupby("country")
    for v in SRC + ["gdp_g", "nl_corp", "nl_hh", "d_nl_corp", "d_nl_hh"]:
        for l in range(1, P + 1):
            d[f"L{l}_{v}"] = g[v].shift(l)
    for v in STATE + ["ca"]:
        d[f"L1_{v}"] = g[v].shift(1)
    need = [f"{y}_h{h}" for y in OWN for h in range(H)] + SRC + [f"L{l}_{v}" for v in SRC + ["gdp_g", "nl_corp", "nl_hh", "d_nl_corp", "d_nl_hh"]
                                                                  for l in range(1, P + 1)] + [f"L1_{v}" for v in STATE + ["ca"]]
    q = d.dropna(subset=need)
    return d, q[q.year <= 2019].copy()


def controls(y, own, p, extra):
    lag = lambda vs: [f"L{l}_{v}" for v in vs for l in range(1, p + 1)]
    C = lag(SRC + ["gdp_g"]) + lag([y if own == "levels" else f"d_{y}"])
    if extra == "+ CA, other-sector S-I":
        C += ["L1_ca", f"L1_{OTHER[y]}"]
    elif extra == "+ state levels":
        C += [f"L1_{v}" for v in STATE]
    return C


def persistence(q, y, X, C):
    M = np.hstack([q[X + C].to_numpy(float), lt._fe_matrix(q, ("country", "year"))])
    e = q[f"{y}_h0"].to_numpy(float) - M @ np.linalg.lstsq(M, q[f"{y}_h0"].to_numpy(float), rcond=None)[0]
    q = q.assign(w=1 / q.country.map(pd.Series(e, index=q.index).groupby(q.country).var()))
    B = np.array([lt.ols(q, f"{y}_h{h}", X, C, se=False, w="w")[0] for h in range(H)])
    return {x: (B[1:3, k].mean() / B[0, k], B[3:, k].mean() / B[0, k]) for k, x in enumerate(X) if x in OWN[y]}


def bic(q, y, p):
    X = [f"L{l}_{v}" for v in SRC + ["gdp_g", y] for l in range(1, p + 1)]
    M = np.hstack([q[X].to_numpy(float), lt._fe_matrix(q, ("country", "year"))])
    u = q[y].to_numpy(float) - M @ np.linalg.lstsq(M, q[y].to_numpy(float), rcond=None)[0]
    n, k = len(u), np.linalg.matrix_rank(M)
    return n * np.log((u ** 2).mean()) + k * np.log(n), n * np.log((u ** 2).mean()) + 2 * k


if __name__ == "__main__":
    d, q = data()
    print(f"sample: N = {len(q)}, {q.country.nunique()} economies")
    for y in OWN:
        print(y, "BIC / AIC by p:", {p: tuple(round(v, 1) for v in bic(q, y, p)) for p in range(1, P + 1)})
    rows = []
    for own, p, cross, extra in itertools.product(("changes", "levels"), range(1, P + 1), ("joint", "own only"),
                                                  ("none", "+ CA, other-sector S-I", "+ state levels")):
        for y in OWN:
            X = SRC if cross == "joint" else OWN[y]
            C = controls(y, own, p, extra) + ([] if cross == "joint" else [])
            for x, (s, m) in persistence(q, y, X, C).items():
                rows.append(dict(own=own, p=p, cross=cross, extra=extra, source=x, short=s, medium=m))
    r = pd.DataFrame(rows)
    r.to_csv(OUT / "ep_source_grid.csv", index=False)
    pd.set_option("display.width", 200)
    for own in ("changes", "levels"):
        t = r[r.own == own].pivot_table(index=["p", "cross", "extra"], columns="source", values=["short", "medium"], sort=False)
        t = t.reindex(columns=pd.MultiIndex.from_product([["short", "medium"], SRC]))
        print(f"\n== own-sector S-I in {own}")
        print(t.round(2).to_string())
    print("\nranges (min / median / max) by own:")
    print(r.groupby(["own", "source"])[["short", "medium"]].describe().loc[:, (slice(None), ["min", "50%", "max"])].round(2).to_string())
