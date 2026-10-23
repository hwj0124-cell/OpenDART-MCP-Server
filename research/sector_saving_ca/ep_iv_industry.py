"""산업 이익 도구변수(Bartik·GIV)의 1단계 강도 점검. 결과: 둘 다 약함(군집 강건 F < 5).

내생변수: 시작 연도 기업저축 변화(Δs^C) 또는 기업 영업잉여 변화(Δπ), GDP 대비 %p.
통제·가중치: 원천 분해 제안 사양과 같음(네 성분·성장률·자기 부문 S−I 시차, 국가·연도 FE, 2단계 가중치).
도구: iv_industry.py가 만든 z_bartik, z_giv(명목 증가율 기준), zm_bartik, zm_giv(이익률 기준).
→ output/ep_iv_industry.csv
"""
import numpy as np
import pandas as pd

import ep_source_decomp as sd
import lp_tools as lt
from make_report import OUT


def data():
    q, ctrl = sd.sample()
    iv = pd.read_csv(OUT.parent / "data" / "industry_iv.csv")
    q = q.merge(iv, on=["country", "year"], how="left").sort_values(["country", "year"])
    q["dsv"], q["dprof"] = q.sv_corp_h0, q.prof_h0
    return q, ctrl["nl_corp"] + ["x_R", "x_H", "x_Y"]


def first_stage(q, Z, C, endog):
    r = q.dropna(subset=[endog, "w_nl_corp"] + Z + C)
    w = r.w_nl_corp.to_numpy(float); sw = np.sqrt(w)[:, None]
    M = np.hstack([r[C].to_numpy(float), lt._fe_matrix(r, ("country", "year"))])
    part = lambda v: v - M @ np.linalg.lstsq(M * sw, v * (sw if v.ndim > 1 else sw[:, 0]), rcond=None)[0]
    Zt, xt = part(r[Z].to_numpy(float)), part(r[endog].to_numpy(float))
    ZWZ = (Zt * w[:, None]).T @ Zt
    pi = np.linalg.solve(ZWZ, (Zt * w[:, None]).T @ xt)
    e = xt - Zt @ pi
    S = pd.DataFrame(Zt * (w * e)[:, None]).groupby(r.country.to_numpy()).sum().to_numpy()
    V = np.linalg.solve(ZWZ, np.linalg.solve(ZWZ, S.T @ S).T)
    return pi @ np.linalg.solve(V, pi) / len(pi), len(r), r.country.nunique()


if __name__ == "__main__":
    q, C = data()
    rows = []
    for Z in (["z_bartik"], ["z_giv"], ["zm_bartik"], ["zm_giv"], ["zm_bartik", "zm_giv"]):
        for endog in ("dsv", "dprof"):
            F, n, k = first_stage(q, Z, C, endog)
            rows.append(dict(instrument="+".join(Z), endog=endog, F=F, n=n, countries=k))
    r = pd.DataFrame(rows)
    r.to_csv(OUT / "ep_iv_industry.csv", index=False)
    print(r.round(1).to_string(index=False))
    print(q[["z_bartik", "z_giv", "zm_bartik", "zm_giv", "dm_actual", "dprof", "dsv"]].corr().round(2).loc[["z_bartik", "z_giv", "zm_bartik", "zm_giv"], ["dm_actual", "dprof", "dsv"]])
