"""고정 비중 shift-share 도구변수(Bartik)와 LP-IV.

z_it = Σ_k s_ik0 · ḡ_kt^(−i)
  s_ik0 : 산업 k 영업잉여 / 전체 GVA, 국가별 기준기간(처음 3년 또는 1995–99) 평균으로 고정
  ḡ     : 자국 제외 다른 나라 같은 산업의 증가율(또는 미국), 연도별 산업 간 평균을 뺀 값
shift 후보: 이익 증가율, 이익률 변화, 부가가치 증가율, 이익 변화/GVA. 부가가치 증가율(세계 산업 수요)만 1단계가 강함.
LP-IV: 내생변수 = 시작 연도 기업저축 변화(dsv) 또는 영업잉여 변화(dprof). 기준기간은 추정에서 제외.
통제·가중치는 원천 분해 제안 사양과 같음. 군집 강건 표준오차.
→ output/ep_iv_shiftshare_fs.csv(1단계), output/ep_iv_shiftshare.csv(LP-IV)
"""
import numpy as np
import pandas as pd

import ep_iv_industry as e
import iv_industry as ii
import lp_tools as lt
from make_report import OUT

SHIFTS = {"g": "profit growth", "dm": "margin change", "gv": "value-added growth", "dprof_k": "profit change/GVA"}
BASES = {"first 3 years": None, "1995-99": (1995, 1999)}


def industry():
    w = ii.partition(ii.load()).sort_values(["country", "ind", "year"])
    g = w.groupby(["country", "ind"])
    w["gos_l1"], w["gva_l1"], w["gtot_l1"] = g.gos.shift(1), g.gva.shift(1), g.gva_tot.shift(1)
    w = w[w.year - g.year.shift(1) == 1].copy()
    sym = lambda a, b: (a - b) / ((a.abs() + b.abs()) / 2)
    w["g"] = sym(w.gos, w.gos_l1)                                   # 이익 대칭 증가율
    w["dm"] = (w.gos / w.gva - w.gos_l1 / w.gva_l1).clip(-0.3, 0.3)  # 이익률 변화
    w["gv"] = sym(w.gva, w.gva_l1)                                  # 부가가치 증가율
    w["dprof_k"] = (w.gos - w.gos_l1) / w.gtot_l1                   # 산업 이익 변화 / 전체 GVA
    return w


def shares(w, base):
    if base is None:
        b = w[w.year <= w.groupby("country").year.transform("min") + 2]
    else:
        b = w[w.year.between(*base)]
    return (b.gos_l1 / b.gtot_l1).groupby([b.country, b.ind]).mean().rename("s0")


def shift(w, col, source):
    if source == "US":
        us = w[w.country == "USA"].set_index(["ind", "year"])[col]
        s = pd.Series([us.get((k, y), np.nan) for k, y in zip(w.ind, w.year)], index=w.index).where(w.country != "USA")
    else:
        sm = w.groupby(["ind", "year"])[col].transform("sum")
        n = w.groupby(["ind", "year"])[col].transform("count")
        s = ((sm - w[col]) / (n - 1)).where(n > 1)
    return s - s.groupby(w.year).transform("mean")


def instrument(w, base, col, source="LOO"):
    t = w.join(shares(w, base), on=["country", "ind"])
    t["sh"] = shift(w, col, source)
    z = (100 * t.s0 * t.sh).groupby([t.country, t.year]).sum(min_count=1).rename("z").reset_index()
    first = w.groupby("country").year.min()
    z["in_base"] = z.year <= z.country.map(first) + 2 if base is None else z.year <= base[1]
    return z


def wiv(q, y, h, Z, C, endog):
    r = q.dropna(subset=[f"{y}_h{h}", endog, "w_nl_corp"] + Z + C)
    w = r.w_nl_corp.to_numpy(float)
    sw = np.sqrt(w)[:, None]
    M = np.hstack([r[C].to_numpy(float), lt._fe_matrix(r, ("country", "year"))])
    part = lambda v: v - M @ np.linalg.lstsq(M * sw, v * (sw if v.ndim > 1 else sw[:, 0]), rcond=None)[0]
    Zt, xt, yt = part(r[Z].to_numpy(float)), part(r[endog].to_numpy(float)), part(r[f"{y}_h{h}"].to_numpy(float))
    G = r.country.to_numpy()
    ZWZ = (Zt * w[:, None]).T @ Zt
    pi = np.linalg.solve(ZWZ, (Zt * w[:, None]).T @ xt)
    xh, e1 = Zt @ pi, xt - Zt @ pi
    S = pd.DataFrame(Zt * (w * e1)[:, None]).groupby(G).sum().to_numpy()
    F = pi @ np.linalg.solve(np.linalg.solve(ZWZ, np.linalg.solve(ZWZ, S.T @ S).T), pi) / len(pi)
    b = (w * xh * yt).sum() / (w * xh * xt).sum()
    se = np.sqrt((pd.Series(w * xh * (yt - b * xt)).groupby(G).sum() ** 2).sum()) / abs((w * xh * xt).sum())
    return b, se, F, len(r)


if __name__ == "__main__":
    q, C = e.data()
    w = industry()
    fs = []
    for bl, base in BASES.items():
        for col, lab in SHIFTS.items():
            for src in ("LOO", "US"):
                z = instrument(w, base, col, src)
                qq = q.merge(z, on=["country", "year"], how="left")
                qq = qq[qq.in_base.eq(False)]   # 기준기간과 도구 없는 해 제외
                for endog in ("dprof", "dsv"):
                    F, n, k = e.first_stage(qq, ["z"], C, endog)
                    fs.append(dict(base=bl, shift=lab, source=src, endog=endog, F=F, n=n, countries=k))
    fs = pd.DataFrame(fs)
    fs.to_csv(OUT / "ep_iv_shiftshare_fs.csv", index=False)
    print(fs.pivot_table(index=["base", "shift", "source"], columns="endog", values="F").round(1).to_string())

    rows = []
    for bl, base in BASES.items():
        qq = q.merge(instrument(w, base, "gv"), on=["country", "year"], how="left")
        qq = qq[qq.in_base.eq(False)]   # 기준기간과 도구 없는 해 제외
        for endog in ("dsv", "dprof"):
            for y in ("nl_corp", "sv_corp", "prof", "gcf_corp", "div_net"):
                for h in range(7):
                    b, se, F, n = wiv(qq, y, h, ["z"], C, endog)
                    rows.append(dict(base=bl, endog=endog, var=y, h=h, b=b, se=se, F=F, n=n))
    r = pd.DataFrame(rows)
    r.to_csv(OUT / "ep_iv_shiftshare.csv", index=False)
    for (bl, endog), g in r.groupby(["base", "endog"], sort=False):
        print(f"== base {bl}, endog {endog}, F = {g.F.iloc[0]:.1f}, n = {g.n.iloc[0]}")
        print(g.pivot_table(index="var", columns="h", values="b", sort=False).round(2).to_string())
