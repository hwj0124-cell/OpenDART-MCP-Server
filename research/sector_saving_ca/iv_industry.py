"""산업별 영업잉여로 만든 기업 이익 도구변수 두 가지 (OECD Table 6, 비금융 기업 부문에 가까운 산업만).

산업: B, C(중분류 가능하면 C10T12…C31T33), D, E, F, G, H, I, J(가능하면 J58T60·J61·J62_63), M, N
  — 농업(A, 혼합소득 위주)·금융(K)·부동산(L, 귀속임대료)·공공·가계(O–U) 제외
성장률: 대칭 증가율 g = (Π_t − Π_{t−1}) / ((|Π_t| + |Π_{t−1}|)/2)  (음수·0 근처 영업잉여에도 유한)

(1) Bartik: z^B_it = Σ_k (Π_ik,t−1 / GVA_i,t−1)·ĝ^{(−i)}_kt     ĝ^{(−i)} = 자국 제외 다른 나라의 같은 산업 증가율 평균
(2) GIV  : u_ikt = g_ikt − (국가·연도 평균) − (산업·연도 평균)  [양방향 평균 제거]
           z^G_it = (Π_i,t−1 / GVA_i,t−1)·[Σ_k S_ik,t−1 u_ikt − (1/K_i) Σ_k u_ikt],   S = 산업 영업잉여 비중
두 도구 모두 GVA 대비 %p 단위(예측된 이익 변화).
→ data/industry_iv.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SECT = ["B", "D", "E", "F", "G", "H", "I", "M", "N"]
CDIV = ["C10T12", "C13T15", "C16T18", "C19", "C20", "C21", "C22_23", "C24_25", "C26", "C27", "C28", "C29_30", "C31T33"]
JDIV = ["J58T60", "J61", "J62_63"]


def load():
    t = pd.read_csv(HERE / "data" / "raw" / "oecd" / "table6_gos.csv",
                    usecols=["REF_AREA", "TRANSACTION", "ACTIVITY", "UNIT_MEASURE", "PRICE_BASE", "TIME_PERIOD", "OBS_VALUE"])
    t = t[(t.UNIT_MEASURE == "XDC") & (t.PRICE_BASE == "V")]
    w = t.pivot_table(index=["REF_AREA", "TIME_PERIOD", "ACTIVITY"], columns="TRANSACTION", values="OBS_VALUE", aggfunc="first").reset_index()
    w.columns.name = None
    return w.rename(columns={"REF_AREA": "country", "TIME_PERIOD": "year", "ACTIVITY": "ind", "B2A3G": "gos", "B1G": "gva"})


def partition(w):
    """국가·연도마다 겹치지 않는 산업 분할: 제조업·정보통신은 중분류가 다 있으면 중분류, 아니면 대분류 하나."""
    rows = []
    for (c, y), g in w.groupby(["country", "year"]):
        g = g.set_index("ind")
        inds = [k for k in SECT if k in g.index and pd.notna(g.at[k, "gos"])]
        for sec, divs in (("C", CDIV), ("J", JDIV)):
            have = [k for k in divs if k in g.index and pd.notna(g.at[k, "gos"])]
            if len(have) == len(divs) or (sec == "C" and len(have) >= len(divs) - 1 and "C26" in have):
                inds += have
            elif sec in g.index and pd.notna(g.at[sec, "gos"]):
                inds.append(sec)
        if "_T" not in g.index or pd.isna(g.at["_T", "gva"]):
            continue
        for k in inds:
            rows.append(dict(country=c, year=y, ind=k, gos=g.at[k, "gos"], gva=g.at[k, "gva"] if "gva" in g else np.nan,
                             gva_tot=g.at["_T", "gva"]))
    return pd.DataFrame(rows)


def build():
    p = partition(load()).sort_values(["country", "ind", "year"])
    g = p.groupby(["country", "ind"])
    p["gos_l1"] = g.gos.shift(1)
    p["gva_l1"] = g.gva_tot.shift(1)
    p = p[p.year - g.year.shift(1) == 1].copy()
    p["g"] = (p.gos - p.gos_l1) / ((p.gos.abs() + p.gos_l1.abs()) / 2)
    p["share_gva"] = p.gos_l1 / p.gva_l1                       # 산업 영업잉여 / GVA (전년)
    tot = p.groupby(["country", "year"]).gos_l1.transform("sum")
    p["S"] = p.gos_l1 / tot                                      # 산업 영업잉여 비중 (전년)
    p["prof_share"] = tot / p.gva_l1                             # 대상 산업 영업잉여 / GVA (전년)
    # Bartik: 자국 제외 같은 산업 평균 증가율
    s = p.groupby(["ind", "year"]).g.transform("sum"); n = p.groupby(["ind", "year"]).g.transform("count")
    p["g_loo"] = np.where(n > 1, (s - p.g) / (n - 1), np.nan)
    # GIV: 국가·연도, 산업·연도 평균을 번갈아 제거
    u = p.g.copy()
    for _ in range(50):
        u = u - u.groupby([p.country, p.year]).transform("mean")
        u = u - u.groupby([p.ind, p.year]).transform("mean")
    p["u"] = u
    # 마진 기반: m = 영업잉여/부가가치(산업), v = 산업 부가가치 / 전체 GVA (전년)
    p["m"] = p.gos / p.gva
    p["m_l1"] = p.gos_l1 / g.gva.shift(1).reindex(p.index)
    p["dm"] = (p.m - p.m_l1).clip(-0.3, 0.3)
    p["v"] = g.gva.shift(1).reindex(p.index) / p.gva_l1
    s2 = p.groupby(["ind", "year"]).dm.transform("sum"); n2 = p.groupby(["ind", "year"]).dm.transform("count")
    p["dm_loo"] = np.where(n2 > 1, (s2 - p.dm) / (n2 - 1), np.nan)
    um = p.dm.copy()
    for _ in range(50):
        um = um - um.groupby([p.country, p.year]).transform("mean")
        um = um - um.groupby([p.ind, p.year]).transform("mean")
    p["um"] = um
    out = p.groupby(["country", "year"]).apply(lambda q: pd.Series({
        "zm_bartik": 100 * (q.v * q.dm_loo).sum(min_count=1),
        "zm_giv": 100 * ((q.v * q.um).sum() - q.v.sum() * q.um.mean()),
        "dm_actual": 100 * (q.v * q.dm).sum(min_count=1),
        "z_bartik": 100 * (q.share_gva * q.g_loo).sum(min_count=1),
        "z_giv": 100 * q.prof_share.iloc[0] * ((q.S * q.u).sum() - q.u.mean()),
        "n_ind": len(q),
        "top_share": q.S.max(),
    }), include_groups=False).reset_index()
    out.to_csv(HERE / "data" / "industry_iv.csv", index=False)
    return out, p


if __name__ == "__main__":
    out, p = build()
    print(out.describe().round(3).to_string())
    print(out[out.country == "KOR"].set_index("year").loc[2005:2025, ["z_bartik", "z_giv", "n_ind", "top_share"]].round(2).to_string())
    k = p[(p.country == "KOR") & (p.year == 2024)].sort_values("S", ascending=False)[["ind", "S", "g", "u"]].head(6)
    print(k.round(2).to_string(index=False))
