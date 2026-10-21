"""SDMX-CSV 원자료 → 국가×연도 패널 (명목, 자국통화 → GDP 대비 %).

출력: data/panel.csv
변수(모두 GDP 대비 %, hh_rate만 가처분소득 대비 %)
  ca        국민경제 순대출(S1 B9) ≈ 경상수지+자본수지
  nl_corp   비금융법인(S11) 순대출       nl_fin   금융법인(S12)
  nl_gov    일반정부(S13)                nl_hh    가계·비영리(S14_S15)
  sv_corp   비금융법인 총저축(B8G)        sv_hh    가계 총저축
  hh_rate   가계 총저축률(B8G/B6G)        prof     비금융법인 총영업잉여(B2G_B3G)
  div_paid  비금융법인 배당 등 지급(D42, 없으면 D4)  tax_paid 법인세 등(D5) 지급
  inv_corp  비금융법인 총고정자본형성(P51G)  inv_hh  가계 총고정자본형성
  cons_hh   가계 최종소비(P3)             gdp_g    명목 GDP 증가율(%)
"""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RAW = HERE / "data" / "raw"

HH = ["S14_S15", "S1M"]  # OECD 코드 표기가 데이터셋마다 다름

# (변수명, 부문 후보, 거래 후보, 계정기입 후보)
SPEC = [
    ("gdp", ["S1"], ["B1GQ"], ["B", "_Z", "C"]),
    ("nl_s1", ["S1"], ["B9"], ["B", "_Z"]),
    ("nl_s2", ["S2"], ["B9"], ["B", "_Z"]),
    ("nl_corp", ["S11"], ["B9"], ["B", "_Z"]),
    ("nl_fin", ["S12"], ["B9"], ["B", "_Z"]),
    ("nl_gov", ["S13"], ["B9"], ["B", "_Z"]),
    ("nl_hh", HH, ["B9"], ["B", "_Z"]),
    ("sv_corp", ["S11"], ["B8G"], ["B", "_Z"]),
    ("sv_hh", HH, ["B8G"], ["B", "_Z"]),
    ("sv_gov", ["S13"], ["B8G"], ["B", "_Z"]),
    ("yd_hh", HH, ["B6G"], ["B", "_Z"]),
    ("prof", ["S11"], ["B2G_B3G", "B2G"], ["B", "_Z"]),
    ("div_paid", ["S11"], ["D42", "D4"], ["D"]),
    ("tax_paid", ["S11"], ["D5"], ["D"]),
    ("inv_corp", ["S11"], ["P51G", "P5"], ["D", "_Z"]),
    ("inv_hh", HH, ["P51G", "P5"], ["D", "_Z"]),
    ("cons_hh", HH, ["P3"], ["D", "_Z"]),
]

PREF = {  # 중복 행이 있을 때 우선순위
    "UNIT_MEASURE": ["XDC"],
    "PRICE_BASE": ["V", "_Z"],
    "TRANSFORMATION": ["N", "_Z"],
    "COUNTERPART_SECTOR": ["S1", "_Z", "_T", "S2"],
    "CONSOLIDATION": ["N", "S", "C", "_Z"],
    "INSTR_ASSET": ["_Z", "_T"],
    "ACTIVITY": ["_Z", "_T"],
    "EXPENDITURE": ["_Z", "_T"],
}


def load_raw(path):
    df = pd.read_csv(path, dtype=str)
    df["OBS_VALUE"] = pd.to_numeric(df["OBS_VALUE"], errors="coerce")
    df["year"] = pd.to_numeric(df["TIME_PERIOD"].str[:4], errors="coerce")
    df = df[df.get("FREQ", "A") == "A"] if "FREQ" in df else df
    return df.dropna(subset=["OBS_VALUE", "year"])


def pick(df, sectors, trans, entries):
    """후보 코드 중 가용한 것을 고르고, 나머지 차원은 PREF 순서로 한 계열만 남긴다."""
    d = df[df["SECTOR"].isin(sectors) & df["TRANSACTION"].isin(trans)].copy()
    if "ACCOUNTING_ENTRY" in d:
        d = d[d["ACCOUNTING_ENTRY"].isin(entries)]
        d["_e"] = d["ACCOUNTING_ENTRY"].map({e: i for i, e in enumerate(entries)})
    else:
        d["_e"] = 0
    if d.empty:
        return pd.Series(dtype=float)
    d["_s"] = d["SECTOR"].map({s: i for i, s in enumerate(sectors)})
    d["_t"] = d["TRANSACTION"].map({t: i for i, t in enumerate(trans)})
    rank = ["_t", "_s", "_e"]
    for col, order in PREF.items():
        if col in d:
            d[f"_{col}"] = d[col].map({v: i for i, v in enumerate(order)}).fillna(len(order))
            rank.append(f"_{col}")
    # 국가별로 하나의 계열(코드 조합)을 고른다 — 연도별로 섞이지 않게
    d["_rank"] = list(zip(*[d[c] for c in rank]))
    best = d.groupby("REF_AREA")["_rank"].transform("min")
    d = d[d["_rank"] == best]
    return d.groupby(["REF_AREA", "year"])["OBS_VALUE"].first()


def build(nonfin=RAW / "nonfin.csv"):
    raw = load_raw(nonfin)
    cols = {name: pick(raw, s, t, e) for name, s, t, e in SPEC}
    p = pd.DataFrame(cols).reset_index().rename(columns={"REF_AREA": "country"})
    p = p.sort_values(["country", "year"])
    # 국민경제 순대출: S1 B9, 없으면 −S2 B9
    p["nl_nat"] = p["nl_s1"].fillna(-p["nl_s2"])
    level = [c for c, *_ in SPEC if c not in ("gdp", "yd_hh", "nl_s1", "nl_s2")] + ["nl_nat"]
    out = p[["country", "year"]].copy()
    for c in level:
        out[c] = 100 * p[c] / p["gdp"]
    out = out.rename(columns={"nl_nat": "ca"})
    out["hh_rate"] = 100 * p["sv_hh"] / p["yd_hh"]
    out["gdp_g"] = 100 * p.groupby("country")["gdp"].pct_change(fill_method=None)
    return out


if __name__ == "__main__":
    panel = build()
    path = HERE / "data" / "panel.csv"
    panel.to_csv(path, index=False)
    cov = panel.dropna(subset=["ca", "sv_corp", "sv_hh"]).groupby("country")["year"].agg(["min", "max", "count"])
    print(cov.to_string())
    print(f"\n{len(cov)} countries, {int(cov['count'].sum())} obs → {path}")
