"""SDMX-CSV 원자료 → 국가×연도 패널 (명목, 자국통화 → GNI 대비 %).

분모: OECD 국민계정 Table 1의 명목 GDP(B1GQ, 자국통화, data/raw/oecd/gdp.csv)가 있으면 GDP,
없으면 Table 14A의 국민총소득(S1 B5G).
출력: data/panel.csv (GDP 대비), data/panel_gni.csv (GNI 대비, 비교용)
변수(모두 분모 대비 %, hh_rate만 가처분소득 대비 %)
  ca        국민경제 순대출(S1 B9) ≈ 경상수지+자본수지 (이하 'CA')
  nl_corp   비금융법인(S11) 순대출       nl_fin   금융법인(S12)
  nl_gov    일반정부(S13)                nl_hh    가계·비영리(S14_S15)
  sv_corp   비금융법인 총저축(B8G)        sv_hh    가계 총저축      sv_gov 정부 총저축
  hh_rate   가계 총저축률(B8G/B6G)        prof     비금융법인 총영업잉여(B2G)
  div_paid  비금융법인 배당 지급(D42)      div_recv_hh 가계 배당 수취(D42)
  tax_paid  비금융법인 소득세 등(D5) 지급   int_paid 비금융법인 이자 지급(D41)
  inv_corp  비금융법인 총고정자본형성(P51G)  inv_hh  가계   inv_gov 정부
  cons_hh   가계 최종소비(P3)          yd_hh 가계 총가처분소득(B6G)   sv_nat 국민총저축(S1 B8G)
  gdp_g     실질 GDP 성장률(WDI, 없으면 명목 GNI 증가율)   tot_g 교역조건 변화율(WDI 상품 순교역조건, 2006~)
  tot_na_g  교역조건 변화율(WDI 국민계정 수출 디플레이터/수입 디플레이터, 상품·서비스, 1970년대~)
  tot_goods_g 교역조건 변화율(OECD 국민계정 상품 수출 디플레이터/상품 수입 디플레이터, 1970~)
"""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RAW = HERE / "data" / "raw"

HH = ["S1M", "S14_S15"]  # OECD 코드 표기가 데이터셋마다 다름
BAL = ["B", "_Z", "C", "D"]  # 균형항목: 추출본에 따라 C·D 양쪽에 같은 값
AGGREGATES = {"EU", "EA20", "EA19", "EU27_2020", "OECD"}

# (변수명, 부문 후보, 거래 후보, 계정기입 후보)
SPEC = [
    ("gdp", ["S1"], ["B5G"], BAL),  # 분모: GNI
    ("nl_s1", ["S1"], ["B9"], BAL),
    ("nl_s2", ["S2"], ["B9"], BAL),
    ("nl_corp", ["S11"], ["B9"], BAL),
    ("nl_fin", ["S12"], ["B9"], BAL),
    ("nl_gov", ["S13"], ["B9"], BAL),
    ("nl_hh", HH, ["B9"], BAL),
    ("sv_corp", ["S11"], ["B8G"], BAL),
    ("sv_hh", HH, ["B8G"], BAL),
    ("sv_gov", ["S13"], ["B8G"], BAL),
    ("sv_nat", ["S1"], ["B8G"], BAL),
    ("yd_hh", HH, ["B6G"], BAL),
    ("prof", ["S11"], ["B2G_B3G", "B2G"], BAL),
    ("div_paid", ["S11"], ["D42"], ["D"]),
    ("div_recv_hh", HH, ["D42"], ["C"]),
    ("tax_paid", ["S11"], ["D5"], ["D"]),
    ("int_paid", ["S11"], ["D41"], ["D"]),
    ("inv_corp", ["S11"], ["P51G", "P5"], ["D", "_Z"]),
    ("inv_hh", HH, ["P51G", "P5"], ["D", "_Z"]),
    ("inv_gov", ["S13"], ["P51G", "P5"], ["D", "_Z"]),
    ("cons_hh", HH, ["P3"], ["D", "_Z"]),
]

PREF = {  # 중복 행이 있을 때 우선순위
    "UNIT_MEASURE": ["XDC"],
    "PRICE_BASE": ["V", "_Z"],
    "TRANSFORMATION": ["N", "_Z"],
    "COUNTERPART_SECTOR": ["S1", "_Z", "_T", "S2"],
    "CONSOLIDATION": ["N", "S", "C", "_Z"],
    "INSTR_ASSET": ["_Z", "N11G", "_T"],  # P51G는 고정자산 총계 N11G (세부 자산 N111G 등 제외)
    "ACTIVITY": ["_Z", "_T"],
    "EXPENDITURE": ["_Z", "_T"],
}


def load_raw(path):
    df = pd.read_csv(path, dtype=str)
    df["OBS_VALUE"] = pd.to_numeric(df["OBS_VALUE"], errors="coerce")
    df["year"] = pd.to_numeric(df["TIME_PERIOD"].str[:4], errors="coerce")
    if "FREQ" in df:
        df = df[df["FREQ"] == "A"]
    if "PRICE_BASE" in df:  # 현재가격만 (일부 P51G는 연쇄가격 L로만 제공)
        df = df[df["PRICE_BASE"].isin(["V", "_Z"])]
    df = df[~df["REF_AREA"].isin(AGGREGATES)]
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


def tot_na(folder=RAW / "wb"):
    """WDI 국민계정 수출입(경상·불변 자국통화) → 수출 디플레이터 / 수입 디플레이터 변화율(%).
    상품·서비스 포함, 1970년대부터. 반환: country, year, tot_na_g"""
    def read(code):
        f = next(Path(folder).glob(f"API_{code}_DS2_*.csv"))
        w = pd.read_csv(f, skiprows=4).drop(columns=["Country Name", "Indicator Name", "Indicator Code"])
        w = w.melt(id_vars="Country Code", var_name="year", value_name=code).rename(columns={"Country Code": "country"})
        w = w[w.year.str.isdigit()].astype({"year": int})
        return w.set_index(["country", "year"])[code]
    w = pd.concat([read(c) for c in ("NE.EXP.GNFS.CN", "NE.EXP.GNFS.KN", "NE.IMP.GNFS.CN", "NE.IMP.GNFS.KN")], axis=1)
    tot = (w["NE.EXP.GNFS.CN"] / w["NE.EXP.GNFS.KN"]) / (w["NE.IMP.GNFS.CN"] / w["NE.IMP.GNFS.KN"])
    g = 100 * tot.groupby(level=0).pct_change(fill_method=None)
    return g.rename("tot_na_g").reset_index()


def tot_goods(path=RAW / "oecd" / "goods_trade.csv"):
    """OECD 국민계정 Table 1(지출 측면) 상품 수출(P61)·수입(P71), 경상(V)·연쇄 실질(L) 자국통화
    → 상품 수출 디플레이터 / 상품 수입 디플레이터 변화율(%). 반환: country, year, tot_goods_g"""
    d = pd.read_csv(path, usecols=["REF_AREA", "TRANSACTION", "PRICE_BASE", "TIME_PERIOD", "OBS_VALUE"])
    w = d.pivot_table(index=["REF_AREA", "TIME_PERIOD"], columns=["TRANSACTION", "PRICE_BASE"], values="OBS_VALUE")
    tot = (w[("P61", "V")] / w[("P61", "L")]) / (w[("P71", "V")] / w[("P71", "L")])
    g = 100 * tot.groupby(level=0).pct_change(fill_method=None)
    g.index.names = ["country", "year"]
    return g.rename("tot_goods_g").reset_index()


def gdp_oecd(path=RAW / "oecd" / "gdp.csv"):
    """OECD Table 1 명목 GDP(B1GQ, 경상가격, 자국통화, 백만) → country, year, b1gq"""
    d = pd.read_csv(path, usecols=["REF_AREA", "TRANSACTION", "PRICE_BASE", "TIME_PERIOD", "OBS_VALUE"])
    d = d[(d.TRANSACTION == "B1GQ") & (d.PRICE_BASE == "V")]
    return d.rename(columns={"REF_AREA": "country", "TIME_PERIOD": "year", "OBS_VALUE": "b1gq"})[["country", "year", "b1gq"]]


NONFIN = RAW / "oecd" / "table14_raw.csv"  # OECD API 응답 원본(모든 차원 열 유지). 없으면 예전 추출본
if not NONFIN.exists():
    NONFIN = RAW / "nonfin.csv"


def build(nonfin=NONFIN, wdi=RAW / "wdi.csv", denom="gdp"):
    raw = load_raw(nonfin)
    cols = {name: pick(raw, s, t, e) for name, s, t, e in SPEC}
    cols = {k: v for k, v in cols.items() if not v.empty}
    p = pd.DataFrame(cols)
    p.index.names = ["country", "year"]
    p = p.reset_index()
    for k, *_ in SPEC:
        if k not in p:
            p[k] = np.nan
    p = p.sort_values(["country", "year"])
    # 국민경제 순대출: S1 B9, 없으면 −S2 B9
    p["nl_nat"] = p["nl_s1"].fillna(-p["nl_s2"])
    level = [c for c, *_ in SPEC if c not in ("gdp", "nl_s1", "nl_s2")] + ["nl_nat"]
    gdp_file = RAW / "oecd" / "gdp.csv"
    if denom == "gdp" and gdp_file.exists():
        p = p.merge(gdp_oecd(gdp_file), on=["country", "year"], how="left")
        den = p["b1gq"]  # 두 표 모두 백만 단위(UNIT_MULT 6)
    else:
        den = p["gdp"]
    out = p[["country", "year"]].copy()
    for c in level:
        out[c] = 100 * p[c] / den
    out = out.rename(columns={"nl_nat": "ca"})
    # 금융법인 순대출은 대부분 국가에서 미제공 → 항등식 잔차로 보완
    out["nl_fin"] = out["nl_fin"].fillna(out["ca"] - out["nl_corp"] - out["nl_gov"] - out["nl_hh"])
    out["hh_rate"] = 100 * p["sv_hh"] / p["yd_hh"]
    out["gdp_g"] = 100 * p.groupby("country")["gdp"].pct_change(fill_method=None)
    if Path(wdi).exists():
        w = pd.read_csv(wdi).rename(columns={"REF_AREA": "country", "TIME_PERIOD": "year"})
        w["tot_g"] = 100 * w.groupby("country")["terms_of_trade_index"].pct_change(fill_method=None)
        out = out.merge(w[["country", "year", "real_gdp_growth", "tot_g", "old_age_dependency", "real_interest_rate"]],
                        on=["country", "year"], how="left")
        out["gdp_g"] = out["real_gdp_growth"].fillna(out["gdp_g"])
        out = out.drop(columns="real_gdp_growth")
    if (RAW / "wb").exists():
        out = out.merge(tot_na(), on=["country", "year"], how="left")
    if (RAW / "oecd" / "goods_trade.csv").exists():
        out = out.merge(tot_goods(), on=["country", "year"], how="left")
    return out


if __name__ == "__main__":
    import sys
    panel = build(*[Path(a) for a in sys.argv[1:3]])
    path = HERE / "data" / "panel.csv"
    panel.to_csv(path, index=False)
    build(*[Path(a) for a in sys.argv[1:3]], denom="gni").to_csv(HERE / "data" / "panel_gni.csv", index=False)
    cov = panel.dropna(subset=["ca", "sv_corp", "sv_hh"]).groupby("country")["year"].agg(["min", "max", "count"])
    print(cov.to_string())
    print(f"\n{len(cov)} countries, {int(cov['count'].sum())} obs → {path}")
