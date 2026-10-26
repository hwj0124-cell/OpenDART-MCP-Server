"""삼성전자·SK하이닉스(별도재무제표, OpenDART) 현금흐름과 한국 부문별 순자금운용(OECD) 비교.

두 기업 합산, GNI 대비 %:
  영업이익, 영업현금흐름(OCF), 설비투자(유형+무형 취득), 잉여현금(FCF = OCF − 설비투자),
  법인세 납부, 배당 지급, 자사주 매입
한국 부문: 국민경제 순대출(CA), 비금융법인·가계·정부 순대출, 부문저축
→ output/korea_firms.csv, 콘솔 요약
"""
import numpy as np
import pandas as pd

import run_analysis as ra

HERE = ra.HERE


def gni_korea():
    raw = pd.read_csv(HERE / "data" / "raw" / "nonfin.csv", dtype=str)
    g = raw[(raw.REF_AREA == "KOR") & (raw.SECTOR == "S1") & (raw.TRANSACTION == "B5G")
            & (raw.ACCOUNTING_ENTRY == "C") & (raw.PRICE_BASE == "V")]
    return pd.Series(g.OBS_VALUE.astype(float).values / 1e3, index=g.TIME_PERIOD.astype(int).values)  # 십억원


def build(fs="ofs"):
    f = pd.read_csv(HERE / "data" / "firms" / f"firm_cf{'_ofs' if fs == 'ofs' else ''}.csv")
    f["capex"] = f.capex_ppe + f.capex_intang
    f["fcf"] = f.ocf - f.capex
    f["payout"] = f.dividends_paid + f.buyback
    two = f.groupby("year")[["op_profit", "ocf", "capex", "fcf", "tax_paid", "dividends_paid", "buyback", "payout"]].sum()
    gni = gni_korea().reindex(two.index)
    two = two.div(gni, axis=0) * 100
    two.columns = ["f_" + c for c in two.columns]
    p = pd.read_csv(HERE / "data" / "panel.csv")
    k = p[p.country == "KOR"].set_index("year")[["ca", "nl_corp", "nl_hh", "nl_gov", "nl_fin", "sv_corp", "sv_hh", "sv_gov",
                                                  "inv_corp", "prof", "hh_rate"]]
    d = two.join(k, how="left")
    d["nl_corp_other"] = d.nl_corp - d.f_fcf  # 두 기업 외 비금융법인(근사)
    return d


def main():
    d = build()
    d.round(2).to_csv(ra.OUT / "korea_firms.csv")
    print(d.round(2).to_string())
    dd = d.diff().dropna()
    cols = ["f_op_profit", "f_fcf", "nl_corp", "nl_corp_other", "nl_hh", "nl_gov", "sv_hh", "ca"]
    print("\n[연간 변화의 표준편차, GNI %p, 2015–2025]")
    print(dd[cols].std().round(2).to_string())
    print("\n[연간 변화의 1차 자기상관 (음수 = 되돌림)]")
    print(dd[cols].apply(lambda s: s.autocorr()).round(2).to_string())
    print("\n[두 기업 FCF 변화와의 상관 / 회귀계수]")
    for c in ("nl_corp", "ca", "nl_hh", "nl_gov"):
        b = np.polyfit(dd.f_fcf, dd[c], 1)[0]
        print(f"  {c:12s} corr={dd.f_fcf.corr(dd[c]):.2f}  slope={b:.2f}")
    print("\n[시차: 법인세 납부·설비투자·배당 vs 영업이익]")
    for c in ("f_tax_paid", "f_capex", "f_dividends_paid"):
        print(f"  {c:16s} corr(t, OP_t)={d[c].corr(d.f_op_profit):.2f}  corr(t, OP_t-1)={d[c].corr(d.f_op_profit.shift(1)):.2f}")


if __name__ == "__main__":
    main()
