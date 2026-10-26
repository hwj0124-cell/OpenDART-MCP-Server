"""외생적 원천으로 식별: 교역조건(ToT) 충격 → 기업이익, 고령화 → 가계저축률 (41개국 전체).

1) 1단계 강도: Δ이익 ← ToT 변화율, 가계저축률 ← 노년부양비 (국가·연도 FE, 국가 군집)
2) ToT +10% 충격의 축약형 LP: 이익·부문저축·국민저축·경상수지·투자·순자금운용 경로
→ output/tot_first_stage.csv, output/tot_lp.csv
"""
import numpy as np
import pandas as pd

import persistence as P
import run_analysis as ra
from lp_tools import ols

H = ra.H
OUTS = ["prof", "sv_corp", "sv_hh", "sv_gov", "sv_nat", "ca", "inv_corp", "inv_hh", "nl_corp", "nl_hh", "nl_gov"]


def main():
    d = P.prepare(pd.read_csv(ra.HERE / "data" / "panel.csv"))
    g = d.groupby("country")
    d["L2_tot_g"] = g["tot_g"].shift(2)
    d["d_prof"], d["d_hh_rate"] = g["prof"].diff(), g["hh_rate"].diff()
    d["d_oad"] = g["old_age_dependency"].diff()
    fs = []
    for y, x in (("d_prof", "tot_g"), ("d_hh_rate", "d_oad"), ("hh_rate", "old_age_dependency"),
                 ("ca", "old_age_dependency")):
        b, V, n = ols(d, y, [x], [])
        fs.append(dict(y=y, x=x, b=b[0], se=np.sqrt(V[0, 0]), F=b[0] ** 2 / V[0, 0], n=n))
    pd.DataFrame(fs).to_csv(ra.OUT / "tot_first_stage.csv", index=False)
    ctrl = ["L1_tot_g", "L2_tot_g", "L1_d_ca", "L2_d_ca", "L1_gdp_g", "L2_gdp_g",
            "L1_sv_corp", "L1_sv_hh", "L1_ca", "L1_prof", "L1_hh_rate"]
    rows = []
    for y in OUTS:
        for h in range(H + 1):
            b, V, n = ols(d, f"{y}_h{h}", ["tot_g"], ctrl)
            rows.append(dict(outcome=y, h=h, b=10 * b[0], se=10 * np.sqrt(V[0, 0]), n=n))  # ToT +10%
    pd.DataFrame(rows).to_csv(ra.OUT / "tot_lp.csv", index=False)
    print(pd.DataFrame(fs).round(3).to_string())


if __name__ == "__main__":
    main()
