"""LP 이외의 방법(전체 41개국): 장기차분, 주기 분해(HP), 패널 오차수정모형(ECM).

두 쌍을 비교: (1) 부문 총저축 sv_corp vs sv_hh
             (2) 원천: 기업이익×국가평균 유보율 vs 가계저축률×국가평균 가처분소득비중 (GNI %p)
→ output/alt_methods.csv
"""
import numpy as np
import pandas as pd
from statsmodels.tsa.filters.hp_filter import hpfilter

import run_analysis as ra
from lp_tools import balance, lagged, ols

PAIRS = {"부문 총저축": ("sv_corp", "sv_hh"), "원천(이익 vs 저축률)": ("profS", "rateS")}


def prepare():
    d = balance(pd.read_csv(ra.HERE / "data" / "panel.csv"))
    g = d.groupby("country")
    d["profS"] = d["prof"] * g["sv_corp"].transform("mean") / g["prof"].transform("mean")
    d["rateS"] = d["hh_rate"] * g["yd_hh"].transform("mean") / 100
    return d


def row(method, spec, pair, b, V, n, i=0, j=1):
    se = np.sqrt(np.diag(V))
    return dict(method=method, spec=spec, pair=pair, n=n, corp=b[i], se_corp=se[i], hh=b[j], se_hh=se[j],
                diff=b[j] - b[i], se_diff=np.sqrt(V[i, i] + V[j, j] - 2 * V[i, j]))


def main():
    d = prepare()
    rows = []
    for name, (c, h) in PAIRS.items():
        for k in (1, 2, 3, 5, 7, 10):  # 장기차분
            dd = d.copy()
            for v in ("ca", c, h):
                dd[f"D_{v}"] = dd[v] - dd.groupby("country")[v].shift(k)
            b, V, n = ols(dd, "D_ca", [f"D_{c}", f"D_{h}"], [], fe=("year",))
            rows.append(row("장기차분", f"k={k}", name, b, V, n))
        for lam in (6.25, 100):  # HP 분해
            parts = []
            for ctry, gg in d.groupby("country"):
                z = gg.dropna(subset=["ca", c, h])
                if len(z) < 15:
                    continue
                r = z[["country", "year"]].copy()
                for v in ("ca", c, h):
                    r[v + "_c"], r[v + "_t"] = hpfilter(z[v].to_numpy(), lamb=lam)
                parts.append(r)
            hp = pd.concat(parts)
            for comp, lab in (("c", "순환"), ("t", "추세")):
                b, V, n = ols(hp, f"ca_{comp}", [f"{c}_{comp}", f"{h}_{comp}"], [])
                rows.append(row("HP 분해", f"λ={lam} {lab}", name, b, V, n))
        dd = d.copy()  # ECM
        gg = dd.groupby("country")
        dd["dca"], dd["l_ca"] = gg["ca"].diff(), gg["ca"].shift(1)
        for v in (c, h):
            dd["d_" + v], dd["l_" + v] = gg[v].diff(), gg[v].shift(1)
        lag = lagged(dd, ["dca", "d_" + c, "d_" + h], 1)
        dd = pd.concat([dd, lag], axis=1)
        X = ["d_" + c, "d_" + h, "l_ca", "l_" + c, "l_" + h]
        b, V, n = ols(dd, "dca", X, list(lag.columns))
        rows.append(row("ECM", "단기(당해)", name, b, V, n, 0, 1))
        phi = b[2]
        J = np.zeros((2, 5))  # 장기계수 −θ/φ 의 델타법
        for r_, i in ((0, 3), (1, 4)):
            J[r_, i], J[r_, 2] = -1 / phi, b[i] / phi ** 2
        lr = np.array([-b[3] / phi, -b[4] / phi])
        rows.append(row("ECM", f"장기(조정속도 {phi:.2f})", name, lr, J @ V @ J.T, n))
    out = pd.DataFrame(rows)
    out.to_csv(ra.OUT / "alt_methods.csv", index=False)
    print(out.round(2).to_string())


if __name__ == "__main__":
    main()
