"""사례 선정 규칙 민감도(주 사양: FGLS, S−I 정규화, 시차 통제)와 사례 선정 없는 부문 수지 지속성.

(1) 필터를 하나씩 빼며 Figure 1 비율(γ_h/γ_0)과 시작 연도 반응의 t값을 비교
(2) 무조건부 지속성: y_{t+h} − y_{t−1}을 Δy_t(+2시차, 국가·연도 FE)에 회귀 → 원천을 구분하지 않은 평균 지속성
→ output/ep_selection_check_lags.csv
"""
import numpy as np
import pandas as pd

import ep_report as rp
import lp_tools as lt
from make_report import OUT

VARIANTS = {
    "main": {},
    "no S-I direction filter": dict(align="shock"),
    "no profit condition": dict(tot_prof=False),
    "neither corporate filter": dict(align="shock", tot_prof=False),
    "no consumption-led condition": dict(hh_def=None),
    "minimal rule": dict(align="shock", tot_prof=False, hh_def=None),
}


def selection():
    rp.FGLS = rp.WSV = rp.LAGS = True
    rp.NORM = "nl"
    rp.HHV, rp.COV = ["nl_hh"], ["nl_corp"]
    base, rows = dict(rp.CH), []
    for name, ch in VARIANTS.items():
        rp.CH = {**base, **ch}
        P, d = rp.prepare()
        for s in ("hh", "corp"):
            g = [lt.ols(d, f"nl_{s}_h{h}", [f"z_{s}"], rp.ctrl(s), se=False, w=f"w_{s}")[0][0] for h in range(7)]
            b, V, _ = lt.ols(d, f"nl_{s}_h0", [f"z_{s}"], rp.ctrl(s), w=f"w_{s}")
            rows += [dict(kind="selection", spec=name, sector=s, n=len(P[s]), t0=b[0] / np.sqrt(V[0, 0]), h=h, ratio=g[h] / g[0])
                     for h in range(7)]
    rp.CH = base
    return rows


def unconditional():
    d = lt.balance(pd.read_csv(OUT.parent / "data" / "panel.csv"))
    g, rows = d.groupby("country"), []
    for v in ("nl_hh", "nl_corp", "sv_hh", "sv_corp"):
        x = g[v].diff()
        d["dx"], d["l1"], d["l2"] = x, x.groupby(d.country).shift(1), x.groupby(d.country).shift(2)
        for h in range(7):
            d["y"] = g[v].shift(-h) - g[v].shift(1)
            rows.append(dict(kind="unconditional", spec=v, sector=v.split("_")[1], h=h,
                             ratio=lt.ols(d, "y", ["dx"], ["l1", "l2"], se=False)[0][0]))
    return rows


if __name__ == "__main__":
    r = pd.DataFrame(selection() + unconditional())
    r.to_csv(OUT / "ep_selection_check_lags.csv", index=False)
    print(r.pivot_table(index=["kind", "spec", "sector"], columns="h", values="ratio", sort=False).round(2).to_string())
