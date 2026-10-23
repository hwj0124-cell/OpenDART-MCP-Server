"""기업 행태 비율 반응(교역조건 사례, 주 사양: FGLS·S−I 기준 가중치·시차 통제), 정규화 없이 사례 1건당 %p.

  ret_corp   유보율      = 기업 총저축 / 총영업잉여
  reinv_corp 재투자율    = 기업 총자본형성 / 총영업잉여
  pay_corp   순배당 성향 = 순배당 지급(지급−수취) / 총영업잉여
  prof_corp  영업잉여(GDP 대비, %p) — 분모 자체의 움직임 확인용
→ output/ep_corp_ratios_lags.csv
"""
import numpy as np
import pandas as pd

import ep_detail as ed
import ep_report as rp
import lp_tools as lt
from make_report import OUT

RATIOS = ["ret_corp", "reinv_corp", "pay_corp", "prof_corp"]
KEYS = [(v, h) for v in RATIOS for h in range(rp.K + 1)]
_panel = ed.er.panel


def panel(*a, **k):
    x = _panel(*a, **k)
    x["ret_corp"] = 100 * x.sv_corp / x.prof
    x["reinv_corp"] = 100 * x.gcf_corp / x.prof
    x["pay_corp"] = 100 * (x.div_paid - x.div_recv_corp) / x.prof
    x["prof_corp"] = x.prof
    return x


def stats(d):
    return np.array([lt.ols(d, f"{v}_h{h}", ["z_corp"], rp.ctrl("corp"), se=False, w="w_corp")[0][0] for v, h in KEYS])


def main():
    ed.er.panel = panel
    rp.FGLS = rp.WSV = rp.LAGS = True
    rp.NORM = "nl"
    rp.HHV, rp.COV = ["nl_hh"], ["nl_corp", *RATIOS]
    P, d = rp.prepare()
    pt = stats(d)
    bs = lt.cluster_bootstrap(d, stats, reps=499)
    r = pd.DataFrame(KEYS, columns=["var", "h"]).assign(
        ratio=pt, lo68=np.nanpercentile(bs, 16, axis=0), hi68=np.nanpercentile(bs, 84, axis=0), n_corp=len(P["corp"]))
    r.to_csv(OUT / "ep_corp_ratios_lags.csv", index=False)
    for c in ("ratio", "lo68", "hi68"):
        print(c); print(r.pivot(index="var", columns="h", values=c).loc[RATIOS].round(2).to_string())


if __name__ == "__main__":
    main()
