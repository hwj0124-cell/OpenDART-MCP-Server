"""본문 사양(시차 통제, 2단계 가중최소제곱, S−I 정규화)의 보조 결과.

1) 진단용 세부 변수 점추정치(S−I 정규화): 기업 저축·영업잉여·순배당·총투자·고정투자, 가계 저축·가처분소득·소비·총투자
   → output/ep_report_diag_lags.csv
2) 명목 누적 증가율(로그×100, 정규화 없음, 68% 구간): 가계 소비, 가처분소득, 명목 GDP — 두 사례 모두
   GDP 대비 비율의 분모 효과 없이 소비 반응을 비교하는 용도 → output/ep_report_growth_lags.csv
"""
import numpy as np
import pandas as pd

import ep_report as rp
import lp_tools as lt
from make_report import OUT

K = rp.K
DIAG = {"hh": ["nl_hh", "sv_hh", "yd_hh", "cons_hh", "gcf_hh"],
        "corp": ["nl_corp", "sv_corp", "prof", "div_net", "gcf_corp", "inv_corp"]}
GROW = ["lcons_hh", "lyd_hh", "lgdp"]
KEYS = [(s, v, h) for s in ("hh", "corp") for v in GROW for h in range(K + 1)]


def growth(d):
    return np.array([lt.ols(d, f"{v}_h{h}", [f"z_{s}"], rp.ctrl(s), se=False, w=f"w_{s}")[0][0] for s, v, h in KEYS])


def main():
    rp.FGLS = rp.WSV = rp.LAGS = True
    rp.NORM = "nl"
    rp.HHV, rp.COV = DIAG["hh"] + GROW, DIAG["corp"] + GROW
    P, d = rp.prepare()
    rows = []
    for s, vs in DIAG.items():
        c, w = rp.ctrl(s), f"w_{s}"
        g0 = lt.ols(d, f"nl_{s}_h0", [f"z_{s}"], c, se=False, w=w)[0][0]
        for v in vs:
            rows += [dict(sector=s, var=v, h=h, ratio=lt.ols(d, f"{v}_h{h}", [f"z_{s}"], c, se=False, w=w)[0][0] / g0)
                     for h in range(K + 1)]
    diag = pd.DataFrame(rows).assign(n_hh=len(P["hh"]), n_corp=len(P["corp"]))
    diag.to_csv(OUT / "ep_report_diag_lags.csv", index=False)
    print(diag.pivot_table(index=["sector", "var"], columns="h", values="ratio", sort=False).round(2).to_string())
    pt = growth(d)
    bs = lt.cluster_bootstrap(d, growth, reps=499)
    g = pd.DataFrame(KEYS, columns=["sector", "var", "h"]).assign(
        ratio=pt, lo68=np.nanpercentile(bs, 16, axis=0), hi68=np.nanpercentile(bs, 84, axis=0),
        n_hh=len(P["hh"]), n_corp=len(P["corp"]))
    g.to_csv(OUT / "ep_report_growth_lags.csv", index=False)
    print(g.pivot_table(index=["sector", "var"], columns="h", values="ratio", sort=False).round(2).to_string())


if __name__ == "__main__":
    main()
