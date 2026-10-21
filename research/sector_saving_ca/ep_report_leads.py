"""그림 1의 사전 시점(lead) 점검: h = −3, −2의 반응(y_{t+h} − y_{t−1})을 같은 사양으로 추정.

h = −1은 기준 시점이라 0. 사전 반응이 0 근처면 사건 전에 이미 움직이고 있지 않았다는 뜻(사전 추세 없음).
사양은 ep_report --fgls --nl과 같음(S−I 정규화, 2단계 가중최소제곱, 국가·연도FE), 국가 단위 부트스트랩 499회.
→ output/ep_report_leads.csv
"""
import numpy as np
import pandas as pd

import ep_report as rp
import lp_tools as lt
from make_report import OUT

LEADS = (3, 2)
HS = [-k for k in LEADS] + list(range(rp.K + 1))
KEYS = [(s, h) for s in ("hh", "corp") for h in HS]


def col(s, h):
    return f"nl_{s}_h{h}" if h >= 0 else f"nl_{s}_lead{-h}"


def stats(d):
    out = []
    for s in ("hh", "corp"):
        g0 = lt.ols(d, col(s, 0), [f"z_{s}"], [], se=False, w=f"w_{s}")[0][0]
        out += [lt.ols(d, col(s, h), [f"z_{s}"], [], se=False, w=f"w_{s}")[0][0] / g0 for h in HS]
    return np.array(out)


def main():
    rp.FGLS = rp.WSV = True
    rp.NORM = "nl"
    rp.HHV, rp.COV = ["nl_hh"], ["nl_corp"]
    P, d = rp.prepare()
    for s in ("hh", "corp"):
        g = d.groupby("country")[f"nl_{s}"]
        for k in LEADS:
            d[f"nl_{s}_lead{k}"] = g.shift(k) - g.shift(1)  # y_{t−k} − y_{t−1}
    pt = stats(d)
    bs = lt.cluster_bootstrap(d, stats, reps=499)
    r = pd.DataFrame(KEYS, columns=["sector", "h"]).assign(
        var=lambda q: "nl_" + q.sector, ratio=pt, lo68=np.nanpercentile(bs, 16, axis=0), hi68=np.nanpercentile(bs, 84, axis=0),
        n_hh=len(P["hh"]), n_corp=len(P["corp"]))
    r.to_csv(OUT / "ep_report_leads.csv", index=False)
    print(r.pivot_table(index="sector", columns="h", values="ratio").round(2).to_string())


if __name__ == "__main__":
    main()
