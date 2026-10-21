"""보고서 보조 결과(구간 포함): 가계저축률과 경상수지 반응. 사양은 ep_report --fgls --nl과 같음.

가계저축률(src_hh = 가계 총저축/가처분소득, %p): 정규화 없이 사례 1건당 원단위 반응(γ_h)
경상수지(ca = 국민경제 순대출, GDP 대비): 시작 연도 원천 부문 S−I 반응(γ_0)으로 나눈 값
→ output/ep_report_extra.csv
"""
import numpy as np
import pandas as pd

import ep_full_check as fc
import ep_report as rp
import lp_tools as lt
from make_report import OUT

K = rp.K
KEYS = [(s, v, h) for s in ("hh", "corp") for v in ("ca", "src_hh") for h in range(K + 1)]


def stats(d):
    g0 = {s: lt.ols(d, f"nl_{s}_h0", [f"z_{s}"], rp.ctrl(s), se=False, w=f"w_{s}")[0][0] for s in ("hh", "corp")}
    out = []
    for s, v, h in KEYS:
        g = lt.ols(d, f"{v}_h{h}", [f"z_{s}"], rp.ctrl(s), se=False, w=f"w_{s}")[0][0]
        out.append(g if v == "src_hh" else g / g0[s])
    return np.array(out)


def main():
    import sys
    rp.LAGS = "--lags" in sys.argv
    rp.FGLS = rp.WSV = True
    rp.NORM = "nl"
    rp.HHV, rp.COV = ["nl_hh", "ca", "src_hh"], ["nl_corp", "ca", "src_hh"]
    P, d = rp.prepare()
    pt = stats(d)
    bs = lt.cluster_bootstrap(d, stats, reps=499)
    r = pd.DataFrame(KEYS, columns=["sector", "var", "h"]).assign(
        ratio=pt, lo68=np.nanpercentile(bs, 16, axis=0), hi68=np.nanpercentile(bs, 84, axis=0),
        lo90=np.nanpercentile(bs, 5, axis=0), hi90=np.nanpercentile(bs, 95, axis=0), n_hh=len(P["hh"]), n_corp=len(P["corp"]))
    r.to_csv(OUT / f"ep_report_extra{'_lags' if rp.LAGS else ''}.csv", index=False)
    print(r.pivot_table(index=["sector", "var"], columns="h", values="ratio").round(2).to_string())


if __name__ == "__main__":
    main()
