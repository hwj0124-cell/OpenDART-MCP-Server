"""최종 비교(신뢰구간): 가계 성향(기존 또는 소비 주도) vs 상품·서비스 교역조건.

기본 사양: 새 원자료 전체 기간(불균형 패널), GDP 대비, 이벤트 더미 국소투영, 국가·연도FE, 가중치 없음.
y_{t+h} − y_{t−1} = γ_h · z_it + 국가FE + 연도FE,  정규화 = γ_h ÷ γ_0(원천 부문 순자금운용)
차이 = 가계 사례(가계 부문) − 교역조건 사례(기업 부문): 순자금운용·저축, 그리고 경상수지(두 사례 각각).
구간: 국가 단위 부트스트랩 499회(비율·차이를 각 표본에서 다시 계산). 68%·90% 저장.
사용: python ep_final_ci.py base|cons  → output/ep_final_ci_<정의>.csv
"""
import sys

import numpy as np
import pandas as pd

import ep_detail as ed
import ep_full_check as fc
import ep_lpiv as el
import lp_tools as lt
from make_report import OUT

K = ed.K
HHV = ["nl_hh", "sv_hh", "invtot_hh", "inv_hh", "yd_hh", "cons_hh", "ca", "nl_corp", "sv_corp", "nl_gov"]
COV = ["nl_corp", "sv_corp", "invtot_corp", "inv_corp", "prof", "div_paid", "ca", "nl_hh", "nl_gov"]
KEYS = [("hh", v, h) for v in HHV for h in range(K + 1)] + [("corp", v, h) for v in COV for h in range(K + 1)]
DIFFS = [("nl", "nl_hh", "nl_corp"), ("sv", "sv_hh", "sv_corp"), ("ca", "ca", "ca")]


def stats(d):
    g = {k: lt.ols(d, f"{k[1]}_h{k[2]}", [f"z_{k[0]}"], [], se=False)[0][0] for k in KEYS}
    r = {k: g[k] / g[(k[0], f"nl_{k[0]}", 0)] for k in KEYS}
    out = [r[k] for k in KEYS]
    for _, a, b in DIFFS:
        out += [r[("hh", a, h)] - r[("corp", b, h)] for h in range(K + 1)]
    return np.array(out)


def main():
    defn = sys.argv[1] if len(sys.argv) > 1 else "base"
    x = ed.er.panel()
    fc.VARS["hh"], fc.VARS["corp"] = HHV, COV
    P = ed.paths(x, {**el.SPEC, "tot": fc.TOT, "hh_def": None if defn == "base" else defn})
    d = fc.design(x, P)
    pt = stats(d)
    bs = lt.cluster_bootstrap(d, stats, reps=499)
    labels = [dict(kind="resp", sector=s, var=v, h=h) for s, v, h in KEYS] + \
             [dict(kind="diff", sector="diff", var=name, h=h) for name, _, _ in DIFFS for h in range(K + 1)]
    r = pd.DataFrame(labels).assign(
        defn=defn, ratio=pt, lo68=np.nanpercentile(bs, 16, axis=0), hi68=np.nanpercentile(bs, 84, axis=0),
        lo90=np.nanpercentile(bs, 5, axis=0), hi90=np.nanpercentile(bs, 95, axis=0),
        p=np.minimum(1, 2 * np.minimum((bs <= 0).mean(0), (bs >= 0).mean(0))),
        n_hh=len(P["hh"]), n_corp=len(P["corp"]))
    r.to_csv(OUT / f"ep_final_ci_{defn}.csv", index=False)
    q = r[r.kind == "diff"]
    print(defn, "n", len(P["hh"]), len(P["corp"]))
    print(q.pivot_table(index="var", columns="h", values=["ratio", "p"]).round(2).to_string())


if __name__ == "__main__":
    main()
