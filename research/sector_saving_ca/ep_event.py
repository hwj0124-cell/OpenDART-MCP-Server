"""이벤트 더미 국소투영(정규화): 가계 성향 vs 교역조건.

y_{t+h} − y_{t−1} = γ_h(y) · z_it + 국가FE [+ 연도FE] + e
  z_it = 에피소드 해의 충격 부호(±1), 그 외 0  (증가·감소를 부호로 맞춰 함께 추정)
  γ_h(y): 에피소드 1건의 평균 반응 (GNI 대비 %p)
정규화 비율 = γ_h(y) / γ_0(자기 부문 순자금운용)  → 시작 시점 순자금운용 변화 1단위당 반응
에피소드는 ep_lpiv.SPEC(국가별 상·하위 10%, 같은 방향, 4년 내 중복 제외, 시작 2020–22 제외,
가계·교역조건 모두 그해 성장률 ≥ 0, 교역조건은 영업잉여 같은 방향)에 아래 사양별 변경.
구간·차이 검정은 국가 단위 부트스트랩 499회(비율을 각 표본에서 다시 계산).
→ output/ep_event.csv
"""
import sys

import numpy as np
import pandas as pd

import ep_detail as ed
import ep_lpiv as el
import lp_tools as lt
import run_analysis as ra

K = ed.K
VARS = el.VARS
SPECS = {
    "상품 교역조건, 국가·연도FE (기본)": (dict(tot="tot_goods_g"), ("country", "year")),
    "상품 교역조건, 국가FE만": (dict(tot="tot_goods_g"), ("country",)),
    "상품·서비스 교역조건, 국가·연도FE": (dict(tot="tot_na_g"), ("country", "year")),
    "상품 교역조건, 개선·상승만, 국가·연도FE": (dict(tot="tot_goods_g", dir="up"), ("country", "year")),
}
KEYS = [(s, v, h) for s in VARS for v in VARS[s] for h in range(K + 1)]


def gammas(d, fe):
    return np.array([lt.ols(d, f"{v}_h{h}", [f"z_{s}"], [], fe=fe, se=False)[0][0] for s, v, h in KEYS])


def ratios(g):
    g = pd.Series(g, index=pd.MultiIndex.from_tuples(KEYS))
    return np.array([g[(s, v, h)] / g[(s, f"nl_{s}", 0)] for s, v, h in KEYS])


def run(name, ch, fe, x):
    P = ed.paths(x, {**el.SPEC, **ch})
    d = el.design(x, P)
    g, se = [], []
    for s, v, h in KEYS:
        b, V, n = lt.ols(d, f"{v}_h{h}", [f"z_{s}"], [], fe=fe)
        g.append(b[0]); se.append(np.sqrt(V[0, 0]))
    bs = lt.cluster_bootstrap(d, lambda b: ratios(gammas(b, fe)), reps=499)
    r = ratios(np.array(g))
    out = pd.DataFrame(KEYS, columns=["sector", "var", "h"]).assign(
        spec=name, n_ep=[len(P[s]) for s, _, _ in KEYS], gamma=g, gamma_se=se, ratio=r,
        lo68=np.nanpercentile(bs, 16, axis=0), hi68=np.nanpercentile(bs, 84, axis=0))
    ih = [i for i, k in enumerate(KEYS) if k[1] == "nl_hh"]
    ic = [i for i, k in enumerate(KEYS) if k[1] == "nl_corp"]
    diff = bs[:, ih] - bs[:, ic]
    dd = pd.DataFrame(dict(sector="diff", var="nl_diff", h=range(K + 1), spec=name, ratio=r[ih] - r[ic],
                           lo68=np.nanpercentile(diff, 16, axis=0), hi68=np.nanpercentile(diff, 84, axis=0),
                           lo90=np.nanpercentile(diff, 5, axis=0), hi90=np.nanpercentile(diff, 95, axis=0),
                           p=2 * np.minimum((diff <= 0).mean(0), (diff >= 0).mean(0))))
    yrs = P["corp"].year.value_counts()
    print(f"== {name}: 가계 {len(P['hh'])}, 교역조건 {len(P['corp'])}건, 교역조건 최다 연도 {yrs.head(3).to_dict()}", flush=True)
    return pd.concat([out, dd], ignore_index=True)


def main():
    x = ed.er.panel()
    names = sys.argv[1:] or list(SPECS)
    res = pd.concat([run(n, *SPECS[n], x) for n in names], ignore_index=True)
    res.to_csv(ra.OUT / "ep_event.csv", index=False)
    for n in names:
        r = res[(res.spec == n) & res["var"].isin(["nl_hh", "nl_corp", "nl_diff"])]
        t = r.pivot_table(index="var", columns="h", values="ratio", sort=False).round(2)
        t.loc["diff_p"] = r[r.sector == "diff"].set_index("h").p.round(3)
        t.loc["γ_hh(%p)"] = r[r["var"] == "nl_hh"].set_index("h").gamma.round(2)
        t.loc["γ_corp(%p)"] = r[r["var"] == "nl_corp"].set_index("h").gamma.round(2)
        print(f"-- {n}\n{t.to_string()}")


if __name__ == "__main__":
    main()
