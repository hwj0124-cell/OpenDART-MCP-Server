"""단방향 에피소드(가계저축률 상승·교역조건 개선만)와 이벤트 더미 회귀(축약형) 대비 LP-IV.

사양: ep_lpiv와 같음(국가별 분위, 같은 방향, 2020–22 제외, 가계·교역조건 모두 성장률 ≥ 0) + dir="up", q ∈ {0.10, 0.20}.
축약형: y_{t+h} − y_{t−1} = γ_h z_it + 국가FE + 연도FE  (γ_h: 에피소드 1건의 평균 반응, GNI 대비 %p)
LP-IV : β_h = γ_h(y) / γ_0(nl)  — 축약형을 시작 시점 순자금운용 반응으로 나눈 것과 같다(같은 표본일 때 정확히).
→ output/ep_oneway.csv
"""
import numpy as np
import pandas as pd

import ep_detail as ed
import ep_lpiv as el
import lp_tools as lt
import run_analysis as ra

K = ed.K
SPECS = {"양방향 10% (ep_lpiv 기본)": dict(), "상승·개선만 10%": dict(dir="up"), "상승·개선만 20%": dict(dir="up", q=0.20)}


def nl_betas(d):
    return np.array([lt.iv(d, f"nl_{s}_h{h}", f"dx_{s}", f"z_{s}", [], se=False)[0] for s in ("hh", "corp") for h in range(K + 1)])


def main():
    x = ed.er.panel()
    rows = []
    for name, ch in SPECS.items():
        P = ed.paths(x, {**el.SPEC, **ch})
        d = el.design(x, P)
        for s, E in P.items():
            for h in range(K + 1):
                y = f"nl_{s}_h{h}"
                b, se, n, F = lt.iv(d, y, f"dx_{s}", f"z_{s}", [])
                ep = E[f"nl_{s}_{h}"].dropna().sum() / E.loc[E[f"nl_{s}_{h}"].notna(), "den"].sum()
                rows.append(dict(spec=name, sector=s, h=h, n_ep=len(E), beta=b, se=se, F=F, episode_avg=ep))
        bs = lt.cluster_bootstrap(d, nl_betas, reps=499)
        diff = bs[:, :K + 1] - bs[:, K + 1:]
        for h in range(K + 1):
            rows.append(dict(spec=name, sector="diff", h=h, beta=np.nan, lo90=np.nanpercentile(diff[:, h], 5),
                             hi90=np.nanpercentile(diff[:, h], 95),
                             p=2 * min((diff[:, h] <= 0).mean(), (diff[:, h] >= 0).mean())))
    res = pd.DataFrame(rows)
    pt = res[res.sector != "diff"].pivot_table(index=["spec", "h"], columns="sector", values="beta")
    m = res.sector == "diff"
    res.loc[m, "beta"] = [pt.loc[(sp, h), "hh"] - pt.loc[(sp, h), "corp"] for sp, h in zip(res.spec[m], res.h[m])]
    res.to_csv(ra.OUT / "ep_oneway.csv", index=False)
    for name in SPECS:
        r = res[res.spec == name]
        print(f"== {name}  사례: 가계 {int(r[r.sector=='hh'].n_ep.iloc[0])}, 교역조건 {int(r[r.sector=='corp'].n_ep.iloc[0])}")
        t = r.pivot_table(index="sector", columns="h", values="beta", sort=False).round(2)
        t.loc["hh_se"] = r[r.sector == "hh"].set_index("h").se.round(2)
        t.loc["corp_se"] = r[r.sector == "corp"].set_index("h").se.round(2)
        t.loc["diff_p"] = r[r.sector == "diff"].set_index("h").p.round(3)
        t.loc["hh_epavg"] = r[r.sector == "hh"].set_index("h").episode_avg.round(2)
        t.loc["corp_epavg"] = r[r.sector == "corp"].set_index("h").episode_avg.round(2)
        print(t.to_string()); print("F:", r.groupby("sector").F.min().round(1).to_dict())


if __name__ == "__main__":
    main()
