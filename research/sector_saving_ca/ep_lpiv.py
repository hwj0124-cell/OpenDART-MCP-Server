"""에피소드를 도구로 한 패널 LP-IV (국가·연도 고정효과): 가계 성향 vs 교역조건(비침체).

y_{t+h} − y_{t−1} = β_h (nl_t − nl_{t−1}) + 국가FE + 연도FE [+ 통제] + e
1단계 도구 z_it = 에피소드 해의 충격 부호(±1), 그 외 0. 에피소드는 ep_detail 기본 기준에
교역조건 쪽 "그해 성장률 ≥ 0"(가계와 같은 침체 제외 조건)을 더한 것.
β_h = 시작 시점 자기 부문 순자금운용 1단위 변화 중 h년 뒤 남는 몫(y = 순자금운용일 때).
통제(선택): 자기 부문 Δ순자금운용 2개 시차, 전년 성장률. 구간은 국가 단위 부트스트랩 499회.
→ output/ep_lpiv_fe.csv
"""
import numpy as np
import pandas as pd

import ep_detail as ed
import lp_tools as lt
import run_analysis as ra

K = ed.K
SPEC = {**ed.BASE, "tot_growth": True}
VARS = {"hh": ["nl_hh", "sv_hh", "invtot_hh", "inv_hh"], "corp": ["nl_corp", "sv_corp", "invtot_corp", "inv_corp"]}


def design(x, P):
    d = x.copy()
    g = lambda v: d.groupby("country")[v]
    d["gdp_g_l1"] = g("gdp_g").shift(1)
    for s, E in P.items():
        z = E.set_index(["country", "year"]).sign
        d[f"z_{s}"] = [z.get((c, y), 0.0) for c, y in zip(d.country, d.year)]
        d[f"dx_{s}"] = d[f"nl_{s}"] - g(f"nl_{s}").shift(1)
        d[f"dx_{s}_l1"] = g(f"dx_{s}").shift(1)
        d[f"dx_{s}_l2"] = g(f"dx_{s}").shift(2)
        for v in VARS[s]:
            for h in range(K + 1):
                d[f"{v}_h{h}"] = g(v).shift(-h) - g(v).shift(1)
    return d


def ctrl(s, on):
    return [f"dx_{s}_l1", f"dx_{s}_l2", "gdp_g_l1"] if on else []


def betas(d, on=False):
    return np.array([lt.iv(d, f"{v}_h{h}", f"dx_{s}", f"z_{s}", ctrl(s, on), se=False)[0]
                     for s in VARS for v in VARS[s] for h in range(K + 1)])


def main():
    x = ed.er.panel()
    P = ed.paths(x, SPEC)
    d = design(x, P)
    keys = [(s, v, h) for s in VARS for v in VARS[s] for h in range(K + 1)]
    rows = []
    for on in (False, True):
        for s, v, h in keys:
            b, se, n, F = lt.iv(d, f"{v}_h{h}", f"dx_{s}", f"z_{s}", ctrl(s, on))
            rows.append(dict(model="고정효과+통제" if on else "고정효과", sector=s, var=v, h=h, beta=b, se=se, nobs=n, F=F))
    res = pd.DataFrame(rows)
    bs = lt.cluster_bootstrap(d, betas, reps=499)
    base = res[res.model == "고정효과"].index
    res.loc[base, "lo68"] = np.nanpercentile(bs, 16, axis=0)
    res.loc[base, "hi68"] = np.nanpercentile(bs, 84, axis=0)
    # 가계 − 교역조건 차이 (순자금운용), 같은 부트스트랩 표본에서
    ih = [i for i, k in enumerate(keys) if k[1] == "nl_hh"]
    ic = [i for i, k in enumerate(keys) if k[1] == "nl_corp"]
    diff = bs[:, ih] - bs[:, ic]
    pt = res.loc[base].set_index("var").beta
    dd = pd.DataFrame(dict(model="차이(가계−교역조건)", sector="diff", var="nl_diff", h=range(K + 1),
                           beta=pt["nl_hh"].values - pt["nl_corp"].values,
                           lo68=np.nanpercentile(diff, 16, axis=0), hi68=np.nanpercentile(diff, 84, axis=0),
                           lo90=np.nanpercentile(diff, 5, axis=0), hi90=np.nanpercentile(diff, 95, axis=0),
                           p=2 * np.minimum((diff <= 0).mean(0), (diff >= 0).mean(0))))
    # 에피소드 평균(고정효과 없음)과 비교
    ep = pd.DataFrame([dict(model="에피소드 평균", sector=s, var=f"nl_{s}", h=h,
                            beta=(E[f"nl_{s}_{h}"] * 1).dropna().sum() / E.loc[E[f"nl_{s}_{h}"].notna(), "den"].sum())
                       for s, E in P.items() for h in range(K + 1)])
    res = pd.concat([res, dd, ep], ignore_index=True)
    res.to_csv(ra.OUT / "ep_lpiv_fe.csv", index=False)
    print("사례 수:", {s: len(E) for s, E in P.items()})
    show = res[res["var"].isin(["nl_hh", "nl_corp", "nl_diff"])]
    print(show.pivot_table(index=["model", "var"], columns="h", values="beta", sort=False).round(2).to_string())
    print(res[(res.model == "고정효과") & res["var"].str.startswith("nl")].pivot_table(
        index="var", columns="h", values="se").round(2).to_string())
    print("1단계 F:", res[res.model == "고정효과"].groupby("sector").F.agg(["min", "max"]).round(1).to_dict())
    print(dd[["h", "beta", "lo90", "hi90", "p"]].round(3).to_string(index=False))
    for kind in ("sv", "invtot", "inv"):
        print(kind, res[(res.model == "고정효과") & res["var"].str.startswith(kind + "_")].pivot_table(
            index="var", columns="h", values="beta").round(2).to_string())


if __name__ == "__main__":
    main()
