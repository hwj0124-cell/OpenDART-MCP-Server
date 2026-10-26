"""표본 구성 강건성: 구조가 다른 국가군을 제외했을 때 결론이 바뀌는가.

  python robustness_samples.py [--reps 299]
→ output/samples_contrasts.csv, output/samples_passthrough.csv, output/loo.csv
"""
import argparse

import numpy as np
import pandas as pd

import run_analysis as ra

MNE = ["IRL", "LUX", "NLD"]  # 다국적기업 이익 유보·재투자 수익이 부문계정을 왜곡
TRANSITION = ["BGR", "ROU", "LVA", "LTU", "EST", "HRV", "SVK", "SVN", "CZE", "HUN", "POL"]
EM_COMMODITY = ["CHL", "COL", "MEX", "ZAF", "RUS", "BRA", "CHN", "TUR", "CRI"]
CRISIS = ["ISL"]  # 2008년 은행위기로 부문수지 급변


def volatile_countries(panel, k=2.0):
    """부문저축·순대출·CA 변화의 표준편차 중 하나라도 국가 간 중위수의 k배를 넘는 국가."""
    d = ra.balance(panel)
    sd = d.groupby("country")[["sv_corp", "sv_hh", "nl_corp", "nl_hh", "ca"]].apply(
        lambda g: g.diff().std())
    return sorted(sd.index[(sd > k * sd.median()).any(axis=1)])


def samples(panel):
    allc = sorted(panel.dropna(subset=["ca", "sv_corp", "sv_hh"])["country"].unique())
    vol = volatile_countries(panel)
    core = [c for c in allc if c not in set(MNE + TRANSITION + EM_COMMODITY + CRISIS)]
    return {
        "전체": [],
        "MNE 허브 제외 (IRL·LUX·NLD)": MNE,
        "체제전환국 제외": TRANSITION,
        "신흥·원자재국 제외": EM_COMMODITY,
        f"고변동 국가 제외 ({'·'.join(vol)})": vol,
        "선진 핵심국만 (" + "·".join(core) + ")": [c for c in allc if c not in core],
    }


def main(reps):
    panel = pd.read_csv(ra.HERE / "data" / "panel.csv")
    d = ra.prepare(panel)
    rows, pts = [], []
    for name, drop in samples(panel).items():
        ds = ra.sample(d, (), drop)
        for ss in ("baseline", "profit_propensity"):
            pt = ra.passthrough(ds, ss).assign(sample=name, set=ss)
            pts.append(pt)
            c = ra.bootstrap_contrasts(ds, ss, reps)
            row = dict(sample=name, set=ss, countries=ds.dropna(subset=["ca_h0", "d_sv_corp"])["country"].nunique())
            for k in ("impact_corp_short", "impact_corp_medium", "impact_hh_short", "impact_hh_medium",
                      "impact_dd", "horizon_corp_medium", "horizon_hh_medium", "horizon_dd", "impact_diff_h6"):
                row[k] = c[k]["point"]
                row[k + "_lo"], row[k + "_hi"] = c[k]["lo90"], c[k]["hi90"]
            rows.append(row)
            print(name, ss, f"dd={row['impact_dd']:.2f} [{row['impact_dd_lo']:.2f},{row['impact_dd_hi']:.2f}]",
                  flush=True)
    pd.DataFrame(rows).to_csv(ra.OUT / "samples_contrasts.csv", index=False)
    pd.concat(pts).to_csv(ra.OUT / "samples_passthrough.csv", index=False)

    # 한 국가씩 빼기: 어느 한 나라가 결론을 좌우하는가
    loo = []
    for ss in ("baseline", "profit_propensity"):
        for c in sorted(d["country"].unique()):
            con = ra.pt_contrasts(ra.passthrough(ra.sample(d, (), [c]), ss, se=False))
            loo.append(dict(set=ss, dropped=c, impact_dd=con["impact_dd"], diff_h6=con["impact_diff_h6"],
                            corp_medium=con["impact_corp_medium"], hh_medium=con["impact_hh_medium"]))
    pd.DataFrame(loo).to_csv(ra.OUT / "loo.csv", index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=299)
    main(ap.parse_args().reps)
