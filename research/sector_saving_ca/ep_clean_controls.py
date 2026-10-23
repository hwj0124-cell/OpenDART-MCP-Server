"""LP-DiD식 '깨끗한 대조군' 점검 (Dube, Girardi, Jordà & Taylor의 비흡수 처치 설정을 따름).

시점 h 회귀마다 표본을 다음으로 제한:
  처치 관측치: 사례 해(z ≠ 0)
  대조 관측치: 같은 나라에 [t−3, t+h] 동안 같은 종류의 큰 충격이 없는 국가·연도
    큰 충격 = 사례 선정 변수(가계저축률 변화 / 교역조건 변화)가 국가별 상·하위 10%에 든 모든 해
             (다른 선정 조건 통과 여부와 무관하게 '오염'으로 간주 → 보수적)
그 외 사양은 ep_report --fgls --nl --lags와 같음(S−I 정규화, 2단계 가중최소제곱 가중치, 국가·연도FE, 시차 통제).
→ output/ep_clean_controls.csv  (python ep_clean_controls.py --boot 이면 구간 포함)
"""
import sys

import numpy as np
import pandas as pd

import ep_full_check as fc
import ep_report as rp
import lp_tools as lt
from make_report import OUT

K = rp.K
PRE = 3
VARS = {"hh": ["nl_hh", "sv_hh"], "corp": ["nl_corp", "sv_corp", "prof", "div_net", "gcf_corp"]}


def shock_flags(d):
    g = d.groupby("country")
    sel = {"hh": d.hh_rate - g.hh_rate.shift(1), "corp": d[fc.TOT]}
    out = {}
    for s, v in sel.items():
        lo = v.groupby(d.country).transform(lambda x: x.quantile(0.10))
        hi = v.groupby(d.country).transform(lambda x: x.quantile(0.90))
        out[s] = ((v <= lo) | (v >= hi)).fillna(False)
    return out


def clean_masks(d):
    """(s, h) → 표본 마스크: 처치 관측치 ∪ [t−PRE, t+h]에 큰 충격이 없는 대조 관측치."""
    flags = shock_flags(d)
    m = {}
    for s in ("hh", "corp"):
        f = flags[s].astype(int)
        g = f.groupby(d.country)
        past = sum(g.shift(k).fillna(0) for k in range(0, PRE + 1))  # t−PRE … t
        for h in range(K + 1):
            fut = sum(g.shift(-k).fillna(0) for k in range(1, h + 1)) if h else 0
            m[(s, h)] = (d[f"z_{s}"] != 0) | ((past + fut) == 0)
    return m


def stats(d):
    m = clean_masks(d)
    out = []
    for s, vs in VARS.items():
        c, w = rp.ctrl(s), f"w_{s}"
        g0 = lt.ols(d[m[(s, 0)]], f"nl_{s}_h0", [f"z_{s}"], c, se=False, w=w)[0][0]
        for v in vs:
            out += [lt.ols(d[m[(s, h)]], f"{v}_h{h}", [f"z_{s}"], c, se=False, w=w)[0][0] / g0 for h in range(K + 1)]
    return np.array(out)


def main():
    rp.FGLS = rp.WSV = rp.LAGS = True
    rp.NORM = "nl"
    rp.HHV, rp.COV = VARS["hh"], VARS["corp"]
    P, d = rp.prepare()
    m = clean_masks(d)
    for s in ("hh", "corp"):
        nz = int((d[f"z_{s}"] != 0).sum())
        print(f"{s}: 처치 {nz}건, 대조 관측치 수 h=0 {int(m[(s, 0)].sum()) - nz}, h=6 {int(m[(s, 6)].sum()) - nz} (전체 {len(d)})")
    pt = stats(d)
    keys = [(s, v, h) for s, vs in VARS.items() for v in vs for h in range(K + 1)]
    r = pd.DataFrame(keys, columns=["sector", "var", "h"]).assign(ratio=pt)
    if "--boot" in sys.argv:
        bs = lt.cluster_bootstrap(d, stats, reps=499)
        r = r.assign(lo68=np.nanpercentile(bs, 16, axis=0), hi68=np.nanpercentile(bs, 84, axis=0))
    r.to_csv(OUT / "ep_clean_controls.csv", index=False)
    base = pd.read_csv(OUT / "ep_report_fgls_nl_lags.csv")
    base = base[base.kind == "resp"]
    for s, v in (("hh", "nl_hh"), ("corp", "nl_corp"), ("hh", "sv_hh"), ("corp", "sv_corp")):
        a = base[(base.sector == s) & (base["var"] == v)].sort_values("h").ratio.round(2).values
        b = r[(r.sector == s) & (r["var"] == v)].sort_values("h").ratio.round(2).values
        print(f"{v:8s} 기본  {a}\n{'':8s} 깨끗한 {b}")


if __name__ == "__main__":
    main()
