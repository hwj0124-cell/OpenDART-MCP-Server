"""저축 기준 정규화·σ 문턱 점검(점추정치). 국가별 1σ·1.28σ 밖 비율과 사양별 반응을 출력한다."""
import sys
sys.path.insert(0, ".")
import numpy as np, pandas as pd, ep_detail as ed, ep_lpiv as el, ep_full_check as fc, lp_tools as lt
x = ed.er.panel()
g = x.groupby("country")
# 실제로 몇 %가 1σ·1.28σ 밖인가
dh = x.hh_rate - g.hh_rate.shift(1); tt = x[fc.TOT]
for k in (1.0, 1.28):
    for nm, v in (("가계저축률 변화", dh), ("교역조건 변화", tt)):
        z = (v - v.groupby(x.country).transform("mean")) / v.groupby(x.country).transform("std")
        print(f"{k}σ 밖 비율 {nm}: {(z.abs() > k)[v.notna()].mean():.2f}")
HHV = ["sv_hh", "nl_hh", "yd_hh", "cons_hh", "ca"]; COV = ["sv_corp", "nl_corp", "prof", "div_paid", "inv_corp", "ca"]
fc.VARS["hh"], fc.VARS["corp"] = HHV, COV
def run(name, ch):
    P = ed.paths(x, {**el.SPEC, "tot": fc.TOT, **ch}); d = fc.design(x, P)
    out = {}
    for s, vs in (("hh", HHV), ("corp", COV)):
        b = ch.get("base", "nl")
        g0 = lt.ols(d, f"{b}_{s}_h0", [f"z_{s}"], [], se=False)[0][0]
        out[s] = {v: np.round([lt.ols(d, f"{v}_h{h}", [f"z_{s}"], [], se=False)[0][0] / g0 for h in (0, 1, 2, 3, 4, 6)], 2) for v in vs}
    print(f"===== {name}: 가계 {len(P['hh'])}건, 교역조건 {len(P['corp'])}건  (h=0,1,2,3,4,6; 시작 {ch.get('base','nl')}=1)")
    for s in out:
        for v, a in out[s].items():
            print(f"  {s:4s} {v:9s} {a}")
run("상·하위 10%, 소비 주도, 저축 기준", dict(hh_def="cons", base="sv"))
run("1σ, 소비 주도, 저축 기준", dict(hh_def="cons", base="sv", sd=1.0))
run("1σ, 기존 정의, 저축 기준", dict(base="sv", sd=1.0))
run("상·하위 10%, 소비 주도, 저축 기준, 같은 방향 조건 없음", dict(hh_def="cons", base="sv", align="shock"))
run("1σ, 소비 주도, 저축 기준, 같은 방향 조건 없음", dict(hh_def="cons", base="sv", align="shock", sd=1.0))
