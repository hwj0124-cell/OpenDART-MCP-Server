"""가계 성향 vs 교역조건 에피소드: 세부 변수 반응과 선정 조건별 민감도.

기본 사양(BASE): 1년 변화(t−1 = 0), 국가별 상·하위 10%, 연도평균 미제거, 전체 국가.
선정 조건은 모두 OPTS의 키로 드러내 하나씩 바꿔 볼 수 있게 했다.
  q, by_country : 선정 변수의 상·하위 q (국가별 또는 전체 통합 분위)
  align         : "same" = 시작 시점 순자금운용이 충격과 같은 방향으로 움직인 사례만, 충격 부호로 정렬(기본)
                  "nl" = 순자금운용 변화의 부호로 정렬(반대 방향 사례도 부호를 뒤집어 포함, 이전 방식)
                  "shock" = 모든 사례를 충격 부호로 정렬(평균 충격 반응)
  min_den       : |시작 시점 순자금운용 변화| ≤ min_den(%p) 사례 제외 (이전 방식 0.3)
  gap           : 같은 나라에서 gap년 이내 재발 사례 제외
  hh_growth     : 가계 사례는 그해 실질성장률 ≥ 0 (침체 제외)   tot_growth: 교역조건 사례에도 같은 조건
  covid         : 시작 연도 제외 목록 (가계·교역조건 공통, 기본 2020–22)
  tot_prof      : 교역조건 사례는 영업잉여 비중이 같은 방향
  balanced      : t+K까지 모두 관측되는 사례만 (시점별 구성 고정)
  dir           : "both"(기본) = 증가·감소 모두 / "up" = 가계저축률 상승·교역조건 개선만 / "down" = 반대만
  tot           : 교역조건 변화율 변수. "tot_na_g" = 국민계정 수출/수입 디플레이터(상품·서비스, 1970년대~, 기본)
                  "tot_g" = WDI 상품 순교역조건 지수(2006~, 이전 기본)
값 = 사례 평균[부호·(v_{t+k} − v_{t−1})] / 사례 평균[부호·(nl_t − nl_{t−1})] (시작 = 1). 68% 구간은 사례 부트스트랩 499회.
→ output/ep_detail.csv (세부 변수, 기본 사양), output/ep_detail_sens.csv (조건별 순자금운용)
"""
import numpy as np
import pandas as pd

import ep_robust as er
import run_analysis as ra

K = 6
BASE = dict(q=0.10, by_country=True, align="same", min_den=0, gap=4, hh_growth=True, covid=(2020, 2021, 2022),
            tot_prof=True, tot_growth=False, mne=True, balanced=False, tot="tot_na_g")
# 부문별 세부 변수 (모두 GNI 대비 %)
VARS = {"hh": ["nl_hh", "sv_hh", "invtot_hh", "inv_hh", "yd_hh", "cons_hh"],
        "corp": ["nl_corp", "sv_corp", "invtot_corp", "inv_corp", "prof", "div_paid", "tax_paid", "int_paid"]}
SENS = {
    "기본": {},
    "교역조건 = WDI 상품지수 (2006~)": dict(tot="tot_g"),
    "교역조건 = 디플레이터, 2006년 이후만": dict(start=2006),
    "이전 방식 (순자금운용 부호 정렬·0.3%p 문턱·2020–21만 제외·WDI 상품지수)": dict(
        align="nl", min_den=0.3, covid=(2020, 2021), covid_tot=False, tot="tot_g"),
    "반대 방향 사례도 포함 (순자금운용 부호 정렬)": dict(align="nl"),
    "모든 사례 충격 부호 정렬": dict(align="shock"),
    "코로나 시기 포함": dict(covid=()),
    "t+6까지 관측된 사례만": dict(balanced=True),
    "중복 제외 2년": dict(gap=2),
    "교역조건: 영업잉여 조건 없음": dict(tot_prof=False),
    "교역조건: 침체 제외 추가": dict(tot_growth=True),
    "가계: 침체 포함": dict(hh_growth=False),
    "전체 통합 분위": dict(by_country=False),
    "IRL·LUX 제외": dict(mne=False),
}


def episodes(x, o):
    g = x.groupby("country")
    x = x.assign(d_hh=x.hh_rate - g.hh_rate.shift(1), d_prof=x.prof - g.prof.shift(1), tot_x=x[o["tot"]])

    def cut(col):
        if o["by_country"]:
            gg = x.groupby("country")[col]
            return gg.transform(lambda s: s.quantile(o["q"])), gg.transform(lambda s: s.quantile(1 - o["q"]))
        return x[col].quantile(o["q"]), x[col].quantile(1 - o["q"])

    up, dn = o.get("dir", "both") in ("both", "up"), o.get("dir", "both") in ("both", "down")
    lo, hi = cut("d_hh")
    Fh = ((x.d_hh >= hi) & up) | ((x.d_hh <= lo) & dn)
    if o["hh_growth"]:
        Fh &= x.gdp_g >= 0
    Fh &= ~x.year.isin(o["covid"])
    lo, hi = cut("tot_x")
    Ft = ((x.tot_x >= hi) & up) | ((x.tot_x <= lo) & dn)
    if o.get("start"):
        Ft &= x.year >= o["start"]
    if o["tot_prof"]:
        Ft &= np.sign(x.d_prof) == np.sign(x.tot_x)
    if o["tot_growth"]:
        Ft &= x.gdp_g >= 0
    if o.get("covid_tot", True):
        Ft &= ~x.year.isin(o["covid"])
    if not o["mne"]:
        Ft &= ~x.country.isin(er.MNE)
    out = {}
    for s, F, shock in (("hh", Fh, "d_hh"), ("corp", Ft, "tot_x")):
        last, eps = {}, []
        sel = x[F.fillna(False)]
        for c, y, sh in zip(sel.country, sel.year, sel[shock]):
            if y - last.get(c, -99) > o["gap"]:
                eps.append((c, y, np.sign(sh)))
                last[c] = y
        out[s] = eps
    return out


def paths(x, o):
    xp = x.set_index(["country", "year"])
    eps = episodes(x, o)
    res = {}
    for s, el in eps.items():
        nl = xp[f"nl_{s}"]
        rows = []
        for c, y, ssh in el:
            den = nl.get((c, y), np.nan) - nl.get((c, y - 1), np.nan)
            if np.isnan(den) or abs(den) <= o["min_den"]:
                continue
            if o["align"] == "same" and np.sign(den) != ssh:
                continue
            sg = np.sign(den) if o["align"] == "nl" else ssh
            if o["balanced"] and np.isnan(nl.get((c, y + K), np.nan)):
                continue
            r = dict(country=c, year=y, sign=sg, shock_sign=ssh, den=sg * den)
            for v in VARS[s]:
                ser = xp[v]
                b = ser.get((c, y - 1), np.nan)
                for k in range(-1, K + 1):
                    r[f"{v}_{k}"] = sg * (ser.get((c, y + k), np.nan) - b)
            # 원천 변수 경로(원단위): 가계저축률 %p, 교역조건 누적 %
            if s == "hh":
                hb = xp.hh_rate.get((c, y - 1), np.nan)
                for k in range(-1, K + 1):
                    r[f"src_{k}"] = sg * (xp.hh_rate.get((c, y + k), np.nan) - hb)
            else:
                lv = np.exp(np.log1p(xp[o["tot"]].loc[c] / 100).cumsum())
                for k in range(-1, K + 1):
                    r[f"src_{k}"] = sg * 100 * (lv.get(y + k, np.nan) / lv.get(y - 1, np.nan) - 1)
            rows.append(r)
        res[s] = pd.DataFrame(rows)
    return res


def summarize(E, s, v, rng, norm=True):
    out = []
    for k in range(-1, K + 1):
        p = E[[f"{v}_{k}", "den"]].dropna().to_numpy(float)
        if len(p) < 3:
            continue
        f = (lambda a: a[:, 0].mean() / a[:, 1].mean()) if norm else (lambda a: a[:, 0].mean())
        bs = [f(p[rng.integers(0, len(p), len(p))]) for _ in range(499)]
        out.append(dict(sector=s, var=v, k=k, ratio=f(p), lo68=np.percentile(bs, 16), hi68=np.percentile(bs, 84), n=len(p)))
    return out


def lpiv(x, P):
    """에피소드 경로 = 패널 LP-IV 확인.
    도구 z_it = 에피소드 해의 부호(±1), 나머지 0. 내생변수 = nl_t − nl_{t−1}, 종속변수 = nl_{t+h} − nl_{t−1}.
    (a) 상수 없음 → 사례 평균 비율과 정확히 같음  (b) 상수 포함  (c) 국가·연도 고정효과(+1단계 F)"""
    import lp_tools as lt
    out = []
    for s, E in P.items():
        z = E.set_index(["country", "year"]).sign
        d = x[["country", "year", f"nl_{s}"]].copy()
        g = d.groupby("country")[f"nl_{s}"]
        d["z"] = [z.get((c, y), 0.0) for c, y in zip(d.country, d.year)]
        d["dx"] = g.shift(0) - g.shift(1)
        for h in range(K + 1):
            d["dy"] = g.shift(-h) - g.shift(1)
            q = d.dropna(subset=["dx", "dy"])
            a = (q.z * q.dy).sum() / (q.z * q.dx).sum()
            zc = q.z - q.z.mean()
            b = (zc * q.dy).sum() / (zc * q.dx).sum()
            c, se, nobs, F = lt.iv(q, "dy", "dx", "z", [])
            out.append(dict(sector=s, h=h, no_const=a, const=b, twfe=c, twfe_se=se, F=F, nobs=nobs))
    return pd.DataFrame(out)


def main():
    x = er.panel()
    rng = np.random.default_rng(0)
    P = paths(x, BASE)
    rows = []
    for s, E in P.items():
        for v in VARS[s]:
            rows += summarize(E, s, v, rng)
        rows += summarize(E, s, "src", rng, norm=False)
        E.to_csv(ra.OUT / f"ep_detail_cases_{s}.csv", index=False)
    det = pd.DataFrame(rows)
    iv = lpiv(x, P)
    iv.to_csv(ra.OUT / "ep_lpiv.csv", index=False)
    print(iv.round(2).to_string(index=False))
    det.to_csv(ra.OUT / "ep_detail.csv", index=False)
    sens = []
    for name, ch in SENS.items():
        o = {**BASE, **ch}
        for s, E in paths(x, o).items():
            for v in (f"nl_{s}", f"sv_{s}", f"invtot_{s}"):
                sens += [dict(r, spec=name) for r in summarize(E, s, v, rng)]
    sens = pd.DataFrame(sens)
    sens.to_csv(ra.OUT / "ep_detail_sens.csv", index=False)
    print(det[det.k.isin([0, 1, 2, 3, 4, 6])].pivot_table(index=["sector", "var"], columns="k", values="ratio", sort=False).round(2).to_string())
    t = sens[sens.k.isin([0, 1, 2, 3, 4, 6]) & sens["var"].str.startswith("nl")].pivot_table(
        index=["spec", "sector"], columns="k", values="ratio", sort=False).round(2)
    t["n"] = sens[(sens.k == 0) & sens["var"].str.startswith("nl")].set_index(["spec", "sector"]).n
    t["n6"] = sens[(sens.k == 6) & sens["var"].str.startswith("nl")].set_index(["spec", "sector"]).n
    print(t.to_string())


if __name__ == "__main__":
    main()
