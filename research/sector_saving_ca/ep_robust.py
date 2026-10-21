"""가계 성향 vs 교역조건 에피소드: 순자금운용·저축·투자 경로와 사양 민감도.

에피소드(상·하위 Q, 증가·감소 통합, 국가별 4년 내 중복 제외):
  w년 변화 기준(기본 w=1: t−1→t, 기준시점 t−1 = 0)
  가계 성향: 가계저축률 w년 변화 상·하위 Q, 그 기간 성장률 ≥ 0, 시작 2020–21 제외 → 가계 부문
  교역조건 : 교역조건 w년 누적 변화 상·하위 Q, 영업잉여 비중 같은 방향 → 비금융법인
경로: 사례 평균 [sign·(v_{t+k} − v_{t−w})] / 사례 평균 |nl_t − nl_{t−w}|  (평균의 비율; 작은 분모에 덜 민감)
      68% 구간은 사례 단위 부트스트랩 499회. 개별 비율의 평균·중위수도 함께 저장
      v =  v = 순자금운용, 저축, 총투자(= 저축 − 순자금운용; 재고투자·자본이전 포함), 고정투자
사양: 기본(1년, 연도평균 제거, IRL·LUX 제외) / 2년 정의 / IRL·LUX 포함 / 연도평균 미제거 / 코로나 제외(시작 2019–21) / 상·하위 20%
→ output/ep_robust.csv
"""
import numpy as np
import pandas as pd

import ca_episodes as ce
import run_analysis as ra

K = 6
MNE = {"IRL", "LUX"}
SECT = {"가계 성향": ("nl_hh", "sv_hh", "inv_hh"), "교역조건": ("nl_corp", "sv_corp", "inv_corp")}
SPECS = {
    "기본 (1년·평균제거·IRL/LUX 제외·10%)": dict(w=1, demean=True, mne=False, covid=False, q=0.10),
    "2년 정의": dict(w=2, demean=True, mne=False, covid=False, q=0.10),
    "IRL·LUX 포함": dict(w=1, demean=True, mne=True, covid=False, q=0.10),
    "연도평균 미제거": dict(w=1, demean=False, mne=False, covid=False, q=0.10),
    "코로나 제외": dict(w=1, demean=True, mne=False, covid=True, q=0.10),
    "상·하위 20%": dict(w=1, demean=True, mne=False, covid=False, q=0.20),
}


def panel():
    d = ce.prepare()
    for s in ("hh", "corp"):
        d[f"invtot_{s}"] = d[f"sv_{s}"] - d[f"nl_{s}"]  # 총투자(재고·자본이전 등 포함)
    return d


def run(d, w, demean, mne, covid, q):
    g = d.groupby("country")
    x = d.copy()
    x["d_hh"] = x.hh_rate - g.hh_rate.shift(w)
    x["d_prof"] = x.prof - g.prof.shift(w)
    x["tot_w"] = g.tot_g.transform(lambda s: s.rolling(w).sum())
    x["gmin"] = g.gdp_g.transform(lambda s: s.rolling(w).min())
    vars_ = ["nl_hh", "sv_hh", "inv_hh", "invtot_hh", "nl_corp", "sv_corp", "inv_corp", "invtot_corp"]
    for v in vars_:
        x[v + "_a"] = x[v] - x.groupby("year")[v].transform("mean") if demean else x[v]
    lo, hi = x.d_hh.quantile(q), x.d_hh.quantile(1 - q)
    Fh = ((x.d_hh >= hi) | (x.d_hh <= lo)) & (x.gmin >= 0) & ~x.year.isin([2020, 2021])
    lo, hi = x.tot_w.quantile(q), x.tot_w.quantile(1 - q)
    Ft = ((x.tot_w >= hi) & (x.d_prof > 0)) | ((x.tot_w <= lo) & (x.d_prof < 0))
    if not mne:
        Ft &= ~x.country.isin(MNE)
    if covid:
        Fh &= ~x.year.isin([2019, 2020, 2021])
        Ft &= ~x.year.isin([2019, 2020, 2021])
    xp = x.set_index(["country", "year"])
    out = []
    for name, F, s in (("가계 성향", Fh.fillna(False), "hh"), ("교역조건", Ft.fillna(False), "corp")):
        last, eps = {}, []
        for c, y in zip(x.country[F], x.year[F]):
            if y - last.get(c, -99) > 4:
                eps.append((c, y))
                last[c] = y
        rows = {k: [] for k in ("순자금운용", "저축", "총투자", "고정투자")}
        n_used = 0
        for c, y in eps:
            nl = xp[f"nl_{s}_a"]
            den = nl.get((c, y), np.nan) - nl.get((c, y - w), np.nan)
            if not abs(den) > 0.3:
                continue
            n_used += 1
            sg, ad = np.sign(den), abs(den)
            for kind, v in (("순자금운용", f"nl_{s}"), ("저축", f"sv_{s}"), ("총투자", f"invtot_{s}"), ("고정투자", f"inv_{s}")):
                ser = xp[v + "_a"]
                b = ser.get((c, y - w), np.nan)
                # 부호를 맞춘 변화(분자)와 시작 시점 |Δ순자금운용|(분모)를 따로 저장 → 평균의 비율
                rows[kind].append([(sg * (ser.get((c, y + k), np.nan) - b), ad) for k in range(-w, K + 1)])
        rng = np.random.default_rng(0)
        for kind, M in rows.items():
            for j, k in enumerate(range(-w, K + 1)):
                pairs = np.array([m[j] for m in M], float)
                pairs = pairs[~np.isnan(pairs[:, 0])]
                if len(pairs) < 3:
                    continue
                est = pairs[:, 0].mean() / pairs[:, 1].mean()
                bs = [pairs[i, 0].mean() / pairs[i, 1].mean()
                      for i in (rng.integers(0, len(pairs), len(pairs)) for _ in range(499))]
                ratio = pairs[:, 0] / pairs[:, 1]
                out.append(dict(type=name, kind=kind, k=k, n=len(pairs), n_ep=n_used, ratio=est,
                                lo68=np.percentile(bs, 16), hi68=np.percentile(bs, 84),
                                mean=ratio.mean(), median=np.median(ratio)))
    return pd.DataFrame(out)


def main():
    d = panel()
    res = pd.concat([run(d, **cfg).assign(spec=name) for name, cfg in SPECS.items()], ignore_index=True)
    res.to_csv(ra.OUT / "ep_robust.csv", index=False)
    t = res[res.k.isin([1, 2, 3, 4, 6]) & res.kind.isin(["순자금운용", "저축", "총투자"])].pivot_table(
        index=["spec", "type", "kind"], columns="k", values="ratio", sort=False).round(2)
    t["n"] = res[res.k == 0].set_index(["spec", "type", "kind"]).n_ep
    print(t.to_string())


if __name__ == "__main__":
    main()
