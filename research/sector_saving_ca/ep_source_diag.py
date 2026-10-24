"""원천 분해 성분(x_P, x_R, x_H, x_Y)이 얼마나 '충격'다운지 점검. 제안 사양과 같은 표본·통제·가중치.

(1) 같은 해 거시변수와의 관계: 성분을 거시변수 하나씩에 회귀(LP 통제·국가·연도 FE 포함), 1 표준편차당 %p와 군집 t.
(2) 예측 가능성: LP 통제 밖의 전기 변수(수준·교역조건·실질금리)가 성분을 예측하는지(군집 강건 F),
    LP 통제가 FE 외에 설명하는 몫(부분 R²).
(3) 유보율 성분의 회계적 구성: x_R을 순배당·법인세·이자 지급 변화에 회귀.
(4) 사전 백색화 비교: 자기 시차 AR 잔차만 쓰는 LP vs 제안 사양 vs 같은 해 교역조건 통제.
→ output/ep_source_diag.csv (1·2), 화면 출력 (3·4)
"""
import numpy as np
import pandas as pd

import ep_iv_industry as e
import ep_source_check as c
import ep_source_fig as sf
import lp_tools as lt
from make_report import OUT

SEC = {"x_P": "nl_corp", "x_R": "nl_corp", "x_H": "nl_hh", "x_Y": "nl_hh"}
CONTEMP = {"gdp_g": "real GDP growth", "tot_goods_g": "goods ToT change", "d_rr": "real rate change",
           "d_old": "old-age dependency change", "recession": "recession year"}
PREDET = {"levels": ["L1_sv_corp", "L1_sv_hh", "L1_ca", "L1_prof", "L1_hh_rate"],
          "ToT": ["L1_tot_goods_g", "L2_tot_goods_g"], "real rate": ["L1_d_rr", "L1_real_interest_rate"]}


def data():
    q, ctrl = sf.sample()
    p = lt.balance(pd.read_csv(OUT.parent / "data" / "panel.csv"))
    g = p.groupby("country")
    p["d_rr"], p["d_old"] = g.real_interest_rate.diff(), g.old_age_dependency.diff()
    for v in ("div_paid", "tax_paid", "int_paid"):
        p[f"d_{v}"] = g[v].diff()
    p["L1_d_rr"] = p.groupby("country").d_rr.shift(1)
    for l in (1, 2):
        p[f"L{l}_tot_goods_g"] = g.tot_goods_g.shift(l)
    p["L1_real_interest_rate"] = g.real_interest_rate.shift(1)
    new = [x for x in p.columns if x not in q.columns or x in ("country", "year")]
    q = q.merge(p[new], on=["country", "year"], how="left")
    q["recession"] = (q.gdp_g < 0).astype(float)
    return q, ctrl


def partial(q, y, Z, C, w):
    r = q.dropna(subset=[y, w] + Z + C)
    sw = np.sqrt(r[w].to_numpy(float))[:, None]
    M = np.hstack([r[C].to_numpy(float), lt._fe_matrix(r, ("country", "year"))]) if C else lt._fe_matrix(r, ("country", "year"))
    res = lambda v: v - M @ np.linalg.lstsq(M * sw, v * sw, rcond=None)[0]
    return r, res(r[Z].to_numpy(float)), res(r[[y]].to_numpy(float))[:, 0]


if __name__ == "__main__":
    q, ctrl = data()
    rows = []
    for x, sec in SEC.items():
        C, w = c.SRC.copy(), f"w_{sec}"
        C = [v for v in c.SRC if v != x] + ctrl[sec]
        sd = q[x].std()
        for z, lab in CONTEMP.items():
            b, V, n = lt.ols(q.rename(columns={w: "w"}), x, [z], C, w="w")
            rows.append(dict(test="contemporaneous", source=x, var=lab, b_per_sd=b[0] * q[z].std(), t=b[0] / np.sqrt(V[0, 0]),
                             share_of_sd=b[0] * q[z].std() / sd, n=n))
        for lab, Z in PREDET.items():
            F, n, k = e.first_stage(q.assign(w_nl_corp=q[w]), Z, C, x)
            rows.append(dict(test="predictability", source=x, var=lab, F=F, n=n, countries=k))
        r, _, fe_only = partial(q, x, [x], [], w)
        _, _, with_c = partial(q, x, [x], C, w)
        ww = r[w].to_numpy(float)
        rows.append(dict(test="partial R2 of LP controls", source=x, var="controls beyond FE",
                         F=1 - (ww * with_c ** 2).sum() / (ww * fe_only ** 2).sum(), n=len(r)))
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "ep_source_diag.csv", index=False)
    print(out[out.test == "contemporaneous"].pivot_table(index="var", columns="source", values="t", sort=False).round(1).to_string())
    print(out[out.test == "contemporaneous"].pivot_table(index="var", columns="source", values="share_of_sd", sort=False).round(2).to_string())
    print(out[out.test != "contemporaneous"].pivot_table(index="var", columns="source", values="F", sort=False).round(2).to_string())

    # 잔차 상관(같은 해 네 성분)
    R = {}
    for x, sec in SEC.items():
        _, _, R[x] = partial(q, x, [x], ctrl["nl_corp"], "w_nl_corp")
    print(pd.DataFrame(R).corr().round(2).to_string())

    # (3) 유보율 성분의 회계적 구성
    for v in ("d_div_paid", "d_tax_paid", "d_int_paid", "gdp_g"):
        b, V, n = lt.ols(q, "x_R", [v], [x for x in c.SRC if x != "x_R"] + ctrl["nl_corp"], w="w_nl_corp")
        print(f"x_R on {v:12s}: b = {b[0]:6.2f}  t = {b[0] / np.sqrt(V[0, 0]):5.1f}  n = {n}")
    b, V, n = lt.ols(q, "x_R", ["d_div_paid", "d_tax_paid", "d_int_paid"], [x for x in c.SRC if x != "x_R"] + ctrl["nl_corp"], w="w_nl_corp")
    print("joint:", np.round(b, 2), np.round(b / np.sqrt(np.diag(V)), 1))

    # (4) 사전 백색화 비교: 지속성(1–2년, 3–6년 평균, 시작 연도 = 1)
    def persist(qq, X, C, sec, k):
        bs = np.array([lt.ols(qq, f"{sec}_h{h}", X, C, se=False, w=f"w_{sec}")[0][k] for h in range(c.H)])
        return bs[1:3].mean() / bs[0], bs[3:].mean() / bs[0], bs[0]
    pw = q.copy()
    for x in c.SRC:
        _, _, res = partial(pw, x, [x], [f"L1_{x}", f"L2_{x}"], "w_nl_corp")
        pw.loc[pw.dropna(subset=[x, f"L1_{x}", f"L2_{x}"]).index, f"u_{x}"] = res
    U = [f"u_{x}" for x in c.SRC]
    tq = q.dropna(subset=["tot_goods_g"])
    for x, sec in SEC.items():
        k = c.SRC.index(x)
        a = persist(q, c.SRC, ctrl[sec], sec, k)
        b = persist(pw, U, [], sec, k)
        t0 = persist(tq, c.SRC, ctrl[sec], sec, k)
        t1 = persist(tq, c.SRC, ctrl[sec] + ["tot_goods_g"], sec, k)
        print(f"{x}: main {a[0]:.2f}/{a[1]:.2f} | AR(2) residuals, no controls {b[0]:.2f}/{b[1]:.2f} | "
              f"ToT sample {t0[0]:.2f}/{t0[1]:.2f} (n={len(tq)}) + current ToT {t1[0]:.2f}/{t1[1]:.2f}")
