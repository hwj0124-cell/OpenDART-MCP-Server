"""이익 횡재 vs 저축 행태 변화: 정부 부문 점검.

정부 저축 분해: 정부 저축률(저축/처분가능소득)은 적자국에서 음수라 '소득 × 성향' 분해가 뒤집힘 → 가법 분해 사용.
  Δs^G = Δ처분가능소득(수입 원천) − Δ최종소비(지출 원천)        (모두 GNI 대비 %)
(1) 통제: 기업·가계 식에 정부 두 성분의 시차(t−1, t−2)를 추가 / 정부 S−I 수준 시차를 추가.
    같은 해 정부 성분은 넣지 않음: 같은 해에 횡재에 반응할 수 있는 변수라 나쁜 통제.
    예측력 검정: 정부 시차가 충격(x_P, x_H)과 결과(h0, h3)를 예측하는지(군집 강건 F).
(2) 이익 횡재 뒤 정부: 정부 S−I = 처분가능소득 − 소비 − 총자본형성 + 기타, 조세 수취(D5)도 별도로.
    기업 식 설계, 기업 S−I 시작 연도 반응 = 1로 정규화.
구간: 국가 군집 델타 방법(ep_wb_extra.Delta).
→ output/ep_wb_gov.csv, output/paper/fig_wb_gov.(png|pdf)
"""
import sys

import numpy as np
import pandas as pd

import build_panel as bp
import ep_iv_industry as e
import ep_wb_extra as ex
import ep_windfall_behavior as wb
import lp_tools as lt
from make_report import OUT

H, SEC = wb.H, wb.SEC
GOV = {"yd_gov": ("B6G", bp.BAL), "cons_gov": ("P3", ["D", "_Z"]), "tax_recv_gov": ("D5", ["C"])}


def gov_series():
    raw = bp.load_raw(bp.RAW / "oecd" / "table14_raw.csv")
    gni = bp.pick(raw, ["S1"], ["B5G"], bp.BAL)
    out = {v: 100 * bp.pick(raw, ["S13"], [t], ent) / gni for v, (t, ent) in GOV.items()}
    g = pd.DataFrame(out).reset_index().rename(columns={"REF_AREA": "country"})
    return g


def data():
    d, q = wb.data()
    g = gov_series()
    d = d.drop(columns=[c for c in g.columns if c in d.columns and c not in ("country", "year")]).merge(g, on=["country", "year"], how="left")
    d = d.sort_values(["country", "year"]).reset_index(drop=True)
    gr = d.groupby("country")
    d["D_yd_gov"], d["D_cons_gov"] = gr.yd_gov.diff(), gr.cons_gov.diff()
    new = {}
    for v in ("D_yd_gov", "D_cons_gov", "nl_gov"):
        for l in (1, 2):
            new[f"L{l}_{v}"] = d.groupby("country")[v].shift(l)
    for v in ("yd_gov", "cons_gov", "tax_recv_gov", "inv_gov", "nl_gov"):
        for h in range(H):
            new[f"{v}_h{h}"] = lt.add_lp_columns(d, v, h)
    d = pd.concat([d.drop(columns=[c for c in new if c in d.columns]), pd.DataFrame(new, index=d.index)], axis=1)
    keep = q[["country", "year"]]
    q = keep.merge(d, on=["country", "year"], how="left")
    return q


GL = [f"L{l}_{v}" for v in ("D_yd_gov", "D_cons_gov") for l in (1, 2)]
GN = [f"L{l}_nl_gov" for l in (1, 2)]


def run():
    q = data()
    full = len(q)
    q = q.dropna(subset=GL + GN).copy()
    q = wb.weights(q)
    print(f"sample with government lags: {len(q)} of {full}, {q.country.nunique()} economies")
    # 예측력
    chk = []
    for s, m in SEC.items():
        for dep in (m["x"], f"{m['y']}_h0", f"{m['y']}_h3"):
            C = wb.ctrl(s) + ([m["z"]] if dep == m["x"] else [m["x"], m["z"]])
            F, n, k = e.first_stage(q.assign(w_nl_corp=q[f"w_{s}"]), GL, C, dep)
            chk.append(dict(sector=s, dependent=dep, F_gov_lags=F))
    print(pd.DataFrame(chk).round(2).to_string(index=False))

    D = ex.Delta()
    for s, m in SEC.items():
        for lab, extra in (("base", []), ("govlags", GL), ("govlags_nl", GL + GN)):
            D.add(f"{s}_{lab}", *ex.design(q, [m["x"], m["z"]], wb.ctrl(s) + extra, f"w_{s}", ex.ycols(m["y"])))
    gv = sum((ex.ycols(v) for v in ("nl_corp", "nl_gov", "yd_gov", "cons_gov", "inv_gov", "tax_recv_gov")), [])
    D.add("gov", *ex.design(q, [SEC["corp"]["x"], SEC["corp"]["z"]], wb.ctrl("corp"), "w_corp", gv))
    D.finish()
    rows = []

    def emit(key, f):
        for h in range(H):
            b, se = D.fn(lambda d, h=h: f(d, h))
            rows.append(dict(key=key, h=h, b=b, se=se, lo68=b - se, hi68=b + se, lo90=b - 1.645 * se, hi90=b + 1.645 * se))
    for s, m in SEC.items():
        for lab in ("base", "govlags", "govlags_nl"):
            emit(f"{s}_{lab}", lambda d, h, t=f"{s}_{lab}", y=m["y"]: d[f"{t}|{y}_h{h}"] / d[f"{t}|{y}_h0"])
    for v, sg in (("nl_gov", 1), ("yd_gov", 1), ("cons_gov", -1), ("inv_gov", -1), ("tax_recv_gov", 1)):
        emit(f"gov_{v}", lambda d, h, v=v, sg=sg: sg * d[f"gov|{v}_h{h}"] / d["gov|nl_corp_h0"])
    r = pd.DataFrame(rows).assign(n=len(q), countries=q.country.nunique())
    r.to_csv(OUT / "ep_wb_gov.csv", index=False)
    return r


def figure(r):
    import fig_paper as fp
    plt = fp.plt
    g = lambda k: r[r.key == k].sort_values("h")
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4))
    ax = axes[0]
    for s, cl in (("corp", fp.CO), ("hh", fp.HH)):
        b = g(f"{s}_base")
        ax.fill_between(b.h, b.lo68, b.hi68, color=cl, alpha=0.15, lw=0)
        ax.plot(b.h, b.b, color=cl, lw=1.9, marker="o", ms=3.2, label=("Profit windfall" if s == "corp" else "Saving-rate shift") + ", baseline")
        for lab, ls in (("govlags", "--"), ("govlags_nl", ":")):
            c = g(f"{s}_{lab}")
            ax.plot(c.h, c.b, color=cl, lw=1.2, ls=ls,
                    label=("+ lags of Δgov. income, Δgov. consumption" if lab == "govlags" else "+ also lags of gov. balance") if s == "hh" else None)
    fp.finish(ax, "Own-sector S–I (impact = 1)", h0=0)
    ax.set_xlabel("Years after the saving change ($h$)")
    ax.set_title("(a) Controlling for past fiscal developments", loc="left", fontsize=9.3)
    ax.set_ylim(-0.6, 1.1)
    ax.legend(loc="upper right", fontsize=6.8, frameon=False)
    ax = axes[1]
    pal = {"nl_gov": ("0.1", "Government balance"), "yd_gov": ("#c98a1b", "Disposable income"), "tax_recv_gov": ("#bc5090", "  of which income taxes received"),
           "cons_gov": ("#7a5195", "Consumption (−)"), "inv_gov": ("#2b6c8f", "Investment (−)")}
    for v, (cl, lab) in pal.items():
        q = g(f"gov_{v}")
        if v in ("nl_gov", "yd_gov"):
            ax.fill_between(q.h, q.lo68, q.hi68, color=cl, alpha=0.13, lw=0)
        ax.plot(q.h, q.b, color=cl, lw=2.0 if v == "nl_gov" else 1.3, ls=":" if v == "tax_recv_gov" else "-", marker="o" if v == "nl_gov" else None, ms=3.2, label=lab)
    fp.finish(ax, "Cumulative change\n(impact corporate S–I = 1)", h0=0)
    ax.set_xlabel("Years after the saving change ($h$)")
    ax.set_title("(b) Government after a profit windfall", loc="left", fontsize=9.3)
    ax.legend(loc="upper left", fontsize=6.8, frameon=False)
    fig.suptitle("The government sector", x=0.0, ha="left", fontweight="bold", y=1.03)
    fig.tight_layout()
    n, k = int(r.n.iloc[0]), int(r.countries.iloc[0])
    fig.text(0.0, -0.06, f"Notes: N = {n}, {k} economies with government accounts. (a) Baseline local projections, adding two lags of the changes in "
             "government disposable income and final consumption (dashed) and two lags of the government balance (dotted); government items at t are "
             "excluded because they can respond to the windfall within the year. (b) Cumulative change in government items after a profit windfall, from the "
             "corporate equation; outlays with a negative sign. Shaded: 68 percent bands, country-cluster delta method.", fontsize=7, wrap=True, va="top")
    fp.save(fig, "fig_wb_gov", sfx="")


if __name__ == "__main__":
    r = pd.read_csv(OUT / "ep_wb_gov.csv") if "--render" in sys.argv else run()
    figure(r)
    print(r.pivot_table(index="key", columns="h", values="b", sort=False).round(2).to_string())
    a = r.assign(per=np.where(r.h.between(1, 2), "1-2", np.where(r.h >= 3, "3-6", "0")))
    print(a[a.per != "0"].pivot_table(index="key", columns="per", values="b", sort=False).round(2).to_string())
