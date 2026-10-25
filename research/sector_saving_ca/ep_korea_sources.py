"""한국 국민저축 변화의 원천: 소득형(이익·소득·정부 수입) vs 행태형(유보율·저축률·정부 지출), 순대출이 아닌 저축 기준.

기업 Δs^C = ρ_{t−1}Δπ (이익) + π_tΔρ (유보율)       가계 Δs^H = y_{t−1}Δσ (저축률) + σ_tΔy (소득)
정부 Δs^G = Δ처분가능소득 (수입) − Δ최종소비 (지출)   기타 = 국민저축 변화 − 세 부문 합(금융법인 등). 모두 GNI 대비 %p.
구간
  2012→2015, 2023→2025: OECD 연간 부문계정(패널, 연도별 성분의 누적합).
  2025 H1 → 2026 H1: 한국은행 분기 국민계정(KOSIS). 2026년은 일부 추정:
    가계 처분가능소득·저축: 「주요지표(분기)」 실측. GNI: 2025년 분기 수준 × (1 + 전년동기비).
    국민저축: 2025년 분기 저축/GNDI(원계열) + 계절조정 총저축률의 전년동기 변화.
    비금융법인 영업잉여: 총영업잉여 증가분에서 비기업 부문을 2025년 증가율로 늘리고 나머지를 배분.
    정부·금융법인 저축: GNI 대비 비중을 2025년 같은 분기로 고정(기본) / 정부 +1.9%p(자금순환 정부 순대출 변화, 민감도).
    비금융법인 저축 = 국민저축 − 가계 − 정부 − 금융법인.
→ output/ep_korea_sources.csv, output/paper/fig_korea_sources.(png|pdf)
"""
import numpy as np
import pandas as pd

import ep_wb_gov as eg
import source_shocks as ss
from make_report import OUT

RAW = OUT.parent / "data" / "raw" / "kosis"
COMP = ["Corporate profits", "Household income", "Government revenue", "Corporate retention", "Household saving rate", "Government spending", "Other"]
INCOME, BEHAV = COMP[:3], COMP[3:6]


def annual(a, b):
    d = ss.prepare(pd.read_csv(OUT.parent / "data" / "panel.csv"))
    k = d[d.country == "KOR"].merge(eg.gov_series(), on=["country", "year"], how="left").set_index("year").sort_index()
    yrs = range(a + 1, b + 1)
    c = {"Corporate profits": k.loc[yrs, "x_P"].sum(), "Corporate retention": k.loc[yrs, "x_R"].sum(),
         "Household saving rate": k.loc[yrs, "x_H"].sum(), "Household income": k.loc[yrs, "x_Y"].sum(),
         "Government revenue": k.loc[b, "yd_gov"] - k.loc[a, "yd_gov"], "Government spending": -(k.loc[b, "cons_gov"] - k.loc[a, "cons_gov"])}
    dnat = k.loc[b, "sv_nat"] - k.loc[a, "sv_nat"]
    c["Other"] = dnat - sum(c.values())
    return pd.Series(c)[COMP], dnat


def quarterly(gov_shift=0.0):
    s = pd.read_csv(RAW / "bok_sector_income_q.csv", comment="#").set_index("quarter")
    m = pd.read_csv(RAW / "bok_main_q.csv", comment="#").set_index("quarter")
    rows = {}
    for q in ("Q1", "Q2"):
        b, p = s.loc[f"2025{q}"], s.loc[f"2024{q}"]
        mm, mb = m.loc[f"2026{q}"], m.loc[f"2025{q}"]
        gni = b.gni * (1 + mm.gni_yoy / 100)
        gndi = gni * b.gndi / b.gni
        nat = gndi * (b.nat_saving / b.gndi + (mm.nat_saving_rate_sa - mb.nat_saving_rate_sa) / 100)
        non_nfc_g = (b.gos_total - b.nfc_gos) / (p.gos_total - p.nfc_gos) - 1
        gos_tot = b.gos_total * (1 + mm.gos_yoy / 100)
        nfc_gos = gos_tot - (b.gos_total - b.nfc_gos) * (1 + non_nfc_g)
        gov = gni * (b.gov_saving / b.gni + gov_shift / 100)
        fc = gni * b.fc_saving / b.gni
        rows[q] = dict(gni0=b.gni, gni1=gni, nfc_gos0=b.nfc_gos, nfc_gos1=nfc_gos, nfc_s0=b.nfc_saving,
                       nfc_s1=nat - mm.hh_saving - gov - fc, hh_yd0=b.hh_gndi, hh_yd1=mm.hh_gndi, hh_s0=b.hh_saving, hh_s1=mm.hh_saving,
                       gov_s0=b.gov_saving, gov_s1=gov, fc_s0=b.fc_saving, fc_s1=fc, nat0=b.nat_saving, nat1=nat)
    h = pd.DataFrame(rows).sum(axis=1)
    sh = lambda v, t: 100 * h[f"{v}{t}"] / h[f"gni{t}"]
    ret0 = h.nfc_s0 / h.nfc_gos0
    sig0, sig1 = h.hh_s0 / h.hh_yd0, h.hh_s1 / h.hh_yd1
    c = {"Corporate profits": ret0 * (sh("nfc_gos", 1) - sh("nfc_gos", 0)),
         "Household saving rate": sh("hh_yd", 0) * (sig1 - sig0), "Household income": sig1 * (sh("hh_yd", 1) - sh("hh_yd", 0))}
    c["Corporate retention"] = (sh("nfc_s", 1) - sh("nfc_s", 0)) - c["Corporate profits"]
    c["Government revenue"], c["Government spending"] = np.nan, np.nan
    c["Government (total)"] = sh("gov_s", 1) - sh("gov_s", 0)
    dnat = sh("nat", 1) - sh("nat", 0)
    c["Other"] = dnat - (c["Corporate profits"] + c["Corporate retention"] + c["Household saving rate"] + c["Household income"] + c["Government (total)"])
    info = dict(gos_share_nfc=h.nfc_gos1 / (h.gni1) * 100, nfc_gos_growth=100 * (h.nfc_gos1 / h.nfc_gos0 - 1), sig0=100 * sig0, sig1=100 * sig1,
                ret0=ret0, ret1=h.nfc_s1 / h.nfc_gos1, nat_share0=sh("nat", 0), nat_share1=sh("nat", 1))
    return pd.Series(c), dnat, info


def figure(res):
    import fig_paper as fp
    plt = fp.plt
    pal = {"Corporate profits": fp.CO, "Household income": "#e0a37a", "Government revenue": "#f2cfb5", "Government (total)": "#6a8f3c",
           "Corporate retention": "#7a5195", "Household saving rate": fp.HH, "Government spending": "#9db6d9", "Other": "0.78"}
    lab = {"Corporate profits": "Corporate profits (income)", "Household income": "Household income (income)",
           "Government revenue": "Government revenue (income)", "Corporate retention": "Corporate retention (behavior)",
           "Household saving rate": "Household saving rate (behavior)", "Government spending": "Government spending (behavior)",
           "Government (total)": "Government, assumed share of GNI (2026)", "Other": "Other (financial corporations, discrepancies)"}
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    seen = set()
    for x, (name, c, dnat) in enumerate(res):
        pos = neg = 0.0
        for k_, v in c.items():
            if not np.isfinite(v) or abs(v) < 1e-9:
                continue
            ax.bar(x, v, bottom=pos if v >= 0 else neg, color=pal[k_], width=0.55, edgecolor="white", lw=0.3,
                   label=lab[k_] if k_ not in seen else None, hatch="//" if k_ == "Government (total)" else None)
            seen.add(k_)
            pos, neg = (pos + v, neg) if v >= 0 else (pos, neg + v)
        ax.scatter(x, dnat, color="k", s=24, zorder=5, label="Change in national saving" if x == 0 else None)
    ax.axhline(0, color=".3", lw=.6)
    ax.axvline(1.5, color=".5", lw=.7, ls=":")
    ax.set_xticks(range(len(res)))
    ax.set_xticklabels([r[0] for r in res], fontsize=8.5)
    ax.set_xlim(-0.6, len(res) + 1.6)
    ax.set_ylabel("Percentage points of GNI")
    ax.legend(loc="center right", fontsize=6.9, frameon=False)
    ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1] + 1.6)
    ax.text(0.5, ax.get_ylim()[1] * 0.97, "OECD annual accounts", ha="center", va="top", fontsize=7, color="0.35")
    ax.text(2.0, ax.get_ylim()[1] * 0.97, "BOK quarterly, partly estimated", ha="center", va="top", fontsize=7, color="0.35")
    ax.set_title("Sources of the change in Korea's gross saving", loc="left", fontsize=10, fontweight="bold")
    fig.tight_layout()
    fig.text(0.0, -0.02, "Notes: Gross saving by sector, percent of GNI. Corporate saving splits into a profit component (the change in operating surplus at the "
             "earlier retention ratio) and a retention component; household saving into a saving-rate component (the change in the saving rate at earlier "
             "income) and an income component; government saving into the change in disposable income and minus the change in final consumption. "
             "2012→2015 and 2023→2025: annual OECD sector accounts, cumulated over the years. H1 2026: Bank of Korea quarterly national accounts; household "
             "income and saving are published, while GNI, national saving and corporate operating surplus for 2026 are extrapolated from published growth "
             "rates and saving rates, and government and financial-corporation saving are held at their 2025 shares of GNI, so corporate saving is a residual. "
             "The two sources differ in coverage and vintage.", fontsize=6.8, wrap=True, va="top")
    fp.save(fig, "fig_korea_sources", sfx="")


if __name__ == "__main__":
    a1, n1 = annual(2012, 2015)
    a2, n2 = annual(2023, 2025)
    q, nq, info = quarterly()
    qb, nqb, _ = quarterly(gov_shift=1.9)
    out = pd.DataFrame({"2012→2015": a1, "2023→2025": a2}).reindex(COMP + ["Government (total)"])
    out["H1 2025→H1 2026 (est.)"] = q.reindex(out.index)
    out["… gov +1.9pp"] = qb.reindex(out.index)
    out.loc["Change in national saving"] = [n1, n2, nq, nqb]
    out.to_csv(OUT / "ep_korea_sources.csv")
    pd.set_option("display.width", 200)
    print(out.round(2).to_string())
    print({k: round(v, 2) for k, v in info.items()})
    figure([("2012→2015", a1, n1), ("2023→2025", a2, n2), ("H1 2025→H1 2026", q, nq)])
