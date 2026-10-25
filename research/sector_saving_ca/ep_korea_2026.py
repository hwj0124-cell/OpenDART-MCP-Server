"""2026년 상반기 한국 국민저축 증가의 원천: 관측치와 추정치를 분리한 정합 추정 + 동료 방식(자본계정 경로) 교차 점검.

항등식(부문 j, 기간 합계, 십억원)
  S_j = I_j + NL_j^FoF + D_j        D_j: 자본이전·비생산비금융자산·실물–금융 통계 불일치(자금순환 순자금운용과 자본계정 순대출의 차이)
  S   = Σ_j S_j,  Σ_j NL_j^FoF = −NL_ROW^FoF,   D ≡ S − I − Σ_j NL_j^FoF  (총량은 관측)
2026 상반기에 관측되는 것: S, I(총자본형성), GNDI(DT_200Y156) / GNI 증가율, 총영업잉여 증가율, 가계 처분가능소득·저축(DT_200Y102)
  / 정부 최종소비·총고정자본형성(DT_200Y109) / 부문별 순자금운용(DT_281Y001).
관측되지 않는 것: 비금융법인·금융법인·정부 저축, 부문별 영업잉여, 민간 투자의 부문 구성.
가정(전년 같은 반기 기준)
  A1 정부 D_G의 GNI 대비 비중 고정 → S_G = NL_G^FoF + I_G + D_G,   정부 처분가능소득 = S_G + C_G
  A2 금융법인 저축의 GNI 대비 비중 고정
  A3 비금융법인 영업잉여 = 총영업잉여 − 비기업 영업잉여(전년 증가율로 연장)
  A4 (경로 II만) (총자본형성 − 정부 총고정자본형성) 중 비금융법인·금융법인·가계 비중을 전년 연간값으로 고정, D_NFC의 GNI 대비 비중 고정
경로 I (기준, 소득 측 잔차):   S_NFC = S − S_HH − S_G − S_FC
경로 II (동료 방식, 자본계정): S_NFC = NL_NFC^FoF + I_NFC + D_NFC
두 경로의 차이 = 비금융법인에 배분되지 않은 통계 불일치 변화 + 가정 오차. 관측된 총 불일치 변화 ΔD를 함께 보고.
검증: 부문 저축이 공표된 2024→2025(상반기·하반기)에 같은 절차를 적용해 실제값과 비교, 연간 2010–2025 자료로 가정별 오차 크기 산출.
→ output/ep_korea_2026.csv, output/ep_korea_2026_check.csv, output/paper/fig_korea_2026.(png|pdf)
"""
import numpy as np
import pandas as pd

from make_report import OUT

RAW = OUT.parent / "data" / "raw" / "kosis"
SEC = ["nfc", "fc", "gov", "hh"]


def load():
    rd = lambda f, k: pd.read_csv(RAW / f, comment="#").set_index(k)
    q = rd("bok_sector_income_q.csv", "quarter")
    q = q.join(rd("bok_saving_invest_q.csv", "quarter").drop(columns=["nat_saving", "gndi"]), how="outer")
    t = rd("bok_saving_invest_q.csv", "quarter")
    q["nat_saving"], q["gndi"] = t.nat_saving, t.gndi
    q = q.join(rd("bok_gov_exp_q.csv", "quarter").rename(columns={"gov_cons": "gov_cons_x"}), how="outer")
    q = q.join(rd("bok_fof_nl_q.csv", "quarter").add_prefix("nl_"), how="outer")
    m = rd("bok_main_q.csv", "quarter")
    for k in ("2026Q1", "2026Q2"):
        p = f"2025{k[-2:]}"
        q.loc[k, "gni"] = q.loc[p, "gni"] * (1 + m.loc[k, "gni_yoy"] / 100)
        q.loc[k, "gos_total"] = q.loc[p, "gos_total"] * (1 + m.loc[k, "gos_yoy"] / 100)
        q.loc[k, ["hh_gndi", "hh_saving"]] = m.loc[k, ["hh_gndi", "hh_saving"]].to_numpy()
    a = rd("bok_saving_invest_a.csv", "year")
    fa = rd("bok_fof_nl_a.csv", "year")
    return q, a, fa


def estimate(q, a, base, new, prev):
    """base/new/prev: 분기 리스트(전년 같은 반기, 대상 반기, 전전년 같은 반기). new 기간의 부문 저축은 쓰지 않음."""
    b, n, p = q.loc[base].sum(), q.loc[new].sum(), q.loc[prev].sum()
    yb = int(base[0][:4])
    sh = lambda x, g: x / g * n.gni
    e = {}
    # 정부(A1): D_G = S_G − NL_G − I_G
    d_gov = b.gov_saving - b.nl_gov - b.gov_gfcf
    e["gov"] = n.nl_gov + n.gov_gfcf + sh(d_gov, b.gni)
    e["gov_yd"], e["gov_cons"] = e["gov"] + n.gov_cons_x, n.gov_cons_x
    e["gov_old"] = sh(b.gov_saving, b.gni)                     # 이전 버전: 비중 고정
    # 금융법인(A2)
    e["fc"] = sh(b.fc_saving, b.gni)
    e["fc_fof"] = n.nl_fc + sh(b.fc_saving - b.nl_fc, b.gni)   # 대안: 순자금운용 + 나머지 비중 고정
    e["hh"], e["hh_yd"] = n.hh_saving, n.hh_gndi
    # 경로 I
    e["nfc_I"] = n.nat_saving - e["hh"] - e["gov"] - e["fc"]
    # 경로 II (A4)
    yq = q[q.index.str.startswith(str(yb))]
    w = a.loc[yb, ["i_nfc", "i_fc", "i_hh"]] / (yq.gcf.sum() - yq.gov_gfcf.sum())   # 분기 기준과 같은 정의(총투자 − 정부 GFCF)의 연간 비율
    i_priv_b, i_priv_n = b.gcf - b.gov_gfcf, n.gcf - n.gov_gfcf
    d_nfc = b.nfc_saving - b.nl_nfc - w.i_nfc * i_priv_b
    e["nfc_II"] = n.nl_nfc + w.i_nfc * i_priv_n + sh(d_nfc, b.gni)
    e["i_nfc"] = w.i_nfc * i_priv_n
    # A3: 비금융법인 영업잉여
    g_non = (b.gos_total - b.nfc_gos) / (p.gos_total - p.nfc_gos) - 1
    e["nfc_gos"] = n.gos_total - (b.gos_total - b.nfc_gos) * (1 + g_non)
    # 총 불일치 D
    nl_dom = lambda x: x.nl_nfc + x.nl_fc + x.nl_gov + x.nl_hh
    e["D"], e["D_base"] = n.nat_saving - n.gcf - nl_dom(n), b.nat_saving - b.gcf - nl_dom(b)
    e["nat"], e["gni"], e["gcf"], e["nl_dom"] = n.nat_saving, n.gni, n.gcf, nl_dom(n)
    return b, n, e


def shares(b, n, e, route="nfc_I", gov="gov"):
    """GNI 대비 %p 기여. 기업 이익/유보율, 가계 저축률/소득, 정부 수입/지출, 금융법인."""
    P = lambda x, g: 100 * x / g
    ret0 = b.nfc_saving / b.nfc_gos
    sig0, sig1 = b.hh_saving / b.hh_gndi, e["hh"] / e["hh_yd"]
    c = {"Corporate profits": ret0 * (P(e["nfc_gos"], n.gni) - P(b.nfc_gos, b.gni))}
    c["Corporate retention"] = P(e[route], n.gni) - P(b.nfc_saving, b.gni) - c["Corporate profits"]
    c["Household saving rate"] = P(b.hh_gndi, b.gni) * (sig1 - sig0)
    c["Household income"] = sig1 * (P(e["hh_yd"], n.gni) - P(b.hh_gndi, b.gni))
    if gov == "gov":
        c["Government revenue"] = P(e["gov_yd"], n.gni) - P(b.gov_gndi, b.gni)
        c["Government spending"] = -(P(e["gov_cons"], n.gni) - P(b.gov_cons, b.gni))
    else:
        c["Government revenue"], c["Government spending"] = P(e[gov], n.gni) - P(b.gov_saving, b.gni), 0.0
    c["Financial corporations"] = P(e["fc"], n.gni) - P(b.fc_saving, b.gni)
    dnat = P(e["nat"], n.gni) - P(b.nat_saving, b.gni)
    c["Unallocated"] = dnat - sum(c.values())
    info = dict(ret0=ret0, ret1=e[route] / e["nfc_gos"], sig0=100 * sig0, sig1=100 * sig1,
                nfc_gos_growth=100 * (e["nfc_gos"] / b.nfc_gos - 1))
    return pd.Series(c), dnat, info


def sector_changes(b, n, e):
    """부문 저축의 GNI 대비 변화(%p): 경로별·관측/추정 표시."""
    P = lambda x, g: 100 * x / g
    r = {"National saving (observed)": P(e["nat"], n.gni) - P(b.nat_saving, b.gni),
         "Households (observed)": P(e["hh"], n.gni) - P(b.hh_saving, b.gni),
         "Government: FoF route (A1)": P(e["gov"], n.gni) - P(b.gov_saving, b.gni),
         "Government: share held (old)": 0.0,
         "Financial corporations: share held (A2)": 0.0,
         "Financial corporations: FoF route": P(e["fc_fof"], n.gni) - P(b.fc_saving, b.gni),
         "NFC: route I (income residual)": P(e["nfc_I"], n.gni) - P(b.nfc_saving, b.gni),
         "NFC: route II (FoF + investment)": P(e["nfc_II"], n.gni) - P(b.nfc_saving, b.gni),
         "Total discrepancy D (observed)": P(e["D"], n.gni) - P(e["D_base"], b.gni),
         "Investment (observed)": P(e["gcf"], n.gni) - P(b.gcf, b.gni),
         "Domestic FoF net lending (observed)": P(e["nl_dom"], n.gni) - P(b.nl_nfc + b.nl_fc + b.nl_gov + b.nl_hh, b.gni)}
    return pd.Series(r)


def backtest(q, a):
    """2024→2025: 부문 저축이 공표된 기간에 같은 절차를 적용. 오차 = 추정 − 실제(GNI 대비 %p 변화)."""
    rows = []
    for lab, base, new in (("H1 2024→H1 2025", ["2024Q1", "2024Q2"], ["2025Q1", "2025Q2"]),
                           ("H2 2024→H2 2025", ["2024Q3", "2024Q4"], ["2025Q3", "2025Q4"])):
        b, n, e = estimate(q, a, base, new, base)          # 영업잉여(A3)는 여기서 쓰지 않음
        P = lambda x, g: 100 * x / g
        act = lambda v: P(n[v], n.gni) - P(b[v], b.gni)
        est = lambda v: P(e[v], n.gni) - P(b[{"nfc_I": "nfc_saving", "nfc_II": "nfc_saving", "gov": "gov_saving",
                                                "gov_old": "gov_saving", "fc": "fc_saving", "fc_fof": "fc_saving"}[v]], b.gni)
        for k, v in (("Government: FoF route (A1)", "gov"), ("Government: share held (old)", "gov_old"),
                     ("Financial corporations: share held (A2)", "fc"), ("Financial corporations: FoF route", "fc_fof"),
                     ("NFC: route I (income residual)", "nfc_I"), ("NFC: route II (FoF + investment)", "nfc_II")):
            s = {"gov": "gov_saving", "gov_old": "gov_saving", "fc": "fc_saving", "fc_fof": "fc_saving"}.get(v, "nfc_saving")
            rows.append(dict(period=lab, item=k, actual=act(s), estimate=est(v)))
    r = pd.DataFrame(rows)
    r["error"] = r.estimate - r.actual
    return r


def annual_errors(a, fa):
    """연간 2010–2025: 각 가정이 틀린 정도(GNDI 대비 %p, 전년 대비 변화의 오차) — 연간 부문 투자가 관측되므로 A4 오차는 제외."""
    P = lambda x: 100 * x / a.gndi
    dG = P(a.s_gov - a.i_gov - fa.gov).diff()               # A1 오차
    dFC = P(a.s_fc).diff()                                   # A2 오차(비중 고정)
    dFCf = P(a.s_fc - fa.fc).diff()                          # 금융법인 FoF 경로 오차
    dN2 = P(a.s_nfc - a.i_nfc - fa.nfc).diff()               # 경로 II 오차(D_NFC 비중 고정)
    eI = -(dG + dFC)                                         # 경로 I 오차 = −(정부 오차 + 금융법인 오차)
    rms = lambda x: float(np.sqrt((x.dropna() ** 2).mean()))
    return pd.Series({"Government: FoF route (A1)": rms(dG), "Government: share held (old)": rms(P(a.s_gov).diff()),
                      "Financial corporations: share held (A2)": rms(dFC), "Financial corporations: FoF route": rms(dFCf),
                      "NFC: route I (income residual)": rms(eI), "NFC: route II (FoF + investment)": rms(dN2)})


def annual_ols(a, b):
    import ep_korea_sources as ks
    return ks.annual(a, b)


def figure(res, sc, rmse, bt, info):
    import fig_paper as fp
    plt = fp.plt
    pal = {"Corporate profits": fp.CO, "Corporate retention": "#7a5195", "Household saving rate": fp.HH, "Household income": "#7f9cc9",
           "Government revenue": "#6a8f3c", "Government spending": "#b3cf8f", "Financial corporations": "0.55", "Unallocated": "0.82",
           "Other": "0.82"}
    lab = {"Corporate profits": "Corporate profits (income)", "Corporate retention": "Corporate retention (behavior)",
           "Household saving rate": "Household saving rate (behavior)", "Household income": "Household income (income)",
           "Government revenue": "Government disposable income", "Government spending": "Government consumption (−)",
           "Financial corporations": "Financial corporations", "Unallocated": "Other / unallocated", "Other": "Other / unallocated"}
    est = {"Corporate profits", "Corporate retention", "Government revenue", "Financial corporations"}
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.3), gridspec_kw=dict(width_ratios=[1.25, 1]))
    ax = axes[0]
    seen = set()
    for x, (name, c, dnat) in enumerate(res):
        pos = neg = 0.0
        for k_, v in c.items():
            if not np.isfinite(v) or abs(v) < 1e-9:
                continue
            k2 = "Unallocated" if k_ == "Other" else k_
            hatch = "////" if (x >= 2 and k_ in est) else None
            ax.bar(x, v, bottom=pos if v >= 0 else neg, color=pal[k_], width=0.58, edgecolor="white", lw=0.3, hatch=hatch,
                   label=lab[k_] if k2 not in seen else None)
            seen.add(k2)
            pos, neg = (pos + v, neg) if v >= 0 else (pos, neg + v)
        ax.scatter(x, dnat, color="k", s=24, zorder=5, label="Change in national saving" if x == 0 else None)
    ax.axhline(0, color=".3", lw=.6)
    ax.axvline(1.5, color=".5", lw=.7, ls=":")
    ax.set_xticks(range(len(res)))
    ax.set_xticklabels([r[0] for r in res], fontsize=8.3)
    ax.set_xlim(-0.6, len(res) - 0.4)
    ax.axvline(2.5, color=".85", lw=.5)
    ax.set_ylabel("Percentage points of GNI")
    ax.set_title("(a) Sources of the change in gross national saving", loc="left", fontsize=9.3)
    ax.legend(loc="upper left", fontsize=6.6, frameon=False)
    ax.set_ylim(-4, 12.5)
    ax.text(0.5, -3.6, "OECD annual", ha="center", fontsize=7, color="0.35")
    ax.text(2.5, -3.6, "BOK quarterly, H1 2025→H1 2026", ha="center", fontsize=7, color="0.35")

    ax = axes[1]
    items = ["Households (observed)", "Government: FoF route (A1)", "Government: share held (old)",
             "Financial corporations: share held (A2)", "Financial corporations: FoF route",
             "NFC: route I (income residual)", "NFC: route II (FoF + investment)"]
    names = {"Households (observed)": "Households\n(published)", "Government: FoF route (A1)": "Government\nFoF route",
             "Government: share held (old)": "Government\nshare held (old)", "Financial corporations: share held (A2)": "Financial corp.\nshare held",
             "Financial corporations: FoF route": "Financial corp.\nFoF route", "NFC: route I (income residual)": "Nonfin. corp.\nroute I: residual",
             "NFC: route II (FoF + investment)": "Nonfin. corp.\nroute II: FoF+inv."}
    col = {"Households": fp.HH, "Government": "#6a8f3c", "Financial": "0.45", "NFC": fp.CO}
    ys = np.arange(len(items))[::-1]
    for y, k in zip(ys, items):
        cl = col[k.split(":")[0].split(" ")[0]]
        v = sc[k]
        bk = bt[bt.item == k]
        e_ = float(np.sqrt((bk.error ** 2).mean())) if len(bk) else 0.0
        ax.barh(y, v, color=cl, alpha=0.85 if "old" not in k else 0.35, height=0.6)
        if e_ > 0:
            ax.errorbar(v, y, xerr=e_, color="k", lw=0.9, capsize=2.5)
        if k in rmse:
            ax.errorbar(v, y, xerr=rmse[k], color="0.55", lw=2.2, capsize=0, alpha=0.6)
    ax.axvline(0, color=".3", lw=.6)
    ax.set_yticks(ys)
    ax.set_yticklabels([names[k] for k in items], fontsize=7.4)
    ax.set_xlabel("Change in sector saving, H1 2025→H1 2026 (pp of GNI)")
    d = sc["Total discrepancy D (observed)"]
    ax.set_title("(b) Pinning down sector saving in H1 2026", loc="left", fontsize=9.3)
    ax.text(0.98, 0.97, f"National saving {sc['National saving (observed)']:+.1f}\nInvestment {sc['Investment (observed)']:+.1f}\n"
            f"FoF net lending, domestic {sc['Domestic FoF net lending (observed)']:+.1f}\nReal–financial discrepancy {d:+.1f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=6.8, color="0.25")
    fig.suptitle("Korea's saving surge: households then, corporate profits now", x=0.0, ha="left", fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.text(0.0, -0.02,
             "Notes: Gross saving, percent of GNI. (a) Corporate saving splits into a profit component (change in operating surplus at the earlier "
             "retention ratio) and a retention component; household saving into a saving-rate and an income component; government saving into "
             "disposable income and minus final consumption. H1 2026: national saving, investment, household income and saving, government "
             "consumption and investment, and sector net lending (flow of funds) are published; hatched components rest on assumptions: government "
             "saving = its flow-of-funds net lending + its investment + the gap between the two accounts held at its H1 2025 share of GNI; "
             "financial-corporation saving held at its H1 2025 share; nonfinancial-corporate saving is the residual (route I); its operating surplus "
             f"is total operating surplus less other sectors' at their 2025 growth (implied growth {info['nfc_gos_growth']:.0f}%). (b) Route II "
             "follows the capital-account approach: corporate flow-of-funds net lending + corporate investment (private investment split at 2025 "
             "shares) + the gap held at its H1 2025 share; in (a) the part of national saving it leaves unexplained is 'unallocated'. Black whiskers: "
             "root-mean-square error when the same procedure is applied to the 2024→2025 half-years, for which sector saving is published; grey: "
             "the same for annual changes, 2010–2025.",
             fontsize=6.6, wrap=True, va="top")
    fp.save(fig, "fig_korea_2026", sfx="")


if __name__ == "__main__":
    q, a, fa = load()
    b, n, e = estimate(q, a, ["2025Q1", "2025Q2"], ["2026Q1", "2026Q2"], ["2024Q1", "2024Q2"])
    cI, dnat, info = shares(b, n, e, "nfc_I")
    cII, _, infoII = shares(b, n, e, "nfc_II")
    cOld, _, _ = shares(b, n, e, "nfc_I", gov="gov_old")
    sc = sector_changes(b, n, e)
    rm = annual_errors(a, fa)
    bt = backtest(q, a)
    a1, n1 = annual_ols(2012, 2015)
    a2, n2 = annual_ols(2023, 2025)
    pd.set_option("display.width", 220)
    out = pd.DataFrame({"route I (base)": cI, "route II": cII, "gov share held (old)": cOld})
    out.loc["Change in national saving"] = dnat
    out.to_csv(OUT / "ep_korea_2026.csv")
    chk = pd.concat([sc.rename("H1 2026 change").to_frame().join(rm.rename("annual RMSE")),
                     bt.pivot_table(index="item", columns="period", values="error", sort=False).add_prefix("backtest error ")], axis=1)
    chk.to_csv(OUT / "ep_korea_2026_check.csv")
    print(out.round(2).to_string())
    print(chk.round(2).to_string())
    print(bt.round(2).to_string())
    print({k: round(v, 3) for k, v in info.items()}, "route II retention", round(infoII["ret1"], 3))
    print("levels ₩tn H1:", {k: round(e[k] / 1e3, 1) for k in ("nfc_I", "nfc_II", "gov", "gov_old", "fc", "nfc_gos", "i_nfc", "D", "D_base")},
          {k: round(b[k] / 1e3, 1) for k in ("nfc_saving", "gov_saving", "fc_saving", "nfc_gos", "gni")}, round(n.gni / 1e3, 1))
    # 2012→2015 교차 확인: 한국은행 연간 부문 저축(DT_200Y156)과 순자금운용(DT_281Y001), GNDI 대비 %p
    P = lambda x: 100 * x / a.gndi
    past = pd.DataFrame({s_: P(a[f"s_{s_}"]) for s_ in SEC} | {f"fof_{s_}": 100 * fa[s_] / a.gndi for s_ in SEC})
    print("BOK annual 2012→2015 change (pp GNDI):", (past.loc[2015] - past.loc[2012]).round(2).to_dict())
    # A3 민감도: 비기업 영업잉여 증가율 0%, 5%, 10%
    for g in (0.0, 0.05, 0.10):
        nfc_gos = n.gos_total - (b.gos_total - b.nfc_gos) * (1 + g)
        print(f"non-NFC GOS growth {g:.0%}: profit component", round(info["ret0"] * 100 * (nfc_gos / n.gni - b.nfc_gos / b.gni), 2))
    figure([("2012→2015", a1, n1), ("2023→2025", a2, n2), ("Route I\n(baseline)", cI, dnat), ("Route II\n(FoF + inv.)", cII, dnat)],
           sc, rm, bt, info)
