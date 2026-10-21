"""경상수지 에피소드 분석 → output/report.html (보고서 본문). 회귀 분석은 report_regressions.html 로 보관."""
import numpy as np
import pandas as pd

from make_report import HERE, OUT, Z, svg_lines

CLS = {"기업저축↑": "c_sv_corp", "가계저축↑": "c_sv_hh"}
CRISIS = [1997, 1998, 1999, 2008, 2009, 2010, 2020, 2021]


def classify(ep):
    f = ep[ep.persist != "미확정"].copy()
    f["cls"] = np.select([f[c] >= 0.5 for c in CLS.values()], list(CLS), "기타")
    return f


def pooled(f, var, cls_col="cls"):
    out = {}
    for c in CLS:
        g = f[f[cls_col] == c]
        ks = list(range(-3, 7))
        m = [g[f"{var}_{k}"].mean() for k in ks]
        se = [g[f"{var}_{k}"].std() / np.sqrt(g[f"{var}_{k}"].notna().sum()) for k in ks]
        out[c] = (np.array(ks), np.array(m), np.array(se), len(g))
    return out


def stat_table(f, by):
    rows = []
    for c, g in f.groupby(by):
        rows.append(f"<tr><th>{c}</th><td>{len(g)}</td><td>{g.R3.mean():.2f} / {g.R3.median():.2f}</td>"
                    f"<td>{g.R5.mean():.2f} / {g.R5.median():.2f}</td>"
                    f"<td>{(g.persist == '단명').mean():.0%}</td><td>{(g.persist == '지속').mean():.0%}</td></tr>")
    return "\n".join(rows)


def main():
    ep3 = pd.read_csv(OUT / "ca_episodes_thr3.csv")
    f = classify(ep3)
    p = pooled(f, "ca")
    fig_ca = svg_lines([("corp" if c == "기업저축↑" else "hh", c, k, m, m - Z * s, m + Z * s)
                        for c, (k, m, s, n) in p.items()], "에피소드 시작 기준 연도 (t)", "ca")
    q = pooled(f, "sv_corp")
    r = pooled(f, "sv_hh")
    fig_sv = svg_lines([("corp", "", *q["기업저축↑"][:2], None, None),
                        ("hh", "", *r["가계저축↑"][:2], None, None),
                        ("corpalt", "", *r["기업저축↑"][:2], None, None),
                        ("hhalt", "", *q["가계저축↑"][:2], None, None)], "에피소드 시작 기준 연도 (t)", "sv")
    n = {c: v[3] for c, v in p.items()}

    t_main = stat_table(f[f.cls != "기타"], "cls")
    t_driver = stat_table(f, "driver")
    nc = f[~f.year.isin(CRISIS)]
    t_nocrisis = stat_table(nc[nc.cls != "기타"], "cls")
    rob = []
    for thr in (2, 3, 4):
        e = classify(pd.read_csv(OUT / f"ca_episodes_thr{thr:g}.csv"))
        for c in CLS:
            g = e[e.cls == c]
            rob.append(f"<tr><th>{thr}%p</th><td>{c}</td><td>{len(g)}</td><td>{g.R3.mean():.2f} / {g.R3.median():.2f}</td>"
                       f"<td>{g.R5.mean():.2f} / {g.R5.median():.2f}</td></tr>")
    comp_cols = [("c_nl_corp", "기업 순자금운용"), ("c_nl_hh", "가계 순자금운용"), ("c_nl_gov", "정부 순자금운용"),
                 ("c_x_P", "이익 원천 기업저축"), ("c_x_R", "유보율 변화"), ("c_x_H", "성향 원천 가계저축"),
                 ("c_inv_tot", "총투자")]
    comp = "\n".join(
        f"<tr><th>{lab}</th>" + "".join(f"<td>{f[f.persist == p_][col].mean():.2f}</td>" for p_ in ("단명", "중간", "지속"))
        + "</tr>" for col, lab in comp_cols)
    big = f.sort_values("size", ascending=False).head(15)
    bigrows = "\n".join(
        f"<tr><th>{r.country} {r.year}</th><td>{r.sign}</td><td>{r.size:.1f}</td><td>{r.cls}</td>"
        f"<td>{r.c_sv_corp:.2f}</td><td>{r.c_sv_hh:.2f}</td><td>{-r.c_inv_tot:.2f}</td><td>{r.R3:.2f}</td><td>{r.R5:.2f}</td></tr>"
        for r in big.itertuples())

    # 한국: 두 기업 사이클
    kf = pd.read_csv(OUT / "korea_firms.csv").set_index("year")
    yrs = kf.index.values
    fig_k1 = svg_lines([("corp", "", yrs, kf.f_fcf.values, None, None),
                        ("hh", "", yrs, kf.sv_hh.values - kf.sv_hh.loc[2014], None, None),
                        ("ca", "", yrs, kf.ca.values - kf.ca.loc[2014], None, None)], "연도", "k1", xs=yrs)
    fig_k2 = svg_lines([("prof", "", yrs, kf.f_op_profit.values, None, None),
                        ("gov", "", yrs, kf.f_tax_paid.values, None, None)], "연도", "k2", xs=yrs)
    dd = kf.diff().dropna()
    kstats = dict(
        K_SD_OP=f"{dd.f_op_profit.std():.2f}", K_SD_HH=f"{dd.sv_hh.std():.2f}", K_SD_NLHH=f"{dd.nl_hh.std():.2f}",
        K_CORR_CA=f"{dd.f_fcf.corr(dd.ca):.2f}", K_SLOPE_CA=f"{np.polyfit(dd.f_fcf, dd.ca, 1)[0]:.2f}",
        K_TAX0=f"{kf.f_tax_paid.corr(kf.f_op_profit):.2f}", K_TAX1=f"{kf.f_tax_paid.corr(kf.f_op_profit.shift(1)):.2f}",
        K_CAPEX1=f"{kf.f_capex.corr(kf.f_op_profit.shift(1)):.2f}")
    h1 = pd.read_csv(HERE / "data" / "firms" / "firm_h1_2026.csv")
    tot = h1.groupby("period")[["op_profit", "ocf", "capex", "tax_paid", "tax_expense", "dividends_paid", "buyback", "current_tax_liab_end"]].sum(min_count=1) / 1000
    h1rows = "\n".join(f"<tr><th>{lab}</th><td>{tot.loc['2025H1', c]:.1f}</td><td>{tot.loc['2026H1', c]:.1f}</td></tr>"
                        for c, lab in (("op_profit", "영업이익"), ("ocf", "영업현금흐름"), ("capex", "설비투자"),
                                       ("tax_expense", "법인세비용(발생)"), ("tax_paid", "법인세 납부(현금)"),
                                       ("dividends_paid", "배당 지급"), ("buyback", "자사주 매입")))
    h1rows += f"\n<tr><th>미지급 법인세(반기말 잔액)</th><td>–</td><td>{tot.loc['2026H1', 'current_tax_liab_end']:.1f}</td></tr>"

    # 가계저축률 급등 에피소드
    from lp_tools import balance
    pan = balance(pd.read_csv(HERE / "data" / "panel.csv")).set_index(["country", "year"])

    def absp(ep, v):
        out = []
        for r in ep.itertuples():
            out.append([pan[v].get((r.country, r.year + k), np.nan) - pan[v].get((r.country, r.year - 2), np.nan)
                        for k in range(-2, 7)])
        return pd.DataFrame(out, columns=range(-2, 7))

    he = pd.read_csv(OUT / "hh_episodes.csv")
    pe = pd.read_csv(OUT / "prof_episodes.csv")
    TYPES = [("비침체", "hh"), ("부동산 붕괴", "gov"), ("경기침체", "prof"), ("코로나", "ca")]
    ks = np.arange(-2, 7)
    fig_h1 = svg_lines([(cls, t, ks, np.array([he[he.type == t][f"hh_rate_{k}"].median() for k in ks]), None, None)
                        for t, cls in TYPES], "에피소드 시작 기준 연도 (t)", "hh_rate")
    fig_h2 = svg_lines([(cls, t, ks, absp(he[he.type == t], "ca").median().values, None, None)
                        for t, cls in TYPES[:2]] + [("corp", "이익", ks, absp(pe, "ca").median().values, None, None)],
                       "에피소드 시작 기준 연도 (t)", "ca_abs")
    hrows = []
    for t, _ in TYPES:
        g = he[he.type == t]
        ca, cons, nlg = absp(g, "ca").median(), absp(g, "cons_hh").median(), absp(g, "nl_gov").median()
        fmt = lambda x: "–" if pd.isna(x) else f"{x:.2f}"
        hrows.append(f"<tr><th>{t}</th><td>{len(g)}</td><td>{fmt(g.hh_rate_3.median())} / {fmt(g.hh_rate_6.median())}</td>"
                     f"<td>{cons[5]:.1f}</td><td>{ca[0]:.1f} / {ca[3]:.1f} / {ca[5]:.1f}</td><td>{nlg[5]:.1f}</td></tr>".replace("nan", "–"))
    ca_p, cons_p, nlg_p = absp(pe, "ca").median(), absp(pe, "cons_hh").median(), absp(pe, "nl_gov").median()
    hrows.append(f"<tr class='weak'><th>(비교) 기업 이익비중 급등</th><td>{len(pe)}</td><td>이익비중 {pe.prof_3.median():.2f} / {pe.prof_6.median():.2f}</td>"
                 f"<td>{cons_p[5]:.1f}</td><td>{ca_p[0]:.1f} / {ca_p[3]:.1f} / {ca_p[5]:.1f}</td><td>{nlg_p[5]:.1f}</td></tr>")
    hcases = {t: ", ".join(f"{r.country} {r.year}" for r in he[he.type == t].sort_values("size", ascending=False).head(10).itertuples())
              for t, _ in TYPES}

    # 비침체 가계저축률 급등: 변수별 경로(연도평균 제거, 평균 ± 90% 구간)
    dm = balance(pd.read_csv(HERE / "data" / "panel.csv"))
    PV = [("hh_rate", "가계저축률 (가처분소득 대비 %p)"), ("sv_hh", "가계저축"), ("cons_hh", "가계소비"),
          ("inv_hh", "가계투자"), ("nl_hh", "가계 순자금운용"), ("nl_gov", "정부 순자금운용"),
          ("sv_corp", "기업저축"), ("ca", "경상수지")]
    for v, _ in PV:
        dm[v + "_dm"] = dm[v] - dm.groupby("year")[v].transform("mean")
    dm = dm.set_index(["country", "year"])

    def epath(ep, v):
        M = [[dm[v + "_dm"].get((r.country, r.year + k), np.nan) - dm[v + "_dm"].get((r.country, r.year - 2), np.nan)
              for k in range(-2, 7)] for r in ep.itertuples()]
        P = pd.DataFrame(M, columns=range(-2, 7))
        m, se = P.mean(), P.std() / np.sqrt(P.notna().sum())
        return m.values, (m - Z * se).values, (m + Z * se).values, P.notna().sum().values
    nonrec = he[he.type == "비침체"]
    panels, ptab = [], []
    for v, lab in PV:
        mh, lh, uh, nh = epath(nonrec, v)
        mp, lp_, up, _ = epath(pe, v)
        svg = svg_lines([("corp", "", ks, mp, lp_, up), ("hh", "", ks, mh, lh, uh)], "시작 기준 연도 (t)", lab)
        panels.append(f'<figure><div class="panel-title">{lab}</div>{svg}</figure>')
        ptab.append(f"<tr><th>{lab}</th>" + "".join(f"<td>{mh[k + 2]:.2f} <span class='ci'>[{lh[k + 2]:.2f}, {uh[k + 2]:.2f}]</span></td>"
                                                    for k in (0, 3, 6)) + "</tr>")
    nonrec_n = f"{len(nonrec)}개 (t+6까지 관측 {int(nh[-1])}개)"

    # 부문 자체 지속성: 기업이익 주도 vs 가계
    sp = pd.read_csv(OUT / "sector_persistence.csv")
    G = [("가계: 비침체 저축률 급등", "sv_hh", "nl_hh", "hh"), ("기업: 이익 주도 기업저축 급증", "sv_corp", "nl_corp", "corp"),
         ("기업: 교역조건 횡재", "sv_corp", "nl_corp", "gov")]
    def sps(g, v, cls):
        x = sp[(sp.group == g) & (sp["var"] == v) & (sp.k >= 0)]
        return (cls, "", x.k.values, x["mean"].values, (x["mean"] - Z * x.se.fillna(0)).values, (x["mean"] + Z * x.se.fillna(0)).values)
    fig_s1 = svg_lines([sps(g, sv, c) for g, sv, _, c in G], "시작 기준 연도 (t)", "own_sv")
    fig_s2 = svg_lines([sps(g, v, "prof") for g, v in [("기업: 이익 주도 기업저축 급증", "prof")]]
                       + [sps("기업: 교역조건 횡재", "prof", "gov")], "시작 기준 연도 (t)", "prof")
    srows = []
    for g, sv, nl, _ in G + [("기업: 이익비중 급등", "sv_corp", "nl_corp", "")]:
        def cell(v, k):
            x = sp[(sp.group == g) & (sp["var"] == v) & (sp.k == k)]
            return "–" if x.empty or pd.isna(x["mean"].iloc[0]) else f"{x['mean'].iloc[0]:.2f}"
        n0 = int(sp[(sp.group == g) & (sp["var"] == sv) & (sp.k == 0)].n.iloc[0])
        n6 = int(sp[(sp.group == g) & (sp["var"] == sv) & (sp.k == 6)].n.iloc[0])
        pr = f"{cell('prof', 3)} / {cell('prof', 6)}" if g.startswith("기업") else "–"
        srows.append(f"<tr><th>{g}</th><td>{n0} ({n6})</td><td>{cell(sv, 1)} / {cell(sv, 3)} / {cell(sv, 6)}</td>"
                     f"<td>{cell(nl, 3)} / {cell(nl, 6)}</td><td>{pr}</td><td>{cell('sv_nat', 3)} / {cell('sv_nat', 6)}</td></tr>")

    # 대칭·상대 기준 부문 지속성 (상·하위 10%, 증가·감소 통합)
    s2 = {q: pd.read_csv(OUT / f"sector_persistence2_q{q}.csv") for q in (10, 20)}
    T3 = [("가계 성향", "hh"), ("이익 충격", "corp"), ("교역조건", "gov")]
    def s2ser(basis, t, cls, q=10, dirn="전체"):
        x = s2[q][(s2[q].basis == basis) & (s2[q].type == t) & (s2[q].dir == dirn)]
        return (cls, "", x.k.values, x["mean"].values, (x["mean"] - Z * x.se.fillna(0)).values, (x["mean"] + Z * x.se.fillna(0)).values)
    fig_p1 = svg_lines([s2ser("저축", t, c) for t, c in T3], "시작 기준 연도 (t)", "sym_sv")
    fig_p2 = svg_lines([s2ser("순자금운용", t, c) for t, c in T3], "시작 기준 연도 (t)", "sym_nl")
    prow = []
    for q in (10, 20):
        for basis in ("저축", "순자금운용"):
            for t, _ in T3:
                cells = []
                for dirn in ("증가", "감소", "전체"):
                    x = s2[q][(s2[q].basis == basis) & (s2[q].type == t) & (s2[q].dir == dirn)].set_index("k")
                    cells.append(f"{x.loc[3, 'median']:.2f} / {x.loc[6, 'median']:.2f} <span class='ci'>(n {int(x.loc[0, 'n'])})</span>")
                prow.append(f"<tr><th>상·하위 {q}% · {basis}<div class='ci'>{t}</div></th>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
    cs = pd.read_csv(OUT / "sector_persistence2_cases_q10.csv").set_index(["type", "dir"])
    tot_cases = cs.loc[("교역조건", "증가"), "cases"]

    style = (HERE / "report_template.html").read_text()
    style = style[:style.index("<main>")].replace("<title>기업저축·가계저축 경상수지 반응</title>", "<title>경상수지 에피소드 지속성</title>")
    body = (HERE / "episode_template.html").read_text()
    for k, v in {"FIG_CA": fig_ca, "FIG_SV": fig_sv, "T_MAIN": t_main, "T_DRIVER": t_driver, "T_NOCRISIS": t_nocrisis,
                 "T_ROB": "\n".join(rob), "T_COMP": comp, "T_BIG": bigrows,
                 "N_CORP": str(n["기업저축↑"]), "N_HH": str(n["가계저축↑"]), "N_ALL": str(len(f)),
                 "FIG_K1": fig_k1, "FIG_K2": fig_k2, "T_H1": h1rows, **kstats,
                 "FIG_H1": fig_h1, "FIG_H2": fig_h2, "T_HH": "\n".join(hrows), "N_HH_ALL": str(len(he)),
                 "CASES_NONREC": hcases["비침체"], "CASES_HOUSING": hcases["부동산 붕괴"],
                 "NONREC_PANELS": "\n".join(panels), "NONREC_TABLE": "\n".join(ptab), "NONREC_N": nonrec_n,
                 "N_PROF": str(len(pe)), "FIG_S1": fig_s1, "FIG_S2": fig_s2, "T_SECTOR": "\n".join(srows),
                 "FIG_P1": fig_p1, "FIG_P2": fig_p2, "T_SYM": "\n".join(prow), "TOT_CASES": tot_cases}.items():
        body = body.replace("{{" + k + "}}", v)
    body = body.replace("<td>nan</td>", "<td>–</td>")
    (OUT / "report.html").write_text(style + body)
    print("→", OUT / "report.html")


if __name__ == "__main__":
    main()
