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

    style = (HERE / "report_template.html").read_text()
    style = style[:style.index("<main>")].replace("<title>기업저축·가계저축 경상수지 반응</title>", "<title>경상수지 에피소드 지속성</title>")
    body = (HERE / "episode_template.html").read_text()
    for k, v in {"FIG_CA": fig_ca, "FIG_SV": fig_sv, "T_MAIN": t_main, "T_DRIVER": t_driver, "T_NOCRISIS": t_nocrisis,
                 "T_ROB": "\n".join(rob), "T_COMP": comp, "T_BIG": bigrows,
                 "N_CORP": str(n["기업저축↑"]), "N_HH": str(n["가계저축↑"]), "N_ALL": str(len(f))}.items():
        body = body.replace("{{" + k + "}}", v)
    (OUT / "report.html").write_text(style + body)
    print("→", OUT / "report.html")


if __name__ == "__main__":
    main()
