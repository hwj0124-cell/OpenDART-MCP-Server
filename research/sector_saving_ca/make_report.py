"""output/*.csv, summary.json → output/report.html (결과 요약 페이지, 인라인 SVG 그림)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
Z = 1.645


def svg_lines(series, xlabel, ylabel, w=640, h=280, xs=None):
    """series: [(class, label, x, y, lo, hi)] → 반응형 SVG 문자열."""
    ml, mr, mt, mb = 52, 16, 16, 44
    allx = np.concatenate([np.asarray(s[2], float) for s in series])
    ally = np.concatenate([np.concatenate([np.asarray(s[i], float) for i in (3, 4, 5) if s[i] is not None])
                           for s in series])
    x0, x1 = allx.min(), allx.max()
    lo, hi = min(0, np.nanmin(ally)), max(0, np.nanmax(ally))
    step = [0.1, 0.2, 0.25, 0.5, 1, 2][np.searchsorted([0.6, 1.2, 1.6, 3, 6], hi - lo)]
    lo, hi = np.floor(lo / step) * step, np.ceil(hi / step) * step
    X = lambda v: ml + (v - x0) / (x1 - x0) * (w - ml - mr)
    Y = lambda v: mt + (hi - v) / (hi - lo) * (h - mt - mb)
    o = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{ylabel}">']
    for t in np.arange(lo, hi + step / 2, step):
        cls = "zero" if abs(t) < 1e-9 else "grid"
        o.append(f'<line class="{cls}" x1="{ml}" x2="{w-mr}" y1="{Y(t):.1f}" y2="{Y(t):.1f}"/>')
        o.append(f'<text class="tick" x="{ml-6}" y="{Y(t)+4:.1f}" text-anchor="end">{t:.2g}</text>')
    for t in (xs if xs is not None else np.unique(allx)):
        o.append(f'<text class="tick" x="{X(t):.1f}" y="{h-mb+18}" text-anchor="middle">{int(t)}</text>')
    o.append(f'<text class="axis" x="{(ml+w-mr)/2}" y="{h-6}" text-anchor="middle">{xlabel}</text>')
    for cls, label, x, y, l, u in series:
        x, y = np.asarray(x, float), np.asarray(y, float)
        if l is not None:
            pts = [f"{X(a):.1f},{Y(b):.1f}" for a, b in zip(x, u)] + \
                  [f"{X(a):.1f},{Y(b):.1f}" for a, b in zip(x[::-1], np.asarray(l)[::-1])]
            o.append(f'<polygon class="band {cls}" points="{" ".join(pts)}"/>')
    for cls, label, x, y, l, u in series:
        x, y = np.asarray(x, float), np.asarray(y, float)
        ok = ~np.isnan(y)  # 결측 연도는 선에서 제외
        x, y = x[ok], y[ok]
        pts = " ".join(f"{X(a):.1f},{Y(b):.1f}" for a, b in zip(x, y))
        o.append(f'<polyline class="line {cls}" points="{pts}"/>')
        for a, b in zip(x, y):
            o.append(f'<circle class="dot {cls}" cx="{X(a):.1f}" cy="{Y(b):.1f}" r="2.6"/>')
    o.append("</svg>")
    return "".join(o)


def legend(items):
    return '<div class="legend">' + "".join(
        f'<span><i class="sw {c}"></i>{t}</span>' for c, t in items) + "</div>"


def main():
    pt = pd.read_csv(OUT / "passthrough.csv")
    jl = pd.read_csv(OUT / "joint_lp.csv")
    s = json.loads((OUT / "summary.json").read_text())

    # 그림 1: 충격시점 부문저축 1%p당 CA 반응 (IRL·LUX 제외)
    p = pt[pt.variant == "baseline_exMNE"]
    ser = []
    for sec, cls, lab in [("corp", "corp", "기업저축"), ("hh", "hh", "가계저축")]:
        q = p[p.sector == sec]
        ser.append((cls, lab, q.h, q.impact, q.impact - Z * q.se_impact, q.impact + Z * q.se_impact))
    fig1 = svg_lines(ser, "충격 후 연수", "CA 반응")
    ser = []
    for sec, cls, lab in [("corp", "corp", "기업저축"), ("hh", "hh", "가계저축")]:
        q = p[p.sector == sec]
        ser.append((cls, lab, q.h, q.horizon, q.horizon - Z * q.se_horizon, q.horizon + Z * q.se_horizon))
    fig1b = svg_lines(ser, "충격 후 연수", "누적 전가율")

    # 그림 2: 에피소드 사건연구
    es = pd.read_csv(OUT / "episodes_baseline_exMNE_event.csv")
    def ev(var, sec):
        q = es[(es["var"] == var) & (es.sector == sec)]
        return q.k, q["mean"], q["mean"] - Z * q.se, q["mean"] + Z * q.se
    fig2a = svg_lines([("corp", "", *ev("ca", "corp")), ("hh", "", *ev("ca", "hh"))],
                      "에피소드 시작 기준 연도 (t−1 = 0)", "CA 변화")
    fig2b = svg_lines([("corp", "", *ev("sv_corp", "corp")), ("corpalt", "", *ev("sv_hh", "corp")),
                       ("hh", "", *ev("sv_hh", "hh")), ("hhalt", "", *ev("sv_corp", "hh"))],
                      "에피소드 시작 기준 연도 (t−1 = 0)", "부문저축 변화")
    n_ep = {k: len(v) for k, v in s["episodes_baseline_exMNE"]["episodes"].items()}

    # 그림 3: 기업이익 1%p 충격의 배분 경로
    j = jl[jl.variant == "profit_propensity_exMNE"]
    def ch(v):
        q = j[j.outcome == v]
        return q.h, q.b_corp, q.b_corp - Z * q.se_corp, q.b_corp + Z * q.se_corp
    fig3a = svg_lines([("prof", "", *ch("prof")), ("corp", "", *ch("sv_corp")), ("ca", "", *ch("ca"))],
                      "충격 후 연수", "")
    fig3b = svg_lines([("div", "", *ch("div_paid")), ("tax", "", *ch("tax_paid")), ("gov", "", *ch("nl_gov"))],
                      "충격 후 연수", "")

    # 표: 단기→중기 하락폭 비교
    names = {"baseline": "기본 (부문저축, 41개국)", "baseline_exMNE": "아일랜드·룩셈부르크 제외",
             "baseline_exCOVID": "2020–21년 제외", "baseline_growthctrl": "당해 실질성장률 통제",
             "profit_propensity": "기업이익 vs 가계저축률", "profit_propensity_exMNE": "기업이익 vs 가계저축률, IRL·LUX 제외",
             "profit_propensity_exCOVID": "기업이익 vs 가계저축률, 2020–21년 제외"}
    f = lambda d: f'{d["point"]:.2f}'
    ci = lambda d: f'<span class="ci">[{d["lo90"]:.2f}, {d["hi90"]:.2f}]</span>'
    rows = []
    for k, lab in names.items():
        c = s[f"contrasts_{k}"]
        rows.append(f"<tr><th>{lab}</th>"
                    f"<td>{f(c['impact_corp_short'])} → {f(c['impact_corp_medium'])}</td>"
                    f"<td>{f(c['impact_hh_short'])} → {f(c['impact_hh_medium'])}</td>"
                    f"<td class='em'>{f(c['impact_dd'])} {ci(c['impact_dd'])}</td>"
                    f"<td>{f(c['horizon_corp_medium'])} / {f(c['horizon_hh_medium'])}</td>"
                    f"<td>{f(c['horizon_dd'])} {ci(c['horizon_dd'])}</td></tr>")
    table = "\n".join(rows)

    # 표본 강건성 표
    sc = pd.read_csv(OUT / "samples_contrasts.csv")
    g = lambda r, k: f'{r[k]:.2f} <span class="ci">[{r[k + "_lo"]:.2f}, {r[k + "_hi"]:.2f}]</span>'
    srows = []
    for name, grp in sc.groupby("sample", sort=False):
        b_, p_ = grp[grp.set == "baseline"].iloc[0], grp[grp.set == "profit_propensity"].iloc[0]
        label = name.split(" (")[0]
        members = name[len(label):].strip(" ()")
        sub = f'<div class="ci" style="white-space:normal">{members}</div>' if members else ""
        srows.append(f"<tr><th>{label}{sub}</th><td>{int(b_.countries)}</td>"
                     f"<td>{b_.impact_corp_short:.2f} → {b_.impact_corp_medium:.2f}</td>"
                     f"<td>{b_.impact_hh_short:.2f} → {b_.impact_hh_medium:.2f}</td>"
                     f"<td class='em'>{g(b_, 'impact_dd')}</td><td class='em'>{g(p_, 'impact_dd')}</td></tr>")
    loo = pd.read_csv(OUT / "loo.csv")
    lb, lp = loo[loo.set == "baseline"].impact_dd, loo[loo.set == "profit_propensity"].impact_dd

    # 원천별 비교
    si = pd.read_csv(OUT / "source_irf.csv")
    def src(sample_prefix, y):
        r = si[si["sample"].str.startswith(sample_prefix) & (si.outcome == y)]
        return [("corp", "", r.h, r.b_P, r.b_P - Z * r.se_P, r.b_P + Z * r.se_P),
                ("hh", "", r.h, r.b_H, r.b_H - Z * r.se_H, r.b_H + Z * r.se_H)]
    figs = {f"{{{{SRC_{k}}}}}": svg_lines(src(pref, y), "충격 후 연수", y)
            for k, pref, y in [("CA_W", "MNE", "ca"), ("SN_W", "MNE", "sv_nat"),
                               ("CA_C", "선진", "ca"), ("SN_C", "선진", "sv_nat")]}
    sc2 = pd.read_csv(OUT / "source_contrasts.csv")
    s2 = []
    for _, r in sc2.iterrows():
        label = r["sample"].split(" (")[0]
        s2.append(f"<tr><th>{label}</th><td>{int(r.countries)}</td>"
                  f"<td>{r.ca_P_short:.2f} → {r.ca_P_medium:.2f}</td>"
                  f"<td>{r.ca_H_short:.2f} → {r.ca_H_medium:.2f}</td>"
                  f"<td class='em'>{g(r, 'ca_medium_diff')}</td>"
                  f"<td>{r.ptc_P:.2f} / {r.ptc_H:.2f}</td>"
                  f"<td class='em'>{g(r, 'ptc_diff')}</td></tr>")

    tpl = (HERE / "report_template.html").read_text()
    for k, v in figs.items():
        tpl = tpl.replace(k, v)
    tpl = tpl.replace("{{SOURCE_TABLE}}", "\n".join(s2))
    # 경상수지 1단위 정규화 지속성
    pp = pd.read_csv(OUT / "persistence_paths.csv")
    def pth(pref, variant, shock, y, cls):
        r = pp[pp["sample"].str.startswith(pref) & (pp.variant == variant) & (pp.shock == shock) & (pp.outcome == y)]
        return (cls, "", r.h, r.b, r.b - Z * r.se, r.b + Z * r.se)
    for key, pref in (("ALL", "전체"), ("MNE", "MNE"), ("CORE", "선진")):
        tpl = tpl.replace(f"{{{{PER_CA_{key}}}}}", svg_lines(
            [pth(pref, "base", "P", "ca", "corp"), pth(pref, "base", "H", "ca", "hh")], "충격 후 연수", "ca"))
        tpl = tpl.replace(f"{{{{PER_P_{key}}}}}", svg_lines(
            [pth(pref, "base", "P", "prof", "prof"), pth(pref, "base", "P", "sv_nat", "gov"),
             pth(pref, "base", "P", "ca", "corp")], "충격 후 연수", ""))
        tpl = tpl.replace(f"{{{{PER_H_{key}}}}}", svg_lines(
            [pth(pref, "base", "H", "hh_rate", "prof"), pth(pref, "base", "H", "sv_nat", "gov"),
             pth(pref, "base", "H", "ca", "hh")], "충격 후 연수", ""))
    ps = pd.read_csv(OUT / "persistence.csv")
    vlab = {"base": "기본", "growth": "+ 당해 성장률", "tot": "+ 교역조건", "macro": "+ 실질금리·고령화·재정",
            "lags4": "시차 4년", "big": "대형충격만"}
    s4 = []
    for (smp, var), grp in ps.groupby(["sample", "variant"], sort=False):
        gg = grp.set_index("key")
        pt_ = lambda k: f'{gg.loc[k, "point"]:.2f}'
        cell = lambda k: f'{gg.loc[k, "point"]:.2f} <span class="ci">[{gg.loc[k, "lo90"]:.2f}, {gg.loc[k, "hi90"]:.2f}]</span>'
        f_ = gg.iloc[0]
        weak = " weak" if min(f_.F_P, f_.F_H) < 10 else ""
        s4.append(f"<tr class='{weak.strip()}'><th>{smp.split(' (')[0]}<div class='ci'>{vlab[var]}</div></th>"
                  f"<td>{int(f_.n)}</td><td>{f_.F_P:.0f} / {f_.F_H:.0f}</td>"
                  f"<td>{pt_('P_ca_13')} / {pt_('H_ca_13')}</td><td class='em'>{cell('diff_ca_13')}</td>"
                  f"<td>{pt_('P_ca_46')} / {pt_('H_ca_46')}</td><td>{cell('diff_ca_46')}</td>"
                  f"<td>{pt_('P_prof_46')}</td></tr>")
    tpl = tpl.replace("{{PERSIST_TABLE}}", "\n".join(s4))
    am = pd.read_csv(OUT / "alt_methods.csv")
    s5 = [f"<tr><th>{r.method}<div class='ci'>{r.spec}</div></th><td>{r.pair}</td>"
          f"<td>{r.corp:.2f} <span class='ci'>({r.se_corp:.2f})</span></td><td>{r.hh:.2f} <span class='ci'>({r.se_hh:.2f})</span></td>"
          f"<td class='em'>{r.diff:.2f} <span class='ci'>({r.se_diff:.2f})</span></td></tr>" for r in am.itertuples()]
    tpl = tpl.replace("{{ALT_TABLE}}", "\n".join(s5))
    tl = pd.read_csv(OUT / "tot_lp.csv")
    def tser(y, cls):
        r = tl[tl.outcome == y]
        return (cls, "", r.h, r.b, r.b - Z * r.se, r.b + Z * r.se)
    tpl = tpl.replace("{{TOT_A}}", svg_lines([tser("prof", "prof"), tser("ca", "ca"), tser("sv_nat", "gov")], "충격 후 연수", ""))
    tpl = tpl.replace("{{TOT_B}}", svg_lines([tser("nl_corp", "corp"), tser("nl_gov", "gov"), tser("inv_corp", "prof")], "충격 후 연수", ""))
    ip = pd.read_csv(OUT / "lp_ipw.csv")
    s3 = []
    for name, grp in ip.groupby("sample", sort=False):
        gg = grp.set_index("key")
        cell = lambda k: f'{gg.loc[k, "point"]:.2f} <span class="ci">[{gg.loc[k, "lo90"]:.2f}, {gg.loc[k, "hi90"]:.2f}]</span>'
        s3.append(f"<tr><th>{name.split(' (')[0]}</th><td>{cell('D_P_ca_short')}</td><td>{cell('D_P_ca_medium')}</td>"
                  f"<td>{cell('D_H_ca_short')}</td><td>{cell('D_H_ca_medium')}</td><td class='em'>{cell('medium_diff')}</td></tr>")
    tpl = tpl.replace("{{IPW_TABLE}}", "\n".join(s3))
    tpl = (tpl.replace("{{SAMPLES}}", "\n".join(srows))
           .replace("{{LOO_B}}", f"{lb.min():.2f} ~ {lb.max():.2f}")
           .replace("{{LOO_P}}", f"{lp.min():.2f} ~ {lp.max():.2f}"))
    html = (tpl.replace("{{FIG1}}", fig1).replace("{{FIG1B}}", fig1b)
            .replace("{{FIG2A}}", fig2a).replace("{{FIG2B}}", fig2b)
            .replace("{{FIG3A}}", fig3a).replace("{{FIG3B}}", fig3b)
            .replace("{{TABLE}}", table)
            .replace("{{N_CORP}}", str(n_ep["corp"])).replace("{{N_HH}}", str(n_ep["hh"])))
    (OUT / "report_regressions.html").write_text(html)
    print("→", OUT / "report_regressions.html")


if __name__ == "__main__":
    main()
