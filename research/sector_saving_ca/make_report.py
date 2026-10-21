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

    tpl = (HERE / "report_template.html").read_text()
    html = (tpl.replace("{{FIG1}}", fig1).replace("{{FIG1B}}", fig1b)
            .replace("{{FIG2A}}", fig2a).replace("{{FIG2B}}", fig2b)
            .replace("{{FIG3A}}", fig3a).replace("{{FIG3B}}", fig3b)
            .replace("{{TABLE}}", table)
            .replace("{{N_CORP}}", str(n_ep["corp"])).replace("{{N_HH}}", str(n_ep["hh"])))
    (OUT / "report.html").write_text(html)
    print("→", OUT / "report.html")


if __name__ == "__main__":
    main()
