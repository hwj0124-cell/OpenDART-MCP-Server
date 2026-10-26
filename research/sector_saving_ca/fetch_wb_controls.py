"""World Bank WDI 추가 통제 후보: 소비자물가 상승률, 실업률(ILO 모형 추정, 1991~), 민간신용/GDP.
→ data/raw/wb_controls.csv (country, year, infl, unemp, credit)
"""
import json
import urllib.request

import pandas as pd

from make_report import OUT

CODES = {"FP.CPI.TOTL.ZG": "infl", "SL.UEM.TOTL.ZS": "unemp", "FS.AST.PRVT.GD.ZS": "credit"}


def get(code):
    url = f"https://api.worldbank.org/v2/country/all/indicator/{code}?format=json&per_page=20000&date=1960:2025"
    with urllib.request.urlopen(url, timeout=60) as r:
        meta, rows = json.loads(r.read())
    return pd.DataFrame([dict(country=x["countryiso3code"], year=int(x["date"]), value=x["value"]) for x in rows if x["countryiso3code"]])


if __name__ == "__main__":
    out = None
    for code, name in CODES.items():
        d = get(code).rename(columns={"value": name})
        out = d if out is None else out.merge(d, on=["country", "year"], how="outer")
    out = out.dropna(subset=list(CODES.values()), how="all")
    out.to_csv(OUT.parent / "data" / "raw" / "wb_controls.csv", index=False)
    print(out.describe().round(1).to_string())
