"""도구변수·순저축용 추가 OECD 자료.

  table6_gos.csv : 산업별(ISIC Rev.4) 총영업잉여·혼합소득(B2A3G)과 총부가가치(B1G), 경상가격 자국통화 — Table 6
  table14_cfc.csv: 부문별 고정자본소모(P51C) — 순저축은 B8G − P51C로 계산 — Table 14
→ data/raw/oecd/
"""
from pathlib import Path

import fetch_oecd as f

OUT = Path(__file__).resolve().parent / "data" / "raw" / "oecd"
CSV = {"Accept": "application/vnd.sdmx.data+csv; charset=utf-8; version=1.0.0"}


def pull(flow, ver, sel, name, start="1970"):
    dims = ["FREQ", "REF_AREA", "SECTOR", "COUNTERPART_SECTOR", "TRANSACTION", "INSTR_ASSET", "ACTIVITY", "EXPENDITURE",
            "UNIT_MEASURE", "PRICE_BASE", "TRANSFORMATION", "TABLE_IDENTIFIER"] if "TABLE6" in flow else None
    if dims is None:  # Table 14 차원 순서(원자료 열 순서와 같음)
        dims = ["FREQ", "REF_AREA", "SECTOR", "COUNTERPART_SECTOR", "ACCOUNTING_ENTRY", "TRANSACTION", "INSTR_ASSET",
                "EXPENDITURE", "UNIT_MEASURE", "VALUATION", "PRICE_BASE", "TRANSFORMATION", "TABLE_IDENTIFIER"]
    key = ".".join(sel.get(d, "") for d in dims)
    r = f.get(f"{f.BASE}/data/OECD.SDD.NAD,{flow},{ver}/{key}",
              params={"startPeriod": start, "dimensionAtObservation": "AllDimensions"}, headers=CSV)
    (OUT / name).write_bytes(r.content)
    print(name, len(r.content) // 1024, "KB")


if __name__ == "__main__":
    import sys
    if "--cfc-only" not in sys.argv:
        pull("DSD_NAMAIN10@DF_TABLE6", "2.0", {"FREQ": "A", "TRANSACTION": "B2A3G+B1G", "PRICE_BASE": "V"}, "table6_gos.csv")
    pull("DSD_NASEC10@DF_TABLE14", "", {"FREQ": "A", "SECTOR": "S1+S11+S14_S15", "TRANSACTION": "P51C"}, "table14_cfc.csv")
