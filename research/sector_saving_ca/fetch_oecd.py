"""OECD 부문별 비금융계정(연간) 다운로드.

OECD Data Explorer(SDMX REST)에서 제도부문별 저축·순대출·소득분배 항목을 받아
data/raw/ 에 SDMX-CSV로 저장한다. DSD의 차원 순서는 실행 시점에 구조 메타데이터를
조회해 결정하므로, 차원 이름만 맞으면 키를 하드코딩하지 않아도 된다.

    python fetch_oecd.py            # 기본 데이터플로우
    python fetch_oecd.py --list     # 부문계정 관련 데이터플로우 후보 출력
"""
import argparse
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import requests

BASE = "https://sdmx.oecd.org/public/rest"
RAW = Path(__file__).resolve().parent / "data" / "raw"

# 비금융계정(부문별, 연간). 필요 시 --flow 로 덮어쓴다.
FLOWS = {
    "nonfin": ("OECD.SDD.NAD", "DSD_NASEC10@DF_TABLE14", "latest"),
    # 금융계정 순자금운용(B9F) — 강건성 점검용
    "fin": ("OECD.SDD.NAD", "DSD_NASEC20@DF_T720R_A", "latest"),
}

SECTORS = ["S1", "S11", "S12", "S13", "S14_S15", "S1M", "S14", "S15", "S2"]
TRANSACTIONS = {
    "nonfin": ["B1GQ", "B9", "B8G", "B6G", "B2G_B3G", "B2G", "P3", "P51G", "P5",
               "D4", "D42", "D5", "D8", "D41"],
    "fin": ["B9F", "B9"],
}

NS = {
    "s": "http://www.sdmx.org/resources/sdmxml/schemas/v2_1/structure",
    "c": "http://www.sdmx.org/resources/sdmxml/schemas/v2_1/common",
    "m": "http://www.sdmx.org/resources/sdmxml/schemas/v2_1/message",
}


def get(url, **kw):
    for attempt in range(5):
        r = requests.get(url, timeout=300, **kw)
        if r.status_code == 429:  # OECD 요청 한도
            time.sleep(30 * (attempt + 1))
            continue
        r.raise_for_status()
        return r
    r.raise_for_status()


def list_flows(pattern=("sector", "financial account", "non-financial")):
    xml = get(f"{BASE}/dataflow/all?detail=allstubs").content
    root = ET.fromstring(xml)
    for df in root.iter(f"{{{NS['s']}}}Dataflow"):
        name = " ".join(n.text or "" for n in df.findall("c:Name", NS))
        if any(p in name.lower() for p in pattern):
            print(f"{df.get('agencyID')},{df.get('id')},{df.get('version')}  {name}")


def dimension_order(agency, flow, version):
    """데이터플로우가 참조하는 DSD의 차원 순서(TIME_PERIOD 제외)."""
    url = f"{BASE}/dataflow/{agency}/{flow}/{version}?references=datastructure"
    root = ET.fromstring(get(url).content)
    dims = []
    for d in root.iter(f"{{{NS['s']}}}Dimension"):
        dims.append((int(d.get("position")), d.get("id")))
    return [d for _, d in sorted(dims)]


def fetch(kind, agency, flow, version, start="1990"):
    dims = dimension_order(agency, flow, version)
    print(f"[{kind}] {flow} dims: {dims}")
    sel = {"FREQ": "A", "SECTOR": "+".join(SECTORS),
           "TRANSACTION": "+".join(TRANSACTIONS[kind])}
    key = ".".join(sel.get(d, "") for d in dims)
    url = f"{BASE}/data/{agency},{flow},{version}/{key}"
    r = get(url, params={"startPeriod": start, "dimensionAtObservation": "AllDimensions"},
            headers={"Accept": "application/vnd.sdmx.data+csv; charset=utf-8; version=1.0.0"})
    RAW.mkdir(parents=True, exist_ok=True)
    out = RAW / f"{kind}.csv"
    out.write_bytes(r.content)
    print(f"[{kind}] saved {out} ({len(r.content)/1e6:.1f} MB)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--only", choices=list(FLOWS))
    a = ap.parse_args()
    if a.list:
        list_flows()
        sys.exit()
    for kind, (agency, flow, ver) in FLOWS.items():
        if a.only and kind != a.only:
            continue
        try:
            fetch(kind, agency, flow, ver)
        except requests.HTTPError as e:
            print(f"[{kind}] 실패: {e}. --list 로 데이터플로우 ID를 확인하세요.")
