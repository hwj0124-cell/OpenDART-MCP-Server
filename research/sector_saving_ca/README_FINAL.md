# OECD 패널 최종 분석: 이익 횡재 vs 가계 저축 성향 변화

이 폴더의 최종(본문용) 결과와 재현 방법. 이전 README.md는 초기 설계(부문저축 충격, LP-IV)로, 최종 사양과 다르다.

## 요약

| 항목 | 내용 |
|---|---|
| 자료 | OECD 연간 부문계정(Table 14A), 38개국, 1973–2025(비균형). 모든 비중은 **명목 GDP 대비 %** (`data/panel.csv`) |
| 충격 | 기업: 이익 성분 x_P = ρ_{t−1}Δπ_t (유보율 고정, 영업잉여/GDP 변화) · 가계: 저축률 성분 x_H = y_{t−1}Δσ_t |
| 함께 넣는 성분 | 기업: 유보율 성분 x_R · 가계: 소득 성분 x_Y |
| 종속변수 | 자기 부문 순대출 n^j_{t+h} − n^j_{t−1}, h = 0…6 |
| 통제 | 자기 부문 순대출 수준·두 성분·GDP 성장률·정부 순대출 수준(각 2시차) + 국가·연도 FE |
| 추정 | 2단계 WLS(시작 연도 식 잔차의 국가별 분산 역수), 시점별 비균형 표본(표본 B: 2020년 이후 시작 연도 포함) |
| 구간 | 국가 군집 부트스트랩 999회(`fastboot.py`: 국가별 정규방정식 블록, 느린 부트스트랩과 정확히 일치) |
| 정규화 | 시작 연도(h = 0) 순대출 반응 = 1 |

주요 결과(`output/ep_final.csv`): 순대출 반응 h = 1, 2, 4, 6
- 가계(저축률 변화): 0.74, 0.49, 0.20, 0.24 (90% 구간 h = 6까지 0 위)
- 기업(이익 횡재): 0.31, 0 근처(h ≥ 2)
- 기업 분해: 영업잉여 2.1 → 1.1, 투자 +0.4 → +0.9, 순배당 +0.4 → +0.6
- 가계 분해: 소비 감소 −0.55 → −0.3 수준 유지, 소득 +0.36 → 0, 투자 +0.2

## 실행 순서

```bash
pip install pandas numpy matplotlib scipy requests
# (선택) 원자료 다시 받기 — 이미 data/raw/에 있음
python fetch_oecd.py && python fetch_oecd_extra.py && python fetch_wb_controls.py
python build_panel.py            # → data/panel.csv (GDP 대비), data/panel_gni.csv (GNI 대비, 비교용)

python ep_final.py               # 기준 사양 + 부트스트랩 999회(약 30초) → output/ep_final.csv, fig_final, fig_final_items
python ep_final.py --render      # 저장된 CSV로 그림만
python ep_cum.py                 # 누적 순대출·반감기 → output/ep_cum.csv
python ep_pretrend.py            # 사전 추세(h = −5…6) → output/ep_pretrend.csv, fig_pretrend
python ep_specs.py               # 표본 A–D × 정부 통제 유무 → output/ep_specs.csv, fig_spec_*
python ep_ratio_history.py       # 부문 저축 성향의 역사 → fig_ratio_history
python ep_fig_story.py           # 본문용 2×2 그림(fig_story5: GDP 대비, 정부 합계)
```

## 주요 파일

| 파일 | 역할 |
|---|---|
| `source_shocks.py` | 패널 준비, 성분 x_P·x_R·x_H·x_Y 계산 |
| `ep_windfall_behavior.py` | 부문 블록 설계, 통제, 가중치, 분해 항목 정의 |
| `ep_specs.py` | 표본 A–D, 정부 통제(GOV), 시점별 설계 |
| `fastboot.py` | 빠른 국가 군집 부트스트랩 |
| `ep_final.py` | **기준 사양 결과**(표본 B + 정부 통제) |
| `ep_cum.py`, `ep_pretrend.py` | 누적·반감기, 사전 추세 |
| `ep_wb_gov.py`, `ep_wb_three.py`, `ep_wb_candidates.py` | 정부 점검, 세 부문 분산 분해, 후보 통제변수 |
| `ep_fig_story.py` | 한국 에피소드(2012→15, 2026 상반기) + 패널 결과 결합 그림 |
| `ep_korea_2026*.py` | 2026 상반기 국민저축 원천(한국은행 자료, `data/raw/kosis/`) |
| `model_check.py`, `ep_rho_paths.py`, `fig_two_sources_schematic.py` | 모형 수치 검증, 식 (1)의 ρ 표현, 개념도 |
| `output/paper/` | 논문용 그림(png·pdf). 본문 후보: `fig_story5`, `fig_final`, `fig_final_items`, `fig_pretrend`, `fig_spec_overview` |
| `output/notes/` | 모형 도출 노트(PDF) |

## 참고
- 2026 상반기 한국 분해는 일부 추정치를 포함(부문 저축: 투자 + 순자금운용 + 실물–금융 불일치(2025 상반기 GDP 비중 고정), 나머지는 잔차).
- KOSIS 자료는 서로 다른 통계표를 연결해 계산했으므로 정의 차이에 유의.
