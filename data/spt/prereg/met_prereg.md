# SPT 사전등록 예측 (P1–P6)

등록일: **2026-08-12** (DMS 점수 분포를 보기 **전**에 기록)
대상 방법: Structure-Position Triage (SPT) — `met_classify.py`, 계획 §13
금표준: Yee et al., *Mol Cell* 2024 OCT1 DMS (abundance / uptake)
실험 구조: PDB **8SC1** (apo inward-open, 2.92 Å; Zeng et al. *Nat Commun* 2023)

이 파일의 예측을 **WP3 상관분석 전에 고정**한다. 임계값·분류 규칙·반증 조건을
사후에 바꾸지 않는다. 민감도 분석(다른 임계값)은 보충으로만 보고한다.

---

## 잠금 규칙 (SPT)

입력: 단백질 구조(PDB) + UniProt 지형(TM / Extracellular / Cytoplasmic)

상대 SASA = residue SASA / Tien et al. 2013 이론 최대 ASA × 100

| 계급 | 정의 (사람 판단 개입 금지) |
|---|---|
| **CORE** | rel.SASA **< 10%** |
| **EXPOSED** | rel.SASA **> 30%** AND topology ∈ {Extracellular, Cytoplasmic} |
| **GREY** | 그 외 (부분 매몰, TM 표면/공동 노출 등) |

도구 적용 정책:

| 계급 | 정책 |
|---|---|
| CORE | AlphaMissense + ΔΔG를 1차 근거로 해석 **유효** |
| EXPOSED | 두 점수를 1차 근거로 **사용 금지**. 트래피킹 / 이황화 / PTM / 계면 가설로 이동 |
| GREY | 두 경로 병행, **불확실**로 보고 |

AF2 WT-vs-변이 RMSD는 **동일 서열 WT 모델 간 최대 CA RMSD(노이즈 기준선)** 를
초과할 때만 해석한다. 현재 OCT1 WT 5모델: 최대 **3.284 Å** (§12-A.12).

"hydrophobic core"는 UniProt TM 여부보다 **매몰(rel.SASA < 10%)** 로 조작적 정의한다.
짧은 cytoplasmic loop가 TM 다발에 패킹된 경우(예: G401)도 CORE다. TM이면서
지질/공동 쪽으로 노출된 잔기(rel.SASA ≥ 10%)는 GREY다.

---

## 설계 세트 (검증에서 제외)

아래 5변이는 규칙을 **만든** 사례다. WP3 군별 상관·WP4 held-out에서 **제외**한다.

rank-1 AF2 (§12-A.12)에 잠금 규칙을 적용하면 아래가 나와야 한다.
이 표가 뒤집히면 구현 버그이거나 규칙이 비결정적이다 (WP1 실패).

| 변이 | UniProt 지형 | rel.SASA (rank-1) | 기대 계급 |
|---|---|---|---|
| R61C | Extracellular (43–149) | 65.6% | **EXPOSED** |
| C88R | Extracellular (43–149) | 14.9% | **GREY** |
| G401S | Cytoplasmic (398–402) | 0.0% | **CORE** |
| M420del | Transmembrane (403–423) | 5.5% | **CORE** |
| G465R | Transmembrane (465–485) | 0.0% | **CORE** |

C88R을 EXPOSED가 아니라 GREY로 두는 이유: ECD 안 부분 매몰 Cys이며 실험 구조
(8SC1 SSBOND)에서 **C88–C142 이황화**를 이룬다. 완전 노출 루프 잔기가 아니다.

---

## 사전등록 예측

| # | 예측 | 분석 단위 | 반증 조건 |
|---|---|---|---|
| **P1** | CORE 위치의 missense는 DMS **abundance** 감소(WT 대비 더 낮은 점수)가 EXPOSED보다 크다 | 잔기 중앙값 또는 변이 전체 분포, CORE vs EXPOSED (GREY 제외 또는 별도) | 두 군 차이 없음 (양측, 사전 정한 검정에서 p≥0.05) 또는 반대 방향 |
| **P2** | AlphaMissense ↔ DMS abundance **스피어만 |ρ|** 가 CORE에서 EXPOSED보다 크다. AM 점수는 높을수록 유해이므로 ρ 자체는 **음수**가 기대된다. 즉 ρ_CORE < 0 이고 \|ρ_CORE\| > \|ρ_EXPOSED\|. | 위치×치환 missense (설계 5변이 제외). 군별 ρ 및 Δ\|ρ\|의 잔기-묶음 부트스트랩 CI | Δ\|ρ\|=\|ρ_CORE\|−\|ρ_EXPOSED\| 의 95% CI가 0을 포함하거나 음수. 또는 ρ_CORE ≥ 0 |
| **P3** | ΔΔG(안정성 손실) 중앙값이 CORE에서 크고 EXPOSED에서 ≈0 | ThermoMPNN 또는 FoldX, 동일 missense 세트 | EXPOSED 중앙값이 CORE와 구분되지 않거나 CORE보다 큼 |
| **P4** | AlphaMissense가 **benign**으로 부른 기능저하 변이(DMS abundance 또는 문헌 기능저하)는 EXPOSED에 편중된다 | 기능저하 ∩ AM-benign 집합의 계급 분포 vs 전체 missense 배경 | CORE에도 고르게 분포 ( enrichment p≥0.05) |
| **P5** | OCT1 점돌연변이의 AF2 WT-vs-변이 전역 CA RMSD는 노이즈 기준선(3.284 Å) **이내** | 설계 5변이 + WP4 held-out 점돌연변이 | 1개 이상이 기준선을 초과 |
| **P6** | 양성 대조(큰 구조 변화: 큰 결실/절단, 또는 inward vs outward 실험 구조 쌍)의 AF2 또는 실험 구조 RMSD는 노이즈 기준선을 **초과** | WP5에서 사전 지정한 양성 대조 1종 이상 | 초과하지 못함 → 기준선이 과도하게 관대 |

**이 논문의 핵심 주장은 P2와 P4다.** P1은 기전 전제, P3는 ΔΔG 도구의 정합성, P5+P6은
AF2 RMSD 기준선의 음성·양성 대조다.

P3는 ΔΔG 도구 설치 후에만 판정한다. 도구가 없으면 P3는 "미시행"으로 보고하고
P1/P2/P4로 논문을 진행할 수 있다.

---

## 검정 세부 (WP3에서 그대로 사용)

- 설계 5변이 및 해당 위치의 **모든 치환**은 상관· enrichment 계산에서 제외.
- Missense만. 동의/결실/절단은 별도(AM 적용 범위 밖).
- Abundance와 uptake는 **따로** 계산. 주 가설은 abundance (접힘/단백질량).
- 군 비교: CORE vs EXPOSED. GREY는 기술통계만 보고하거나 민감도에서 포함.
- Δρ 신뢰구간: 잔기 단위 부트스트랩 10,000회 (치환을 잔기에 묶음).
- 다중비교: P1–P4를 주 가족으로 보고, 보정은 Holm (4개).

---

## WP4 held-out 후보 (설계 5변이 제외)

문헌/PharmGKB에서 기능이 보고된 OCT1 변이. 맹검 분류 후 대조:

S14F, Q97K, P117L, S189L, R206C, G220V, P283L, R287G, P341L, V408M
(+ `literature_variants.csv`에 있는 추가 항목, 설계 5개만 빼고 전부).

---

## WP5 양성 대조 후보 (P6)

아래 중 **최소 1개**를 사전에 지정해 돌린다.

1. OCT1 inward-open (8SC1) vs outward-open 실험/모델 구조 쌍의 CA RMSD
2. 큰 결실 또는 N/C 절단 변이의 AF2 vs WT RMSD
3. 접힘 붕괴로 문헌에 확립된 변이의 AF2 RMSD (점돌연변이가 기준선을 넘지 못하면 이 항으로 대체)

**지정 (2026-08-12, RMSD 계산 전):** 옵션 1을 쓴다.
주 대조 = PDB **8SC1** (inward WT) vs **8ET6** (outward OCT1CS).
건전성 = 8SC1 vs **8SC4** (둘 다 inward). 세부·반증은 `met_prereg_grey.md`.

---

## 성공 / 중단 (계획 §13.5와 동일)

- **투고 가능:** P2 성립 + WP4 재현 + AF2↔8SC1 분류 일치 ≥80% (공통 잔기, 쇄 절단 인접 제외) + P5·P6
- **범위 축소 투고:** P2는 성립, WP6 일반화 실패 → OCT1/SLC22 한정
- **재설계:** P2 반증 → "AF2/AM은 PGx 기전 층화에 쓸 수 없다"는 순수 음성 방법론으로 전환
