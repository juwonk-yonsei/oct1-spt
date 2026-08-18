# SPT 추가 사전등록 — GREY 우위 · WP4 · P6 대조 지정

등록일: **2026-08-12** (WP4/WP5/WP6 분석 **전**)
선행: `met_prereg.md` P1–P6 (변경하지 않음). P2는 이미 기각됨(§13-A.3).
이 문서는 (i) 탐색적 GREY 관찰을 **독립 검증용으로 재등록**하고,
(ii) WP4 held-out와 P6 양성 대조를 **어떤 구조/검정으로 할지 고정**한다.

SPT 임계값(10% / 30%)과 설계 5변이 제외 규칙은 그대로다.

---

## G — GREY 우위 (WP6에서만 판정, OCT1 DMS로 재튜닝 금지)

OCT1 DMS에서 GREY의 |ρ(AM, GFP)|가 CORE·EXPOSED보다 컸다. 이는 **사후 관찰**이다.
아래는 독립 단백질에서만 판정한다. OCT1 점수를 보고 임계값을 바꾸지 않는다.

| # | 예측 | 데이터 | 반증 |
|---|---|---|---|
| **G1** | \|ρ(AM, 기능점수)\| 순위가 **GREY ≥ CORE > EXPOSED**, 그리고 ρ_GREY < 0 | OCT2 또는 다른 SLC22/막단백질의 **독립 DMS/활성 점수** (문헌 소규모 n은 G1에 쓰지 않음) | GREY가 CORE·EXPOSED보다 작거나, ρ_GREY ≥ 0 |
| **G2** | P4 복제: AM-benign ∩ 문헌 기능저하가 EXPOSED에 편중 | OCT2/MATE1 등 문헌 특성화 변이 + 해당 단백질 SPT | enrichment p≥0.05 또는 CORE에 더 많음 |
| **G3** | AF2 vs 실험 구조 SPT 일치율 ≥80% | OCT2 실험 구조(있으면, 예: 8ET9) 또는 OCT1 8ET6(outward) vs AF2 | <80% |

G1을 시험할 DMS가 없으면 G1은 **미시행**으로 보고하고 G2+G3만 진행한다.
OCT1 DMS로 G1을 “통과”시키지 않는다.

---

## WP4 held-out (OCT1 문헌 변이, 설계 5개 제외)

출처: `literature_variants.csv` 전부 − 위치 61/88/401/420/465.
결실·불확실 노트(???)는 1차에 포함, 민감도에서 제외.

기능 묶음:
- **loss*** = `loss` + `partial_loss` (reduced function)
- **neutral**, **gain**은 따로. gain은 H4에 넣지 않고 기술만.

| # | 예측 | 반증 |
|---|---|---|
| **H4.1** | loss* 변이의 AM **pathogenic 분율**이 CORE > EXPOSED (단측 Fisher) | p≥0.05 또는 반대 |
| **H4.2** | EXPOSED에 있는 loss* 중 AM pathogenic 분율 **≤ 0.5** (AM을 1차 근거로 쓰면 안 됨) | pathogenic > 0.5 |
| **H4.3** | **neutral**은 loss*보다 CORE에 덜 있다 (단측 Fisher, CORE vs not-CORE) | p≥0.05 또는 반대 |

n이 작으면 점추정+정확 검정을 보고하고, 기각/채택을 과장하지 않는다.

---

## P6 양성 대조 (이번 실행에서 지정)

노이즈 기준선: OCT1 AF2 WT 5모델 최대 CA RMSD = **3.284 Å**.

| 대조 | 구조 쌍 | 기대 | 역할 |
|---|---|---|---|
| **P6 주** | **8SC1** (OCT1 WT, inward-open) vs **8ET6** (OCT1CS, outward-facing apo, Suo *NSMB* 2023) | CA RMSD **> 3.284 Å** | 진짜 컨포메이션 변화는 기준선을 넘는다 |
| **P6 건전성** | 8SC1 vs **8SC4** (OCT1 WT + metformin, 둘 다 inward) | CA RMSD **≤ 3.284 Å** | 같은 상태+리간드만으로는 기준선을 넘지 않는다 |

겹침 잔기: 양쪽 CA가 있고 **아미노산이 같은** 위치만 (OCT1CS 공학 치환 제거).
추가로 TM-only RMSD도 보고. 주 판정은 **전역 identical-AA CA RMSD**.

8ET6를 못 쓰면 대체: 7ZH0 (OCT3 outward) vs 8SC1. 서열이 다르므로 2차만.

P5(점돌연변이 AF2 RMSD ≤ 기준선)는 설계 5변이에서 이미 성립. held-out AF2는 이번 라운드에서 돌리지 않는다.
