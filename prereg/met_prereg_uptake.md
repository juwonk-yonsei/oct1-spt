# Uptake-Substrate Model (USM) 사전등록

등록일: **2026-08-12** (USM fit · helix-LOPO **전**)
선행: `met_prereg.md`, `met_prereg_tpt.md` (T1 uptake FAIL 확정). **변경하지 않음.**
SPT 10/30 · 설계 5위치 · WP4 문헌 위치 학습 제외 · helix 클러스터는 TPT와 **동일**.

동기: TPT U-head·AM·ΔΔG가 SM73_0에서 ρ≈0. 이 트랙만 **기질·게이트 물리 특징**으로
uptake를 다시 친다. ProteinGym SOTA·abundance 재도전은 범위 밖.

---

## 주장

**이기면:** helix-LOPO에서 USM Spearman(SM73_0) > AM_fitness 및 > TPT U-head 및 > ΔΔG_fitness.
Δρ 잔기부트스트랩 95% CI가 0을 넘는다.

**쓰지 않음:** 전체 VE SOTA, SPT 재튜닝, GFP/abundance에서 AM 이기기.

---

## 방법 (fit 전 고정)

구조: 8SC1, 8ET6 (identical-AA CA만), 8SC4 MF8, AF2 WT(pLDDT만 보조).

**USM 입력 (AM 제외):**
- 게이트: 잔기별 CA 변위 ‖x_8ET6 − x_8SC1‖ (동일 AA만; 없으면 NaN)
- 포켓: dist_MF8, pocket(≤4.5Å), pocket×|Δcharge|
- 치환 물리: Δcharge, Δvolume(Å³), Δhydropathy(Kyte–Doolittle)
- 컨포메이션: ΔSASA(8ET6−8SC1), tm_interface, topology one-hot
- (보조) pLDDT, rel.SASA_8SC1

**타깃:**
- 주: `SM73_0_score`
- 민감도: GFP로 train-fold만 맞춘 선형 잔차 `SM73_resid` (누수 방지: 계수는 학습 helix 밖만)

**모델:** `StandardScaler + Ridge(α=1)` 주 판정. HGBR(max_depth=3, max_iter=100)은 민감도만.

**누수:** leave-one-helix-out (TPT 클러스터). 설계+WP4 위치 학습 제외. AM은 대조만.

---

## 게이트

| # | 예측 | 반증 |
|---|---|---|
| **U1** (주) | ρ(USM, SM73_0) > AM 및 > TPT U-head 및 > ΔΔG; 각 Δρ CI 하단 > 0 | 하나라도 실패 |
| **U2** | ρ(USM, SM73_resid) > AM 및 > ΔΔG; Δρ CI > 0 | 실패 → 잔차 타깃은 기각 |
| **U3** (2차) | 포켓 변이만(n≥50)에서 ρ(USM, SM73_0) > ρ(AM, SM73_0) | 포켓에서도 못 이기면 기질가설 약화 |

U1 실패 시 컷·특징을 사후 바꾸지 않고 USM 음성 보고.

---

## 체크리스트

- [x] `spt/prereg/` 복사 + timestamp
- [x] 특징·Ridge α 고정 후 `met_uptake_features.py` / `met_uptake_lopo.py`
