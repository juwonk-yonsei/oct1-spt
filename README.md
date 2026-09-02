# OCT1 / SERT AlphaMissense calibration archive

Analysis code and frozen tables for:

> Kang J, Choi J. ClinVar-calibrated AlphaMissense thresholds miss loss-of-function alleles at solvent-exposed residues in two human drug transporters. *PLOS Computational Biology* (in preparation).

This repository is the **code availability** archive for that article. It rebuilds the display items from frozen tables. It is not a new variant-effect predictor and not a pharmacogene-discovery paper.

A previous working title used “structure-position triage (SPT)” and targeted *The Pharmacogenomics Journal*. That identity is withdrawn. Geometric class labels (buried / exposed / grey) remain the same locked rule. **The repository name retains the original project code and does not reflect the current article’s framing.**

## Authors

- Juwon Kang (0009-0009-2186-0038) — first author
- Junjeong Choi (0000-0003-1339-593X) — corresponding author (`junjeong@yonsei.ac.kr`)
  College of Pharmacy, Yonsei Institute of Pharmaceutical Sciences, Yonsei University, Incheon 21983, Republic of Korea

## Provenance of the 12 August 2026 lock

This public repository was created on **18 August 2026** from the private working directory. GitHub history begins at that date (`432abab`); there is no 12 August 2026 commit here.

The lock itself is the dated file `prereg/met_prereg.md` (written 12 August 2026, before DMS–AM correlations). A retrospective OSF registration of that snapshot is in progress and will be cited in the manuscript. The preregistered primary AM test was rank correlation; the recall-gap analysis in the paper is a secondary description, promoted post hoc.

## What is included

- `prereg/` — locked hypotheses, including `met_prereg.md`
- `data/spt/` — frozen class labels, validation missense table (`n = 9,711`), ΔΔG, literature compilation, and locked result JSON
- `data/spt/ms1_feedback2_freeze.json` plus `ms1_feedback2_addendum.json` … `addendum5.json` — numbers used in the PLOS manuscript
- `data/spt/fb260901/` — gnomAD GFP-loss records and SERT cutoff sweep used by Fig. 5–6
- `make_figures.py` — rebuild Figs 1–6 from the freeze (PDF / PNG / TIFF)

`dms_loss` (GFP ≤ −0.814) is the loss definition used for every reported result. `func_loss` (`dms_loss` ∪ literature loss) appears in earlier exploratory JSON files and is not used in the manuscript.

Primary-label ColabFold wild-type coordinates (rank-1 and the five-model ensemble, with pLDDT) are **not** in this repository. They are deposited with the freeze on Zenodo. GitHub is not the archival copy.

## Display-item files

| Display item | Source |
|---|---|
| Fig 1A | `data/spt/wp3_residue_median_gfp.tsv` |
| Fig 1B | `data/spt/wp3_p3_thermompnn_residue_median.tsv` |
| Fig 2A, Table 1, clustered CIs | `data/spt/ms1_feedback2_freeze.json` |
| Fig 2B | `data/spt/ms1_feedback2_addendum2.json` (phyloP quintiles; values also hardcoded in `make_figures.py`) |
| Fig 3 | `data/spt/ms1_feedback2_freeze.json` (`A2_confusion_AF2_x_8SC1`, `A2_A3_three_label`) |
| Fig 4 | `data/spt/wp3_validation_missense.tsv`; Youden gap from `ms1_feedback2_freeze.json` |
| Fig 5A | `data/spt/ms1_feedback2_addendum5.json` (clustered RR) |
| Fig 5B | `data/spt/fb260901/addendum5/SLC6A4_cutoff_sweep.tsv` |
| Fig 6, Table 3 | `data/spt/fb260901/addendum/gnomad_gfp_loss.tsv` |

## What is not a result of the paper

Scripts for an uptake / residual-function classifier and held-out literature AlphaFold mutant models (preregistered P5) failed or were not completed. The classifier scripts are in `history/` (`met_uptake_features.py`, `met_uptake_lopo.py`, `met_r_residual.py`). They are not display items.

## What is not redistributed

Yee et al. raw DMS files, the AlphaMissense proteome table, and experimental PDB files (8SC1, 8SC4, 8ET6, 8ET9) remain with their original sources. Derived columns in `data/` (including AlphaMissense scores in the validation table) are covered by `DATA_LICENSE.md`.

Do not retune the class cuts (buried: relative SASA < 10%; exposed: relative SASA > 30% and extra- or cytoplasmic). Design positions 61 / 88 / 401 / 420 / 465 were used to draft the geometric rule and are excluded from validation statistics.

## Rebuild figures

Tested on **Python 3.12.7**. Residue-clustered intervals in the freeze used `numpy.random.default_rng(20260812)` with 10,000 resamples, not `np.random.seed` / `RandomState`. Exact CI digits require the pinned numpy in `requirements.txt`.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python make_figures.py
```

`make_figures.py` reads `data/spt/` unless `MET_HDD` is set. Outputs go to `figures/`. Point estimates on the figures are locked; the script does not re-run the clustered bootstrap.

## License

Code is MIT (see `LICENSE`). Frozen tables under `data/` contain values derived from third-party sources and are subject to their terms: AlphaMissense predictions (CC BY 4.0, DeepMind Technologies Limited), gnomAD v4, and the Yee et al. deep mutational scan. Attribution requirements of those sources apply to the derived columns. See `DATA_LICENSE.md`.
