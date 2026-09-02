# OCT1 / SERT AlphaMissense calibration archive

Analysis code and frozen tables for:

> Kang J, Choi J. ClinVar-calibrated AlphaMissense thresholds miss loss-of-function alleles at solvent-exposed residues in two human drug transporters. *PLOS Computational Biology* (in preparation).

This repository is the **code availability** archive for that article. It rebuilds the display items from frozen tables. It is not a new variant-effect predictor and not a pharmacogene-discovery paper.

A previous working title used “structure-position triage (SPT)” and targeted *The Pharmacogenomics Journal*. That identity is withdrawn. Geometric class labels (buried / exposed / grey) remain the same locked rule.

## Authors

- Juwon Kang (0009-0009-2186-0038) — first author
- Junjeong Choi (0000-0003-1339-593X) — corresponding author (`junjeong@yonsei.ac.kr`)
  College of Pharmacy, Yonsei Institute of Pharmaceutical Sciences, Yonsei University, Incheon 21983, Republic of Korea

## What is included

- `prereg/` — hypotheses locked on **12 August 2026** (before DMS–AM correlations). The preregistered primary AM test was rank correlation; the recall-gap analysis in the paper is a secondary description, promoted post hoc.
- `data/spt/` — frozen SPT labels, validation missense table (`n = 9,711`), ΔΔG, literature compilation, and locked result JSON
- `data/spt/ms1_feedback2_freeze.json` plus `ms1_feedback2_addendum.json` … `addendum5.json` — numbers used in the PLOS manuscript
- `data/spt/fb260901/` — gnomAD GFP-loss table and SERT cutoff sweep used by Fig. 5–6
- `make_figures.py` — rebuild Figs 1–6 from the freeze (PDF / PNG / TIFF)

Manuscript P4 uses **`dms_loss` n = 485**. Do **not** use `wp3_p1_p2_p4.json` `n_hit` = 493 (`func_loss`).

Primary-label ColabFold wild-type coordinates (rank-1 and the five-model ensemble, with pLDDT) are **not** in this repository. They are deposited with the freeze on Zenodo. GitHub is not the archival copy.

## What is not a result of the paper

Scripts for an uptake / residual-function classifier (`met_uptake_*`, `met_r_residual.py`) and held-out literature AlphaFold mutant models (preregistered P5) failed or were not completed. They remain in the tree as the analysis history. They are not display items.

## What is not redistributed

Yee et al. DMS scores, the AlphaMissense proteome table, and experimental PDB files (8SC1, 8SC4, 8ET6, 8ET9) remain with their original sources.

Do not retune the class cuts (buried: relative SASA < 10%; exposed: relative SASA > 30% and extra- or cytoplasmic). Design positions 61 / 88 / 401 / 420 / 465 were used to draft the geometric rule and are excluded from validation statistics.

## Rebuild figures

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python make_figures.py
```

`make_figures.py` reads `data/spt/` unless `MET_HDD` is set. Outputs go to `figures/`.

## License

MIT (see `LICENSE`).
