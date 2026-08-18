# OCT1 structure-position triage (SPT)

Analysis code and frozen tables for:

> Kang J, Choi J. Structure-position triage of OCT1 (`SLC22A1`) missense variants with AlphaMissense, AlphaFold and deep mutational scanning. *The Pharmacogenomics Journal* (submitted).

This repository is the **code availability** archive for that article. It rebuilds the display items from frozen SPT tables. It is not a new variant-effect predictor.

## Authors

- Juwon Kang (0009-0009-2186-0038) — first author  
- Junjeong Choi (0000-0003-1339-593X) — corresponding author (`junjeong@yonsei.ac.kr`)  
  College of Pharmacy, Yonsei Institute of Pharmaceutical Sciences, Yonsei University, Incheon 21983, Republic of Korea

## What is included

- `prereg/` — SPT hypotheses P1–P6 locked on **12 August 2026** (before DMS–AM correlations), plus the uptake / residual / I1C notes used in the supplement
- `data/spt/` — frozen SPT labels, validation missense table (`n = 9711`), ΔΔG, literature set, and P1–P4 / P6 verdicts
- `data/challenge/` — residual-ensemble and I1C verdict JSON used for Supplementary Fig. 1 / Table S1
- Python scripts used to assign SPT classes, join DMS/AlphaMissense/ΔΔG, and run the locked tests
- `make_figures.py` — rebuild Fig. 1–5 and Supplementary Fig. 1 from the freeze

Held-out literature AlphaFold mutant models (preregistered P5) were **not completed** and are not in this archive.

## What is not redistributed

Yee et al. DMS scores, the AlphaMissense proteome table, AlphaFold2 PDBs, and experimental PDB files (8SC1, 8SC4, 8ET6, 8ET9) remain with their original sources. The freeze already contains the derived SPT tables needed to check the manuscript numbers.

Manuscript P4 uses **`data/spt/ms1_feedback1_freeze.json`** (`dms_loss` n = **485**, OR 1.39). Do **not** use `wp3_p1_p2_p4.json` `n_hit` = 493 (`func_loss`). Residue-clustered P4 sensitivity (CI includes 1) is in the same freeze file. ColabFold models used for SPT are protocol-specific and are **not** the AlphaFold DB entry; they are not redistributed.

Do not retune the SPT cuts (CORE rel.SASA &lt; 10%; EXPOSED rel.SASA &gt; 30% and extra-/cytoplasmic). Design-set positions 61 / 88 / 401 / 420 / 465 are rule-development only and are excluded from validation statistics.

## Rebuild figures

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python make_figures.py
```

Outputs go to `figures/` (PDF + TIFF). Fig. 2a bootstrap CIs are cached in `data/spt/ms1_figure_stats.json`.

## Environment for a full re-run

Full SPT from coordinates needs local AlphaFold2 models and PDB files. `env.sh` points `MET_HDD` at `./data` by default:

```bash
source env.sh
python met_classify.py
python met_dms.py
python met_p3.py
python make_figures.py
```

Scripts that need raw DMS or AlphaMissense files will look under `$MET_HDD/dms` and `$MET_HDD/alphamissense`. Those directories are empty here on purpose.

## License

MIT (see `LICENSE`).
