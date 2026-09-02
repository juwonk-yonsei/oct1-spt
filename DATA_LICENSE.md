# Data licence

Code in this repository is MIT-licensed (`LICENSE`).

Frozen tables under `data/` contain values derived from third-party sources. Those derived columns are not covered by the MIT licence alone. Attribution and other terms of the original sources apply:

- **AlphaMissense** pathogenicity scores (Cheng et al., *Science* 2023), CC BY 4.0, DeepMind Technologies Limited. The public proteome table is not redistributed here; scores joined into the validation missense table (`data/spt/wp3_validation_missense.tsv`, column `am_pathogenicity`) and related freeze files are derived from that resource.
- **Yee et al.** OCT1 deep mutational scan (abundance / SM73 scores). Raw DMS files are not redistributed; derived columns (`GFP_score`, `SM73_*`) appear in the validation table.
- **gnomAD v4** allele frequencies for SLC22A1 missense. Counts in `data/spt/fb260901/addendum/gnomad_gfp_loss.tsv` are gnomAD variant records joined to the scan.
- **SERT / SLC6A4** surface-expression scores from Young et al. (GEO GSE109499), used to build `data/spt/fb260901/addendum5/SLC6A4_cutoff_sweep.tsv`.

Experimental PDB coordinates (8SC1, 8SC4, 8ET6, 8ET9) and ColabFold wild-type models are not in this repository. The latter are on Zenodo.
