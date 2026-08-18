#!/usr/bin/env bash
# Point MET_* paths at this repository's data/ freeze.
# Does not load a GPU/ColabFold stack.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MET_HDD="${MET_HDD:-${ROOT}/data}"
export MET_SPT="${MET_SPT:-${MET_HDD}/spt}"
export MET_STRUCT="${MET_STRUCT:-${MET_HDD}/structures}"
export MET_SEQ="${MET_SEQ:-${MET_HDD}/sequences}"
export MET_PDB="${MET_PDB:-${MET_HDD}/pdb}"
export MET_DMS="${MET_DMS:-${MET_HDD}/dms}"
export MET_AM="${MET_AM:-${MET_HDD}/alphamissense}"
export MET_DDG="${MET_DDG:-${MET_HDD}/ddg}"
echo "[oct1-spt] MET_HDD=${MET_HDD}"
