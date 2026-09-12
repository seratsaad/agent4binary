#!/bin/bash
# Per-system eccentricity likelihoods L(e) on the real deep sample
# (scripts/ecc_real_marginal.py; needs orbit_deep_posteriors.csv and the deep
# table, or $A4B_DEEP_TABLE set at submission). 40 fixed shards, 60k draws per
# e-grid point. A shard writes its CSV only when it ends. Cost not measured yet:
# submit one task first (sbatch --array=0 ...), check sacct, then the rest.
# Concatenate afterwards with
#   python scripts/concat_shards.py --shards resources/census/ecc_real_shards --expect 40 \
#       --key sdss_id --out resources/census/ecc_marginal_real.csv
#SBATCH --job-name=a4b_eccreal
#SBATCH --account=PAS2823
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=03:00:00
#SBATCH --array=0-39
#SBATCH --output=logs/eccreal_%A_%a.out
set -e
module load python/3.12 2>/dev/null || true
source "$HOME/a4b_venv/bin/activate"
cd "$HOME/agent4binary"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p resources/census/ecc_real_shards
python scripts/ecc_real_marginal.py --shard "$SLURM_ARRAY_TASK_ID" --n-shards 40 \
  --out "resources/census/ecc_real_shards/shard_${SLURM_ARRAY_TASK_ID}.csv"
