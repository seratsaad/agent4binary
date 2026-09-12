#!/bin/bash
# Injection validation of the eccentricity-difference estimator
# (scripts/ecc_val_suite.py) at the real deep-table cadences (or $A4B_DEEP_TABLE
# set at submission). 16 configurations (4 true differences x 4 realizations),
# one per task; each fits 2 x 250 mock systems at 60k draws per e-grid point and
# writes its CSV when it ends. Cost not measured yet: submit one task first
# (sbatch --array=0 ...), check sacct, then the rest.
# Concatenate afterwards with
#   python scripts/concat_shards.py --shards resources/census/ecc_val_shards --expect 16 \
#       --key da_true,rep --rows 16 --out resources/census/ecc_validation.csv
#SBATCH --job-name=a4b_eccval
#SBATCH --account=PAS2823
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=06:00:00
#SBATCH --array=0-15
#SBATCH --output=logs/eccval_%A_%a.out
set -e
module load python/3.12 2>/dev/null || true
source "$HOME/a4b_venv/bin/activate"
cd "$HOME/agent4binary"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p resources/census/ecc_val_shards
python scripts/ecc_val_suite.py --shard "$SLURM_ARRAY_TASK_ID" --n-shards 16 \
  --out "resources/census/ecc_val_shards/shard_${SLURM_ARRAY_TASK_ID}.csv"
