#!/bin/bash
# Injection validation of the eccentricity-difference estimator with twins
# defined by the dynamical mass ratio (scripts/ecc_val_suite.py --twin-label qdyn):
# the mock twin and non-twin populations draw q from the deep-table q_dyn of the
# systems with min(q_dyn, 1/q_dyn) > 0.95 and <= 0.95. Same 16 configurations as
# ecc_val_suite.sbatch (4 true differences x 4 realizations, 2 x 250 mocks at 60k
# draws per e-grid point). Measured cost of the best_q run: 80.4 CPU-h for 16
# tasks (4.6-5.4 h each); the 8 h limit leaves room, ceiling 128 CPU-h.
# Concatenate afterwards with
#   python scripts/concat_shards.py --shards resources/census/ecc_val_qdyn_shards --expect 16 \
#       --key da_true,rep --rows 16 --out resources/census/ecc_validation_qdyn.csv
#SBATCH --job-name=a4b_eccvalq
#SBATCH --account=PAS2823
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=08:00:00
#SBATCH --array=0-15
#SBATCH --output=logs/eccvalq_%A_%a.out
set -e
module load python/3.12 2>/dev/null || true
source "$HOME/a4b_venv/bin/activate"
cd "$HOME/agent4binary"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p resources/census/ecc_val_qdyn_shards
python scripts/ecc_val_suite.py --shard "$SLURM_ARRAY_TASK_ID" --n-shards 16 --twin-label qdyn \
  --out "resources/census/ecc_val_qdyn_shards/shard_${SLURM_ARRAY_TASK_ID}.csv"
