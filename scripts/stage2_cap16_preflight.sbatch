#!/bin/bash
# Cost preflight for fitting up to 16 visits (physics.py with the seed fix).
# Submitted twice, once per visit-count group, so sacct gives each group's cost:
#   sbatch --array=0-5 --export=ALL,GRP=hi  (60 catalog SB2 with >= 16 pipeline visits)
#   sbatch --array=0-3 --export=ALL,GRP=mid (40 with 9-15)
# ~15 CPU-h in all. 10 stars per shard, per-star checkpoint.
#SBATCH --job-name=a4b_cap16
#SBATCH --account=PAS2823
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=03:00:00
#SBATCH --output=logs/cap16_%x_%A_%a.out
set -e
module load python/3.12 2>/dev/null || true
source "$HOME/a4b_venv/bin/activate"
cd "$HOME/agent4binary"
export OMP_NUM_THREADS=1
export A4B_S2_IDS=resources/census/stage2_cap16_${GRP}_ids.txt
N=$(( SLURM_ARRAY_TASK_COUNT ))
mkdir -p resources/census/stage2_cap16_shards
python scripts/stage2_census_shard.py --shard "$SLURM_ARRAY_TASK_ID" --n-shards "$N" --max-visits 16 \
  --out "resources/census/stage2_cap16_shards/${GRP}_${SLURM_ARRAY_TASK_ID}.csv"
