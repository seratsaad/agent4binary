#!/bin/bash
# Corrected stage-2 run (rest-frame visits shifted to barycentric, epochs returned)
# on the 26,239 catalog SB2 with two or more epochs. Preflight job 54127600
# measured 14.5 CPU-s per fitted visit -> ~415 CPU-h. 400 shards of ~66 stars;
# each shard checkpoints per star, so a timeout can simply be resubmitted.
#SBATCH --job-name=a4b_s2fixed
#SBATCH --account=PAS2823
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=01:45:00
#SBATCH --array=0-399
#SBATCH --output=logs/s2fixed_%A_%a.out
set -e
module load python/3.12 2>/dev/null || true
source "$HOME/a4b_venv/bin/activate"
cd "$HOME/agent4binary"
export OMP_NUM_THREADS=1
export A4B_S2_IDS=resources/census/stage2_ids_catalog_multi.txt
mkdir -p resources/census/stage2_fixed_shards
python scripts/stage2_census_shard.py --shard "$SLURM_ARRAY_TASK_ID" --n-shards 400 \
  --out "resources/census/stage2_fixed_shards/shard_${SLURM_ARRAY_TASK_ID}.csv"
