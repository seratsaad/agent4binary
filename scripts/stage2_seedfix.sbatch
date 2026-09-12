#!/bin/bash
# Stage-2 rerun with the seed fix (physics.py 3092c6a8d416): the joint binary fit
# is also seeded from the untied per-visit two-component velocities, since without
# them 20.7% of the fits fell back to single with Delta chi2 = 0, including 7.7% of
# the catalog SB2 that published surveys also find double-lined. Seed test job
# 54131762 measured 16.5 CPU-s per fitted visit (9.9 CPU-h for 592 stars), so the
# 26,239 catalog SB2 with two or more epochs (102,777 fitted visits) come to
# ~470 CPU-h. Approved by the user on 2026-09-11 (full rerun, cap 8). 500 shards
# of ~53 stars (~60 min each); per-star checkpoint, so a timeout can be resubmitted.
#SBATCH --job-name=a4b_s2seedfix
#SBATCH --account=PAS2823
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --array=0-499
#SBATCH --output=logs/s2seedfix_%A_%a.out
set -e
module load python/3.12 2>/dev/null || true
source "$HOME/a4b_venv/bin/activate"
cd "$HOME/agent4binary"
export OMP_NUM_THREADS=1
export A4B_S2_IDS=resources/census/stage2_ids_catalog_multi.txt
mkdir -p resources/census/stage2_seedfix_shards
python scripts/stage2_census_shard.py --shard "$SLURM_ARRAY_TASK_ID" --n-shards 500 \
  --out "resources/census/stage2_seedfix_shards/shard_${SLURM_ARRAY_TASK_ID}.csv"
