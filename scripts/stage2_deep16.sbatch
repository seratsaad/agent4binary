#!/bin/bash
# Deep refit for the eccentricity analysis: the 3,983 catalog SB2 with eight or
# more DR19 pipeline visits, refit with up to 16 visits (the highest S/N ones) by
# the fixed fitter (rest-frame visits, dates returned, seeds from the untied
# per-visit fits; physics.py 3092c6a8d416). Replaces the July deep table, which
# was made with the broken frame handling and had no visit dates. Cost preflight
# (jobs 54131879/80) measured <= ~20 CPU-s per visit at these visit counts, so
# ~49,600 visits come to ~275 CPU-h. Approved by the user on 2026-09-11.
# 250 shards of ~16 stars; per-star checkpoint, so a timeout can be resubmitted.
# Build the table afterwards with scripts/build_stage2_deep.py.
#SBATCH --job-name=a4b_deep16
#SBATCH --account=PAS2823
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=02:30:00
#SBATCH --array=0-249
#SBATCH --output=logs/deep16_%A_%a.out
set -e
module load python/3.12 2>/dev/null || true
source "$HOME/a4b_venv/bin/activate"
cd "$HOME/agent4binary"
export OMP_NUM_THREADS=1
export A4B_S2_IDS=resources/census/stage2_deep_ids.txt
mkdir -p resources/census/stage2_deep16_shards
python scripts/stage2_census_shard.py --shard "$SLURM_ARRAY_TASK_ID" --n-shards 250 --max-visits 16 \
  --out "resources/census/stage2_deep16_shards/shard_${SLURM_ARRAY_TASK_ID}.csv"
