#!/bin/bash
# Refit of the 141 catalog SB2 whose fitted visits included one with no DR19
# pipeline velocity. The fitter used to fit such a visit in the rest frame,
# unshifted, which added a spurious velocity jump; it now leaves the visit out
# (physics.dr19_visit_single_vs_binary). Same settings as stage2_seedfix.sbatch
# (cap 8 visits). Measured cost of that run: ~87 CPU-s per star, so ~3.4 CPU-h.
# Rebuild the supplement with A4B_S2_SHARDS=stage2_seedfix_shards and
# A4B_S2_OVERRIDE=stage2_framefix_shards.
#SBATCH --job-name=a4b_s2ff
#SBATCH --account=PAS2823
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --array=0-5
#SBATCH --output=logs/s2ff_%A_%a.out
set -e
module load python/3.12 2>/dev/null || true
source "$HOME/a4b_venv/bin/activate"
cd "$HOME/agent4binary"
export OMP_NUM_THREADS=1
export A4B_S2_IDS=resources/census/stage2_ids_framefix.txt
mkdir -p resources/census/stage2_framefix_shards
python scripts/stage2_census_shard.py --shard "$SLURM_ARRAY_TASK_ID" --n-shards 6 \
  --out "resources/census/stage2_framefix_shards/shard_${SLURM_ARRAY_TASK_ID}.csv"
