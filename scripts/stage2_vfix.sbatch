#!/bin/bash
# Refit of the stars whose visits include a failed DR19 pipeline velocity
# (|v_rad| > 400 km/s, up to ~1000 km/s). The fitter now leaves such visits out
# (physics.VRAD_OK_MAX), as the SB1 selection already did. 73 catalog SB2
# (stage2_ids_vfix.txt) and 3 benchmark controls + 3 random singles
# (stage2_ids_vfix_ctlsng.txt). Same settings as stage2_seedfix.sbatch (cap 8);
# ~87 CPU-s per star, ~2 CPU-h in all.
#SBATCH --job-name=a4b_s2vf
#SBATCH --account=PAS2823
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=01:30:00
#SBATCH --array=0-4
#SBATCH --output=logs/s2vf_%A_%a.out
set -e
module load python/3.12 2>/dev/null || true
source "$HOME/a4b_venv/bin/activate"
cd "$HOME/agent4binary"
export OMP_NUM_THREADS=1
if [ "$SLURM_ARRAY_TASK_ID" -lt 4 ]; then
  export A4B_S2_IDS=resources/census/stage2_ids_vfix.txt
  mkdir -p resources/census/stage2_vfix_shards
  python scripts/stage2_census_shard.py --shard "$SLURM_ARRAY_TASK_ID" --n-shards 4 \
    --out "resources/census/stage2_vfix_shards/shard_${SLURM_ARRAY_TASK_ID}.csv"
else
  export A4B_S2_IDS=resources/census/stage2_ids_vfix_ctlsng.txt
  mkdir -p resources/census/stage2_vfix_ctlsng_shards
  python scripts/stage2_census_shard.py --shard 0 --n-shards 1 \
    --out "resources/census/stage2_vfix_ctlsng_shards/shard_0.csv"
fi
