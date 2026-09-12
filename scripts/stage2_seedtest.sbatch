#!/bin/bash
# Seed test for the joint visit fit: 592 catalog SB2 (192 published SB2 and 200
# others whose corrected fit fell back to single with Delta chi2 = 0, plus 100
# confirmed and 100 unconfirmed controls), rerun with the two added seeds from
# the untied per-visit fits. ~20 CPU-h. 20 shards, per-star checkpoint.
#SBATCH --job-name=a4b_s2seed
#SBATCH --account=PAS2823
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=01:45:00
#SBATCH --array=0-19
#SBATCH --output=logs/s2seed_%A_%a.out
set -e
module load python/3.12 2>/dev/null || true
source "$HOME/a4b_venv/bin/activate"
cd "$HOME/agent4binary"
export OMP_NUM_THREADS=1
export A4B_S2_IDS=resources/census/stage2_seedtest_ids.txt
mkdir -p resources/census/stage2_seedtest_shards
python scripts/stage2_census_shard.py --shard "$SLURM_ARRAY_TASK_ID" --n-shards 20 \
  --out "resources/census/stage2_seedtest_shards/shard_${SLURM_ARRAY_TASK_ID}.csv"
