#!/bin/bash
# Deep orbit sampling (scripts/orbit_deep_shard.py) over the 16-visit deep table
# (resources/census/stage2_deep.csv, or $A4B_DEEP_TABLE set at submission).
# 100 fixed shards, strided by sdss_id. A shard writes its CSV only when it ends,
# so a timed-out shard must be rerun whole. Cost not measured yet: submit one
# task first (sbatch --array=0 scripts/orbit_deep.sbatch), read its CPU time from
# sacct, and scale before submitting the rest.
# Concatenate afterwards with
#   python scripts/concat_shards.py --shards resources/census/orbit_deep_shards --expect 100 \
#       --key sdss_id --check-deep --out resources/census/orbit_deep_posteriors.csv
#SBATCH --job-name=a4b_orbdeep
#SBATCH --account=PAS2823
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=03:00:00
#SBATCH --array=0-99
#SBATCH --output=logs/orbdeep_%A_%a.out
set -e
module load python/3.12 2>/dev/null || true
source "$HOME/a4b_venv/bin/activate"
cd "$HOME/agent4binary"
export OMP_NUM_THREADS=1
mkdir -p resources/census/orbit_deep_shards
python scripts/orbit_deep_shard.py --shard "$SLURM_ARRAY_TASK_ID" --n-shards 100 \
  --out "resources/census/orbit_deep_shards/shard_${SLURM_ARRAY_TASK_ID}.csv"
