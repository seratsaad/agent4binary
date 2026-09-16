#!/bin/bash
# Visit-fit false-positive rate: the joint multi-epoch fit run on the benchmark
# single-star controls with two or more visits (1,643 stars; 1,151 of the 2,520
# staged controls are held-out controls, see resources/census/stage2_controls_split.csv),
# with the same fitter and settings as the catalog SB2 rerun (cap 8, seed fix,
# physics.py 3092c6a8d416). Visit files are staged in data/dr19_raw_visits, so no
# download. ~7,700 visits x 19 CPU-s = ~41 CPU-h. 60 shards of ~27 stars.
#SBATCH --job-name=a4b_s2ctl
#SBATCH --account=PAS2823
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=01:45:00
#SBATCH --array=0-59
#SBATCH --output=logs/s2ctl_%A_%a.out
set -e
module load python/3.12 2>/dev/null || true
source "$HOME/a4b_venv/bin/activate"
cd "$HOME/agent4binary"
export OMP_NUM_THREADS=1
export A4B_S2_IDS=resources/census/stage2_controls_ids.txt
mkdir -p resources/census/stage2_controls_shards
python scripts/stage2_census_shard.py --shard "$SLURM_ARRAY_TASK_ID" --n-shards 60 \
  --out "resources/census/stage2_controls_shards/shard_${SLURM_ARRAY_TASK_ID}.csv"
