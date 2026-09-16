#!/bin/bash
# Independent false-positive test of the joint multi-epoch fit: 3,000 dwarfs drawn at
# random from the 238,205 searched, outside the SB2 catalog, outside the training set
# of the visit fit's single-star network (resources/dr19_raw_train_manifest.csv),
# outside the sigma_v-triaged candidate list and the benchmark. Visits are
# downloaded per shard; stars with fewer than two usable visits carry no test.
# Same fitter and settings as the catalog rerun (cap 8, seed fix, physics.py
# 3092c6a8d416). ~1,950 fitted stars x ~4.7 visits x 19 CPU-s = ~48 CPU-h.
#SBATCH --job-name=a4b_s2single
#SBATCH --account=PAS2823
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=01:45:00
#SBATCH --array=0-99
#SBATCH --output=logs/s2single_%A_%a.out
set -e
module load python/3.12 2>/dev/null || true
source "$HOME/a4b_venv/bin/activate"
cd "$HOME/agent4binary"
export OMP_NUM_THREADS=1
export A4B_S2_IDS=resources/census/stage2_singles_ids.txt
mkdir -p resources/census/stage2_singles_shards
python scripts/stage2_census_shard.py --shard "$SLURM_ARRAY_TASK_ID" --n-shards 100 \
  --out "resources/census/stage2_singles_shards/shard_${SLURM_ARRAY_TASK_ID}.csv"
