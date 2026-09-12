#!/bin/bash
#SBATCH --job-name=a4b_s2pre
#SBATCH --account=PAS2823
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=03:00:00
#SBATCH --output=logs/s2pre_%j.out
set -e
module load python/3.12 2>/dev/null || true
source "$HOME/a4b_venv/bin/activate"
cd "$HOME/agent4binary"
export OMP_NUM_THREADS=1
export A4B_S2_IDS=resources/census/stage2_preflight60.txt
mkdir -p resources/census/stage2_preflight logs
python scripts/stage2_census_shard.py --shard 0 --n-shards 1 \
  --out resources/census/stage2_preflight/shard_0.csv
