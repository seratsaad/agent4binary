#!/bin/bash
#SBATCH --job-name=a4b_astrarv
#SBATCH --account=PAS2823
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:40:00
#SBATCH --output=logs/astrarv_%j.out
set -e
module load python/3.12 2>/dev/null || true
source "$HOME/a4b_venv/bin/activate"
cd "$HOME/agent4binary"
python scripts/stage2_astra_rv.py
