#!/bin/bash
#SBATCH --job-name=ecc_null
#SBATCH --account=PAS2823
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --time=01:00:00
#SBATCH --output=logs/ecc_null_%j.out
module load python/3.12 2>/dev/null || true
source "$HOME/a4b_venv/bin/activate"
export WORKERS=40 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p logs
cd "$HOME/agent4binary"
python scripts/ecc_null_full.py
