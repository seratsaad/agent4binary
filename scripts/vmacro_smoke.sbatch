#!/bin/bash
#SBATCH --job-name=a4b_vmsmoke
#SBATCH --account=PAS2823
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:40:00
#SBATCH --output=logs/vmsmoke_%j.out
set -e
module load python/3.12 2>/dev/null || true
source "$HOME/a4b_venv/bin/activate"
cd "$HOME/agent4binary"
export OMP_NUM_THREADS=1
export AB_MLP_NET=models/payne_dr19_sfbig_holdA.pt
python src/run_census_stage1.py --detector binspec_mlp --no-download \
    --manifest resources/census/vmacro_smoke_manifest.csv \
    --raw-dir /fs/scratch/PAS2823/a4b_census272k_raw \
    --out resources/census/vmacro_smoke.csv --workers 2 --checkpoint-every 500
